"""One synchronous scheduler owns both models, queues and KV transactions."""
from collections import Counter, deque
from dataclasses import dataclass

from nanovllm.engine.block_manager import BlockManager, reserve_atomic
from nanovllm.engine.sequence import Sequence, SequenceStatus as Status


@dataclass
class ActiveBatch:
    seqs: list
    k: int
    limits: dict
    snapshots: dict


class Scheduler:
    def __init__(self, config, target_runner, draft_runner, eos=()):
        self.config = config
        self.target_runner, self.draft_runner = target_runner, draft_runner
        self.target_blocks = BlockManager(target_runner.num_blocks, config.kvcache_block_size)
        self.draft_blocks = BlockManager(draft_runner.num_blocks, config.kvcache_block_size)
        self.eos = {eos} if isinstance(eos, int) else set(eos)
        self.requests = {}
        self.waiting, self.ready = deque(), deque()
        self.draft_sync_pending, self.verify_pending = deque(), deque()
        self.finished_outputs = []
        self.active_batch = None
        self.stats = Counter()
        self.last_step = {}

    def add(self, seq: Sequence):
        if not seq.token_ids:
            raise ValueError('prompt must contain at least one token')
        if len(seq) > self.config.max_model_len:
            raise ValueError('prompt exceeds max_model_len')
        capacity = self.target_blocks.num_blocks * self.config.kvcache_block_size
        if min(len(seq) + seq.max_tokens, self.config.max_model_len) > capacity:
            raise ValueError('one request cannot fit the Target KV pool; reduce its length or increase KV')
        self.requests[seq.request_id] = seq
        if seq.is_done(self.eos, self.config.max_model_len):
            self._finish(seq)
        else:
            self.waiting.append(seq)
        return seq.request_id

    def is_finished(self):
        return not self.requests

    def cancel(self, request_id):
        if self.active_batch is not None:
            raise RuntimeError('cancel is supported between synchronous step() calls')
        seq = self.requests.get(request_id)
        if seq is not None:
            seq.finish_reason = 'cancelled'
            self._finish(seq)

    def _finish(self, seq):
        for queue in (self.waiting, self.ready, self.draft_sync_pending):
            if seq in queue:
                queue.remove(seq)
        self.target_blocks.deallocate(seq.target)
        self.draft_blocks.deallocate(seq.draft)
        seq.draft_tokens.clear()
        seq.status = Status.FINISHED
        self.finished_outputs.append((seq.request_id, list(seq.completion_token_ids)))
        self.requests.pop(seq.request_id, None)

    def _preempt(self, seq):
        assert seq.status == Status.READY and not seq.draft_tokens
        self.ready.remove(seq)
        self.target_blocks.deallocate(seq.target)
        self.draft_blocks.deallocate(seq.draft)
        seq.prefill_end = len(seq) - 1
        seq.status = Status.WAITING
        self.waiting.append(seq)
        self.stats['preemptions'] += 1

    def _prefill_tick(self):
        """Batch Target prefill up to dispatch/token limits; Draft stays untouched.

        Prefill has native-style priority until residency/KV prevents admission.
        Reserve only requests selected for this dispatch, never the whole queue.
        """
        active = [s for s in self.waiting if s.status == Status.PREFILL]
        resident = len(active) + len(self.ready)
        capacity = self.config.max_active_seqs
        budget = self.config.max_num_batched_tokens
        rows = []
        for seq in list(self.waiting):
            if len(rows) >= self.config.batch or budget == 0:
                break
            cost = seq.prefill_end - seq.target.num_cached_tokens
            # Batch full prompts; only split a prompt when it is the first row.
            if rows and cost > budget:
                break
            if seq.status == Status.WAITING:
                if capacity is not None and resident >= capacity:
                    break
                if not self.target_blocks.reserve(seq.target, seq.prefill_end):
                    break
                seq.status = Status.PREFILL
                resident += 1
            end = seq.target.num_cached_tokens + min(cost, budget)
            rows.append((seq, end))
            budget -= end - seq.target.num_cached_tokens
        if not rows:
            return False
        groups = {False: [], True: []}
        for seq, end in rows:
            project = not seq.initialized and end == seq.prefill_end
            groups[project].append((seq, end))
        scheduled_tokens = 0
        for project, group in groups.items():
            if not group:
                continue
            results = self.target_runner.append([s.view('target') for s, _ in group],
                [s.token_ids[:end] for s, end in group], project=project)
            for (seq, end), prediction in zip(group, results):
                cost = end - seq.target.num_cached_tokens
                scheduled_tokens += cost
                self.stats['target_prefill_tokens'] += cost
                seq.target.num_cached_tokens = end
                if project:
                    seq.token_ids.append(prediction[-1])
                    seq.initialized = True
                    self.stats['output_tokens'] += 1
                    if seq.is_done(self.eos, self.config.max_model_len):
                        self._finish(seq)
        for seq, end in rows:
            if seq.request_id not in self.requests:
                continue
            if seq.target.num_cached_tokens == seq.prefill_end:
                self.waiting.remove(seq)
                seq.status = Status.READY
                self.ready.append(seq)
        self.stats['target_prefill_steps'] += 1
        self.stats['peak_resident_requests'] = max(self.stats['peak_resident_requests'], resident)
        self.last_step = {'mode': 'prefill', 'batch': len(rows), 'scheduled_tokens': scheduled_tokens,
                          'ready_requests': len(self.ready)}
        return True

    def _select(self):
        budget = self.config.max_num_batched_tokens
        seqs = list(self.ready)[:min(self.config.batch, budget)]
        while seqs:
            k = self.config.choose_k(len(seqs))
            limits = {s.request_id: min(k, max(0, s.remaining_tokens - 1),
                                       max(0, self.config.max_model_len - len(s) - 1),
                                       max(0, budget - 1)) for s in seqs}
            if sum(limits[s.request_id] + 1 for s in seqs) > budget:
                seqs.pop()
                continue
            minimal = [(self.target_blocks, s.target, len(s)) for s in seqs]
            if reserve_atomic(minimal):
                break
            # Preempt the youngest ready request, only at a committed boundary.
            victim = self.ready[-1]
            self._preempt(victim)
            seqs = [s for s in seqs if s is not victim]
        if not seqs:
            return None
        snapshots = {s.request_id: (len(s), s.target.num_cached_tokens,
                                    s.draft.num_cached_tokens, s.finish_reason) for s in seqs}
        speculative = [s for s in seqs if limits[s.request_id]]
        reservation = []
        for seq in speculative:
            m = limits[seq.request_id]
            reservation.extend([(self.target_blocks, seq.target, len(seq) + m),
                                (self.draft_blocks, seq.draft, len(seq) + m - 1)])
        if not reserve_atomic(reservation):
            # Keep minimal Target reservation and continue without Draft this round.
            limits = {s.request_id: 0 for s in seqs}
            self.stats['kv_fallback_rounds'] += 1
            k = 0
        for seq in seqs:
            self.ready.remove(seq)
        return ActiveBatch(seqs, k, limits, snapshots)

    def _sync_draft(self, seqs):
        # Called only AFTER final B/K/limits and both pools' reservation succeed.
        assert self.active_batch is not None
        assert all(self.active_batch.limits[s.request_id] > 0 for s in seqs)
        for seq in seqs:
            # Even a one-token prompt must first build a real cached prefix.
            # Missing=1/2 graphs are reserved for requests with existing Draft KV.
            if seq.draft.num_cached_tokens == 0 or len(seq) - seq.draft.num_cached_tokens > 2:
                seq.status = Status.DRAFT_SYNC
                self.draft_sync_pending.append(seq)
        while self.draft_sync_pending:
            budget = self.config.max_num_batched_tokens
            rows = []
            for seq in self.draft_sync_pending:
                end = min(len(seq) - 1, seq.draft.num_cached_tokens + budget)
                rows.append((seq, end))
                budget -= end - seq.draft.num_cached_tokens
                if budget == 0:
                    break
            self.draft_runner.append([s.view('draft') for s, _ in rows],
                                     [s.token_ids[:end] for s, end in rows], project=False)
            for seq, end in rows:
                start = seq.draft.num_cached_tokens
                prompt_tokens = max(0, min(end, seq.num_prompt_tokens) - start)
                self.stats['draft_prefill_tokens'] += prompt_tokens
                self.stats['draft_sync_tokens'] += end - start - prompt_tokens
                self.stats['draft_prepare_tokens'] += end - start
                seq.draft.num_cached_tokens = end
                if end == len(seq) - 1:
                    self.draft_sync_pending.remove(seq)
            self.stats['draft_prepare_calls'] += 1

    def _draft(self, seqs, limits):
        self._sync_draft(seqs)
        active = list(seqs)
        while active:
            for seq in active:
                seq.status = Status.DRAFT
            token_lists = [s.token_ids + s.draft_tokens for s in active]
            predictions = self.draft_runner.next_tokens([s.view('draft') for s in active], token_lists)
            following = []
            for seq, tokens, token in zip(active, token_lists, predictions):
                seq.draft.num_cached_tokens = len(tokens)
                seq.draft_tokens.append(token)
                self.stats['draft_candidates'] += 1
                if (len(seq.draft_tokens) < limits[seq.request_id]
                        and (seq.ignore_eos or token not in self.eos)):
                    following.append(seq)
            active = following

    def _verify_commit(self, seqs):
        self.verify_pending.append(seqs)
        for seq in seqs:
            seq.status = Status.VERIFY
        predictions = self.target_runner.verify([s.view('target') for s in seqs])
        for seq, predicted in zip(seqs, predictions):
            old_len = len(seq)
            candidates = seq.draft_tokens
            if len(predicted) != len(candidates) + 1:
                raise RuntimeError('Verify must return one prediction per K+1 query position')
            seq.target.num_cached_tokens = old_len + len(candidates)
            accepted = 0
            stopped = False
            for candidate, token in zip(candidates, predicted):
                if candidate != token:
                    break
                seq.token_ids.append(candidate)
                accepted += 1
                if seq.is_done(self.eos, self.config.max_model_len):
                    stopped = True
                    break
            if not stopped:
                seq.token_ids.append(predicted[accepted])
            # Rejection is committed NOW, before ready/target-only can see this request.
            self.target_blocks.truncate(seq.target, old_len + accepted)
            self.draft_blocks.truncate(seq.draft, old_len + accepted)
            self.stats['accepted_candidates'] += accepted
            self.stats['output_tokens'] += len(seq) - old_len
            seq.draft_tokens = []
        self.verify_pending.popleft()

    def _target_decode(self, seqs):
        if not seqs:
            return
        predictions = self.target_runner.next_tokens([s.view('target') for s in seqs],
                                                     [s.token_ids for s in seqs])
        for seq, token in zip(seqs, predictions):
            seq.target.num_cached_tokens = len(seq)
            seq.token_ids.append(token)
        self.stats['target_decode_tokens'] += len(seqs)
        self.stats['output_tokens'] += len(seqs)

    def _decode_tick(self):
        batch = self._select()
        if batch is None:
            return False
        self.active_batch = batch
        speculative = [s for s in batch.seqs if batch.limits[s.request_id]]
        direct = [s for s in batch.seqs if not batch.limits[s.request_id]]
        mode = 'speculative' if speculative else 'target_only'
        self.last_step = {'mode': mode, 'batch': len(batch.seqs), 'k': batch.k,
                          'request_ids': [s.request_id for s in batch.seqs]}
        stats_before = self.stats.copy()
        self.stats[mode + '_rounds'] += 1
        try:
            if speculative:
                self._draft(speculative, batch.limits)
                self._verify_commit(speculative)
            self._target_decode(direct)
        except Exception:
            # Preserve confirmed pre-round tokens. Partially written speculative KV is invalid.
            for seq in batch.seqs:
                length, target_len, draft_len, finish_reason = batch.snapshots[seq.request_id]
                del seq.token_ids[length:]
                seq.finish_reason = finish_reason
                self.target_blocks.truncate(seq.target, target_len)
                self.draft_blocks.truncate(seq.draft, draft_len)
                seq.draft_tokens.clear()
                seq.status = Status.READY
                self.ready.append(seq)
            self.draft_sync_pending.clear()
            self.verify_pending.clear()
            self.stats = stats_before
            raise
        finally:
            self.active_batch = None
        for seq in batch.seqs:
            if seq.is_done(self.eos, self.config.max_model_len):
                self._finish(seq)
            else:
                assert seq.target.num_cached_tokens == len(seq) - 1
                assert 0 <= seq.draft.num_cached_tokens <= len(seq) - 1
                seq.status = Status.READY
                self.ready.append(seq)
        return True

    def step(self):
        before = self.stats['output_tokens']
        progressed = False
        if self.waiting:
            progressed = self._prefill_tick()
        if not progressed and self.ready:
            progressed = self._decode_tick()
        if not progressed and self.waiting and not self.ready:
            # _select may have preempted the final ready request to make room.
            progressed = self._prefill_tick()
        if not progressed and self.requests and not self.ready:
            raise RuntimeError('no request can make progress within the KV budget')
        outputs, self.finished_outputs = self.finished_outputs, []
        return outputs, self.stats['output_tokens'] - before
