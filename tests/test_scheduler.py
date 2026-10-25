"""CPU state-machine tests with prefix-dependent, physically paged fake KV.

These tests exercise production Scheduler/BlockManager, not a duplicate scheduler.
They do not establish numerical CUDA correctness.
"""
import random
import unittest
from nanovllm import EngineConfig, SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.block_manager import BlockManager, reserve_atomic


def fingerprint(tokens):
    state = 7
    for token in tokens:
        state = (state * 101 + token + 1) % 1000000007
    return state


def predict(tokens):
    return fingerprint(tokens) % 97 + 1


def oracle(prompt, count, eos=(), ignore_eos=False, context=4096):
    tokens = list(prompt)
    out = []
    for _ in range(min(count, context - len(prompt))):
        token = predict(tokens)
        out.append(token)
        tokens.append(token)
        if token in eos and not ignore_eos:
            break
    return out


class FakeRunner:
    def __init__(self, role, blocks=128, block_size=256, wrong=None):
        self.role, self.num_blocks, self.block_size = role, blocks, block_size
        self.wrong = wrong or (lambda tokens: False)
        self.memory, self.calls = {}, []
        self.fail_verify = False

    def check(self, view, tokens, end=None):
        end = view.num_cached_tokens if end is None else end
        for pos in range(end):
            slot = view.block_table[pos // self.block_size] * self.block_size + pos % self.block_size
            expected = (tokens[pos], fingerprint(tokens[:pos + 1]))
            assert self.memory.get(slot) == expected, (self.role, view.request_id, pos, expected, self.memory.get(slot))

    def forward(self, view, tokens):
        assert 0 <= view.num_cached_tokens < len(tokens)
        self.check(view, tokens)
        out = []
        for pos in range(view.num_cached_tokens, len(tokens)):
            slot = view.block_table[pos // self.block_size] * self.block_size + pos % self.block_size
            self.memory[slot] = (tokens[pos], fingerprint(tokens[:pos + 1]))
            token = predict(tokens[:pos + 1])
            if self.role == 'draft' and self.wrong(tokens[:pos + 1]):
                token = token % 97 + 1
            out.append(token)
        return out

    def append(self, views, token_lists, project=False):
        self.calls.append(('append', [len(t) - v.num_cached_tokens for v, t in zip(views, token_lists)], project))
        out = [self.forward(v, t) for v, t in zip(views, token_lists)]
        return [[p[-1]] if project else [] for p in out]

    def next_tokens(self, views, token_lists):
        self.calls.append(('next', [len(t) - v.num_cached_tokens for v, t in zip(views, token_lists)]))
        return [self.forward(v, t)[-1] for v, t in zip(views, token_lists)]

    def verify(self, views):
        self.calls.append(('verify', [len(v.draft_tokens) + 1 for v in views]))
        results = [self.forward(v, v.token_ids + v.draft_tokens) for v in views]
        if self.fail_verify:
            self.fail_verify = False
            raise RuntimeError('injected failure after speculative KV writes')
        return results


class SchedulerTests(unittest.TestCase):
    def make(self, batch=16, threshold=16, wrong=None, budget=4096, blocks=128, draft_blocks=None, context=4096):
        c = EngineConfig(batch=batch, target_only_threshold=threshold,
                         max_num_batched_tokens=budget, max_model_len=context)
        t = FakeRunner('target', blocks)
        d = FakeRunner('draft', blocks if draft_blocks is None else draft_blocks, wrong=wrong)
        return Scheduler(c, t, d), t, d

    def check_state(self, scheduler):
        if scheduler.config.max_active_seqs is not None:
            self.assertLessEqual(len(scheduler.ready) + sum(s.status == SequenceStatus.PREFILL
                                 for s in scheduler.waiting), scheduler.config.max_active_seqs)
        for seq in scheduler.ready:
            self.assertEqual(seq.target.num_cached_tokens, len(seq) - 1)
            self.assertLessEqual(seq.draft.num_cached_tokens, len(seq) - 1)
            self.assertEqual(seq.draft_tokens, [])
            scheduler.target_runner.check(seq.view('target'), seq.token_ids)
            scheduler.draft_runner.check(seq.view('draft'), seq.token_ids)
        for role in ('target', 'draft'):
            manager = getattr(scheduler, role + '_blocks')
            used = [page for seq in scheduler.requests.values() for page in getattr(seq, role).block_table]
            free = list(manager.free_block_ids)
            self.assertEqual(len(set(used + free)), manager.num_blocks)
            self.assertEqual(len(used + free), manager.num_blocks)

    def run_all(self, scheduler, max_steps=1000):
        outputs = {}
        for _ in range(max_steps):
            done, _ = scheduler.step()
            outputs.update(done)
            self.check_state(scheduler)
            if scheduler.is_finished():
                return outputs
        self.fail('scheduler failed to complete')

    def test_dynamic_policy(self):
        c = EngineConfig()
        self.assertEqual([c.choose_k(b) for b in [1, 2, 3, 4, 5, 8, 9, 15, 16]], [4, 4, 3, 3, 2, 2, 1, 1, 0])

    def test_all_accept_uses_two_token_catchup(self):
        s, t, d = self.make(batch=1)
        seq = Sequence([1, 3], SamplingParams(max_tokens=20))
        s.add(seq)
        s.step()
        s.step()
        self.assertEqual(len(seq) - seq.draft.num_cached_tokens, 2)
        self.assertEqual(seq.target.num_cached_tokens, len(seq) - 1)
        s.step()
        self.assertIn(('next', [2]), d.calls)
        self.check_state(s)

    def test_first_and_middle_rejections_match_target(self):
        for wrong in (lambda t: True, lambda t: len(t) % 3 == 0):
            s, _, _ = self.make(batch=4, wrong=wrong)
            rows = [Sequence([i + 1, 9], SamplingParams(max_tokens=25)) for i in range(4)]
            for row in rows:
                s.add(row)
            out = self.run_all(s)
            for row in rows:
                self.assertEqual(out[row.request_id], oracle(row.token_ids[:2], 25))

    def test_target_only_to_speculative_long_chunked_sync(self):
        s, _, d = self.make(batch=16, budget=16)
        rows = [Sequence([2, 4], SamplingParams(max_tokens=40)) for _ in range(16)]
        for row in rows:
            s.add(row)
        while len(s.ready) != 16:
            s.step()
        for _ in range(7):
            s.step()
            self.assertEqual(s.last_step['mode'], 'target_only')
        cached = rows[0].draft.num_cached_tokens
        for row in rows[2:]:
            s.cancel(row.request_id)
        self.assertEqual(rows[0].draft.num_cached_tokens, cached)
        s.step()
        self.assertEqual(s.last_step['k'], 4)
        self.assertGreater(s.stats['draft_sync_tokens'], 0)
        self.assertTrue(all(sum(call[1]) <= 16 for call in d.calls))
        self.check_state(s)

    def test_both_switch_directions_after_accept_and_reject(self):
        for wrong in (lambda t: False, lambda t: True):
            s, _, _ = self.make(batch=4, threshold=4, wrong=wrong)
            first = Sequence([3, 7], SamplingParams(max_tokens=40))
            s.add(first)
            s.step(); s.step()
            self.check_state(s)
            added = [Sequence([6], SamplingParams(max_tokens=40)) for _ in range(3)]
            for row in added:
                s.add(row)
            while len(s.ready) < 4:
                s.step()
            for _ in range(4):
                s.step()
                self.assertEqual(s.last_step['mode'], 'target_only')
            for row in added:
                s.cancel(row.request_id)
            s.step()
            self.assertEqual(s.last_step['mode'], 'speculative')
            self.check_state(s)
            result = self.run_all(s)
            self.assertEqual(result[first.request_id], oracle([3, 7], 40))

    def test_eos_and_remaining_output_limits(self):
        for ignore in (False, True):
            for wrong in (lambda t: False, lambda t: True):
                for count in (0, 1, 2, 5, 8):
                    s, _, _ = self.make(batch=1, wrong=wrong)
                    eos = predict([3, 4, predict([3, 4])])
                    s.eos = {eos}
                    row = Sequence([3, 4], SamplingParams(max_tokens=count, ignore_eos=ignore))
                    s.add(row)
                    out = self.run_all(s)
                    self.assertEqual(out[row.request_id], oracle([3, 4], count, {eos}, ignore))

    def test_unverified_draft_eos_does_not_finish_request(self):
        s, _, _ = self.make(batch=1, wrong=lambda t: True)
        prompt = [3, 4]
        first = predict(prompt)
        draft_eos = predict(prompt + [first]) % 97 + 1
        s.eos = {draft_eos}
        row = Sequence(prompt, SamplingParams(max_tokens=12))
        s.add(row)
        out = self.run_all(s)
        self.assertEqual(out[row.request_id], oracle(prompt, 12, {draft_eos}))

    def test_cancellation_releases_both_caches_during_prefill(self):
        s, _, _ = self.make(batch=1, budget=3)
        row = Sequence([2] * 20, SamplingParams(max_tokens=10))
        s.add(row); s.step()
        self.assertEqual(row.status, SequenceStatus.PREFILL)
        s.cancel(row.request_id)
        self.assertTrue(s.is_finished())
        self.check_state(s)

    def test_request_batch_cap_and_per_call_token_budget(self):
        s, target, draft = self.make(batch=4, budget=16)
        for i in range(17):
            s.add(Sequence([i + 1] * 19, SamplingParams(max_tokens=14)))
        self.run_all(s)
        for runner in (target, draft):
            for call in runner.calls:
                self.assertLessEqual(len(call[1]), 4)
                self.assertLessEqual(sum(call[1]), 16)

    def test_context_limit_and_chunked_prefill(self):
        s, _, _ = self.make(batch=3, budget=3, context=20)
        rows = [Sequence([i + 1] * 11, SamplingParams(max_tokens=50)) for i in range(3)]
        for row in rows:
            s.add(row)
        out = self.run_all(s)
        for row in rows:
            self.assertEqual(out[row.request_id], oracle(row.token_ids[:11], 50, context=20))

    def test_cross_page_rejection_and_reused_pages(self):
        s, _, _ = self.make(batch=2, wrong=lambda t: len(t) % 3 == 0, blocks=4, budget=128, context=512)
        rows = [Sequence([1, 2] * 126 + [7], SamplingParams(max_tokens=20)) for _ in range(5)]
        for row in rows:
            s.add(row)
        out = self.run_all(s)
        for row in rows:
            self.assertEqual(out[row.request_id], oracle([1, 2] * 126 + [7], 20, context=512))

    def test_draft_capacity_falls_back_without_stalling(self):
        s, _, _ = self.make(batch=2, blocks=8, draft_blocks=1, context=512)
        rows = [Sequence([1] * 253, SamplingParams(max_tokens=15)) for _ in range(2)]
        for row in rows:
            s.add(row)
        out = self.run_all(s)
        self.assertGreater(s.stats['kv_fallback_rounds'], 0)
        for row in rows:
            self.assertEqual(out[row.request_id], oracle([1] * 253, 15))

    def test_target_preemption_rebuild_does_not_repeat_output(self):
        s, _, _ = self.make(batch=2, threshold=1, blocks=3, context=512)
        rows = [Sequence([1] * 253, SamplingParams(max_tokens=20)) for _ in range(2)]
        for row in rows:
            s.add(row)
        out = self.run_all(s)
        self.assertGreater(s.stats['preemptions'], 0)
        for row in rows:
            self.assertEqual(out[row.request_id], oracle([1] * 253, 20))

    def test_failed_verify_restores_lengths_and_tokens(self):
        s, target, _ = self.make(batch=1)
        row = Sequence([4, 6], SamplingParams(max_tokens=20))
        s.add(row); s.step()
        before = (list(row.token_ids), row.target.num_cached_tokens, row.draft.num_cached_tokens)
        target.fail_verify = True
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            s.step()
        self.assertEqual((row.token_ids, row.target.num_cached_tokens, row.draft.num_cached_tokens), before)
        self.assertIsNone(s.active_batch)
        self.check_state(s)
        self.assertEqual(self.run_all(s)[row.request_id], oracle([4, 6], 20))

    def test_randomized_workloads_match_independent_target_oracle(self):
        rng = random.Random(600)
        for trial in range(60):
            batch = rng.choice([1, 2, 3, 4, 5, 8, 9, 16])
            s, _, _ = self.make(batch=batch, threshold=rng.choice([1, 4, 8, 16, 17]),
                                budget=rng.choice([7, 32, 64]),
                                wrong=lambda t: fingerprint(t) % 5 == 0, context=128)
            expected = {}
            for _ in range(rng.randint(1, 24)):
                prompt = [rng.randint(1, 50) for _ in range(rng.randint(1, 20))]
                count = rng.randint(0, 35)
                row = Sequence(prompt, SamplingParams(max_tokens=count))
                s.add(row)
                expected[row.request_id] = oracle(prompt, count, context=128)
            self.assertEqual(self.run_all(s), expected, trial)


class ReservationTests(unittest.TestCase):
    def test_atomic_failure_restores_both_pools_and_existing_cache(self):
        from nanovllm.engine.sequence import CacheState
        target, draft = BlockManager(3, 4), BlockManager(1, 4)
        tc, dc = CacheState(), CacheState()
        target.reserve(tc, 4); draft.reserve(dc, 4)
        tc.num_cached_tokens = dc.num_cached_tokens = 3
        before = (list(tc.block_table), list(dc.block_table), len(target.free_block_ids))
        self.assertFalse(reserve_atomic([(target, tc, 10), (draft, dc, 5)]))
        self.assertEqual((tc.block_table, dc.block_table, len(target.free_block_ids)), before)
        self.assertEqual((tc.num_cached_tokens, dc.num_cached_tokens), (3, 3))


if __name__ == '__main__':
    unittest.main()
