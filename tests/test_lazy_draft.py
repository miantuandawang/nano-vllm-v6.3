"""V6.1 admission and lazy Draft regressions, with physical/prefix-dependent KV."""
import unittest
import test_scheduler as helpers
from test_scheduler import FakeRunner, oracle
from nanovllm import EngineConfig, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler


class GuardedDraft(FakeRunner):
    def bind(self, scheduler):
        self.scheduler = scheduler
        self.executed_ids = []

    def forward(self, view, tokens):
        batch = self.scheduler.active_batch
        assert batch is not None, 'Draft ran before a speculative batch was selected'
        assert self.scheduler.last_step['mode'] == 'speculative'
        assert batch.limits[view.request_id] > 0, 'Draft ran for a Target-only request'
        assert view.request_id in [s.request_id for s in batch.seqs]
        self.executed_ids.append(view.request_id)
        return super().forward(view, tokens)


class LazyDraftTests(unittest.TestCase):
    def make(self, batch=16, threshold=16, resident=None, target_pages=128, draft_pages=128, budget=4096):
        config = EngineConfig(batch=batch, target_only_threshold=threshold, max_active_seqs=resident,
                              max_num_batched_tokens=budget)
        target = FakeRunner('target', target_pages)
        draft = GuardedDraft('draft', draft_pages, wrong=lambda t: len(t) % 7 == 0)
        scheduler = Scheduler(config, target, draft)
        draft.bind(scheduler)
        return scheduler, target, draft

    def drain(self, s):
        return helpers.SchedulerTests.run_all(self, s, max_steps=3000)

    def check_state(self, s):
        helpers.SchedulerTests.check_state(self, s)

    def test_120_prompts_admitted_in_dispatch_sized_batches_with_zero_draft_work(self):
        s, target, draft = self.make()
        requests = [Sequence([i + 1] * 8, SamplingParams(max_tokens=20)) for i in range(120)]
        for seq in requests:
            s.add(seq)
        while s.waiting:
            s.step()
            self.assertEqual(s.last_step['mode'], 'prefill')
            self.assertLessEqual(s.last_step['batch'], 16)
            self.check_state(s)
        self.assertEqual(len(s.ready), 120)
        self.assertEqual(s.stats['target_prefill_steps'], 8)
        self.assertEqual([len(call[1]) for call in target.calls], [16] * 7 + [8])
        self.assertEqual(draft.calls, [])
        self.assertTrue(all(not x.draft.block_table and x.draft.num_cached_tokens == 0 for x in requests))
        self.assertEqual(len(s.draft_blocks.free_block_ids), 128)
        s.step()
        self.assertEqual((s.last_step['batch'], s.last_step['k']), (16, 0))
        self.assertEqual(draft.calls, [])

    def test_target_only_entire_workload_never_reserves_or_runs_draft(self):
        s, _, draft = self.make(batch=4, threshold=1, budget=16)
        seqs = [Sequence([i + 1] * 20, SamplingParams(max_tokens=15)) for i in range(19)]
        for seq in seqs:
            s.add(seq)
        def forbidden(*args):
            raise AssertionError('Target-only tried to reserve Draft KV')
        s.draft_blocks.reserve = forbidden
        out = self.drain(s)
        self.assertEqual(draft.calls, [])
        self.assertEqual(s.stats['draft_prepare_tokens'], 0)
        for i, seq in enumerate(seqs):
            self.assertEqual(out[seq.request_id], oracle([i + 1] * 20, 15))

    def test_resident_cap_is_independent_of_dispatch_and_k(self):
        s, _, draft = self.make(batch=2, resident=5)
        for i in range(12):
            s.add(Sequence([i + 1], SamplingParams(max_tokens=10)))
        for _ in range(3):
            s.step()
        self.assertEqual(len(s.ready), 5)
        self.assertEqual(len(s.waiting), 7)
        self.assertEqual(draft.calls, [])
        s.step()
        self.assertEqual((s.last_step['batch'], s.last_step['k']), (2, 4))
        self.assertLessEqual(len(s.ready), 5)
        self.drain(s)

    def test_cold_tail_only_initializes_selected_survivors(self):
        s, _, draft = self.make(batch=16, budget=32)
        seqs = [Sequence([i + 1] * 3, SamplingParams(max_tokens=30)) for i in range(32)]
        for seq in seqs:
            s.add(seq)
        while s.waiting:
            s.step()
        for _ in range(6):
            s.step()
            self.assertEqual(s.last_step['mode'], 'target_only')
        self.assertEqual(draft.calls, [])
        for seq in seqs[4:]:
            s.cancel(seq.request_id)
        s.step()
        self.assertEqual((s.last_step['batch'], s.last_step['k']), (4, 3))
        self.assertEqual(set(draft.executed_ids), {s.request_id for s in seqs[:4]})
        self.assertEqual(s.stats['draft_prefill_tokens'], 12)
        self.assertGreater(s.stats['draft_sync_tokens'], 0)
        self.assertTrue(all(sum(call[1]) <= 32 for call in draft.calls))
        self.check_state(s)

    def test_one_token_cold_prompt_uses_kv_only_append_before_decode(self):
        s, _, draft = self.make(batch=1)
        seq = Sequence([7], SamplingParams(max_tokens=12))
        s.add(seq); s.step()
        self.assertEqual(draft.calls, [])
        self.assertEqual(seq.draft.num_cached_tokens, 0)
        s.step()
        self.assertEqual(draft.calls[0], ('append', [1], False))
        self.assertEqual(draft.calls[1], ('next', [1]))
        self.check_state(s)

    def test_kv_fallback_happens_before_any_draft_prefill(self):
        s, _, draft = self.make(batch=2, draft_pages=1)
        for i in range(2):
            s.add(Sequence([i + 1] * 257, SamplingParams(max_tokens=8)))
        s.step(); s.step()
        self.assertEqual(s.last_step['mode'], 'target_only')
        self.assertEqual(s.stats['kv_fallback_rounds'], 1)
        self.assertEqual(draft.calls, [])
        self.assertEqual(len(s.draft_blocks.free_block_ids), 1)
        self.drain(s)

    def test_prompt_completion_without_speculation_never_initializes_draft(self):
        for limit in (0, 1, 2):
            s, _, draft = self.make(batch=1)
            seq = Sequence([3], SamplingParams(max_tokens=limit))
            s.add(seq)
            self.drain(s)
            self.assertEqual(draft.calls, [])

    def test_partial_cold_sync_failure_restores_empty_draft_and_retries(self):
        s, _, draft = self.make(batch=1, budget=8)
        seq = Sequence([4] * 19, SamplingParams(max_tokens=12))
        s.add(seq)
        while s.waiting:
            s.step()
        before = list(seq.token_ids)
        original = draft.append
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 2:
                raise RuntimeError('injected sync failure')
            return result
        draft.append = fail_second
        with self.assertRaisesRegex(RuntimeError, 'sync failure'):
            s.step()
        self.assertEqual(seq.token_ids, before)
        self.assertEqual(seq.draft.num_cached_tokens, 0)
        self.assertEqual(seq.draft.block_table, [])
        self.assertEqual(s.stats['draft_prepare_tokens'], 0)
        draft.append = original
        self.assertEqual(self.drain(s)[seq.request_id], oracle([4] * 19, 12))

    def test_memory_bounded_admission_does_not_deadlock_on_page_boundary(self):
        s, _, draft = self.make(batch=2, threshold=1, target_pages=4, budget=512)
        rows = [Sequence([2] * 253, SamplingParams(max_tokens=14)) for _ in range(8)]
        for row in rows:
            s.add(row)
        s.step(); s.step()
        self.assertEqual(len(s.ready), 4)
        self.assertEqual(len(s.waiting), 4)
        out = self.drain(s)
        self.assertEqual(len(out), 8)
        self.assertGreater(s.stats['preemptions'], 0)
        self.assertEqual(draft.calls, [])
        for row in rows:
            self.assertEqual(out[row.request_id], oracle([2] * 253, 14))

    def test_invalid_resident_cap_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                EngineConfig(max_active_seqs=value)


if __name__ == '__main__':
    unittest.main()
