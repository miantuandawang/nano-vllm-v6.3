"""A100 V6.1 regression: resident cap, zero-Draft Target-only, cold tail sync.

No throughput claim: assertions and same-shape eager comparisons are enabled.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-model', required=True)
    parser.add_argument('--draft-model', required=True)
    parser.add_argument('--output', default='validation-lazy-cuda.json')
    args = parser.parse_args()
    import torch
    from nanovllm import LLM, EngineConfig, SamplingParams
    from nanovllm.engine.scheduler import Scheduler
    from validate_cuda import uncaptured, compare_outputs
    if hasattr(torch._dynamo.config, 'recompile_limit'):
        torch._dynamo.config.recompile_limit = 64
    else:
        torch._dynamo.config.cache_size_limit = 64
    config = EngineConfig(target_model=args.target_model, draft_model=args.draft_model,
                          batch=16, max_model_len=512, max_num_batched_tokens=512)
    report = {'version': '0.6.1', 'status': 'running', 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(), 'timed_benchmark': False}
    try:
        with torch.inference_mode(), LLM(config) as llm:
            original_append = llm.draft_runner.append
            original_next = llm.draft_runner.next_tokens
            observed_ids = set()
            def guard(views):
                batch = llm.scheduler.active_batch
                assert batch is not None
                assert llm.scheduler.last_step['mode'] == 'speculative'
                assert all(batch.limits[v.request_id] > 0 for v in views)
                observed_ids.update(v.request_id for v in views)
            def append(views, *a, **kw):
                guard(views)
                return original_append(views, *a, **kw)
            def next_tokens(views, *a, **kw):
                guard(views)
                return original_next(views, *a, **kw)
            llm.draft_runner.append, llm.draft_runner.next_tokens = append, next_tokens
            def run(threshold):
                llm.config.target_only_threshold = threshold
                llm.scheduler = Scheduler(llm.config, llm.target_runner, llm.draft_runner, llm.scheduler.eos)
                llm.target_runner.stats.clear(); llm.draft_runner.stats.clear()
                observed_ids.clear()
                ids, prompts, limits = [], [], []
                for i in range(32):
                    prompt = [(i * 13 + j * 7) % 1000 + 100 for j in range(253 if i % 4 == 0 else 9)]
                    count = 32 if i < 2 else 16 if i < 8 else 8
                    prompts.append(prompt); limits.append(count)
                    ids.append(llm.add_request(prompt, SamplingParams(max_tokens=count, ignore_eos=True)))
                trace, outputs, peak = [], {}, 0
                for _ in range(1000):
                    before = llm.draft_runner.stats.copy()
                    done, _ = llm.step()
                    outputs.update(done)
                    step = llm.scheduler.last_step
                    trace.append({k: step[k] for k in ('mode', 'batch', 'k') if k in step})
                    assert step['batch'] <= 16
                    if step['mode'] != 'speculative':
                        assert llm.draft_runner.stats == before, 'Draft execution outside speculation'
                    peak = max(peak, len(llm.scheduler.ready))
                    for seq in llm.scheduler.ready:
                        assert seq.target.num_cached_tokens == len(seq) - 1
                        assert 0 <= seq.draft.num_cached_tokens <= len(seq) - 1
                    if llm.is_finished():
                        break
                else:
                    raise AssertionError('workload did not complete')
                assert peak == 32, peak
                assert len(outputs) == 32
                assert len(llm.scheduler.target_blocks.free_block_ids) == llm.target_runner.num_blocks
                assert len(llm.scheduler.draft_blocks.free_block_ids) == llm.draft_runner.num_blocks
                if threshold == 1:
                    assert not observed_ids and not llm.draft_runner.stats
                else:
                    assert observed_ids and len(observed_ids) < 32
                    assert llm.scheduler.stats['draft_prepare_tokens'] > 0
                    assert llm.scheduler.stats['target_only_rounds'] > 0
                return {'outputs': [outputs[i] for i in ids], 'trace': trace,
                    'peak_ready': peak, 'draft_request_indices': [i for i, rid in enumerate(ids) if rid in observed_ids],
                    'scheduler': dict(llm.scheduler.stats), 'target_runner': dict(llm.target_runner.stats),
                    'draft_runner': dict(llm.draft_runner.stats), 'inputs': prompts, 'max_tokens': limits}
            report['pure_target'] = run(1)
            executors = list(llm.target_runner.graphs.values()) + list(llm.draft_runner.graphs.values())
            with uncaptured(executors):
                reference = run(16)
            actual = run(16)
            compare_outputs(reference['outputs'], actual['outputs'], 'V6.1 same-shape eager vs graph')
            assert reference['trace'] == actual['trace']
            report['adaptive'] = actual
            report['same_shape_eager_outputs'] = reference['outputs']
            report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS V6.1: 32 resident requests, B<=16, zero Draft in Target-only, cold tail sync, eager/graph equality')


if __name__ == '__main__':
    main()
