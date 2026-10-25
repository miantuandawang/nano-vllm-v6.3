"""Opt-in A100 validation; no CUDA validation is implied by the CPU unit tests.

Checks each graph family against the same-shape uncaptured backbone, including
all valid KV positions and scratch isolation. Then checks adaptive scheduling
against same-shape uncaptured execution and independent Target-only generation.
Any token mismatch is a failure, with the first differing position reported.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@contextmanager
def uncaptured(executors):
    """Keep padding, attention metadata, grouping and LM-head selection identical."""
    saved = []
    for ex in executors:
        if ex.phase == 'decode':
            attr, buffers, q = 'graphs', ex.graph_vars, 1
        elif ex.phase == 'catchup':
            attr, buffers, q = 'catchup_graphs', ex.catchup_graph_vars, 2
        else:
            attr, buffers, q = 'verify_graphs', ex.verify_graph_vars, ex.qlen
        original = getattr(ex, attr)
        saved.append((ex, attr, original))
        def reference(count, buffers=buffers, model=ex.model):
            def replay():
                buffers['outputs'][:count] = model(buffers['input_ids'][:count], buffers['positions'][:count])
            return SimpleNamespace(replay=replay)
        setattr(ex, attr, {batch: reference(batch * q) for batch in original})
    try:
        yield
    finally:
        for ex, attr, original in saved:
            setattr(ex, attr, original)


def graph_checks(llm, torch):
    from nanovllm.engine.sequence import RunnerInput
    records = []
    for runner in (llm.target_runner, llm.draft_runner):
        for key, ex in runner.graphs.items():
            cap = ex.config.max_num_seqs
            for batch in sorted({cap, max(1, cap - 1), 1}, reverse=True):
                if runner.num_blocks < batch * 2:
                    raise RuntimeError('CUDA graph checks need at least two usable pages per row')
                q = ex.qlen
                def exercise(reference):
                    views, token_lists, ends = [], [], []
                    for i in range(batch):
                        cached = 255 if i % 2 == 0 else 9
                        tokens = [(j * 31 + i * 17) % 1000 + 100 for j in range(cached + q)]
                        pages = [2 * i, 2 * i + 1]
                        initial = RunnerInput(i, tokens, 0, pages)
                        runner.append([initial], [tokens[:cached]], project=False)
                        if ex.phase == 'verify':
                            views.append(RunnerInput(i, tokens[:cached + 1], cached, pages, tokens[cached + 1:]))
                        else:
                            views.append(RunnerInput(i, tokens, cached, pages))
                        token_lists.append(tokens)
                        ends.append(len(tokens))
                    if reference:
                        with uncaptured([ex]):
                            output = ex.replay(views, token_lists)
                    else:
                        output = ex.replay(views, token_lists)
                    # Full valid prefix and appended KV, not only output-token equality.
                    snapshots = []
                    for view, end in zip(views, ends):
                        pages = torch.tensor([view.block_table[p // 256] for p in range(end)], device='cuda')
                        offsets = torch.arange(end, device='cuda') % 256
                        snapshots.append(runner.kv_cache[:, :, pages, offsets].cpu())
                    assert torch.count_nonzero(runner.kv_cache[:, :, runner.num_blocks:]).item() == 0
                    return output, snapshots
                reference, expected_kv = exercise(True)
                actual, actual_kv = exercise(False)
                assert reference == actual, (runner.config.name, key, batch, 'token mismatch')
                for expected, value in zip(expected_kv, actual_kv):
                    torch.testing.assert_close(value, expected, rtol=0.03, atol=0.03)
                records.append({'model': runner.config.name, 'phase': key[0], 'q': key[1], 'batch': batch})
                print('PASS graph', records[-1], flush=True)
    return records


def drive(llm):
    from nanovllm import SamplingParams
    from nanovllm.engine.scheduler import Scheduler
    # Reuse fixed pools only after the previous run fully completed.
    llm.scheduler = Scheduler(llm.config, llm.target_runner, llm.draft_runner, llm.scheduler.eos)
    ids, prompts, limits = [], [], [64] * 2 + [24] * 2 + [12] * 4 + [6] * 8
    for i, count in enumerate(limits):
        prompt = [(j * 11 + i * 7) % 1000 + 100 for j in range(253 if i % 3 == 0 else 9 + i)]
        prompts.append(prompt)
        ids.append(llm.add_request(prompt, SamplingParams(max_tokens=count, ignore_eos=True)))
    outputs, trace = {}, []
    for _ in range(2000):
        done, _ = llm.step()
        outputs.update(done)
        step = llm.scheduler.last_step
        trace.append({k: step[k] for k in ('mode', 'batch', 'k') if k in step})
        if llm.is_finished():
            return [outputs[i] for i in ids], trace, prompts, limits, dict(llm.scheduler.stats)
    raise AssertionError('adaptive workload did not finish')


def compare_outputs(expected, actual, label):
    for i, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            index = next((j for j, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            raise AssertionError(f'{label}: request={i}, first divergent output position={index}; '
                                 'inspect logits/shape effects before attributing this to BF16')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target-model', required=True)
    p.add_argument('--draft-model', required=True)
    p.add_argument('--target-kv-cache-gib', type=float, default=28)
    p.add_argument('--draft-kv-cache-gib', type=float, default=8)
    p.add_argument('--output', default='validation-cuda.json')
    args = p.parse_args()
    import torch
    from nanovllm import LLM, EngineConfig, SamplingParams
    if not torch.cuda.is_available():
        raise SystemExit('CUDA/PyTorch/FlashAttention and model weights are required')
    # Test-only: many intentional shape variants must not switch the reference to eager kernels.
    if hasattr(torch._dynamo.config, 'recompile_limit'):
        torch._dynamo.config.recompile_limit = 64
    else:
        torch._dynamo.config.cache_size_limit = 64
    torch.manual_seed(600)
    config = EngineConfig(target_model=args.target_model, draft_model=args.draft_model,
                          target_kv_cache_gib=args.target_kv_cache_gib, draft_kv_cache_gib=args.draft_kv_cache_gib,
                          max_model_len=512, max_num_batched_tokens=512)
    report = {'status': 'running', 'device': torch.cuda.get_device_name(), 'torch': torch.__version__}
    try:
        with torch.inference_mode(), LLM(config) as llm:
            report['graph_cases'] = graph_checks(llm, torch)
            executors = list(llm.target_runner.graphs.values()) + list(llm.draft_runner.graphs.values())
            with uncaptured(executors):
                expected, expected_trace, _, _, _ = drive(llm)
            actual, trace, prompts, limits, stats = drive(llm)
            compare_outputs(expected, actual, 'same-shape uncaptured vs graph')
            assert expected_trace == trace
            assert stats.get('target_only_rounds', 0) and stats.get('speculative_rounds', 0)
            assert stats.get('draft_sync_tokens', 0)
            report['adaptive_trace'] = trace
            report['adaptive_stats'] = stats
            llm.config.target_only_threshold = 1
            baseline = llm.generate(prompts, [SamplingParams(max_tokens=n, ignore_eos=True) for n in limits],
                                    use_tqdm=False)
            compare_outputs([row['token_ids'] for row in baseline], actual, 'Target-only vs speculative')
            report['outputs'] = actual
            report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: graph tokens/KV, adaptive switches, and Target-only comparison', flush=True)


if __name__ == '__main__':
    main()
