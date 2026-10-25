"""Throughput measurement excludes model loading and graph capture."""
import json
from pathlib import Path
from time import perf_counter
from example import parser, config_from_args


def main():
    p = parser()
    p.add_argument('--prompts-json', help='JSON list of raw strings or token-ID lists')
    p.add_argument('--requests', type=int, default=32)
    p.add_argument('--output', default='benchmark-v63.json')
    args = p.parse_args()
    from nanovllm import LLM, SamplingParams
    import torch
    prompts = (json.loads(Path(args.prompts_json).read_text(encoding='utf-8')) if args.prompts_json
               else [f'Explain how a computer works. Example {i}:' for i in range(args.requests)])
    if not isinstance(prompts, list):
        raise ValueError('prompts JSON must be a list')
    with LLM(config_from_args(args)) as llm:
        torch.cuda.synchronize()
        start = perf_counter()
        outputs = llm.generate(prompts, SamplingParams(max_tokens=args.max_tokens, ignore_eos=args.ignore_eos),
                               use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        total = sum(len(row['token_ids']) for row in outputs)
        result = {'seconds': elapsed, 'output_tokens': total, 'tokens_per_second': total / elapsed,
                  'config': vars(llm.config), 'scheduler': dict(llm.scheduler.stats),
                  'target': dict(llm.target_runner.stats), 'draft': dict(llm.draft_runner.stats),
                  'prompts': prompts, 'outputs': outputs}
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{total} output tokens in {elapsed:.3f}s: {total / elapsed:.2f} token/s')


if __name__ == '__main__':
    main()
