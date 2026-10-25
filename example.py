"""Single A100 entry point. Model weights are local, matching V5.1."""
import argparse
import json


def parser():
    p = argparse.ArgumentParser(description='nano-vllm V6.3: adaptive single-GPU speculative decode')
    p.add_argument('--target-model', default='/models/Qwen3-14B')
    p.add_argument('--draft-model', default='/models/Qwen3-0.6B')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--max-active-seqs', type=int, default=372,
                   help='resident-request cap; default: 372, also bounded by Target KV capacity')
    p.add_argument('--target-only-threshold', type=int, default=16)
    p.add_argument('--target-kv-cache-gib', type=float, default=28)
    p.add_argument('--draft-kv-cache-gib', type=float, default=8)
    p.add_argument('--target-graph-reserve-mb', type=int, default=1024)
    p.add_argument('--draft-graph-reserve-mb', type=int, default=512)
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--max-num-batched-tokens', type=int, default=4096)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--enforce-eager', action='store_true')
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--ignore-eos', action='store_true')
    p.add_argument('--prompt', action='append', help='repeat for multiple requests; raw text, no implicit chat template')
    return p


def config_from_args(args):
    from dataclasses import fields
    from nanovllm import EngineConfig
    names = {f.name for f in fields(EngineConfig)}
    return EngineConfig(**{key: value for key, value in vars(args).items() if key in names})


def main():
    args = parser().parse_args()
    from nanovllm import LLM, SamplingParams
    with LLM(config_from_args(args)) as llm:
        outputs = llm.generate(args.prompt or ['Explain speculative decoding.'],
                               SamplingParams(max_tokens=args.max_tokens, ignore_eos=args.ignore_eos))
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        print(json.dumps({'scheduler': dict(llm.scheduler.stats),
                          'target': dict(llm.target_runner.stats),
                          'draft': dict(llm.draft_runner.stats)}, indent=2))


if __name__ == '__main__':
    main()
