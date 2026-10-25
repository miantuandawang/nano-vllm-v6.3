import atexit
import gc
import warnings
from pathlib import Path
from types import SimpleNamespace

from nanovllm.config import EngineConfig
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler


class LLMEngine:
    def __init__(self, model=None, **kwargs):
        import torch
        import torch.distributed as dist
        from transformers import AutoConfig, AutoTokenizer
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.memory_budget import GIB, MIB, estimate_weight_bytes, fixed_kv_blocks, kv_block_bytes

        if isinstance(model, EngineConfig):
            if kwargs:
                raise TypeError('pass either EngineConfig or keyword options')
            self.config = model
        else:
            if model is not None:
                kwargs['target_model'] = model
            self.config = EngineConfig(**kwargs)
        c = self.config
        self._closed = False
        self.target_runner = self.draft_runner = None
        self._last_budget_warning = 0.0
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError('V6.3 requires one GPU and TP=1')
        torch.cuda.set_device(c.device)
        name = torch.cuda.get_device_name(c.device)
        if 'A100' not in name:
            warnings.warn(f'V6.3 is designed for one A100; current device is {name}')
        dtype = getattr(torch, c.dtype)
        configs = []
        for role in ('target', 'draft'):
            path = str(Path(getattr(c, role + '_model')).expanduser().resolve())
            hf = AutoConfig.from_pretrained(path, local_files_only=True)
            if hf.model_type != 'qwen3':
                raise ValueError('V6.3 supports Qwen3 models')
            hf.dtype = dtype
            c.max_model_len = min(c.max_model_len, hf.max_position_embeddings)
            configs.append(SimpleNamespace(name=role, model=path, hf_config=hf,
                tensor_parallel_size=1, max_num_seqs=c.batch, max_model_len=c.max_model_len,
                max_num_batched_tokens=c.max_num_batched_tokens, kvcache_block_size=c.kvcache_block_size,
                enforce_eager=c.enforce_eager,
                kv_cache_memory_bytes=int(getattr(c, role + '_kv_cache_gib') * GIB)))
        for config in configs:
            config.max_model_len = c.max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(configs[0].model, local_files_only=True)
        draft_tokenizer = AutoTokenizer.from_pretrained(configs[1].model, local_files_only=True)
        if self.tokenizer.get_vocab() != draft_tokenizer.get_vocab():
            raise ValueError('Target and Draft must share token-to-ID mapping')
        if configs[0].hf_config.vocab_size != configs[1].hf_config.vocab_size:
            raise ValueError('Target and Draft vocab sizes differ')
        del draft_tokenizer
        eos = configs[0].hf_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
        if self.tokenizer.eos_token_id is not None:
            eos.add(self.tokenizer.eos_token_id)
        free, total = torch.cuda.mem_get_info()
        self.device_limit = int(total * c.gpu_memory_utilization)
        reserve = int(c.device_memory_reserve_gib * GIB)
        baseline = total - free
        self.allocator_limit = self.device_limit - baseline - reserve
        graph_reserve = 0 if c.enforce_eager else (c.target_graph_reserve_mb + c.draft_graph_reserve_mb) * MIB
        kv_bytes = sum(fixed_kv_blocks(x, 0 if c.enforce_eager else 1) * kv_block_bytes(x) for x in configs)
        required = sum(estimate_weight_bytes(x.hf_config) for x in configs) + kv_bytes + max(2 * GIB, graph_reserve)
        if required > self.allocator_limit:
            raise RuntimeError(f'weights + fixed KV + runtime reserve need {required / GIB:.2f} GiB; '
                               f'available allocator budget is {self.allocator_limit / GIB:.2f} GiB. '
                               '28/8 GiB defaults require A100-80GB; reduce explicit KV budgets otherwise.')
        # ONE cap for the shared process, never separate per-runner allocator fractions.
        torch.cuda.set_per_process_memory_fraction(self.allocator_limit / total, c.device)
        try:
            self.target_runner = ModelRunner(configs[0])
            self.draft_runner = ModelRunner(configs[1])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.target_runner.warmup()
            self.draft_runner.warmup()
            torch.cuda.synchronize()
            peak = max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            if peak + kv_bytes + graph_reserve > self.allocator_limit:
                raise RuntimeError('measured shared warmup peak + KV + graph reserve exceeds allocator budget')
            torch.cuda.empty_cache()
            for runner in (self.target_runner, self.draft_runner):
                runner.allocate_kv_cache()
            for runner in (self.target_runner, self.draft_runner):
                runner.capture_graphs()
            torch.cuda.synchronize()
            self.scheduler = Scheduler(c, self.target_runner, self.draft_runner, eos)
            self._check_budget()
        except Exception:
            self.exit()
            raise
        atexit.register(self.exit)

    def _check_budget(self):
        import time
        import torch
        free, total = torch.cuda.mem_get_info()
        now = time.monotonic()
        if total - free > self.device_limit and now - self._last_budget_warning > 30:
            warnings.warn('whole-device usage exceeds planned limit; continuing with the shared allocator cap')
            self._last_budget_warning = now

    def add_request(self, prompt, sampling_params=None):
        if self._closed:
            raise RuntimeError('engine is closed')
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if any(type(token) is not int or not 0 <= token < self.target_runner.config.hf_config.vocab_size
               for token in prompt):
            raise ValueError('invalid token ID')
        return self.scheduler.add(Sequence(prompt, sampling_params))

    def step(self):
        if self._closed:
            raise RuntimeError('engine is closed')
        result = self.scheduler.step()
        self._check_budget()
        return result

    def is_finished(self):
        return self.scheduler.is_finished()

    def cancel(self, request_id):
        self.scheduler.cancel(request_id)

    def generate(self, prompts, sampling_params=None, use_tqdm=True):
        from tqdm.auto import tqdm
        if not self.is_finished():
            raise RuntimeError('finish existing streaming requests before generate()')
        # Drain any results previously produced by explicit cancellation.
        self.scheduler.finished_outputs.clear()
        params = sampling_params or SamplingParams()
        params = params if isinstance(params, list) else [params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError('one SamplingParams is required per prompt')
        ids = [self.add_request(prompt, sp) for prompt, sp in zip(prompts, params)]
        outputs = {}
        with tqdm(total=len(ids), disable=not use_tqdm, desc='Generating') as progress:
            while not self.is_finished() or self.scheduler.finished_outputs:
                completed, _ = self.step()
                for request_id, tokens in completed:
                    outputs[request_id] = tokens
                    progress.update(1)
        return [{'text': self.tokenizer.decode(outputs[i]), 'token_ids': outputs[i]} for i in ids]

    def exit(self):
        if self._closed:
            return
        self._closed = True
        import torch
        from nanovllm.layers.rotary_embedding import get_rope
        torch.cuda.synchronize()
        for runner in (self.target_runner, self.draft_runner):
            if runner is not None:
                runner.close()
        get_rope.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
        atexit.unregister(self.exit)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.exit()
