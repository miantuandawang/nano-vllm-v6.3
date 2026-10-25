"""Single-device model execution; scheduling and cache-length ownership stay outside."""
from collections import Counter, defaultdict
from types import SimpleNamespace
from pathlib import Path

import torch

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.memory_budget import fixed_kv_blocks, kv_block_bytes, GIB
from nanovllm.engine.eager_methods import V51EagerMethods
from nanovllm.engine.graph_methods import V51GraphMethods
from nanovllm.engine.sequence import RunnerInput


class GraphExecutor(V51GraphMethods, V51EagerMethods):
    """One V5.1 graph family owns static buffers and a private pool for one Q."""
    def __init__(self, runner, phase, qlen, max_bs):
        self.model = runner.model
        self.config = SimpleNamespace(**vars(runner.config))
        self.config.max_num_seqs = max_bs
        self.rank, self.world_size = 0, 1
        self.block_size = runner.block_size
        self.phase, self.qlen = phase, qlen
        if phase == 'decode':
            self.capture_cudagraph()
        elif phase == 'catchup':
            self.catchup_query_len = qlen
            self.catchup_graph_max_bs = max_bs
            self.catchup_scratch_blocks = 1
            self.capture_catchup_graphs()
        else:
            self.verify_query_len = qlen
            self.verify_graph_max_bs = max_bs
            self.verify_scratch_blocks = 1
            self.capture_verify_graphs()

    def replay(self, seqs, tokens=None):
        if self.phase == 'decode':
            return self.replay_decode_graph(seqs, tokens)
        if self.phase == 'catchup':
            return self.replay_catchup_graph(seqs, tokens)
        return self.replay_verify_graph(seqs)


class ModelRunner(V51EagerMethods):
    def __init__(self, config):
        self.config = config
        self.rank, self.world_size = 0, 1
        self.block_size = config.kvcache_block_size
        self.graphs = {}
        self.stats = Counter()
        if not list(Path(config.model).glob('*.safetensors')):
            raise ValueError(f'{config.name}: no safetensors weights in {config.model}')
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(config.hf_config.dtype)
            with torch.device('cuda'):
                self.model = Qwen3ForCausalLM(config.hf_config).eval()
                load_model(self.model, config.model)
        finally:
            torch.set_default_dtype(previous)

    @torch.inference_mode()
    def warmup(self):
        # Warm the maximum prefill workload before either KV pool is allocated.
        length = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        count = min(self.config.max_num_batched_tokens // length, self.config.max_num_seqs)
        rows = [RunnerInput(i, [0] * length, 0, []) for i in range(count)]
        self._predict(rows, [s.token_ids for s in rows], project=False)
        hidden = torch.zeros(min(128, self.config.max_num_batched_tokens),
                             self.config.hf_config.hidden_size,
                             dtype=self.config.hf_config.dtype, device='cuda')
        self._greedy_predictions(hidden)

    @torch.inference_mode()
    def allocate_kv_cache(self):
        config, hf = self.config, self.config.hf_config
        # One shared read-only scratch page per model; Q <= 5 and page size >= 256.
        scratch = 0 if config.enforce_eager else 1
        total = fixed_kv_blocks(config, scratch)
        self.num_blocks = config.num_kvcache_blocks = total - scratch
        heads = hf.num_key_value_heads
        dim = getattr(hf, 'head_dim', hf.hidden_size // hf.num_attention_heads)
        self.kv_cache = torch.empty(2, hf.num_hidden_layers, total, self.block_size,
                                    heads, dim, dtype=hf.dtype, device='cuda')
        if scratch:
            self.kv_cache[:, :, self.num_blocks:].zero_()
        layer = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = self.kv_cache[0, layer]
                module.v_cache = self.kv_cache[1, layer]
                layer += 1
        print(f'[V6.3 KV] {config.name}: {total * kv_block_bytes(config) / GIB:.3f} GiB, '
              f'{self.num_blocks} usable pages + {scratch} scratch', flush=True)

    def capture_graphs(self):
        c = self.config
        if c.enforce_eager:
            return
        # Target captures the full native bucket ladder through 128, independent
        # of a smaller per-dispatch limit. Draft still handles only small batches.
        decode_cap = 128 if c.name == 'target' else min(16, c.max_num_seqs, c.max_num_batched_tokens)
        self.graphs[('decode', 1)] = GraphExecutor(self, 'decode', 1, decode_cap)
        if c.name == 'draft':
            cap = min(16, c.max_num_seqs, c.max_num_batched_tokens // 2)
            if cap and c.max_model_len >= 2:
                self.graphs[('catchup', 2)] = GraphExecutor(self, 'catchup', 2, cap)
        else:
            # Include smaller buckets for shortened candidates/EOS subgroups.
            for qlen, policy_cap in ((2, 16), (3, 8), (4, 4), (5, 2)):
                cap = min(c.max_num_seqs, policy_cap, c.max_num_batched_tokens // qlen)
                if cap and c.max_model_len >= qlen:
                    self.graphs[('verify', qlen)] = GraphExecutor(self, 'verify', qlen, cap)

    def append(self, seqs, token_lists, project=False):
        if not seqs:
            return []
        self.stats['eager_append_calls'] += 1
        return self._predict(seqs, token_lists, project=project)

    def next_tokens(self, seqs, token_lists):
        groups = defaultdict(list)
        for i, (seq, tokens) in enumerate(zip(seqs, token_lists)):
            missing = len(tokens) - seq.num_cached_tokens
            phase = 'decode' if missing == 1 else 'catchup' if missing == 2 else 'eager'
            graph = self.graphs.get((phase, missing))
            groups[graph].append(i)
        results = [None] * len(seqs)
        for graph, indices in groups.items():
            rows = [seqs[i] for i in indices]
            tokens = [token_lists[i] for i in indices]
            if graph is not None and len(rows) <= graph.config.max_num_seqs:
                predictions = graph.replay(rows, tokens)
                self.stats[graph.phase + '_graph_calls'] += 1
            else:
                predictions = self._predict(rows, tokens)
                self.stats['eager_next_calls'] += 1
            for i, prediction in zip(indices, predictions):
                results[i] = prediction[-1]
        return results

    def verify(self, seqs):
        groups = defaultdict(list)
        for i, seq in enumerate(seqs):
            groups[len(seq.draft_tokens) + 1].append(i)
        results = [None] * len(seqs)
        for qlen, indices in groups.items():
            rows = [seqs[i] for i in indices]
            graph = self.graphs.get(('verify', qlen))
            eligible = all(len(s.token_ids) - s.num_cached_tokens == 1 for s in rows)
            if graph is not None and eligible and len(rows) <= graph.config.max_num_seqs:
                predictions = graph.replay(rows)
                self.stats[f'verify_q{qlen}_graph_calls'] += 1
            else:
                predictions = self._predict(rows, [s.token_ids + s.draft_tokens for s in rows],
                                            all_positions=True)
                # General eager suffix, matching V5.1's offset handling.
                predictions = [p[len(s.token_ids) - s.num_cached_tokens - 1:]
                               for s, p in zip(rows, predictions)]
                self.stats['eager_verify_calls'] += 1
            for i, prediction in zip(indices, predictions):
                results[i] = prediction
        return results

    def close(self):
        self.graphs.clear()
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'kv_cache'):
            del self.kv_cache
