from types import SimpleNamespace as NS
import unittest
from nanovllm.utils.memory_budget import GIB, kv_block_bytes, fixed_kv_blocks, estimate_weight_bytes


class MemoryTests(unittest.TestCase):
    def config(self, layers, gib, name):
        hf = NS(num_hidden_layers=layers, num_key_value_heads=8, num_attention_heads=16,
                head_dim=128, hidden_size=2048, dtype=NS(itemsize=2))
        return NS(hf_config=hf, tensor_parallel_size=1, kvcache_block_size=256,
                  kv_cache_memory_bytes=gib * GIB, name=name)

    def test_v51_default_pool_sizes_and_scratch(self):
        target, draft = self.config(40, 28, 'target'), self.config(28, 8, 'draft')
        self.assertEqual(kv_block_bytes(target), 40 * 1024**2)
        self.assertEqual(kv_block_bytes(draft), 28 * 1024**2)
        self.assertEqual(fixed_kv_blocks(target, 1) - 1, 715)
        self.assertEqual(fixed_kv_blocks(draft, 1) - 1, 291)

    def test_scratch_must_fit_inside_fixed_budget(self):
        config = self.config(40, 28, 'target')
        config.kv_cache_memory_bytes = kv_block_bytes(config)
        with self.assertRaises(ValueError):
            fixed_kv_blocks(config, 1)

    def test_tied_weights_are_counted_once(self):
        hf = self.config(28, 8, 'draft').hf_config
        hf.model_type = 'qwen3'
        hf.intermediate_size = 4096
        hf.vocab_size = 151936
        hf.attention_bias = False
        hf.tie_word_embeddings = True
        tied = estimate_weight_bytes(hf)
        hf.tie_word_embeddings = False
        self.assertEqual(estimate_weight_bytes(hf) - tied, hf.vocab_size * hf.hidden_size * 2)
