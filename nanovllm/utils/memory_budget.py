"""Fixed KV sizing helpers reused verbatim from V5.1."""
import math
GIB = 1024**3
MIB = 1024**2
TARGET_KV_BYTES = 28 * GIB
DRAFT_KV_BYTES = 8 * GIB

def gib_to_bytes(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("KV cache GiB must be finite and positive")
    return int(value * GIB)


def kv_block_bytes(config):
    hf = config.hf_config
    if hf.num_key_value_heads % config.tensor_parallel_size:
        raise ValueError("KV heads must be divisible by tensor_parallel_size")
    heads = hf.num_key_value_heads // config.tensor_parallel_size
    dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    return (2 * hf.num_hidden_layers * config.kvcache_block_size
            * heads * dim * hf.dtype.itemsize)


def fixed_kv_blocks(config, scratch_blocks=0):
    count = config.kv_cache_memory_bytes // kv_block_bytes(config)
    if count <= scratch_blocks:
        raise ValueError(f"{config.name}: KV budget cannot hold usable blocks and scratch")
    return count


def estimate_weight_bytes(hf):
    """Resident Qwen3 parameter storage, including tied embedding storage once."""
    if hf.model_type != "qwen3":
        raise ValueError("The V5.1 shared-GPU budget supports Qwen3 models")
    h, intermediate = hf.hidden_size, hf.intermediate_size
    dim = getattr(hf, "head_dim", h // hf.num_attention_heads)
    q = hf.num_attention_heads * dim
    kv = hf.num_key_value_heads * dim
    # Matches Qwen3ForCausalLM, including its attention-bias/QK-norm branch.
    attention_bias = getattr(hf, "attention_bias", True)
    per_layer = h * (q + 2 * kv) + q * h + 3 * h * intermediate + 2 * h
    per_layer += q + 2 * kv if attention_bias else 2 * dim
    params = hf.num_hidden_layers * per_layer + h
    params += hf.vocab_size * h * (1 if hf.tie_word_embeddings else 2)
    return params * hf.dtype.itemsize

