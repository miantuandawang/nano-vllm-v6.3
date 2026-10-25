from dataclasses import dataclass
import math


@dataclass
class EngineConfig:
    target_model: str = '/models/Qwen3-14B'
    draft_model: str = '/models/Qwen3-0.6B'
    batch: int = 128
    # None: resident requests are bounded by Target KV capacity, not dispatch B.
    max_active_seqs: int | None = 372
    target_only_threshold: int = 16
    max_model_len: int = 4096
    max_num_batched_tokens: int = 4096
    kvcache_block_size: int = 256
    target_kv_cache_gib: float = 28.0
    draft_kv_cache_gib: float = 8.0
    gpu_memory_utilization: float = 0.9
    device_memory_reserve_gib: float = 2.0
    target_graph_reserve_mb: int = 1024
    draft_graph_reserve_mb: int = 512
    enforce_eager: bool = False
    device: int = 0
    dtype: str = 'bfloat16'

    def __post_init__(self):
        if not 1 <= self.batch <= 128:
            raise ValueError('batch must be in [1, 128]')
        if self.max_active_seqs is not None and (
                type(self.max_active_seqs) is not int or self.max_active_seqs < 1):
            raise ValueError('max_active_seqs must be None or a positive integer')
        if not 1 <= self.target_only_threshold <= 17:
            raise ValueError('threshold must be in [1, 17]; 1=target-only, 17=always speculative')
        if self.max_model_len < 2 or self.max_num_batched_tokens < 1:
            raise ValueError('invalid context/token budget')
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256:
            raise ValueError('KV page size must be a positive multiple of 256')
        for name in ('target_kv_cache_gib', 'draft_kv_cache_gib', 'device_memory_reserve_gib'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.gpu_memory_utilization <= 0.9:
            raise ValueError('gpu_memory_utilization must be in (0, 0.9]')
        if min(self.target_graph_reserve_mb, self.draft_graph_reserve_mb, self.device) < 0:
            raise ValueError('device and graph reserves must be nonnegative')
        if self.dtype not in ('bfloat16', 'float16'):
            raise ValueError('supported dtypes: bfloat16, float16')

    def choose_k(self, batch):
        if batch >= self.target_only_threshold:
            return 0
        return 4 if batch <= 2 else 3 if batch <= 4 else 2 if batch <= 8 else 1


Config = EngineConfig
