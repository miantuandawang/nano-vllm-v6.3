from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    PREFILL = auto()
    READY = auto()
    DRAFT_SYNC = auto()
    DRAFT = auto()
    VERIFY = auto()
    FINISHED = auto()


@dataclass
class CacheState:
    num_cached_tokens: int = 0
    block_table: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RunnerInput:
    """Execution metadata; only Scheduler changes authoritative cache lengths."""
    request_id: int
    token_ids: list[int]
    num_cached_tokens: int
    block_table: list[int]
    draft_tokens: list[int] = field(default_factory=list)


class Sequence:
    counter = count()

    def __init__(self, token_ids, sampling_params=None):
        params = sampling_params or SamplingParams()
        self.request_id = next(self.counter)
        self.seq_id = self.request_id
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.max_tokens = params.max_tokens
        self.ignore_eos = params.ignore_eos
        self.target = CacheState()
        self.draft = CacheState()
        self.draft_tokens = []
        self.status = SequenceStatus.WAITING
        self.initialized = False
        self.prefill_end = len(token_ids)
        self.finish_reason = None

    def __len__(self):
        return len(self.token_ids)

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def remaining_tokens(self):
        return self.max_tokens - (len(self) - self.num_prompt_tokens)

    def view(self, role):
        cache = getattr(self, role)
        return RunnerInput(self.request_id, self.token_ids, cache.num_cached_tokens,
                           cache.block_table, self.draft_tokens)

    def is_done(self, eos, max_model_len):
        if self.remaining_tokens <= 0:
            self.finish_reason = 'length'
        elif len(self) >= max_model_len:
            self.finish_reason = 'context_length'
        elif self.initialized and not self.ignore_eos and self.token_ids[-1] in eos:
            self.finish_reason = 'eos'
        return self.finish_reason is not None
