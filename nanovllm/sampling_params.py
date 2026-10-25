from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 256
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature != 0:
            raise ValueError('V6.3 implements greedy speculative decoding only (temperature=0)')
        if self.max_tokens < 0:
            raise ValueError('max_tokens must be nonnegative')
