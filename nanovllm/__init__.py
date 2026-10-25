from nanovllm.config import EngineConfig, Config
from nanovllm.sampling_params import SamplingParams

__version__ = '0.6.3'
__all__ = ['LLM', 'EngineConfig', 'Config', 'SamplingParams']


def __getattr__(name):
    if name == 'LLM':
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(name)
