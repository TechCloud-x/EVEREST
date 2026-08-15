import vllm




from roll.third_party.vllm.vllm_0_8_4.llm import Llm084
from roll.third_party.vllm.vllm_0_8_4.v1.async_llm import AsyncLLM084

LLM = Llm084
AsyncLLM = AsyncLLM084












__all__ = ["LLM", "AsyncLLM"]
