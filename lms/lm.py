from dspy import LM, configure
from dspy.adapters import XMLAdapter
from dspy.clients.lm_local import LocalProvider
from config import OPENAI_API_KEY, OLLAMA_API_KEY 

configure(adapter=XMLAdapter())
ollama_gpt = LM(
        model='ollama_chat/gpt-oss:20b',
        api_base='http://localhost:11434',
        api_key='',
        cache=True,
    )
ollama_llama = LM(
    model='ollama_chat/llama3.2',
        api_base='http://localhost:11434',
        api_key='',
        cache=True,
    )

cloud_gpt = LM(
    model='ollama_chat/gpt-oss:120b-cloud',
    api_base='http://localhost:11434',
    api_key=OLLAMA_API_KEY,
    cache=True,
)

local_gpt = LM(
    model=f"openai/local:gpt-oss-20b",
    provider=LocalProvider(),
    max_tokens=20000
)

gpt_5_nano=LM(
    model="openai/gpt-5-nano",
    api_key=OPENAI_API_KEY,
    temperature=1.0,
    max_tokens=50000,
    cache=True,
)
gpt_5_mini=LM(
    model="openai/gpt-5-mini",
    api_key=OPENAI_API_KEY,
    temperature=1.0,
    max_tokens=50000,
    cache=True,
)
gpt_5=LM(
    model="openai/gpt-5",
    api_key=OPENAI_API_KEY,
    temperature=1.0,
    max_tokens=50000,
    cache=True,
)


