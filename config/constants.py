import os
from dotenv import load_dotenv
from log import setup_logging

load_dotenv()
setup_logging()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

