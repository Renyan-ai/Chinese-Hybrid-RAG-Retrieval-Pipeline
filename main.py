import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_deepseek import ChatDeepSeek

from retrieval.vector import build_or_load_index
from retrieval.hybrid import FixedRetriever
from evaluation.ragas_eval import run_ragas_eval, TEST_SET

# ========== 加载环境变量 ==========
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)
