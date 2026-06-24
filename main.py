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

# ========== 配置 ==========
SAVE_DIR = "./storage_data_fixed"
CHROMA_PATH = os.path.join(SAVE_DIR, "chroma_db")
NODES_PATH = os.path.join(SAVE_DIR, "nodes.json")
STORES_PATH = os.path.join(SAVE_DIR, "extra_stores.json")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
BM25_TOP_K = 15
VECTOR_TOP_K = 15
FUSION_TOP_K = 10
RERANK_TOP_N = 5

# 表格分块配置
TABLE_MAX_ROWS = 10          # 每个表格子块最大行数
TABLE_OVERLAP_ROWS = 2 

class Config:
    """全局配置类"""
    
    # ===== LLM 配置 =====
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    # ===== Embedding 配置 =====
    # 支持本地路径或 HuggingFace 模型名
    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh")
    EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
    
    # ===== Reranker 配置 =====
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

config = Config()

# ========== 初始化 LLM ==========
llm = ChatDeepSeek(
    model=config.DEEPSEEK_MODEL,
    api_key=config.DEEPSEEK_API_KEY,
    base_url=config.DEEPSEEK_BASE_URL
)

# ========== 初始化 Embedding ==========
embed_model = HuggingFaceEmbedding(
    model_name=config.EMBED_MODEL,
    device=config.EMBED_DEVICE,
    normalize=True
)

def interactive_loop(retriever):
    print(f"\n{'='*60}")
    print("💬 交互式检索测试 (输入 'q' 退出, 'ablate <query>' 做消融)")
    print(f"{'='*60}")

    while True:
        try:
            inp = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not inp:
            continue
        if inp.lower() in ("q", "quit", "exit"):
            break

        if inp.lower().startswith("ablate "):
            query = inp[7:]
            retriever.retrieve_ablation(query)
        else:
            retriever.retrieve(inp, verbose=True)


# ========== 主入口 ==========
if __name__ == "__main__":
    force = "--rebuild" in sys.argv or "-f" in sys.argv

    print("=" * 60)
    print("🧪 RAG 检索质量测试工具（表格改进版）")
    print("=" * 60)

    # 构建/加载索引
    index, parent_store, table_store, nodes = build_or_load_index(
        PDF_PATH, embed_model, force_rebuild=force
    )

    # 创建检索器
    retriever = FixedRetriever(
        index, nodes, parent_store, table_store,
        llm=llm,
        embed_model=embed_model,
        reranker_model_path=RERANKER_MODEL_PATH
    )

    # RAGAS 评估
    run_ragas_eval(retriever, TEST_SET, llm=llm)

    # 进入交互模式
    print("\n\n" + "=" * 60)
    print("💡 已进入交互模式，可输入任意查询细看检索结果")
    print("   输入 'ablate <query>' 查看向量 vs BM25 的消融对比")
    print("   输入 'q' 退出")
    print("=" * 60)
    interactive_loop(retriever)
