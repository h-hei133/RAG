import os
from dotenv import load_dotenv

load_dotenv()

# --- API 配置 ---
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("SILICON_BASE_URL")

if not API_KEY:
    raise ValueError("请在 .env 文件中配置 DEEPSEEK_API_KEY")

# --- 模型配置 ---
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# --- 存储路径 ---
PERSIST_DIRECTORY = "./chroma_db"
BM25_PERSIST_PATH = "./chroma_db/bm25_documents.pkl"

# --- 上传限制 ---
MAX_FILE_SIZE_MB = 50
MAX_FILES_COUNT = 5