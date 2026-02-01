import os
from dotenv import load_dotenv

load_dotenv()

# --- API 配置 ---
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("SILICON_BASE_URL")

if not API_KEY:
    raise ValueError("请在 .env 文件中配置 API_KEY")

# --- 模型配置 ---
LLM_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
VLM_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# --- 存储路径 ---
# 向量数据库 (存储子块向量)
PERSIST_DIRECTORY = "./chroma_db"

# 父文档存储 (存储父块原始内容)
PARENT_DOC_STORE_PATH = "./doc_store"

# SQLite 数据库路径 (存储父块原始内容 - 替代 pkl 文件)
SQLITE_DB_PATH = "./doc_store.db"

# BM25 索引 (存储子块关键词索引)
BM25_PERSIST_PATH = "./chroma_db/bm25_documents.pkl"

# 反馈日志路径
FEEDBACK_LOG_PATH = "./logs/feedback.jsonl"

# 提取图片存储路径 (用于存放版面分析提取出的图片)
IMG_STORE_PATH = "./extracted_images"

# --- 上传限制 ---
MAX_FILE_SIZE_MB = 50
MAX_FILES_COUNT = 5