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

# RAG 活动日志路径（重构：统一路径配置，避免各模块硬编码）
RAG_LOG_PATH = "./logs/rag_activity.jsonl"

# 语义缓存路径（重构：从硬编码方法体移至统一配置）
SEMANTIC_CACHE_PATH = "./logs/semantic_cache.pkl"

# 字符串缓存路径（重构：从硬编码方法体移至统一配置）
STRING_CACHE_PATH = "./logs/cache.json"

# 提取图片存储路径 (用于存放版面分析提取出的图片)
IMG_STORE_PATH = "./extracted_images"

# 会话根目录（多对话隔离存储）
SESSIONS_ROOT = "./sessions"


def _sanitize_session_id(session_id: str) -> str:
    """将会话 ID 规范化为安全路径片段。"""
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(session_id)
    )
    return safe or "default"


def get_session_paths(session_id: str) -> dict:
    """
    返回指定会话的隔离存储路径。
    每个会话独立维护 data/chroma/sqlite/logs/cache/image。
    """
    sid = _sanitize_session_id(session_id)
    session_root = os.path.join(SESSIONS_ROOT, sid)
    logs_dir = os.path.join(session_root, "logs")
    chroma_dir = os.path.join(session_root, "chroma_db")

    return {
        "session_id": sid,
        "session_root": session_root,
        "data_dir": os.path.join(session_root, "data"),
        "persist_directory": chroma_dir,
        "bm25_persist_path": os.path.join(chroma_dir, "bm25_documents.pkl"),
        "sqlite_db_path": os.path.join(session_root, "doc_store.db"),
        "feedback_log_path": os.path.join(logs_dir, "feedback.jsonl"),
        "rag_log_path": os.path.join(logs_dir, "rag_activity.jsonl"),
        "semantic_cache_path": os.path.join(logs_dir, "semantic_cache.pkl"),
        "string_cache_path": os.path.join(logs_dir, "cache.json"),
        "img_store_path": os.path.join(session_root, "extracted_images"),
    }


def get_default_paths() -> dict:
    """兼容旧逻辑：返回默认全局路径。"""
    return {
        "persist_directory": PERSIST_DIRECTORY,
        "bm25_persist_path": BM25_PERSIST_PATH,
        "sqlite_db_path": SQLITE_DB_PATH,
        "feedback_log_path": FEEDBACK_LOG_PATH,
        "rag_log_path": RAG_LOG_PATH,
        "semantic_cache_path": SEMANTIC_CACHE_PATH,
        "string_cache_path": STRING_CACHE_PATH,
        "img_store_path": IMG_STORE_PATH,
        "data_dir": "./data",
    }


# --- 上传限制 ---
MAX_FILE_SIZE_MB = 50
MAX_FILES_COUNT = 5


# --- 幻觉检测默认参数 ---
# 阈值越高越严格（越容易触发风险提示）
HALLUCINATION_THRESHOLD = 0.5
# 用于检测的最多文档数
HALLUCINATION_MAX_DOCS = 3
# 每个文档用于检测的截断字符数
HALLUCINATION_DOC_CHARS = 500
# 答案用于检测的截断字符数
HALLUCINATION_ANSWER_CHARS = 800
