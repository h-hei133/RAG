# config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件中的变量
load_dotenv()

# 读取配置
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("SILICON_BASE_URL")

if not API_KEY:
    raise ValueError("请在 .env 文件中配置 DEEPSEEK_API_KEY")

# 模型配置
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"

# --- Rerank 模型配置 ---
# 使用 BAAI 的重排序模型，效果极佳
# 如果显存/内存吃紧，可以换成 "BAAI/bge-reranker-base"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# --- 文件上传限制 ---
MAX_FILE_SIZE_MB = 50  # 允许上传的最大总文件大小 (MB)
MAX_FILES_COUNT = 5    # 允许上传的最大文件数量