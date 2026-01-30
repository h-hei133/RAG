# 🤖 Pro-RAG: 工业级自适应知识库助手

这是一个基于 **Parent-Child Indexing (父子索引)** 和 **Query Planning (查询规划)** 深度优化的 RAG 系统。相比传统 RAG，本项目解决了破碎上下文、复杂查询理解力差以及检索响应慢的痛点。

## ✨ 核心特性

### 1. 自适应混合解析 (Smart Hybrid Parsing)
*   **Dual-Mode 引擎**：自动检测 PDF 页面复杂度。
*   **Fast Mode**：纯文本页面秒级提取，并自动还原 Markdown 表格。
*   **Vision Mode**：集成 **Qwen-VL** 多模态大模型，针对复杂图表、PPT、架构图进行版面分析，将视觉信息转译为结构化 Markdown。

### 2. 父子索引置换 (Small-to-Big Retrieval)
*   **精准检索**：将文档切分为 400 字符的“子块”进行向量索引，确保检索的高精度。
*   **完整生成**：在 LLM 读取阶段，自动将子块置换为 2000 字符的“父块”上下文，彻底解决“断章取义”问题。

### 3. 查询规划器 (Query Planner)
*   **意图分发**：自动识别闲聊、事实问题或抽象问题。
*   **HyDE (假设性文档嵌入)**：针对抽象概念问题，先生成假设性回答再检索，显著提升长尾问题的召回率。
*   **历史指代消解**：结合上下文自动重写用户 Query，支持多轮深度对话。

### 4. 生产级优化
*   **语义缓存 (Semantic Cache)**：重复或语义高度相似的问题毫秒级命中，降低 Token 成本。
*   **混合检索 (Hybrid Search)**：结合 BM25 关键词检索与 MMR 向量检索，兼顾字面匹配与语义理解。
*   **Rerank 重排序**：集成 Cross-Encoder 对候选文档进行二次精排，过滤噪声。

## 🛠️ 技术栈
*   **框架**: LangChain
*   **LLM/VLM**: Qwen-VL (30B), Qwen-3
*   **向量库**: ChromaDB
*   **前端**: Streamlit
*   **模型服务**: SiliconFlow API

## 🚀 快速开始

1. **环境准备**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置 API**
   在 `.env` 中填入您的 SiliconFlow `API_KEY`。

3. **运行系统**
   ```bash
   streamlit run main.py
   ```

## 📂 项目结构
```text
RAG/
├── src/
│   ├── ingestion.py    # 文档摄取流：混合解析 -> 父子切分 -> 索引存储
│   └── rag_chain.py    # RAG 核心链：Query Planner -> 混合检索 -> Rerank -> 生成
├── chroma_db/          # 向量数据库与 BM25 索引存储
├── doc_store/          # 父文档原始内容持久化 (Pickle)
├── logs/               # RAG 活动日志与语义缓存
├── config.py           # 全局配置参数
└── main.py             # Streamlit 交互式界面
```
