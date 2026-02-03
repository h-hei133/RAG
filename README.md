# RAG: 自适应知识库助手

这是一个基于 **Parent-Child Indexing (父子索引)** 和 **Query Planning (查询规划)** 深度优化的 RAG 系统。相比传统 RAG，本项目解决了破碎上下文、复杂查询理解力差以及检索响应慢的痛点。

## 核心特性

### 1. 自适应混合解析 (Smart Hybrid Parsing)
*   **Dual-Mode 引擎**：自动检测 PDF 页面复杂度。
*   **Fast Mode**：纯文本页面秒级提取，并自动还原 Markdown 表格。
*   **Vision Mode**：集成 **Qwen-VL** 多模态大模型，针对复杂图表、PPT、架构图进行版面分析，将视觉信息转译为结构化 Markdown。

### 2. 父子索引置换 (Small-to-Big Retrieval)
*   **精准检索**：将文档切分为 400 字符的"子块"进行向量索引，确保检索的高精度。
*   **完整生成**：在 LLM 读取阶段，自动将子块置换为 2000 字符的"父块"上下文，彻底解决"断章取义"问题。

### 3. 查询规划器 (Query Planner)
*   **8类意图识别**：GREETING, SIMPLE, COMPLEX, ABSTRACT, METADATA_QUERY, COMPARE, SUMMARIZE, OUT_OF_DOMAIN
*   **子问题分解**：复杂/对比问题自动分解为可独立回答的子问题
*   **HyDE (假设性文档嵌入)**：针对抽象概念问题，先生成假设性回答再检索
*   **历史指代消解**：结合上下文自动重写用户 Query，支持多轮深度对话

### 4. Contextual Retrieval (Anthropic 方法)
*   **上下文增强**：在 embedding 之前为每个 chunk 添加上下文前缀
*   **效果**：检索失败率降低 35-67% (Anthropic 官方数据)

### 5. CRAG 纠错检索 (Corrective RAG)
*   **质量评估**：LLM 自动评估检索结果的相关性
*   **智能回退**：低质量结果时自动触发 HyDE 重试
*   **过滤噪声**：只保留真正相关的文档进入上下文

### 6. 语义缓存 (Semantic Cache)
*   **向量匹配**：使用 embedding 相似度而非字符串精确匹配
*   **模糊命中**："How are you?" 和 "How are you" 会命中同一缓存
*   **持久化**：缓存自动保存到磁盘，重启后仍有效

### 7. Token 管理与溢出保护
*   **精确计数**：使用 tiktoken 精确计算 token 数
*   **智能裁剪**：上下文过长时自动截断，避免模型溢出
*   **文档限制**：限制返回文档数量，确保上下文质量

### 8. 生产级优化
*   **混合检索 (Hybrid Search)**：BM25 + MMR 向量检索
*   **Rerank 重排序**：Cross-Encoder 二次精排
*   **流式输出**：实时返回生成结果

### 9. RAGAS 评估框架
*   **Faithfulness**：评估答案是否基于上下文生成
*   **Answer Relevancy**：评估答案是否回答了问题
*   **Context Precision**：评估检索的信噪比
*   **双模式**：支持 RAGAS 库和 LLM 自评估

## 技术栈
*   **框架**: LangChain
*   **LLM/VLM**: Qwen-VL (30B), Qwen-3
*   **向量库**: ChromaDB
*   **前端**: Streamlit
*   **模型服务**: SiliconFlow API
*   **评估**: RAGAS (可选)

## 快速开始

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

## 项目结构
```text
RAG/
├── src/
│   ├── ingestion.py    # 文档摄取流：混合解析 -> Contextual Retrieval -> 父子切分 -> 索引
│   ├── rag_chain.py    # RAG 核心链：Query Planner -> CRAG -> 混合检索 -> Rerank -> 生成
│   └── evaluation.py   # RAGAS 评估框架：Faithfulness, Relevancy, Precision
├── chroma_db/          # 向量数据库与 BM25 索引存储
├── doc_store.db        # SQLite 父文档存储
├── logs/               # RAG 活动日志、语义缓存、评估结果
├── config.py           # 全局配置参数
└── main.py             # Streamlit 交互式界面
```

## 优化效果对比

| 能力 | 优化前 | 优化后 |
|------|--------|--------|
| 意图识别 | 4类 | 8类 |
| 语义缓存 | 字符串匹配 | 向量相似度 |
| 检索质量 | 无校验 | CRAG 自动评估+回退 |
| 上下文 | 无处理 | Contextual Retrieval |
| Token 管理 | 无 | 自动裁剪保护 |
| 评估框架 | 无 | RAGAS 集成 |

## 高级配置

### 启用/禁用功能
```python
# 在 rag_chain.py 中
chain.use_crag = True/False           # CRAG 纠错检索
chain.use_semantic_cache = True/False  # 语义缓存

# 在 ingestion.py 中
ingest_document(..., use_contextual_retrieval=True/False)  # Contextual Retrieval
```

### 评估 RAG 质量
```python
from src.evaluation import quick_evaluate, RAGEvaluator

# 快速评估单个响应
result = quick_evaluate(
    question="什么是机器学习？",
    answer="机器学习是...",
    contexts=["文档片段1", "文档片段2"],
    llm=your_llm
)
print(f"整体分数: {result.overall_score}")

# 查看聚合指标
evaluator = RAGEvaluator(llm=your_llm)
metrics = evaluator.get_aggregate_metrics()
print(metrics)
```
