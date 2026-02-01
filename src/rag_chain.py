import os
import torch
import pickle
import sqlite3
import config
import streamlit as st
from operator import itemgetter

# LangChain 组件
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# 检索相关组件
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import json
import time
import uuid
from typing import Literal, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_mrr(retrieved_ids: List[str], relevant_id: str) -> float:
    """
    计算单个查询的倒数排名 (Reciprocal Rank)
    返回 1/K，其中 K 是第一个相关文档的排名位置
    如果相关文档不在列表中，返回 0
    """
    try:
        rank = retrieved_ids.index(relevant_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def log_feedback(run_id: str, score: int, comment: Optional[str] = None):
    """
    记录用户反馈到日志文件
    score: 1 = 正面反馈 (👍), 0 = 负面反馈 (👎)
    """
    import config
    feedback_log = {
        "run_id": run_id,
        "score": score,
        "comment": comment,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    os.makedirs(os.path.dirname(config.FEEDBACK_LOG_PATH), exist_ok=True)
    with open(config.FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_log, ensure_ascii=False) + "\n")

from langchain_core.pydantic_v1 import BaseModel, Field

class RouteQuery(BaseModel):
    """用户查询意图分类"""
    intent: Literal["GREETING", "SIMPLE", "COMPLEX", "ABSTRACT"] = Field(
        description="查询意图类型"
    )
    reasoning: str = Field(
        description="分类理由",
        default=""
    )

class QueryPlanner:
    """
    Query 规划器：负责 Query 重写、分发、HyDE 等高级检索策略
    """
    def __init__(self, llm):
        self.llm = llm
        self.output_parser = StrOutputParser()

    def classify_intent(self, question: str) -> str:
        """
        使用 LLM 对用户问题进行意图分类
        返回: GREETING | SIMPLE | COMPLEX | ABSTRACT
        """
        routing_prompt = f"""分析用户问题，判断其意图类型。

问题: "{question}"

类型判断标准:
1. GREETING: 打招呼或闲聊（如：你好、谢谢、再见、hi、hello）
2. SIMPLE: 事实性简单问题，只需单一概念查询（如：XX是什么时候发布的？）
3. COMPLEX: 涉及对比、多跳推理或需要多角度回答（如：华为和小米哪个好？）
4. ABSTRACT: 概念性问题，适合先生成假设性答案再检索（如：什么是量子纠缠？如何理解XX？）

请只返回一个 JSON 格式:
{{"intent": "类型", "reasoning": "理由"}}
"""
        try:
            response = self.llm.invoke(routing_prompt)
            result_text = self.output_parser.invoke(response)
            # 尝试解析 JSON
            import json
            import re
            # 提取 JSON 部分
            json_match = re.search(r'\{[^}]+\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
                intent = result.get("intent", "SIMPLE").upper()
                if intent in ["GREETING", "SIMPLE", "COMPLEX", "ABSTRACT"]:
                    return intent
            return "SIMPLE"
        except Exception as e:
            print(f"意图分类失败: {e}")
            return "SIMPLE"

    def plan(self, question: str, chat_history: list) -> dict:
        """
        对 Query 进行意图分析和规划
        """
        intent = self.classify_intent(question)
        
        result = {
            "type": intent,
            "queries": [question],
            "use_hyde": intent == "ABSTRACT"
        }
        
        # COMPLEX 和 SIMPLE 问题都进行查询扩展以提高召回率
        if intent in ["COMPLEX", "SIMPLE"]:
            variants = self.expand_query(question)
            result["queries"] = [question] + variants
        
        return result

    def expand_query(self, question: str) -> List[str]:
        """
        生成查询变体，用于多路召回
        返回 3 个语义相同但表述不同的搜索词
        """
        prompt = f"""针对以下用户问题，请生成3个不同角度的搜索查询词，以便更全面地检索相关文档。

用户问题: {question}

要求:
1. 每个查询词占一行
2. 涵盖问题的不同方面
3. 使用不同的关键词组合
4. 只返回查询词，不要编号或解释

示例:
问题: "华为和小米的手机哪个好？"
华为手机参数配置
小米手机性能评测
华为小米对比分析"""
        
        try:
            response = self.llm.invoke(prompt)
            result = self.output_parser.invoke(response)
            variants = [v.strip() for v in result.strip().split("\n") if v.strip()]
            # 过滤掉太短或太长的变体
            variants = [v for v in variants if 2 < len(v) < 50]
            return variants[:3]  # 最多返回3个
        except Exception as e:
            print(f"查询扩展失败: {e}")
            return []

    def generate_hyde_doc(self, question: str) -> str:
        """
        生成假设性文档 (HyDE - Hypothetical Document Embeddings)
        用于抽象概念问题的检索增强
        """
        prompt = f"""你是一位专业的技术文档作者。请针对以下问题，写一段专业、准确的回答草稿。
这段回答将用于帮助搜索引擎找到相关文档，所以请包含尽可能多的专业术语和关键概念。

问题: {question}

要求:
1. 100-200字左右
2. 使用专业术语
3. 涵盖核心概念
4. 不要说"我不知道"之类的话"""
        
        try:
            response = self.llm.invoke(prompt)
            return self.output_parser.invoke(response)
        except Exception as e:
            print(f"HyDE生成失败: {e}")
            return ""


class ManualHistoryRAGChain:
    """
    手动实现的 RAG 链，集成父子索引策略 (Small-to-Big Retrieval)
    """

    def __init__(self, retriever, qa_prompt, history_prompt, llm):
        self.retriever = retriever
        self.qa_prompt = qa_prompt
        self.history_prompt = history_prompt
        self.llm = llm
        self.output_parser = StrOutputParser()
        # 获取父文档存储路径
        self.doc_store_path = getattr(config, "PARENT_DOC_STORE_PATH", "./doc_store")
        self.planner = QueryPlanner(llm)

    def _rewrite_question(self, question, chat_history):

        formatted_history_prompt = self.history_prompt.invoke({
            "chat_history": chat_history,
            "input": question
        })
        response = self.llm.invoke(formatted_history_prompt)
        return self.output_parser.invoke(response)

    def _map_children_to_parents(self, child_docs):
        """
        核心逻辑：将检索到的子块 (Child) 映射回父块 (Parent)
        使用 SQLite 批量查询替代逐个 pickle 文件读取
        """
        parent_docs = []
        seen_ids = set()
        
        # 收集所有需要查询的 doc_ids
        doc_ids_to_fetch = []
        child_fallbacks = {}  # doc_id -> child_doc (用于降级)
        
        for child in child_docs:
            doc_id = child.metadata.get("doc_id")
            
            if not doc_id:
                # 兼容旧数据：没有 ID 的直接添加
                if child.page_content not in [d.page_content for d in parent_docs]:
                    parent_docs.append(child)
                continue
                
            if doc_id not in seen_ids:
                doc_ids_to_fetch.append(doc_id)
                child_fallbacks[doc_id] = child
                seen_ids.add(doc_id)
        
        if not doc_ids_to_fetch:
            return parent_docs
        
        # 批量从 SQLite 获取父文档
        db_path = getattr(config, "SQLITE_DB_PATH", "./doc_store.db")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 使用 IN 子句批量查询
            placeholders = ",".join("?" * len(doc_ids_to_fetch))
            cursor.execute(
                f"SELECT doc_id, data FROM parent_docs WHERE doc_id IN ({placeholders})",
                doc_ids_to_fetch
            )
            
            results = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            
            # 按原始顺序处理结果
            for doc_id in doc_ids_to_fetch:
                if doc_id in results:
                    try:
                        parent_doc = pickle.loads(results[doc_id])
                        parent_docs.append(parent_doc)
                    except Exception as e:
                        print(f"反序列化父文档失败 {doc_id}: {e}")
                        parent_docs.append(child_fallbacks[doc_id])
                else:
                    # SQLite 中找不到，降级使用子块
                    parent_docs.append(child_fallbacks[doc_id])
                    
        except Exception as e:
            print(f"SQLite 查询失败: {e}")
            # 全部降级使用子块
            for doc_id in doc_ids_to_fetch:
                parent_docs.append(child_fallbacks[doc_id])
        
        return parent_docs

    def _get_base_retriever(self):
        """
        获取基础检索器，用于并发多查询检索
        如果是 ContextualCompressionRetriever，返回其 base_retriever
        """
        if hasattr(self.retriever, 'base_retriever'):
            return self.retriever.base_retriever
        return self.retriever

    def _rerank_documents(self, docs, query):
        """
        对文档进行重排序（如果启用了 Reranker）
        """
        if hasattr(self.retriever, 'base_compressor'):
            try:
                return list(self.retriever.base_compressor.compress_documents(docs, query))
            except Exception as e:
                print(f"重排序失败: {e}")
                return docs[:5]
        return docs[:5]

    def _deduplicate_docs(self, docs):
        """
        根据 doc_id 对文档去重
        """
        seen_ids = set()
        unique_docs = []
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            content_hash = hash(doc.page_content[:100]) if not doc_id else None
            key = doc_id or content_hash
            if key and key not in seen_ids:
                seen_ids.add(key)
                unique_docs.append(doc)
            elif not key:
                unique_docs.append(doc)
        return unique_docs

    def _prepare_context(self, input_dict: dict) -> dict:
        """
        准备上下文的辅助方法，抽取检索/规划逻辑供 invoke 和 stream 共用
        返回: {
            "run_id": str,
            "question": str,
            "chat_history": list,
            "search_query": str,
            "planning_type": str,
            "queries": list,
            "child_docs": list,
            "final_docs": list,
            "context_str": str,
            "formatted_qa_prompt": BaseMessage,
            "cache_hit": str | None
        }
        """
        run_id = str(uuid.uuid4())
        question = input_dict.get("input", "")
        chat_history = input_dict.get("chat_history", [])
        
        # 检查缓存
        cache_hit = self._check_cache(question)
        if cache_hit:
            return {
                "run_id": run_id,
                "question": question,
                "chat_history": chat_history,
                "cache_hit": cache_hit
            }
        
        # 历史记录处理 (指代消解)
        if chat_history:
            search_query = self._rewrite_question(question, chat_history)
        else:
            search_query = question
        
        # 智能路由 (使用 LLM 进行意图分类)
        plan_result = self.planner.plan(search_query, chat_history)
        planning_type = plan_result["type"]
        queries = plan_result["queries"]
        
        child_docs = []
        final_docs = []
        context_str = ""
        
        if planning_type == "GREETING":
            # 闲聊模式：不检索，直接生成
            pass
        
        elif planning_type == "COMPLEX":
            # 复杂问题：并发多查询检索
            base_retriever = self._get_base_retriever()
            all_docs = []
            
            with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
                future_to_query = {
                    executor.submit(base_retriever.invoke, q): q 
                    for q in queries
                }
                for future in as_completed(future_to_query):
                    try:
                        docs = future.result()
                        all_docs.extend(docs)
                    except Exception as e:
                        print(f"检索失败: {e}")
            
            child_docs = self._deduplicate_docs(all_docs)
            child_docs = self._rerank_documents(child_docs, search_query)
            final_docs = self._map_children_to_parents(child_docs)
            context_str = "\n\n".join([f"[文档 {i+1}]: {d.page_content}" for i, d in enumerate(final_docs)])
        
        elif planning_type == "ABSTRACT":
            # 抽象问题：使用 HyDE 增强
            hyde_doc = self.planner.generate_hyde_doc(search_query)
            final_query = f"{search_query}\n{hyde_doc}" if hyde_doc else search_query
            
            child_docs = self.retriever.invoke(final_query)
            final_docs = self._map_children_to_parents(child_docs)
            context_str = "\n\n".join([f"[文档 {i+1}]: {d.page_content}" for i, d in enumerate(final_docs)])
        
        else:  # SIMPLE
            base_retriever = self._get_base_retriever()
            all_docs = []
            
            with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
                future_to_query = {
                    executor.submit(base_retriever.invoke, q): q 
                    for q in queries
                }
                for future in as_completed(future_to_query):
                    try:
                        docs = future.result()
                        all_docs.extend(docs)
                    except Exception as e:
                        print(f"检索失败: {e}")
            
            child_docs = self._deduplicate_docs(all_docs)
            child_docs = self._rerank_documents(child_docs, search_query)
            final_docs = self._map_children_to_parents(child_docs)
            context_str = "\n\n".join([f"[文档 {i+1}]: {d.page_content}" for i, d in enumerate(final_docs)])
        
        # 格式化 QA 提示词
        formatted_qa_prompt = self.qa_prompt.invoke({
            "chat_history": chat_history,
            "context": context_str,
            "question": question
        })
        
        return {
            "run_id": run_id,
            "question": question,
            "chat_history": chat_history,
            "search_query": search_query,
            "planning_type": planning_type,
            "queries": queries,
            "child_docs": child_docs,
            "final_docs": final_docs,
            "context_str": context_str,
            "formatted_qa_prompt": formatted_qa_prompt,
            "cache_hit": None
        }

    def stream(self, input_dict: dict):
        """
        流式生成响应的方法
        Yields:
            1. 首先 yield 一个 dict 包含元数据 (source_documents, run_id 等)
            2. 然后 yield 文本 token (str)
        """
        start_time = time.time()
        question = input_dict.get("input", "")
        
        if not question:
            yield {"type": "metadata", "source_documents": [], "run_id": "", "error": "请输入您的问题。"}
            return
        
        # 准备上下文
        ctx = self._prepare_context(input_dict)
        
        # 缓存命中时直接返回
        if ctx.get("cache_hit"):
            yield {
                "type": "metadata",
                "source_documents": [],
                "run_id": ctx["run_id"],
                "cache_hit": True
            }
            yield ctx["cache_hit"]
            return
        
        # Yield 元数据 (包含 source_documents 供前端展示来源)
        yield {
            "type": "metadata",
            "source_documents": ctx["final_docs"],
            "run_id": ctx["run_id"],
            "planning_type": ctx["planning_type"],
            "cache_hit": False
        }
        
        # 流式生成答案
        full_answer = ""
        for chunk in self.llm.stream(ctx["formatted_qa_prompt"]):
            token = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if token:
                full_answer += token
                yield token
        
        end_time = time.time()
        
        # 生成完毕后，处理日志和缓存
        log_data = {
            "run_id": ctx["run_id"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latency": end_time - start_time,
            "question": ctx["question"],
            "rewrite_query": ctx.get("search_query", ctx["question"]),
            "planning_type": ctx.get("planning_type", "UNKNOWN"),
            "expanded_queries": ctx.get("queries", [ctx["question"]]),
            "retrieved_doc_ids": [d.metadata.get("doc_id", "unknown") for d in ctx.get("child_docs", [])],
            "answer": full_answer
        }
        self._save_log(log_data)
        
        # 存入缓存 (闲聊不缓存)
        if ctx.get("planning_type") != "GREETING" and full_answer:
            self._update_cache(ctx["question"], full_answer)

    def invoke(self, input_dict: dict):
        run_id = str(uuid.uuid4())
        start_time = time.time()
        question = input_dict.get("input", "")
        if not question:
            return {"answer": "请输入您的问题。", "source_documents": []}
            
        chat_history = input_dict.get("chat_history", [])

        # 0. 语义缓存 (Phase 4: Semantic Cache - 简单实现)
        cache_hit = self._check_cache(question)
        if cache_hit:
            return {
                "answer": cache_hit,
                "source_documents": [],
                "log_data": {"cache": "hit", "question": question}
            }

        # 1. 历史记录处理 (指代消解)
        if chat_history:
            search_query = self._rewrite_question(question, chat_history)
        else:
            search_query = question

        # 2. 智能路由 (使用 LLM 进行意图分类)
        plan_result = self.planner.plan(search_query, chat_history)
        planning_type = plan_result["type"]
        queries = plan_result["queries"]
        use_hyde = plan_result["use_hyde"]
        
        child_docs = []
        final_docs = []
        context_str = ""
        
        if planning_type == "GREETING":
            # 闲聊模式：不检索，直接生成
            pass
        
        elif planning_type == "COMPLEX":
            # 复杂问题：并发多查询检索
            base_retriever = self._get_base_retriever()
            all_docs = []
            
            # 并发检索所有查询变体
            with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
                future_to_query = {
                    executor.submit(base_retriever.invoke, q): q 
                    for q in queries
                }
                for future in as_completed(future_to_query):
                    try:
                        docs = future.result()
                        all_docs.extend(docs)
                    except Exception as e:
                        print(f"检索失败: {e}")
            
            # 去重
            child_docs = self._deduplicate_docs(all_docs)
            
            # 重排序（如果有 Reranker）
            child_docs = self._rerank_documents(child_docs, search_query)
            
            # 父子索引置换
            final_docs = self._map_children_to_parents(child_docs)
            context_str = "\n\n".join([f"[文档 {i+1}]: {d.page_content}" for i, d in enumerate(final_docs)])
        
        elif planning_type == "ABSTRACT":
            # 抽象问题：使用 HyDE 增强
            hyde_doc = self.planner.generate_hyde_doc(search_query)
            final_query = f"{search_query}\n{hyde_doc}" if hyde_doc else search_query
            
            child_docs = self.retriever.invoke(final_query)
            final_docs = self._map_children_to_parents(child_docs)
            context_str = "\n\n".join([f"[文档 {i+1}]: {d.page_content}" for i, d in enumerate(final_docs)])
        
        else:  # SIMPLE
            # SIMPLE 问题现在也使用并发多查询检索以提高召回率
            base_retriever = self._get_base_retriever()
            all_docs = []
            
            # 并发检索所有查询变体
            with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
                future_to_query = {
                    executor.submit(base_retriever.invoke, q): q 
                    for q in queries
                }
                for future in as_completed(future_to_query):
                    try:
                        docs = future.result()
                        all_docs.extend(docs)
                    except Exception as e:
                        print(f"检索失败: {e}")
            
            # 去重
            child_docs = self._deduplicate_docs(all_docs)
            
            # 重排序（如果有 Reranker）
            child_docs = self._rerank_documents(child_docs, search_query)
            
            # 父子索引置换
            final_docs = self._map_children_to_parents(child_docs)
            context_str = "\n\n".join([f"[文档 {i+1}]: {d.page_content}" for i, d in enumerate(final_docs)])

        # 6. 生成答案
        formatted_qa_prompt = self.qa_prompt.invoke({
            "chat_history": chat_history,
            "context": context_str,
            "question": question
        })

        ai_message = self.llm.invoke(formatted_qa_prompt)
        answer = self.output_parser.invoke(ai_message)

        end_time = time.time()
        
        # 7. 数据埋点记录 (Phase 3: Feedback Loop)
        log_data = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latency": end_time - start_time,
            "question": question,
            "rewrite_query": search_query,
            "planning_type": planning_type,
            "expanded_queries": queries if planning_type in ["COMPLEX", "SIMPLE"] else [search_query],
            "retrieved_doc_ids": [d.metadata.get("doc_id", "unknown") for d in child_docs],
            "answer": answer
        }
        self._save_log(log_data)
        
        # 存入缓存 (闲聊不缓存)
        if planning_type != "GREETING" and answer:
            self._update_cache(question, answer)

        return {
            "answer": answer,
            "source_documents": final_docs,
            "log_data": log_data,
            "run_id": run_id
        }

    def _check_cache(self, question):
        """简单本地缓存检查"""
        cache_path = "./logs/cache.json"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                return cache.get(question)
            except:
                return None
        return None

    def _update_cache(self, question, answer):
        """更新本地缓存"""
        cache_path = "./logs/cache.json"
        os.makedirs("./logs", exist_ok=True)
        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                pass
        
        cache[question] = answer
        # 限制缓存大小
        if len(cache) > 1000:
            first_key = next(iter(cache))
            del cache[first_key]
            
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def _save_log(self, log_data):
        """保存日志用于后续 A/B 测试和评估"""
        if "run_id" not in log_data:
            log_data["run_id"] = str(uuid.uuid4())
        run_id = log_data["run_id"]
        
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "rag_activity.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        return run_id



@st.cache_resource
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )


def get_rag_chain(custom_prompt=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}")

    # 1. 向量检索 (检索子块)
    embeddings = load_embedding_model()
    if not os.path.exists(config.PERSIST_DIRECTORY):
        return None

    vectorstore = Chroma(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    chroma_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # 2. 关键词检索 (检索子块)
    bm25_retriever = None
    if os.path.exists(config.BM25_PERSIST_PATH):
        try:
            with open(config.BM25_PERSIST_PATH, "rb") as f:
                bm25_docs = pickle.load(f)
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = 10
            print(f"BM25 索引已加载，文档数: {len(bm25_docs)}")
        except Exception as e:
            print(f"BM25 加载失败: {e}")

    # 3. 混合检索
    if bm25_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
    else:
        ensemble_retriever = chroma_retriever

    # 4. 重排序 (对子块进行排序)
    # Rerank 应该作用于子块，因为子块语义更集中，评分更准
    final_retriever = ensemble_retriever
    if device == "cpu":
        print("CPU模式：跳过 Rerank 步骤")
    else:
        try:
            print(f"加载 Rerank 模型: {config.RERANKER_MODEL_NAME}")
            rerank_model = HuggingFaceCrossEncoder(
                model_name=config.RERANKER_MODEL_NAME,
                model_kwargs={'device': device}
            )
            compressor = CrossEncoderReranker(model=rerank_model, top_n=5)
            final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )
        except Exception as e:
            print(f"Rerank 初始化失败，降级使用混合检索: {e}")
            final_retriever = ensemble_retriever

    # 5. LLM & Prompt
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        openai_api_key=config.API_KEY,
        openai_api_base=config.BASE_URL,
        temperature=0.1
    )

    # 历史记录重写提示词
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    history_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 问答提示词 (含引用标记指令)
    default_system_prompt = """你是一个专业的助手。请基于下面的【上下文】内容回答用户的问题。
在回答中引用上下文时，请使用 [1], [2] 这样的格式标注来源，对应上下文中的 [文档 1], [文档 2] 等。
如果上下文没有相关信息，且聊天记录也没提到，请承认不知道。

【上下文】:
{context}
"""
    system_template = custom_prompt if custom_prompt else default_system_prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    # 返回支持父子索引的 Chain
    return ManualHistoryRAGChain(final_retriever, qa_prompt, history_prompt, llm)