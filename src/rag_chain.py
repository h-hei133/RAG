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

# ========== CRAG 纠错检索 (Corrective RAG) ==========
class GradeDocuments(BaseModel):
    """检索结果相关性评分"""
    binary_score: str = Field(
        description="文档是否与问题相关, 'yes' 或 'no'"
    )
    reasoning: str = Field(
        description="评分理由",
        default=""
    )


class DocumentGrader:
    """
    CRAG 文档评分器：评估检索结果质量，过滤不相关文档
    当检索质量差时触发回退策略
    """
    
    GRADING_PROMPT = """你是一个检索质量评估专家。判断下面的文档是否与用户问题相关。

用户问题: {question}

检索到的文档:
{document}

评判标准:
1. 文档是否包含与问题相关的关键词或语义信息
2. 文档内容是否能帮助回答这个问题
3. 即使只是部分相关也应该判定为相关

请返回 JSON 格式:
{{"binary_score": "yes/no", "reasoning": "简短理由"}}"""
    
    def __init__(self, llm):
        self.llm = llm
        self.output_parser = StrOutputParser()
    
    def grade_document(self, question: str, doc_content: str) -> bool:
        """
        评估单个文档是否与问题相关
        返回: True=相关, False=不相关
        """
        try:
            prompt = self.GRADING_PROMPT.format(
                question=question,
                document=doc_content[:1000]  # 限制长度
            )
            response = self.llm.invoke(prompt)
            result_text = self.output_parser.invoke(response)
            
            # 解析 JSON
            import re
            json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("binary_score", "no").lower() == "yes"
            return True  # 解析失败时保守处理
        except Exception as e:
            print(f"文档评分失败: {e}")
            return True  # 失败时保守处理
    
    def grade_and_filter(self, question: str, docs: list, threshold: float = 0.5) -> tuple:
        """
        CRAG 核心逻辑：评估并过滤检索结果
        
        Args:
            question: 用户问题
            docs: 检索到的文档列表
            threshold: 不相关文档比例阈值，超过则触发回退
            
        Returns:
            (filtered_docs, need_fallback, stats)
        """
        if not docs:
            return [], True, {"total": 0, "relevant": 0, "irrelevant": 0}
        
        filtered_docs = []
        irrelevant_count = 0
        
        for doc in docs:
            is_relevant = self.grade_document(question, doc.page_content)
            if is_relevant:
                filtered_docs.append(doc)
            else:
                irrelevant_count += 1
        
        total = len(docs)
        relevant_count = total - irrelevant_count
        need_fallback = (irrelevant_count / total) > threshold if total > 0 else True
        
        stats = {
            "total": total,
            "relevant": relevant_count,
            "irrelevant": irrelevant_count,
            "relevance_ratio": relevant_count / total if total > 0 else 0
        }
        
        return filtered_docs, need_fallback, stats


# ========== 语义缓存 (Semantic Cache) ==========
class SemanticCache:
    """
    语义缓存：使用向量相似度匹配缓存
    相比字符串匹配，可以识别语义相似的问题
    例如: "How are you?" 和 "How are you" 会命中同一缓存
    """
    
    def __init__(self, embeddings, threshold: float = 0.92, max_size: int = 1000):
        """
        Args:
            embeddings: 嵌入模型
            threshold: 相似度阈值 (0-1)，越高越严格
            max_size: 最大缓存条目数
        """
        self.embeddings = embeddings
        self.threshold = threshold
        self.max_size = max_size
        self.cache_path = "./logs/semantic_cache.pkl"
        self.cache_vectors = []  # [(question, question_vector, answer, timestamp)]
        self._load_cache()
    
    def _load_cache(self):
        """从磁盘加载缓存"""
        import pickle
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache_vectors = pickle.load(f)
                print(f"✅ 语义缓存已加载: {len(self.cache_vectors)} 条")
            except Exception as e:
                print(f"加载语义缓存失败: {e}")
                self.cache_vectors = []
    
    def _save_cache(self):
        """保存缓存到磁盘"""
        import pickle
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache_vectors, f)
        except Exception as e:
            print(f"保存语义缓存失败: {e}")
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """计算余弦相似度"""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def get(self, question: str) -> Optional[str]:
        """
        语义缓存查询
        返回: 缓存的答案，未命中返回 None
        """
        if not self.cache_vectors:
            return None
        
        try:
            q_vec = self.embeddings.embed_query(question)
            
            best_match = None
            best_similarity = 0
            
            for cached_q, cached_vec, answer, _ in self.cache_vectors:
                similarity = self._cosine_similarity(q_vec, cached_vec)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (cached_q, answer)
            
            if best_similarity >= self.threshold:
                print(f"🎯 语义缓存命中: {best_similarity:.2%} 相似度")
                return best_match[1]
            
            return None
        except Exception as e:
            print(f"语义缓存查询失败: {e}")
            return None
    
    def set(self, question: str, answer: str):
        """添加到语义缓存"""
        try:
            q_vec = self.embeddings.embed_query(question)
            timestamp = time.time()
            
            # 检查是否已存在非常相似的问题
            for i, (cached_q, cached_vec, _, _) in enumerate(self.cache_vectors):
                similarity = self._cosine_similarity(q_vec, cached_vec)
                if similarity >= 0.98:  # 几乎完全相同，更新答案
                    self.cache_vectors[i] = (question, q_vec, answer, timestamp)
                    self._save_cache()
                    return
            
            # 添加新缓存
            self.cache_vectors.append((question, q_vec, answer, timestamp))
            
            # 限制缓存大小 (LRU: 删除最旧的)
            if len(self.cache_vectors) > self.max_size:
                self.cache_vectors.sort(key=lambda x: x[3])  # 按时间排序
                self.cache_vectors = self.cache_vectors[-self.max_size:]
            
            self._save_cache()
        except Exception as e:
            print(f"语义缓存写入失败: {e}")


# ========== Token 管理器 ==========
class TokenManager:
    """
    Token 管理器：确保上下文不超过模型限制
    支持 tiktoken 精确计数和字符估算回退
    """
    
    def __init__(self, model_name: str = "gpt-4", max_context_tokens: int = 6000):
        """
        Args:
            model_name: 模型名称，用于选择正确的 tokenizer
            max_context_tokens: 上下文最大 token 数
        """
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.encoder = None
        self._init_encoder()
    
    def _init_encoder(self):
        """初始化 tokenizer"""
        try:
            import tiktoken
            # 尝试获取模型对应的编码器
            try:
                self.encoder = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # 如果模型不支持，使用 cl100k_base (GPT-4 系列)
                self.encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            print("⚠️ tiktoken 未安装，使用字符估算 token 数")
            self.encoder = None
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # 估算：中文约 2 字符/token，英文约 4 字符/token
            # 保守估计使用 2
            return len(text) // 2
    
    def trim_context(self, context_str: str, reserve_tokens: int = 500) -> str:
        """
        裁剪上下文以适应 token 限制
        
        Args:
            context_str: 原始上下文
            reserve_tokens: 为问题和回答保留的 token 数
            
        Returns:
            裁剪后的上下文
        """
        max_allowed = self.max_context_tokens - reserve_tokens
        current_tokens = self.count_tokens(context_str)
        
        if current_tokens <= max_allowed:
            return context_str
        
        # 需要裁剪
        print(f"⚠️ 上下文过长 ({current_tokens} tokens)，正在裁剪至 {max_allowed} tokens")
        
        if self.encoder:
            # 精确裁剪
            tokens = self.encoder.encode(context_str)
            truncated_tokens = tokens[:max_allowed]
            truncated_text = self.encoder.decode(truncated_tokens)
        else:
            # 字符估算裁剪
            char_limit = max_allowed * 2
            truncated_text = context_str[:char_limit]
        
        return truncated_text + "\n\n[注: 上下文已截断以适应模型限制]"
    
    def trim_documents(self, docs: list, max_docs: int = 5) -> list:
        """
        限制文档数量和总长度
        
        Args:
            docs: 文档列表
            max_docs: 最大文档数
            
        Returns:
            裁剪后的文档列表
        """
        if len(docs) <= max_docs:
            return docs
        
        # 按相关性排序（假设已经排序），取前 N 个
        return docs[:max_docs]

class RouteQuery(BaseModel):
    """用户查询意图分类"""
    intent: Literal["GREETING", "SIMPLE", "COMPLEX", "ABSTRACT", "METADATA_QUERY", "COMPARE", "SUMMARIZE", "OUT_OF_DOMAIN"] = Field(
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
        返回: GREETING | SIMPLE | COMPLEX | ABSTRACT | METADATA_QUERY | COMPARE | SUMMARIZE | OUT_OF_DOMAIN
        """
        routing_prompt = f"""分析用户问题，判断其意图类型。

问题: "{question}"

类型判断标准:
1. GREETING: 打招呼或闲聊（如：你好、谢谢、再见、hi、hello）
2. SIMPLE: 事实性简单问题，只需单一概念查询（如：XX是什么时候发布的？）
3. COMPLEX: 涉及多跳推理或需要多角度回答（如：这个技术如何影响XX？）
4. ABSTRACT: 概念性问题，适合先生成假设性答案再检索（如：什么是量子纠缠？如何理解XX？）
5. METADATA_QUERY: 询问知识库元信息（如：有哪些文档？文件列表？）
6. COMPARE: 对比类问题（如：A和B有什么区别？X好还是Y好？）
7. SUMMARIZE: 总结类问题（如：总结这篇文档、概括主要内容）
8. OUT_OF_DOMAIN: 与知识库无关的问题（如：今天天气怎么样？）

请只返回一个 JSON 格式:
{{"intent": "类型", "reasoning": "理由"}}"""
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
                valid_intents = ["GREETING", "SIMPLE", "COMPLEX", "ABSTRACT", 
                                "METADATA_QUERY", "COMPARE", "SUMMARIZE", "OUT_OF_DOMAIN"]
                if intent in valid_intents:
                    return intent
            return "SIMPLE"
        except Exception as e:
            print(f"意图分类失败: {e}")
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
            "use_hyde": intent == "ABSTRACT",
            "sub_questions": [],
            "skip_retrieval": False
        }
        
        # 根据意图类型选择不同策略
        if intent == "GREETING":
            result["skip_retrieval"] = True
        elif intent == "OUT_OF_DOMAIN":
            result["skip_retrieval"] = True
        elif intent == "METADATA_QUERY":
            result["skip_retrieval"] = True  # 直接查询数据库元信息
        elif intent in ["COMPLEX", "COMPARE"]:
            # 复杂/对比问题使用子问题分解
            sub_questions = self.decompose_complex_query(question)
            result["sub_questions"] = sub_questions
            result["queries"] = [question] + sub_questions
        elif intent == "SIMPLE":
            # SIMPLE 问题使用查询扩展
            variants = self.expand_query(question)
            result["queries"] = [question] + variants
        elif intent == "SUMMARIZE":
            # 总结问题：不扩展查询，使用更大的检索范围
            result["queries"] = [question]
        # ABSTRACT: 使用 HyDE，已在初始化时设置
        
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

    def decompose_complex_query(self, question: str) -> List[str]:
        """
        将复杂问题分解为可独立回答的子问题
        适用于对比类、多跳推理类问题
        例如: "华为和小米的手机哪个好?" -> ["华为手机有什么特点?", "小米手机有什么特点?", "华为小米手机对比"]
        """
        prompt = f"""将以下复杂问题分解为2-4个可独立回答的子问题。
每个子问题应该可以通过单独的知识库检索来回答。

问题: {question}

要求:
1. 每个子问题占一行
2. 子问题应该简洁明了
3. 不要编号
4. 不要添加解释

示例:
问题: "比较Python和Java在机器学习领域的应用"
Python在机器学习中的优势
Java在机器学习中的应用
Python和Java性能对比"""
        
        try:
            response = self.llm.invoke(prompt)
            result = self.output_parser.invoke(response)
            sub_questions = [q.strip() for q in result.strip().split("\n") if q.strip()]
            # 过滤掉太短或太长的子问题
            sub_questions = [q for q in sub_questions if 4 < len(q) < 100]
            return sub_questions[:4]  # 最多返回4个
        except Exception as e:
            print(f"子问题分解失败: {e}")
            return [question]  # 失败时返回原问题


class ManualHistoryRAGChain:
    """
    手动实现的 RAG 链，集成父子索引策略 (Small-to-Big Retrieval)
    """

    def __init__(self, retriever, qa_prompt, history_prompt, llm, embeddings=None):
        self.retriever = retriever
        self.qa_prompt = qa_prompt
        self.history_prompt = history_prompt
        self.llm = llm
        self.output_parser = StrOutputParser()
        # 获取父文档存储路径
        self.doc_store_path = getattr(config, "PARENT_DOC_STORE_PATH", "./doc_store")
        self.planner = QueryPlanner(llm)
        # CRAG 文档评分器
        self.grader = DocumentGrader(llm)
        # 是否启用 CRAG
        self.use_crag = True
        # 语义缓存 (需要传入 embeddings)
        self.semantic_cache = SemanticCache(embeddings) if embeddings else None
        self.use_semantic_cache = embeddings is not None
        # Token 管理器
        self.token_manager = TokenManager(max_context_tokens=6000)

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

    def _apply_crag(self, question: str, docs: list, search_query: str) -> tuple:
        """
        应用 CRAG (Corrective RAG) 纠错检索
        
        Args:
            question: 用户原始问题
            docs: 检索到的文档
            search_query: 重写后的搜索查询
            
        Returns:
            (filtered_docs, crag_stats)
        """
        if not self.use_crag or not docs:
            return docs, {"crag_enabled": False}
        
        # 评估文档质量
        filtered_docs, need_fallback, stats = self.grader.grade_and_filter(
            question, docs, threshold=0.5
        )
        
        crag_stats = {
            "crag_enabled": True,
            "original_count": stats["total"],
            "filtered_count": len(filtered_docs),
            "relevance_ratio": stats["relevance_ratio"],
            "fallback_triggered": need_fallback
        }
        
        # 如果需要回退且过滤后文档太少
        if need_fallback and len(filtered_docs) < 2:
            # 回退策略 1: 使用 HyDE 重试
            print(f"⚠️ CRAG 触发回退: 相关性比例 {stats['relevance_ratio']:.1%}")
            
            hyde_doc = self.planner.generate_hyde_doc(question)
            if hyde_doc:
                enhanced_query = f"{search_query}\n{hyde_doc}"
                base_retriever = self._get_base_retriever()
                retry_docs = base_retriever.invoke(enhanced_query)
                
                # 再次评分
                retry_filtered, _, retry_stats = self.grader.grade_and_filter(
                    question, retry_docs, threshold=0.7
                )
                
                if retry_filtered:
                    filtered_docs.extend(retry_filtered)
                    filtered_docs = self._deduplicate_docs(filtered_docs)
                    crag_stats["hyde_retry"] = True
                    crag_stats["retry_added"] = len(retry_filtered)
        
        # 如果仍然没有足够的文档，返回原始文档的前几个
        if len(filtered_docs) < 1:
            filtered_docs = docs[:3]
            crag_stats["fallback_to_original"] = True
        
        return filtered_docs, crag_stats

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
        crag_stats = {"crag_enabled": False}
        
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
            
            # CRAG: 评估并过滤文档
            child_docs, crag_stats = self._apply_crag(question, child_docs, search_query)
            
            final_docs = self._map_children_to_parents(child_docs)
            # Token 管理：限制文档数量和上下文长度
            final_docs = self.token_manager.trim_documents(final_docs, max_docs=5)
            context_str = "\n\n".join([f"[文档 {i+1}]: {d.page_content}" for i, d in enumerate(final_docs)])
            context_str = self.token_manager.trim_context(context_str)
        
        elif planning_type == "ABSTRACT":
            # 抽象问题：使用 HyDE 增强
            hyde_doc = self.planner.generate_hyde_doc(search_query)
            final_query = f"{search_query}\n{hyde_doc}" if hyde_doc else search_query
            
            child_docs = self.retriever.invoke(final_query)
            
            # CRAG: 评估并过滤文档
            child_docs, crag_stats = self._apply_crag(question, child_docs, search_query)
            
            final_docs = self._map_children_to_parents(child_docs)
            # Token 管理：限制文档数量和上下文长度
            final_docs = self.token_manager.trim_documents(final_docs, max_docs=5)
            context_str = "\n\n".join([f"[文档 {i+1}]: {d.page_content}" for i, d in enumerate(final_docs)])
            context_str = self.token_manager.trim_context(context_str)
        
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
            
            # CRAG: 评估并过滤文档
            child_docs, crag_stats = self._apply_crag(question, child_docs, search_query)
            
            final_docs = self._map_children_to_parents(child_docs)
            # Token 管理：限制文档数量和上下文长度
            final_docs = self.token_manager.trim_documents(final_docs, max_docs=5)
            context_str = "\n\n".join([f"[文档 {i+1}]: {d.page_content}" for i, d in enumerate(final_docs)])
            context_str = self.token_manager.trim_context(context_str)
        
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
            "cache_hit": None,
            "crag_stats": crag_stats
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
        """检查缓存 - 优先使用语义缓存"""
        # 优先使用语义缓存
        if self.use_semantic_cache and self.semantic_cache:
            result = self.semantic_cache.get(question)
            if result:
                return result
        
        # 回退到字符串缓存
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
        """更新缓存 - 同时更新语义缓存和字符串缓存"""
        # 更新语义缓存
        if self.use_semantic_cache and self.semantic_cache:
            self.semantic_cache.set(question, answer)
        
        # 同时更新字符串缓存 (作为备份)
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

    # 返回支持父子索引的 Chain (传入 embeddings 以启用语义缓存)
    return ManualHistoryRAGChain(final_retriever, qa_prompt, history_prompt, llm, embeddings=embeddings)