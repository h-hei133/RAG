import os
import torch
import pickle
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
        """
        parent_docs = []
        seen_ids = set()

        for child in child_docs:
            # 1. 获取关联 ID
            doc_id = child.metadata.get("doc_id")

            # 如果没有 ID 或者已经添加过该父文档，则跳过 (去重)
            if not doc_id or doc_id in seen_ids:
                # 如果没有 ID (兼容旧数据)，直接用子块
                if not doc_id and child.page_content not in [d.page_content for d in parent_docs]:
                    parent_docs.append(child)
                continue

            # 2. 从磁盘加载父文档
            pkl_path = os.path.join(self.doc_store_path, f"{doc_id}.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        parent_doc = pickle.load(f)
                        parent_docs.append(parent_doc)
                        seen_ids.add(doc_id)
                except Exception as e:
                    print(f"读取父文档失败 {doc_id}: {e}")
                    # 降级处理：如果读不到父块，就暂时用子块顶替
                    parent_docs.append(child)
            else:
                # 找不到文件，降级使用子块
                parent_docs.append(child)

        return parent_docs

    def invoke(self, input_dict: dict):
        question = input_dict.get("input")
        chat_history = input_dict.get("chat_history", [])

        # 1. 历史记录处理
        if chat_history:
            search_query = self._rewrite_question(question, chat_history)
        else:
            search_query = question

        # 2. 执行检索 (此时获取的是子块)
        child_docs = self.retriever.invoke(search_query)

        # 3. 父子索引置换 (Small-to-Big)
        # 将精准的子块替换为上下文完整的父块
        final_docs = self._map_children_to_parents(child_docs)

        # 4. 构建上下文 (使用父块内容)
        context_str = "\n\n".join([d.page_content for d in final_docs]) if final_docs else ""

        # 5. 生成答案
        formatted_qa_prompt = self.qa_prompt.invoke({
            "chat_history": chat_history,
            "context": context_str,
            "question": question
        })

        ai_message = self.llm.invoke(formatted_qa_prompt)
        answer = self.output_parser.invoke(ai_message)

        return {
            "answer": answer,
            # 返回父文档，以便前端展示完整的上下文来源
            "source_documents": final_docs
        }


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

    # 问答提示词
    default_system_prompt = """你是一个专业的助手。请基于下面的【上下文】内容回答用户的问题。
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