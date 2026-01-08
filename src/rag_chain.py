# src/rag_chain.py
import os
import torch
import config
import streamlit as st
from operator import itemgetter

# LangChain 核心组件
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# Rerank 相关组件
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 设定向量数据库路径
PERSIST_DIRECTORY = "./chroma_db"


# --- 1. 手动模式：支持历史感知 + Rerank 的 RAG 链 ---
class ManualHistoryRAGChain:
    """
    手动执行带历史记录 + 重排序的 RAG 流程。
    """

    def __init__(self, retriever, qa_prompt, history_prompt, llm):
        self.retriever = retriever
        self.qa_prompt = qa_prompt  # 用于回答问题的 Prompt
        self.history_prompt = history_prompt  # 用于重写问题的 Prompt
        self.llm = llm
        self.output_parser = StrOutputParser()

    def _rewrite_question(self, question, chat_history):
        """
        内部辅助函数：利用 LLM 重写问题
        """
        # print(f"DEBUG: 正在根据历史记录重写问题...")
        formatted_history_prompt = self.history_prompt.invoke({
            "chat_history": chat_history,
            "input": question
        })
        response = self.llm.invoke(formatted_history_prompt)
        standalone_question = self.output_parser.invoke(response)
        # print(f"DEBUG: 问题重写结果: {standalone_question}")
        return standalone_question

    def invoke(self, input_dict: dict):
        """
        执行主流程
        input_dict 结构: {"input": "用户问题", "chat_history": [消息列表]}
        """
        question = input_dict.get("input")
        chat_history = input_dict.get("chat_history", [])

        # --- 步骤 1: 确定用于检索的问题 ---
        if chat_history:
            # 如果有历史，先重写问题
            search_query = self._rewrite_question(question, chat_history)
        else:
            # 如果没历史，直接用原问题
            search_query = question

        # --- 步骤 2: 检索文档 (集成 Rerank) ---
        # 这里的 self.retriever 已经是包含 Rerank 的检索器了
        # print(f"DEBUG: 正在检索: {search_query}")
        docs = self.retriever.invoke(search_query)

        # --- 步骤 3: 格式化上下文 ---
        context_str = "\n\n".join([d.page_content for d in docs]) if docs else ""

        # --- 步骤 4: 生成最终回答 ---
        # 将 chat_history 也传给最终 Prompt，让 LLM 看到完整的上下文
        formatted_qa_prompt = self.qa_prompt.invoke({
            "chat_history": chat_history,
            "context": context_str,
            "question": question  # 回答时使用用户的原始问题
        })

        ai_message = self.llm.invoke(formatted_qa_prompt)
        answer = self.output_parser.invoke(ai_message)

        return {
            "answer": answer,
            "source_documents": docs
        }


# --- 2. 缓存 Embedding 模型 ---
@st.cache_resource
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在初始化 Embedding 模型 (设备: {device})...")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )


# --- 3. 初始化函数 ---
def get_rag_chain(custom_prompt=None):
    # 1. 准备 Embedding
    embeddings = load_embedding_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(PERSIST_DIRECTORY):
        return None

    # 2. 基础检索器 (Chroma)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # 基础检索：先召回 20 个片段 (fetch_k)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.7}
    )

    # 3. 集成 Rerank (重排序)
    try:
        print(f"正在加载 Rerank 模型: {config.RERANKER_MODEL_NAME}")
        rerank_model = HuggingFaceCrossEncoder(
            model_name=config.RERANKER_MODEL_NAME,
            model_kwargs={'device': device}
        )
        # 从 20 个里挑出最相关的 5 个
        compressor = CrossEncoderReranker(model=rerank_model, top_n=5)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    except Exception as e:
        print(f"⚠️ Rerank 模型加载失败 ({e})，降级为普通检索...")
        final_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. 初始化 LLM
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        openai_api_key=config.API_KEY,
        openai_api_base=config.BASE_URL,
        temperature=0.1
    )

    # --- 定义 Prompts ---

    # A. 历史重写 Prompt (System Prompt)
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

    # B. 最终回答 Prompt
    if custom_prompt:
        # 稍微调整以适应 ChatPromptTemplate.from_messages
        # 这里为了简单，如果用户自定义了 prompt，我们假设他是作为一个 System Message 传入
        system_template = custom_prompt
    else:
        system_template = """你是一个专业的助手。请基于下面的【上下文】内容回答用户的问题。
        如果上下文没有相关信息，且聊天记录也没提到，请承认不知道。

        【上下文】:
        {context}
        """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("chat_history"),  # 注入历史
        ("human", "{question}"),
    ])

    print("✅ RAG Chain 构建成功 (History + Rerank + Manual)")

    return ManualHistoryRAGChain(
        retriever=final_retriever,
        qa_prompt=qa_prompt,
        history_prompt=history_prompt,
        llm=llm
    )