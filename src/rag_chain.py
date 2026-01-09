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
    手动实现的 RAG 链，用于处理历史记录重写和问答
    """
    def __init__(self, retriever, qa_prompt, history_prompt, llm):
        self.retriever = retriever
        self.qa_prompt = qa_prompt
        self.history_prompt = history_prompt
        self.llm = llm
        self.output_parser = StrOutputParser()

    def _rewrite_question(self, question, chat_history):
        formatted_history_prompt = self.history_prompt.invoke({
            "chat_history": chat_history,
            "input": question
        })
        response = self.llm.invoke(formatted_history_prompt)
        return self.output_parser.invoke(response)

    def invoke(self, input_dict: dict):
        question = input_dict.get("input")
        chat_history = input_dict.get("chat_history", [])

        # 如果有历史记录，重写问题
        if chat_history:
            search_query = self._rewrite_question(question, chat_history)
        else:
            search_query = question

        # 执行检索
        docs = self.retriever.invoke(search_query)
        context_str = "\n\n".join([d.page_content for d in docs]) if docs else ""

        # 生成答案
        formatted_qa_prompt = self.qa_prompt.invoke({
            "chat_history": chat_history,
            "context": context_str,
            "question": question
        })

        ai_message = self.llm.invoke(formatted_qa_prompt)
        answer = self.output_parser.invoke(ai_message)

        return {
            "answer": answer,
            "source_documents": docs
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

    # 1. 向量检索 (Chroma)
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

    # 2. 关键词检索 (BM25)
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

    # 4. 重排序 (Rerank)
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

    return ManualHistoryRAGChain(final_retriever, qa_prompt, history_prompt, llm)