import os
import torch
import config
import streamlit as st  # 引入 streamlit 用于缓存

# LangChain 核心组件
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# 注意：我们移除了 RunnablePassthrough 等复杂的链式组件，改用手动执行
from langchain_core.output_parsers import StrOutputParser

# 设定向量数据库路径
PERSIST_DIRECTORY = "./chroma_db"


# --- 1. 手动模式的 RAG 链类 (核心修复) ---
class ManualRAGChain:
    """
    手动执行 RAG 流程，不依赖复杂的 LangChain 自动流水线。
    这能彻底避免 'NoneType' 和 Pydantic 版本冲突错误，同时保持极高的灵活性。
    """

    def __init__(self, retriever, prompt, llm):
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm
        self.output_parser = StrOutputParser()

    def invoke(self, question: str):
        """
        手动控制每一步执行
        """
        # 1. 检索 (直接调用检索器，获取 List[Document])
        # print(f"DEBUG: 正在检索... (MMR模式)")
        docs = self.retriever.invoke(question)

        # 2. 格式化上下文 (纯 Python 字符串处理，绝对安全)
        if not docs:
            context_str = ""
        else:
            context_str = "\n\n".join([d.page_content for d in docs])

        # 3. 填充 Prompt
        # 将 context 和 question 填入模板
        formatted_prompt = self.prompt.invoke({
            "context": context_str,
            "question": question
        })

        # 4. LLM 生成
        ai_message = self.llm.invoke(formatted_prompt)

        # 5. 解析结果
        answer = self.output_parser.invoke(ai_message)

        # 6. 返回标准字典格式 (兼容前端)
        return {
            "answer": answer,
            "source_documents": docs
        }


# --- 2. 缓存 Embedding 模型 (保留你的优化) ---
@st.cache_resource
def load_embedding_model():
    """
    使用 Streamlit 缓存加载 Embedding 模型，避免每次刷新页面都重新加载。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在初始化 Embedding 模型 (设备: {device})...")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )


# --- 3. 初始化函数 ---
def get_rag_chain(custom_prompt=None):
    """
    初始化 RAG 链 (Manual Mode + High Performance Cache)
    """
    # 1. 获取缓存的 Embedding 模型
    embeddings = load_embedding_model()

    # 2. 加载向量数据库
    if not os.path.exists(PERSIST_DIRECTORY):
        return None

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # 3. 创建检索器 (保留了你的 MMR 配置)
    # MMR (最大边际相关性) 能在保持相关性的同时增加内容的多样性
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,  # 最终返回 8 个片段
            "fetch_k": 20,  # 初始搜索 20 个片段进行筛选
            "lambda_mult": 0.7  # 多样性系数 (越小越多样)
        }
    )

    # 4. 初始化 LLM
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        openai_api_key=config.API_KEY,
        openai_api_base=config.BASE_URL,
        temperature=0.1
    )

    # 5. 动态 Prompt
    if custom_prompt:
        # 简单的占位符检查
        if "{context}" not in custom_prompt or "{question}" not in custom_prompt:
            custom_prompt += "\n\n【上下文】:\n{context}\n\n【用户问题】:\n{question}"
        template = custom_prompt
    else:
        template = """你是一个专业的各个领域专家。请基于下面的【上下文】内容回答用户的问题。

            注意：
            1. 请专注于正文中的技术原理。
            2. 如果上下文中包含英文内容，请用中文进行总结和回答。

            【上下文】:
            {context}

            【用户问题】:
            {question}

            回答:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 6. 返回手动组装的链对象
    # 这里不再使用 | 符号连接，而是直接实例化我们定义的类
    print("✅ RAG Chain 构建成功 (Manual Mode)")
    return ManualRAGChain(retriever, prompt, llm)