import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 设定向量数据库的存储路径
PERSIST_DIRECTORY = "./chroma_db"


def ingest_document(file_path):
    """
    读取PDF，切分，并存储到本地向量库
    """
    # 1. 加载 PDF
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"已加载文档，共 {len(documents)} 页")

    # 2. 切分文本 (Chunking)
    # chunk_size=500: 每个块约300-500字，适合语义完整性
    # chunk_overlap=50: 重叠50字，防止上下文丢失
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    print(f"文档已切分为 {len(splits)} 个片段")

    # 3. 初始化 Embedding 模型 (本地运行，使用 BAAI 中文小模型)
    print("正在加载本地 Embedding 模型 (首次运行会自动下载)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    # 4. 存入 ChromaDB (如果存在旧数据，先清理，保证是纯净的知识库)
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print("已清理旧的向量数据库")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("向量数据库构建完成！")
    return True


import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config

# 设定向量数据库路径
PERSIST_DIRECTORY = "./chroma_db"


def get_rag_chain():
    """
    初始化 RAG 链
    """
    # 1. 准备 Embedding (必须与 ingestion.py 使用同一个模型)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    # 2. 加载向量数据库
    if not os.path.exists(PERSIST_DIRECTORY):
        return None

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # 3. 创建检索器 (检索最相似的3个片段)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. 初始化 LLM (指向 DeepSeek)
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        openai_api_key=config.API_KEY,
        openai_api_base=config.BASE_URL,
        temperature=0.1  # RAG 需要严谨，温度调低
    )

    # 5. 定义 Prompt 模板
    template = """你是一个专业的知识助手。请基于下面的【上下文】内容回答用户的问题。
    如果上下文中没有相关信息，请直接回答“根据提供的文档，我无法找到相关答案”，不要编造。

    【上下文】:
    {context}

    【用户问题】:
    {question}

    回答:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 6. 构建 LCEL 链 (LangChain Expression Language)
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


import streamlit as st
import os
from src.ingestion import ingest_document
from src.rag_chain import get_rag_chain

st.set_page_config(page_title="个人知识库助手 (DeepSeek版)", layout="wide")

# 标题
st.title("🤖 个人专属知识库助手")
st.caption("Powered by DeepSeek-V3 + Local Embeddings")

# --- 侧边栏：文件上传 ---
with st.sidebar:
    st.header("1. 上传文档")
    uploaded_file = st.file_uploader("请上传 PDF 文档", type=["pdf"])

    if uploaded_file and not st.session_state.get("file_processed", False):
        with st.spinner("正在构建知识库，请稍候..."):
            # 保存临时文件
            temp_path = os.path.join("data", "temp.pdf")
            os.makedirs("data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 调用数据处理逻辑
            success = ingest_document(temp_path)
            if success:
                st.success("知识库构建完成！")
                st.session_state["file_processed"] = True

    if st.button("重置知识库"):
        st.session_state["file_processed"] = False
        st.rerun()

# --- 主区域：聊天 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！请先上传文档，然后问我关于文档的问题。"}]

# 显示历史消息
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 1. 显示用户问题
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. 检查是否已处理文件
    if not st.session_state.get("file_processed"):
        response = "请先在左侧上传 PDF 文档，我才能回答你的问题哦。"
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    else:
        # 3. 调用 RAG 链生成回答
        rag_chain = get_rag_chain()
        if rag_chain:
            with st.chat_message("assistant"):
                with st.spinner("DeepSeek 正在思考..."):
                    try:
                        response = rag_chain.invoke(prompt)
                        st.write(response)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"发生错误: {e}")
        else:
            st.error("知识库初始化失败，请重试。")