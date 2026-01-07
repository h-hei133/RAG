import os
from langchain_community.document_loaders import PyMuPDFLoader  # 【修改】换用更快更准的 PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config
import torch

# 设定向量数据库的存储路径
PERSIST_DIRECTORY = "./chroma_db"


def ingest_document(file_paths):
    """
    读取PDF列表，切分，并存储到本地向量库
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_documents = []
    
    # 1. 批量加载 PDF
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
            
        try:
            # 使用 PyMuPDFLoader，速度快且对双栏论文支持更好
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            # 清洗数据：去除过短的页眉页脚噪音
            cleaned_docs = []
            for doc in docs:
                if len(doc.page_content) > 50: # 忽略只有几十个字的页面
                    doc.metadata["source"] = os.path.basename(file_path)
                    cleaned_docs.append(doc)
            
            all_documents.extend(cleaned_docs)
            print(f"已加载文档 {os.path.basename(file_path)}，有效页数: {len(cleaned_docs)}")
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")

    if not all_documents:
        return False

    # 2. 切分文本 (Chunking)
    # 稍微减小块大小，提高检索精准度
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,    # 降到 800，更聚焦
        chunk_overlap=150, # 保持重叠
        separators=["\n\n", "\n", "。", "！", "？", " ", ""] # 优化中文切分符
    )
    splits = text_splitter.split_documents(all_documents)
    print(f"所有文档已切分为 {len(splits)} 个片段")

    # 3. 初始化 Embedding 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载 Embedding 模型 ({device})...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )

    # 4. 存入 ChromaDB
    print("正在写入向量数据库...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("向量数据库构建完成！")
    return True