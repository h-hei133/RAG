import os
import pickle
import config
import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_document(file_paths):
    """
    读取PDF -> 切分 -> 1. 存入向量库 2. 存入 BM25 本地缓存
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_documents = []

    # 1. 加载文档
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()

            # 简单的清洗：去除过短的页面
            cleaned_docs = []
            for doc in docs:
                if len(doc.page_content) > 20:
                    doc.metadata["source"] = os.path.basename(file_path)
                    cleaned_docs.append(doc)
            all_documents.extend(cleaned_docs)
            print(f"已加载: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"加载失败 {file_path}: {e}")

    if not all_documents:
        return False

    # 2. 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    splits = text_splitter.split_documents(all_documents)

    # 3. 初始化 Embedding 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )

    # 4. 处理 BM25 数据 (增量更新)
    # 尝试加载旧的索引，以免覆盖之前的文档数据
    if os.path.exists(config.BM25_PERSIST_PATH):
        try:
            with open(config.BM25_PERSIST_PATH, "rb") as f:
                existing_splits = pickle.load(f)
            splits = existing_splits + splits
        except Exception:
            pass # 加载失败则从头开始

    # 保存新的完整列表
    os.makedirs(os.path.dirname(config.BM25_PERSIST_PATH), exist_ok=True)
    with open(config.BM25_PERSIST_PATH, "wb") as f:
        pickle.dump(splits, f)
    print("BM25 索引数据已更新")

    # 5. 存入 Chroma 向量数据库
    try:
        print("正在写入向量数据库...")
        #persist_directory 已在 config 中定义，Chroma 会自动进行增量添加
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=config.PERSIST_DIRECTORY
        )
        print("向量数据库写入完成")
    except Exception as e:
        print(f"写入向量数据库时发生错误: {e}")
        # 这里不再尝试删除文件夹，避免文件占用导致的错误
        return False

    return True