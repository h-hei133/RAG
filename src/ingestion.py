import os
import pickle
import uuid
import config
import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def ingest_document(file_paths):
    """
    实现父子索引构建：
    1. 切分父块 (Large Chunks) -> 存入本地 DocStore (用于生成答案)
    2. 切分子块 (Small Chunks) -> 存入 Chroma 和 BM25 (用于检索)
    3. 建立 ID 关联
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # --- 1. 定义切分器 ---
    # 父块：给 LLM 看的上下文，尽量完整 (2000字符)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # 子块：用于搜索的索引，尽量精准 (400字符)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    all_child_docs = []  # 最终要存入向量库的小块

    # 确保父文档存储目录存在
    doc_store_path = getattr(config, "PARENT_DOC_STORE_PATH", "./doc_store")
    os.makedirs(doc_store_path, exist_ok=True)

    # --- 2. 加载与处理 ---
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        try:
            print(f"正在处理文件: {os.path.basename(file_path)}")
            loader = PyMuPDFLoader(file_path)
            raw_docs = loader.load()

            # 步骤 A: 切分成父块
            parent_docs = parent_splitter.split_documents(raw_docs)

            for parent_doc in parent_docs:
                # 步骤 B: 生成唯一 ID 并关联
                doc_id = str(uuid.uuid4())
                parent_doc.metadata["doc_id"] = doc_id
                parent_doc.metadata["source"] = os.path.basename(file_path)

                # 步骤 C: 保存父块到本地存储 (Key-Value 模式: ID -> Document)
                # 使用 pickle 保存，文件名就是 ID
                save_path = os.path.join(doc_store_path, f"{doc_id}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(parent_doc, f)

                # 步骤 D: 将父块切分为子块
                # 注意：这里我们手动切分，以便把 doc_id 传递给子块
                child_docs = child_splitter.split_documents([parent_doc])

                for child in child_docs:
                    # 关键：子块继承父块的 doc_id
                    child.metadata["doc_id"] = doc_id
                    # 也可以继承 source，方便溯源
                    child.metadata["source"] = os.path.basename(file_path)

                all_child_docs.extend(child_docs)

            print(f"  - 生成父块: {len(parent_docs)} 个")
            print(f"  - 生成子块: {len(all_child_docs)} 个 (当前文件累计)")

        except Exception as e:
            print(f"加载处理失败 {file_path}: {e}")

    if not all_child_docs:
        return False

    # --- 3. 初始化 Embedding 模型 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )

    # --- 4. 处理 BM25 数据 (子块) ---
    # 增量更新逻辑：读取旧的 -> 合并 -> 保存
    final_bm25_docs = all_child_docs
    if os.path.exists(config.BM25_PERSIST_PATH):
        try:
            with open(config.BM25_PERSIST_PATH, "rb") as f:
                existing_docs = pickle.load(f)
            final_bm25_docs = existing_docs + all_child_docs
        except Exception:
            pass

    os.makedirs(os.path.dirname(config.BM25_PERSIST_PATH), exist_ok=True)
    with open(config.BM25_PERSIST_PATH, "wb") as f:
        pickle.dump(final_bm25_docs, f)
    print(f"BM25 索引已更新，总计子块数: {len(final_bm25_docs)}")

    # --- 5. 存入 Chroma 向量数据库 (子块) ---
    try:
        print("正在将子块写入向量数据库...")
        vectorstore = Chroma.from_documents(
            documents=all_child_docs,
            embedding=embeddings,
            persist_directory=config.PERSIST_DIRECTORY
        )
        print("向量数据库写入完成")
    except Exception as e:
        print(f"写入向量数据库时发生错误: {e}")
        return False

    return True