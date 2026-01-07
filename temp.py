import streamlit as st
import os
import time  # 导入 time 模块用于延迟
import shutil # 用于删除文件夹
import gc # 用于垃圾回收
import sys # 用于获取命令行参数
from src.ingestion import ingest_document
from src.rag_chain import get_rag_chain
import config
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 命令行参数处理 ---
should_clear = "--clear" in sys.argv

# 只有在脚本第一次运行时才执行清理
if should_clear and "has_cleared" not in st.session_state:
    print("检测到 --clear 参数，正在清理旧数据...")
    # 清理临时文件夹
    if os.path.exists("data"):
        try:
            shutil.rmtree("data")
            print("已删除: data 文件夹")
        except Exception as e:
            print(f"删除 data 文件夹失败: {e}")

    # 清理数据库
    if os.path.exists("./chroma_db"):
        try:
            shutil.rmtree("./chroma_db", ignore_errors=True)
            print("已尝试删除: ./chroma_db")
        except Exception as e:
            print(f"删除数据库文件夹失败: {e}")

    print("清理完成！")
    st.session_state["has_cleared"] = True

st.set_page_config(page_title="个人知识库助手 (DeepSeek版)", layout="wide")

# 标题
st.title("🤖 个人专属知识库助手")
st.caption("Powered by DeepSeek-V3 + Local Embeddings")

# --- 初始化 Session State ---
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()

# --- 侧边栏：文件上传与控制 ---
with st.sidebar:
    st.header("1. 上传文档")

    # 支持多文件上传
    uploaded_files = st.file_uploader(
        "请上传 PDF 文档 (支持多选)",
        type=["pdf"],
        accept_multiple_files=True
    )

    # 检查是否已有已处理的文件（从上次运行遗留）
    # 逻辑：如果没有上传新文件，但本地有 data 目录且里面有文件，且有数据库，说明是旧会话
    has_existing_data = os.path.exists("data") and len(os.listdir("data")) > 0 and os.path.exists("./chroma_db")

    if not uploaded_files and not st.session_state.get("file_processed", False):
        if has_existing_data:
             st.info("检测到上次运行的知识库，已自动加载。")
             st.session_state["file_processed"] = True
             # 尝试恢复已处理文件列表（简单读取 data 目录）
             for f in os.listdir("data"):
                 st.session_state["processed_files"].add(f)

    # 处理新上传的文件 (增量处理逻辑)
    if uploaded_files:
        # 1. 筛选出尚未处理的新文件
        new_files = [f for f in uploaded_files if f.name not in st.session_state["processed_files"]]

        if new_files:
            # 2. 检查文件数量限制 (已处理 + 新增)
            current_count = len(st.session_state["processed_files"])
            if current_count + len(new_files) > config.MAX_FILES_COUNT:
                st.error(f"❌ 文件数量超过限制！当前已处理 {current_count} 个，尝试新增 {len(new_files)} 个，上限 {config.MAX_FILES_COUNT} 个。")
            else:
                # 3. 检查总容量限制 (粗略计算，只计算新增的，严格来说应该累加所有)
                # 这里为了性能，只检查新增文件是否过大，或者累加 uploaded_files 的总大小
                total_size_mb = sum([f.size for f in uploaded_files]) / (1024 * 1024)

                if total_size_mb > config.MAX_FILE_SIZE_MB:
                    st.error(f"❌ 所有文件总大小超过限制！当前: {total_size_mb:.2f}MB, 最大允许: {config.MAX_FILE_SIZE_MB}MB")
                else:
                    # 4. 开始处理新文件
                    with st.spinner(f"正在处理 {len(new_files)} 个新文档，请稍候..."):
                        # 创建数据目录
                        os.makedirs("data", exist_ok=True)

                        saved_file_paths = []
                        for file in new_files:
                            # 保存每个文件
                            file_path = os.path.join("data", file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                            saved_file_paths.append(file_path)

                        # 调用数据处理逻辑 (只传入新文件)
                        success = ingest_document(saved_file_paths)
                        if success:
                            st.success(f"成功添加 {len(saved_file_paths)} 个新文档！")
                            # 更新状态
                            st.session_state["file_processed"] = True
                            for f in new_files:
                                st.session_state["processed_files"].add(f.name)

        # 如果没有新文件，但 uploaded_files 存在，说明都是老文件，不做操作

    # 分割线
    st.divider()

    # --- 【新增】LLM 角色设定 ---
    st.header("2. AI 角色设定")

    default_prompt = """你是一个专业的助手。请基于下面的【上下文】内容回答用户的问题。
    如果不知道，请直接承认。

    【上下文】:
    {context}

    【用户问题】:
    {question}
    """
    user_prompt = st.text_area(
        "自定义系统提示词 (Prompt)",
        value=default_prompt,
        height=200,
        help="你可以在这里定义AI的角色。必须保留 {context} 和 {question} 这两个占位符。"
    )

    st.divider()

    # 重置按钮
    if st.button("🔄 重置知识库"):
        try:
            gc.collect()

            # 1. 删除 data 文件夹下的所有文件
            if os.path.exists("data"):
                try:
                    shutil.rmtree("data")
                    os.makedirs("data", exist_ok=True) # 重建空目录
                    st.toast("临时文件已删除", icon="🗑️")
                except Exception as e:
                    st.warning(f"⚠️ 临时文件被占用: {e}")

            # 2. 清空数据库
            if os.path.exists("./chroma_db"):
                try:
                    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
                    vectorstore = Chroma(
                        persist_directory="./chroma_db",
                        embedding_function=embeddings
                    )
                    vectorstore.delete_collection()

                    vectorstore = None
                    embeddings = None
                    gc.collect()

                    try:
                        shutil.rmtree("./chroma_db")
                        st.toast("✅ 知识库文件已彻底删除", icon="🗑️")
                    except Exception:
                        st.toast("✅ 知识库已清空", icon="👌")

                except Exception as e:
                    st.error(f"数据库清理失败: {e}")
            else:
                st.toast("知识库本来就是空的", icon="🤷")

            # 3. 重置状态
            st.session_state["file_processed"] = False
            st.session_state["processed_files"] = set() # 清空已处理文件列表
            st.session_state["messages"] = [{"role": "assistant", "content": "你好！请先上传文档，然后问我关于文档的问题。"}]

            st.success("重置完成！页面即将刷新...")
            time.sleep(2)
            st.rerun()

        except Exception as e:
            st.error(f"重置过程发生未知错误: {e}")

    # --- 退出按钮 ---
    if st.button("🔴 退出系统"):
        st.warning("程序正在关闭...")
        time.sleep(1)
        os._exit(0)

    # --- 主区域：聊天 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！请先上传文档，然后问我关于文档的问题。"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("请输入你的问题..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not st.session_state.get("file_processed"):
        response = "请先在左侧上传 PDF 文档，我才能回答你的问题哦。"
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    else:
        rag_chain = get_rag_chain(custom_prompt=user_prompt)
        if rag_chain:
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                status_placeholder.markdown("🔍 正在检索文档并生成回答...")

                try:
                    result = rag_chain.invoke(prompt)
                    answer = result["answer"]
                    source_docs = result["source_documents"]

                    status_placeholder.empty()
                    st.markdown(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})

                    with st.expander("📚 参考来源 (点击展开)"):
                        for i, doc in enumerate(source_docs):
                            source = os.path.basename(doc.metadata.get("source", "未知文件"))
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**来源 {i + 1}:** `{source}` (第 {page} 页)")
                            st.caption(f"原文片段: ...{doc.page_content[:150].replace(chr(10), ' ')}...")
                            st.divider()

                except Exception as e:
                    st.error(f"发生错误: {e}")
        else:
            st.error("知识库初始化失败，请重试。")