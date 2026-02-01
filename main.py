import streamlit as st
import os
import time
import shutil
import gc
import sys
import chromadb
from src.ingestion import ingest_document
from src.rag_chain import get_rag_chain, log_feedback
import config
from langchain_core.messages import AIMessage, HumanMessage

# --- 1. 核心状态初始化 ---
if "has_cleared" not in st.session_state:
    st.session_state["has_cleared"] = False

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# --- 2. 命令行参数处理 ---
if "--clear" in sys.argv and not st.session_state["has_cleared"]:
    st.session_state["has_cleared"] = True
    print("检测到 --clear 参数，尝试清理数据...")

    # 物理清理 data
    if os.path.exists("data"):
        try:
            shutil.rmtree("data")
            print("已删除: data 文件夹")
        except Exception as e:
            print(f"删除 data 失败: {e}")

    # 物理清理 数据库
    if os.path.exists(config.PERSIST_DIRECTORY):
        try:
            shutil.rmtree(config.PERSIST_DIRECTORY, ignore_errors=True)
            print(f"已清理数据库: {config.PERSIST_DIRECTORY}")
        except Exception as e:
            print(f"删除数据库失败: {e}")

    # 物理清理 父文档存储 (DocStore)
    doc_store_path = getattr(config, "PARENT_DOC_STORE_PATH", "doc_store")
    if os.path.exists(doc_store_path):
        try:
            shutil.rmtree(doc_store_path, ignore_errors=True)
            print(f"已清理父文档存储: {doc_store_path}")
        except Exception as e:
            print(f"删除父文档存储失败: {e}")

    # 物理清理 提取的图片存储
    img_store_path = getattr(config, "IMG_STORE_PATH", "extracted_images")
    if os.path.exists(img_store_path):
        try:
            shutil.rmtree(img_store_path, ignore_errors=True)
            print(f"已清理提取图片存储: {img_store_path}")
        except Exception as e:
            print(f"删除提取图片存储失败: {e}")


# --- 3. 重置函数定义 ---
def reset_app():
    """
    重置应用：逻辑清空数据库 -> 清理缓存 -> 删除原始文件 -> 删除中间产物
    """
    print("执行重置中...")

    # 1. 逻辑清空向量数据库 (API 方式)
    if os.path.exists(config.PERSIST_DIRECTORY):
        try:
            print("正在通过 API 清空向量库...")
            client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)
            try:
                client.delete_collection("langchain")
                print("✅ 已删除 'langchain' 集合")
            except ValueError:
                print("集合不存在，无需删除")
            except Exception as e:
                print(f"删除集合时出错: {e}")
        except Exception as e:
            print(f"连接数据库失败: {e}")

    # 2. 物理删除 BM25 索引
    if os.path.exists(config.BM25_PERSIST_PATH):
        try:
            os.remove(config.BM25_PERSIST_PATH)
            print("已删除 BM25 索引文件")
        except Exception as e:
            print(f"删除 BM25 失败: {e}")

    # 3. 物理删除 SQLite 父文档存储
    sqlite_path = getattr(config, "SQLITE_DB_PATH", "./doc_store.db")
    if os.path.exists(sqlite_path):
        try:
            os.remove(sqlite_path)
            print(f"已删除 SQLite 数据库: {sqlite_path}")
        except Exception as e:
            print(f"删除 SQLite 数据库失败: {e}")

    # 4. 物理删除旧的 pkl 父文档存储 (向后兼容)
    doc_store_path = getattr(config, "PARENT_DOC_STORE_PATH", "doc_store")
    if os.path.exists(doc_store_path):
        try:
            shutil.rmtree(doc_store_path)
            print(f"已删除旧父文档存储: {doc_store_path}")
        except Exception as e:
            st.error(f"无法删除父文档存储 {doc_store_path}: {e}")

    # 4. 物理删除 提取的图片存储
    img_store_path = getattr(config, "IMG_STORE_PATH", "extracted_images")
    if os.path.exists(img_store_path):
        try:
            shutil.rmtree(img_store_path)
            print(f"已删除提取图片存储: {img_store_path}")
        except Exception as e:
            st.error(f"无法删除提取图片存储 {img_store_path}: {e}")

    # 5. 物理删除 data 文件夹
    target = "data"
    if os.path.exists(target) and os.path.isdir(target):
        try:
            shutil.rmtree(target)
            print(f"已删除文件夹: {target}")
        except Exception as e:
            st.error(f"无法删除 {target}，可能文件正在被查看。")

    # 6. 清理 Streamlit 资源缓存
    try:
        st.cache_resource.clear()
        print("已清理资源缓存")
    except Exception as e:
        print(f"清理缓存失败: {e}")

    # 7. 重置 Session State
    keys_to_keep = ["has_cleared", "uploader_key"]
    for k in list(st.session_state.keys()):
        if k not in keys_to_keep:
            del st.session_state[k]

    # 更新上传组件 Key
    st.session_state["uploader_key"] += 1

    # 强制 GC
    gc.collect()
    time.sleep(1)

    return True


st.set_page_config(page_title="个人知识库助手", layout="wide")

st.title("🤖 个人专属知识库助手")
st.caption(f"Powered by {config.LLM_MODEL_NAME} (混合解析版)")

# --- 初始化 Session State ---
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()

# --- 侧边栏：文件上传与控制 ---
with st.sidebar:
    st.header("1. 上传文档")

    parse_mode_option = st.radio(
        "解析策略",
        ("混合模式 (推荐)", "强制全视觉 (最慢)", "仅快速文本 (最快)"),
        index=0,
        help="混合模式：自动检测页面复杂度，有图表时用视觉模型，纯文本时用快速解析。\n强制全视觉：所有页面都用 Qwen-VL，适合极复杂的扫描件。"
    )

    # 策略映射
    strategy_map = {
        "混合模式 (推荐)": "auto",
        "强制全视觉 (最慢)": "force",
        "仅快速文本 (最快)": "fast"
    }
    selected_strategy = strategy_map[parse_mode_option]

    uploaded_files = st.file_uploader(
        "请上传 PDF 文档",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    # 检查本地是否有存量数据
    has_existing_data = os.path.exists("data") and len(os.listdir("data")) > 0 and os.path.exists(
        config.PERSIST_DIRECTORY)

    if not uploaded_files and not st.session_state.get("file_processed", False):
        if has_existing_data:
            st.info("检测到上次运行的知识库，已自动加载。")
            st.session_state["file_processed"] = True
            for f in os.listdir("data"):
                st.session_state["processed_files"].add(f)

    # 处理新上传的文件
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state["processed_files"]]

        if new_files:
            current_count = len(st.session_state["processed_files"])
            if current_count + len(new_files) > config.MAX_FILES_COUNT:
                st.error(
                    f"❌ 文件数量超过限制！当前 {current_count}，尝试新增 {len(new_files)}，上限 {config.MAX_FILES_COUNT}。")
            else:
                total_size_mb = sum([f.size for f in uploaded_files]) / (1024 * 1024)

                if total_size_mb > config.MAX_FILE_SIZE_MB:
                    st.error(f"❌ 总大小超过限制！当前: {total_size_mb:.2f}MB, 最大: {config.MAX_FILE_SIZE_MB}MB")
                else:
                    # 创建进度显示区域
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()
                        
                        status_text.text(f"正在准备处理 {len(new_files)} 个新文档...")
                        
                        os.makedirs("data", exist_ok=True)

                        saved_file_paths = []
                        for file in new_files:
                            file_path = os.path.join("data", file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                            saved_file_paths.append(file_path)
                        
                        # 定义进度回调函数
                        def progress_callback(current_page, total_pages, message):
                            """
                            进度回调函数，用于更新 Streamlit 进度条
                            current_page: 当前页码
                            total_pages: 总页数
                            message: 状态消息
                            """
                            progress = current_page / total_pages if total_pages > 0 else 0
                            progress_bar.progress(progress)
                            status_text.text(message)
                        
                        # 传递进度回调到 ingest_document
                        success = ingest_document(
                            saved_file_paths, 
                            parsing_strategy=selected_strategy,
                            progress_callback=progress_callback
                        )
                        
                        # 处理完成，清理进度条
                        progress_bar.empty()
                        status_text.empty()

                        if success:
                            st.success(f"成功添加 {len(saved_file_paths)} 个新文档！")
                            st.session_state["file_processed"] = True
                            for f in new_files:
                                st.session_state["processed_files"].add(f.name)

    st.divider()

    # --- AI 角色设定 ---
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
        help="必须保留 {context} 和 {question} 这两个占位符。"
    )

    st.divider()

    # 重置按钮
    if st.button("🧨 重置知识库", type="primary"):
        reset_app()
        st.rerun()

    # 退出按钮
    if st.button("🔴 退出系统", key="exit_btn_sidebar"):
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
                status_placeholder.markdown("🔍 正在检索并思考...")

                try:
                    # 构建历史记录
                    chat_history = []
                    for msg in st.session_state["messages"][:-1]:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))

                    # 使用流式生成
                    stream_gen = rag_chain.stream({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                    
                    # 提取第一个元素 (metadata)
                    metadata = next(stream_gen)
                    source_docs = metadata.get("source_documents", [])
                    run_id = metadata.get("run_id", "")
                    
                    # 检查是否有错误
                    if metadata.get("error"):
                        status_placeholder.empty()
                        st.warning(metadata["error"])
                        st.session_state["messages"].append({"role": "assistant", "content": metadata["error"]})
                    else:
                        # 清除 "正在思考" 状态
                        status_placeholder.empty()
                        
                        # 流式输出文本
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        for token in stream_gen:
                            if isinstance(token, str):
                                full_response += token
                                response_placeholder.markdown(full_response + "▌")
                        
                        # 移除光标，显示最终结果
                        response_placeholder.markdown(full_response)
                        st.session_state["messages"].append({"role": "assistant", "content": full_response})
                        
                        # 保存当前 run_id 用于反馈
                        st.session_state["last_run_id"] = run_id

                        # 来源展示 (带引用编号对应)
                        if source_docs:
                            with st.expander("📚 参考来源 (点击展开)"):
                                for i, doc in enumerate(source_docs):
                                    source = os.path.basename(doc.metadata.get("source", "未知文件"))
                                    page = doc.metadata.get("page", 0) + 1
                                    mode = doc.metadata.get("parsing_mode", "unknown")
                                    st.markdown(f"**[{i + 1}] 来源:** `{source}` (第 {page} 页) | 模式: `{mode}`")
                                    # 这里的 content 是父块（2000字），我们只展示前 150 字预览
                                    content_preview = doc.page_content[:150].replace('\n', ' ')
                                    st.caption(f"原文片段: ...{content_preview}...")
                                    st.divider()
                        
                        # 用户反馈按钮
                        st.markdown("---")
                        st.caption("这个回答对您有帮助吗？")
                        col1, col2, col3 = st.columns([1, 1, 8])
                        with col1:
                            if st.button("👍", key=f"up_{run_id}"):
                                log_feedback(run_id, 1)
                                st.toast("感谢您的反馈！", icon="✅")
                        with col2:
                            if st.button("👎", key=f"down_{run_id}"):
                                log_feedback(run_id, 0)
                                st.toast("感谢您的反馈！我们会继续改进。", icon="📝")

                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"发生错误: {e}")
        else:
            st.error("知识库初始化失败，请重试。")