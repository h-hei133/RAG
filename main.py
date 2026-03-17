import gc
import json
import os
import shutil
import sys
import time
import uuid

import chromadb
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config
from src.ingestion import ingest_document
from src.rag_chain import get_rag_chain, log_feedback

DEFAULT_PROMPT = """你是一个专业的助手。请基于下面的【上下文】内容回答用户的问题。
如果不知道，请直接承认。

【上下文】:
{context}

【用户问题】:
{question}
"""

st.set_page_config(page_title="个人知识库助手", layout="wide")
SESSION_REGISTRY_PATH = os.path.join(config.SESSIONS_ROOT, "session_registry.json")


def build_new_conversation(name: str = None) -> dict:
    title = name or f"会话 {time.strftime('%m-%d %H:%M')}"
    return {
        "title": title,
        "messages": [
            {
                "role": "assistant",
                "content": "你好！请先上传文档，然后问我关于文档的问题。",
            }
        ],
        "processed_files": set(),
        "file_processed": False,
        "prompt": DEFAULT_PROMPT,
        "uploader_key": 0,
        "last_run_id": "",
    }


def _serialize_conversation(conv: dict) -> dict:
    return {
        "title": conv.get("title", "未命名会话"),
        "messages": conv.get("messages", []),
        "processed_files": sorted(list(conv.get("processed_files", set()))),
        "file_processed": conv.get("file_processed", False),
        "prompt": conv.get("prompt", DEFAULT_PROMPT),
        "uploader_key": conv.get("uploader_key", 0),
        "last_run_id": conv.get("last_run_id", ""),
    }


def _deserialize_conversation(raw: dict) -> dict:
    conv = build_new_conversation(raw.get("title", "会话"))
    conv["messages"] = raw.get("messages") or conv["messages"]
    conv["processed_files"] = set(raw.get("processed_files", []))
    conv["file_processed"] = raw.get("file_processed", False)
    conv["prompt"] = raw.get("prompt", DEFAULT_PROMPT)
    conv["uploader_key"] = raw.get("uploader_key", 0)
    conv["last_run_id"] = raw.get("last_run_id", "")
    return conv


def persist_conversations():
    os.makedirs(config.SESSIONS_ROOT, exist_ok=True)
    payload = {
        "current_session_id": st.session_state.get(
            "current_session_id", "session_default"
        ),
        "conversation_order": st.session_state.get("conversation_order", []),
        "conversations": {
            sid: _serialize_conversation(conv)
            for sid, conv in st.session_state.get("conversations", {}).items()
        },
    }
    with open(SESSION_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_conversations_from_disk() -> dict:
    if not os.path.exists(SESSION_REGISTRY_PATH):
        return {}
    try:
        with open(SESSION_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _discover_conversations_from_fs() -> dict:
    if not os.path.exists(config.SESSIONS_ROOT):
        return {}

    conversations = {}
    order = []
    for name in sorted(os.listdir(config.SESSIONS_ROOT)):
        session_root = os.path.join(config.SESSIONS_ROOT, name)
        if not os.path.isdir(session_root):
            continue
        if not name.startswith("session_"):
            continue

        conv = build_new_conversation("默认会话" if name == "session_default" else name)
        data_dir = os.path.join(session_root, "data")
        if os.path.exists(data_dir):
            files = [
                f
                for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f))
            ]
            conv["processed_files"] = set(files)
            conv["file_processed"] = len(files) > 0
        conversations[name] = conv
        order.append(name)

    return {"conversations": conversations, "conversation_order": order}


def init_state():
    if "has_cleared" not in st.session_state:
        st.session_state["has_cleared"] = False

    if "rag_chain_cache" not in st.session_state:
        st.session_state["rag_chain_cache"] = {}

    if "conversations" not in st.session_state:
        disk_state = _load_conversations_from_disk()
        if not disk_state.get("conversations"):
            disk_state = _discover_conversations_from_fs()

        if disk_state.get("conversations"):
            st.session_state["conversations"] = {
                sid: _deserialize_conversation(conv)
                for sid, conv in disk_state.get("conversations", {}).items()
            }
            st.session_state["conversation_order"] = disk_state.get(
                "conversation_order", list(st.session_state["conversations"].keys())
            )
            current = disk_state.get("current_session_id")
            if current not in st.session_state["conversations"]:
                current = st.session_state["conversation_order"][0]
            st.session_state["current_session_id"] = current
        else:
            first_id = "session_default"
            st.session_state["conversations"] = {
                first_id: build_new_conversation("默认会话")
            }
            st.session_state["conversation_order"] = [first_id]
            st.session_state["current_session_id"] = first_id
            persist_conversations()


def get_current_session_id() -> str:
    return st.session_state["current_session_id"]


def get_current_conversation() -> dict:
    return st.session_state["conversations"][get_current_session_id()]


def get_current_paths() -> dict:
    return config.get_session_paths(get_current_session_id())


def ensure_session_dirs(paths: dict):
    os.makedirs(paths["session_root"], exist_ok=True)
    os.makedirs(paths["data_dir"], exist_ok=True)


def clear_legacy_global_data():
    targets = [
        "data",
        config.PERSIST_DIRECTORY,
        config.BM25_PERSIST_PATH,
        config.SQLITE_DB_PATH,
        config.IMG_STORE_PATH,
    ]
    for target in targets:
        if not os.path.exists(target):
            continue
        try:
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                os.remove(target)
            print(f"已清理: {target}")
        except Exception as e:
            print(f"清理失败 {target}: {e}")


def reset_conversation(paths: dict, conv: dict):
    print(f"重置会话: {paths['session_id']}")

    persist_dir = paths["persist_directory"]
    if os.path.exists(persist_dir):
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            try:
                client.delete_collection("langchain")
            except Exception:
                pass
        except Exception as e:
            print(f"清空向量库失败: {e}")

    for target in [
        paths["bm25_persist_path"],
        paths["sqlite_db_path"],
        paths["data_dir"],
        paths["img_store_path"],
        paths["rag_log_path"],
        paths["feedback_log_path"],
        paths["semantic_cache_path"],
        paths["string_cache_path"],
        paths["persist_directory"],
    ]:
        if not os.path.exists(target):
            continue
        try:
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                os.remove(target)
        except Exception as e:
            print(f"删除失败 {target}: {e}")

    try:
        st.cache_resource.clear()
    except Exception:
        pass

    conv["messages"] = [
        {"role": "assistant", "content": "你好！请先上传文档，然后问我关于文档的问题。"}
    ]
    conv["processed_files"] = set()
    conv["file_processed"] = False
    conv["uploader_key"] += 1
    conv["last_run_id"] = ""

    gc.collect()
    time.sleep(0.2)


init_state()

if "--clear" in sys.argv and not st.session_state["has_cleared"]:
    st.session_state["has_cleared"] = True
    print("检测到 --clear 参数，尝试清理历史全局数据...")
    clear_legacy_global_data()

st.title("个人专属知识库助手")
st.caption(f"Powered by {config.LLM_MODEL_NAME} | 多会话隔离 + Agentic RAG")

current_paths = get_current_paths()
ensure_session_dirs(current_paths)
conversation = get_current_conversation()

with st.sidebar:
    st.header("会话管理")

    order = st.session_state["conversation_order"]
    conversations = st.session_state["conversations"]
    labels = []
    for sid in order:
        suffix = sid if len(sid) <= 6 else sid[-6:]
        labels.append(f"{conversations[sid]['title']} ({suffix})")
    current_sid = get_current_session_id()
    current_index = order.index(current_sid)

    selected_label = st.selectbox("选择会话", labels, index=current_index)
    selected_index = labels.index(selected_label)
    selected_sid = order[selected_index]

    if selected_sid != current_sid:
        st.session_state["current_session_id"] = selected_sid
        persist_conversations()
        st.rerun()

    new_title = st.text_input(
        "当前会话名称",
        value=conversations[current_sid]["title"],
        key=f"rename_input_{current_sid}",
    )
    if st.button("重命名会话"):
        trimmed = new_title.strip()
        if not trimmed:
            st.warning("会话名称不能为空")
        else:
            conversations[current_sid]["title"] = trimmed
            persist_conversations()
            st.success("会话名称已更新")
            st.rerun()

    col_new, col_del = st.columns(2)
    with col_new:
        if st.button("新建会话"):
            new_sid = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state["conversations"][new_sid] = build_new_conversation()
            st.session_state["conversation_order"].append(new_sid)
            st.session_state["current_session_id"] = new_sid
            persist_conversations()
            st.rerun()

    with col_del:
        if st.button("删除当前"):
            if len(st.session_state["conversation_order"]) <= 1:
                st.warning("至少保留一个会话")
            else:
                sid = get_current_session_id()
                paths = config.get_session_paths(sid)
                if os.path.exists(paths["session_root"]):
                    shutil.rmtree(paths["session_root"], ignore_errors=True)
                st.session_state["conversation_order"].remove(sid)
                st.session_state["conversations"].pop(sid, None)
                st.session_state["current_session_id"] = st.session_state[
                    "conversation_order"
                ][0]
                persist_conversations()
                st.rerun()

    st.divider()
    st.header("文档与解析")

    parse_mode_option = st.radio(
        "解析策略",
        ("混合模式 (推荐)", "强制全视觉 (最慢)", "仅快速文本 (最快)"),
        index=0,
    )
    strategy_map = {
        "混合模式 (推荐)": "auto",
        "强制全视觉 (最慢)": "force",
        "仅快速文本 (最快)": "fast",
    }
    selected_strategy = strategy_map[parse_mode_option]

    chunk_mode = st.radio(
        "切片策略",
        ("语义分块 (推荐)", "固定长度分块"),
        index=0,
    )
    chunk_strategy = "semantic" if "语义" in chunk_mode else "recursive"

    uploaded_files = st.file_uploader(
        "上传 PDF 文档",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{get_current_session_id()}_{conversation['uploader_key']}",
    )

    has_existing_data = (
        os.path.exists(current_paths["data_dir"])
        and len(os.listdir(current_paths["data_dir"])) > 0
        and os.path.exists(current_paths["persist_directory"])
    )

    if (
        not uploaded_files
        and not conversation.get("file_processed", False)
        and has_existing_data
    ):
        conversation["file_processed"] = True
        for f in os.listdir(current_paths["data_dir"]):
            conversation["processed_files"].add(f)
        persist_conversations()

    if uploaded_files:
        new_files = [
            f for f in uploaded_files if f.name not in conversation["processed_files"]
        ]

        if new_files:
            current_count = len(conversation["processed_files"])
            if current_count + len(new_files) > config.MAX_FILES_COUNT:
                st.error(
                    f"文件数量超过限制。当前 {current_count}，尝试新增 {len(new_files)}，上限 {config.MAX_FILES_COUNT}。"
                )
            else:
                total_size_mb = sum([f.size for f in uploaded_files]) / (1024 * 1024)
                if total_size_mb > config.MAX_FILE_SIZE_MB:
                    st.error(
                        f"总大小超过限制。当前: {total_size_mb:.2f}MB, 最大: {config.MAX_FILE_SIZE_MB}MB"
                    )
                else:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    status_text.text(f"正在准备处理 {len(new_files)} 个新文档...")
                    os.makedirs(current_paths["data_dir"], exist_ok=True)

                    saved_file_paths = []
                    for file in new_files:
                        file_path = os.path.join(current_paths["data_dir"], file.name)
                        with open(file_path, "wb") as fp:
                            fp.write(file.getbuffer())
                        saved_file_paths.append(file_path)

                    def progress_callback(current_page, total_pages, message):
                        progress = current_page / total_pages if total_pages > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(message)

                    success = ingest_document(
                        saved_file_paths,
                        parsing_strategy=selected_strategy,
                        progress_callback=progress_callback,
                        chunking_strategy=chunk_strategy,
                        storage_paths=current_paths,
                    )

                    progress_bar.empty()
                    status_text.empty()

                    if success:
                        st.success(f"成功添加 {len(saved_file_paths)} 个新文档")
                        conversation["file_processed"] = True
                        for f in new_files:
                            conversation["processed_files"].add(f.name)
                        st.session_state["rag_chain_cache"].pop(
                            get_current_session_id(), None
                        )
                        persist_conversations()

    st.divider()
    st.header("AI 角色设定")

    old_prompt = conversation.get("prompt", DEFAULT_PROMPT)
    conversation["prompt"] = st.text_area(
        "自定义系统提示词 (Prompt)",
        value=old_prompt,
        height=190,
        help="必须保留 {context} 和 {question} 两个占位符",
    )
    if conversation["prompt"] != old_prompt:
        persist_conversations()

    st.divider()
    st.header("幻觉检测配置")

    strictness = st.slider(
        "检测严格度",
        min_value=20,
        max_value=80,
        value=int(getattr(config, "HALLUCINATION_THRESHOLD", 0.5) * 100),
        step=5,
        help="越高越严格，越容易触发风险提示。",
    )

    with st.expander("高级参数"):
        hd_max_docs = st.slider(
            "检测文档数",
            min_value=1,
            max_value=8,
            value=getattr(config, "HALLUCINATION_MAX_DOCS", 3),
            step=1,
        )
        hd_doc_chars = st.slider(
            "单文档截断字符",
            min_value=300,
            max_value=2000,
            value=getattr(config, "HALLUCINATION_DOC_CHARS", 500),
            step=100,
        )
        hd_answer_chars = st.slider(
            "答案截断字符",
            min_value=500,
            max_value=3000,
            value=getattr(config, "HALLUCINATION_ANSWER_CHARS", 800),
            step=100,
        )

    hallucination_config = {
        "threshold": strictness / 100.0,
        "max_docs": hd_max_docs,
        "doc_chars": hd_doc_chars,
        "answer_chars": hd_answer_chars,
    }

    st.divider()
    st.header("性能模式")
    speed_mode = st.checkbox(
        "速度优先（减少额外 LLM 调用）",
        value=True,
        help="开启后会关闭部分耗时能力（如 CRAG、多路查询增强、幻觉检测）。",
    )

    enable_hallucination_check = st.checkbox(
        "启用幻觉检测",
        value=not speed_mode,
        help="关闭可减少等待时间；开启后会进行支撑度检测。",
    )

    runtime_options = {
        "speed_mode": speed_mode,
        "enable_hallucination_check": enable_hallucination_check,
    }

    st.divider()

    if st.button("重置当前会话知识库", type="primary"):
        reset_conversation(current_paths, conversation)
        st.session_state["rag_chain_cache"].pop(get_current_session_id(), None)
        persist_conversations()
        st.rerun()

    if st.button("退出系统", key="exit_btn_sidebar"):
        st.warning("程序正在关闭...")
        time.sleep(1)
        os._exit(0)

for msg in conversation["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("请输入你的问题..."):
    conversation["messages"].append({"role": "user", "content": prompt})
    persist_conversations()
    st.chat_message("user").write(prompt)

    if not conversation.get("file_processed"):
        response = "请先在左侧上传 PDF 文档，我才能回答你的问题。"
        conversation["messages"].append({"role": "assistant", "content": response})
        persist_conversations()
        st.chat_message("assistant").write(response)
    else:
        sid = get_current_session_id()
        chain_signature = json.dumps(
            {
                "sid": sid,
                "prompt": conversation.get("prompt", DEFAULT_PROMPT),
                "storage": current_paths,
                "hallucination": hallucination_config,
                "runtime": runtime_options,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_entry = st.session_state["rag_chain_cache"].get(sid)
        if cache_entry and cache_entry.get("signature") == chain_signature:
            rag_chain = cache_entry.get("chain")
        else:
            rag_chain = get_rag_chain(
                custom_prompt=conversation.get("prompt", DEFAULT_PROMPT),
                storage_paths=current_paths,
                hallucination_config=hallucination_config,
            )
            if rag_chain:
                if runtime_options["speed_mode"]:
                    rag_chain.use_crag = False
                    rag_chain.use_multi_query_retriever = False
                    rag_chain.enable_web_fallback = False
                rag_chain.use_hallucination_check = runtime_options[
                    "enable_hallucination_check"
                ]

            st.session_state["rag_chain_cache"][sid] = {
                "signature": chain_signature,
                "chain": rag_chain,
            }
        if rag_chain:
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                status_placeholder.markdown("正在检索并思考...")

                try:
                    chat_history = []
                    for msg in conversation["messages"][:-1]:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history.append(AIMessage(content=msg["content"]))

                    stream_gen = rag_chain.stream(
                        {"input": prompt, "chat_history": chat_history}
                    )
                    metadata = next(stream_gen)
                    source_docs = metadata.get("source_documents", [])
                    run_id = metadata.get("run_id", "")

                    if metadata.get("error"):
                        status_placeholder.empty()
                        st.warning(metadata["error"])
                        conversation["messages"].append(
                            {"role": "assistant", "content": metadata["error"]}
                        )
                        persist_conversations()
                    else:
                        status_placeholder.empty()
                        response_placeholder = st.empty()
                        full_response = ""

                        for token in stream_gen:
                            if isinstance(token, str):
                                full_response += token
                                response_placeholder.markdown(full_response + "▌")
                            elif (
                                isinstance(token, dict)
                                and token.get("type") == "hallucination_check"
                            ):
                                if token.get("hallucination_risk"):
                                    score = token.get("score")
                                    score_text = (
                                        f"{score:.2f}"
                                        if isinstance(score, (int, float))
                                        else "N/A"
                                    )
                                    st.warning(
                                        f"提示：答案支撑度较低 (score={score_text})，请重点核对来源。"
                                    )

                        response_placeholder.markdown(full_response)
                        conversation["messages"].append(
                            {"role": "assistant", "content": full_response}
                        )
                        conversation["last_run_id"] = run_id
                        persist_conversations()

                        if source_docs:
                            with st.expander("参考来源"):
                                for i, doc in enumerate(source_docs):
                                    source = os.path.basename(
                                        doc.metadata.get("source", "未知文件")
                                    )
                                    page = doc.metadata.get("page", 0) + 1
                                    mode = doc.metadata.get("parsing_mode", "unknown")
                                    st.markdown(
                                        f"**[{i + 1}] 来源:** {source} (第 {page} 页) | 模式: {mode}"
                                    )
                                    content_preview = doc.page_content[:150].replace(
                                        "\n", " "
                                    )
                                    st.caption(f"原文片段: ...{content_preview}...")
                                    st.divider()

                        st.markdown("---")
                        st.caption("这个回答对您有帮助吗？")
                        col1, col2, _ = st.columns([1, 1, 8])
                        with col1:
                            if st.button(
                                "👍", key=f"up_{get_current_session_id()}_{run_id}"
                            ):
                                log_feedback(
                                    run_id,
                                    1,
                                    feedback_log_path=current_paths[
                                        "feedback_log_path"
                                    ],
                                )
                                st.toast("感谢您的反馈", icon="✅")
                        with col2:
                            if st.button(
                                "👎", key=f"down_{get_current_session_id()}_{run_id}"
                            ):
                                log_feedback(
                                    run_id,
                                    0,
                                    feedback_log_path=current_paths[
                                        "feedback_log_path"
                                    ],
                                )
                                st.toast("感谢反馈，我们会继续改进", icon="📝")

                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"发生错误: {e}")
        else:
            st.error("知识库初始化失败，请重试")
