import base64
import os
import pickle
import sqlite3
import uuid

import fitz  # PyMuPDF
import torch
from langchain_chroma import Chroma

# LangChain 组件
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def _resolve_storage_paths(storage_paths=None) -> dict:
    """解析存储路径：优先使用会话路径，缺省回退到全局路径。"""
    base = config.get_default_paths()
    if storage_paths:
        base.update(storage_paths)
    return base


def _semantic_paragraph_split(
    text: str, chunk_size: int = 400, overlap_size: int = 80
) -> list:
    """语义化段落切分：优先按段落合并，减少句子硬截断。"""
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text[:chunk_size]]

    chunks = []
    current = ""
    for p in paragraphs:
        candidate = f"{current}\n\n{p}".strip() if current else p
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        # 极长段落的兜底硬切
        if len(p) > chunk_size:
            start = 0
            while start < len(p):
                end = min(start + chunk_size, len(p))
                chunks.append(p[start:end])
                if end == len(p):
                    break
                start = max(0, end - overlap_size)
            current = ""
        else:
            current = p

    if current:
        chunks.append(current)

    return chunks


# ========== Contextual Retrieval (Anthropic 方法) ==========
CONTEXT_PROMPT = """<document>
{whole_document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>
请为这个文本块生成一段简短的上下文说明，用于帮助检索系统更好地理解这个块在整个文档中的位置和含义。
要求：
1. 说明这个块属于哪个文档/章节
2. 补充必要的背景信息（如公司名、时间、主题等）
3. 只返回上下文说明，不要返回其他内容
4. 控制在50-100字以内"""


class ContextualChunkEnricher:
    """
    上下文增强器：为每个 chunk 添加上下文前缀
    根据 Anthropic 的 Contextual Retrieval 方法，可降低检索失败率 35-67%
    """

    def __init__(self, llm=None):
        if llm is None:
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL_NAME,
                openai_api_key=config.API_KEY,
                openai_api_base=config.BASE_URL,
                temperature=0.1,
                max_tokens=200,
            )
        else:
            self.llm = llm
        self.output_parser = StrOutputParser()

    def generate_context(self, whole_doc: str, chunk: str) -> str:
        """
        为单个 chunk 生成上下文前缀
        whole_doc: 完整文档内容（或父块内容）
        chunk: 需要添加上下文的子块
        """
        try:
            # 限制文档长度避免超出上下文窗口
            max_doc_len = 8000
            truncated_doc = (
                whole_doc[:max_doc_len] if len(whole_doc) > max_doc_len else whole_doc
            )

            prompt = CONTEXT_PROMPT.format(
                whole_document=truncated_doc, chunk_content=chunk
            )
            response = self.llm.invoke(prompt)
            context = self.output_parser.invoke(response)
            return context.strip()
        except Exception as e:
            print(f"上下文生成失败: {e}")
            return ""

    def enrich_chunk(self, whole_doc: str, chunk: str) -> str:
        """
        返回带上下文前缀的 chunk
        格式: [上下文说明]\n\n[原始内容]
        """
        context = self.generate_context(whole_doc, chunk)
        if context:
            return f"[背景: {context}]\n\n{chunk}"
        return chunk


def init_db(sqlite_db_path=None):
    """
    初始化 SQLite 数据库，创建父文档存储表
    使用 BLOB 存储 pickle 序列化的 Document 对象，确保完全兼容
    """
    db_path = sqlite_db_path or getattr(config, "SQLITE_DB_PATH", "./doc_store.db")
    # 重构：使用 with 语句管理连接生命周期，确保异常时也能正确关闭
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parent_docs (
                doc_id TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                ingest_time TEXT
            )
        """)
        conn.commit()
    return db_path


class DualModePDFParser:
    """
    自适应混合 PDF 解析器：
    引入“复杂度检测”路由，自动在快速模式和视觉模式间切换。
    """

    def __init__(self, img_store_path=None):
        # 图片/快照存储路径
        self.img_store_path = img_store_path or getattr(
            config, "IMG_STORE_PATH", "./extracted_images"
        )
        os.makedirs(self.img_store_path, exist_ok=True)

        # 初始化 VLM (仅在视觉模式下真正调用)
        self.vlm_client = ChatOpenAI(
            model=config.VLM_MODEL_NAME,
            openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL,
            temperature=0.1,
            max_tokens=2000,  # 增加 Token 数，因为要生成整页的 Markdown
        )

    def _image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _collect_key_visual_regions(self, page, max_regions: int = 3):
        """收集页面中较大的视觉区域，用于局部 VLM 分析。"""
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        candidates = []

        for img in page.get_images(full=True):
            try:
                xref = img[0]
                rects = page.get_image_rects(xref)
                for rect in rects:
                    area = rect.width * rect.height
                    ratio = area / page_area if page_area else 0
                    # 只保留较大视觉块，减少无意义调用
                    if ratio >= 0.08:
                        candidates.append((ratio, rect))
            except Exception:
                continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [rect for _, rect in candidates[:max_regions]]

    def _analyze_visual_region(self, page, rect, filename, page_num, region_idx):
        """对单个局部区域调用 VLM，避免整页高成本解析。"""
        tmp_path = os.path.join(
            self.img_store_path,
            f"{filename}_p{page_num}_r{region_idx}.png",
        )
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
            pix.save(tmp_path)
            base64_img = self._image_to_base64(tmp_path)

            prompt = (
                "你是文档视觉分析助手。请仅分析这块局部区域，输出简洁 Markdown：\n"
                "1. 若是图表，说明核心趋势/结论；\n"
                "2. 若是架构图，说明主要模块与关系；\n"
                "3. 若是截图/插图，提炼与正文相关的信息；\n"
                "4. 不要重复无关内容，控制在120字以内。"
            )
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                    },
                ]
            )
            response = self.vlm_client.invoke([message])
            return (response.content or "").strip()
        except Exception as e:
            print(f"    x 局部视觉解析失败 (P{page_num}, R{region_idx}): {e}")
            return ""
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _is_complex_page(self, page, page_num, filename):
        """
        [侦察兵] 判断页面是否复杂
        返回: (bool, reason)
        """
        # 1. 检测表格
        table_finder = page.find_tables()
        if table_finder.tables:
            return True, f"检测到 {len(table_finder.tables)} 个表格"

        # 2. 检测图片
        image_list = page.get_images(full=True)
        if image_list:
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height

            for img in image_list:
                try:
                    xref = img[0]
                    # 获取图片在页面上的边界框 (可能由多个实例组成)
                    rects = page.get_image_rects(xref)
                    for rect in rects:
                        img_area = rect.width * rect.height
                        # 阈值：如果图片面积超过页面 15%，认为是关键图表/PPT/扫描件
                        if (img_area / page_area) > 0.15:
                            return True, "检测到大尺寸图片/图表"
                except Exception:
                    continue

        return False, "纯文本页面"

    def parse_fast(self, page, filename, page_num):
        """处理单页 - 快速模式"""
        # 即使是快速模式，也尝试提取表格为 Markdown
        table_finder = page.find_tables()
        md_tables = []

        if table_finder.tables:
            for t in table_finder.tables:
                md = t.to_markdown()
                if md:
                    md_tables.append(md)

        text = page.get_text()

        content = text
        if md_tables:
            # 将表格附加在文本后面 (简单处理)
            content += "\n\n【检测到表格 (Fast Mode)】:\n" + "\n".join(md_tables)

        return Document(
            page_content=content,
            metadata={
                "source": filename,
                "page": page_num,
                "parsing_mode": "fast_text",
            },
        )

    def parse_vision(self, page, filename, page_num):
        """处理单页 - 分层视觉模式（文本主干 + 局部视觉补充）"""
        try:
            # 先走快速文本，保留正文主干
            fast_doc = self.parse_fast(page, filename, page_num)
            content = fast_doc.page_content

            # 只对关键视觉区做补充理解，避免整页 VLM
            visual_regions = self._collect_key_visual_regions(page, max_regions=3)
            region_notes = []
            for idx, rect in enumerate(visual_regions, start=1):
                note = self._analyze_visual_region(page, rect, filename, page_num, idx)
                if note:
                    region_notes.append(f"> [视觉补充 R{idx}] {note}")

            if region_notes:
                content = content + "\n\n【视觉补充】\n" + "\n".join(region_notes)

            return Document(
                page_content=content if content else "[VLM 解析为空]",
                metadata={
                    "source": filename,
                    "page": page_num,
                    "parsing_mode": "vision_layered",
                },
            )

        except Exception as e:
            print(f"    x 视觉解析失败 (P{page_num}): {e}，回退到快速模式")
            return self.parse_fast(page, filename, page_num)

    def parse(self, file_path, use_vision_strategy="auto", progress_callback=None):
        """
        入口函数
        use_vision_strategy:
          - "force": 强制全视觉 (最慢)
          - "auto": 智能路由 (推荐)
          - "fast": 强制全文本 (最快)
        """
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)
        parsed_docs = []

        print(f"开始解析: {filename} | 策略: {use_vision_strategy}")

        for page_num, page in enumerate(doc):
            is_complex = False
            reason = "强制纯文本"

            # 策略路由逻辑
            if use_vision_strategy == "force":
                is_complex = True
                reason = "用户强制视觉模式"
            elif use_vision_strategy == "auto":
                is_complex, reason = self._is_complex_page(page, page_num, filename)
            else:
                # "fast" mode
                is_complex = False
                reason = "用户强制快速模式"

            # 执行解析
            if is_complex:
                print(f"  > P{page_num + 1} [{reason}] -> 切换 Qwen-VL 视觉解析")
                res_doc = self.parse_vision(page, filename, page_num)
            else:
                # print(f"  > P{page_num+1} [{reason}] -> 快速文本提取")
                res_doc = self.parse_fast(page, filename, page_num)

            parsed_docs.append(res_doc)

            # 进度回调
            if progress_callback:
                total_pages = len(doc)
                progress_callback(
                    page_num + 1,
                    total_pages,
                    f"正在进行分层视觉补充解析第 {page_num + 1}/{total_pages} 页..."
                    if is_complex
                    else f"快速解析第 {page_num + 1}/{total_pages} 页...",
                )

        doc.close()
        return parsed_docs


def ingest_document(
    file_paths,
    parsing_strategy="auto",
    progress_callback=None,
    use_contextual_retrieval=False,
    chunking_strategy="semantic",
    storage_paths=None,
):
    """
    全流程接入：
    1. 自适应混合解析 (Smart Hybrid Parsing)
    2. 父子切分 (Parent-Child Indexing)
    3. 向量化存储

    注意: use_contextual_retrieval=True 会显著增加处理时间 (每个子块需调用 LLM)
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    paths = _resolve_storage_paths(storage_paths)

    # 初始化 SQLite 数据库
    db_path = init_db(paths["sqlite_db_path"])

    # --- 1. 定义切分器 ---
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n## ", "\n\n# ", "\n\n", "\n", "。", " "],
    )

    all_child_docs = []

    # 初始化解析器
    parser = DualModePDFParser(img_store_path=paths["img_store_path"])

    # 初始化上下文增强器 (Contextual Retrieval) - 默认关闭以加速
    context_enricher = None
    if use_contextual_retrieval:
        context_enricher = ContextualChunkEnricher()
        print(
            "✨ 已启用 Contextual Retrieval (Anthropic 方法) - 注意：这会增加处理时间"
        )

    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        try:
            # 步骤 A: 解析文档 (传入策略和进度回调)
            raw_docs = parser.parse(
                file_path,
                use_vision_strategy=parsing_strategy,
                progress_callback=progress_callback,
            )

            if not raw_docs:
                print(f"文件 {file_path} 未提取到有效内容")
                continue

            # 步骤 B: 切分父块
            parent_docs = parent_splitter.split_documents(raw_docs)

            import time

            ingest_time = time.strftime("%Y-%m-%d %H:%M:%S")

            # 统计总子块数用于进度显示
            total_parent_docs = len(parent_docs)
            processed_parents = 0

            # 重构：将 SQLite 连接提升到循环外部，使用 with 语句管理连接生命周期。
            # 原代码在每次循环中各自 connect/commit/close，N 个父块产生 N 次连接开销。
            # 重构后：1 个文件只建立 1 次连接，全部父块处理完毕后统一提交，性能显著提升。
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                for parent_doc in parent_docs:
                    doc_id = str(uuid.uuid4())
                    parent_doc.metadata["doc_id"] = doc_id
                    parent_doc.metadata["ingest_time"] = ingest_time

                    # 步骤 C: 保存父块到 SQLite (存大块)
                    # 重构：移除循环内的 connect/commit/close，改用外层统一连接
                    cursor.execute(
                        "INSERT OR REPLACE INTO parent_docs (doc_id, data, ingest_time) VALUES (?, ?, ?)",
                        (doc_id, pickle.dumps(parent_doc), ingest_time),
                    )

                    # 步骤 D: 切分子块 (存向量)
                    if chunking_strategy == "semantic":
                        semantic_chunks = _semantic_paragraph_split(
                            parent_doc.page_content, chunk_size=400, overlap_size=80
                        )
                        child_docs = [
                            Document(
                                page_content=chunk, metadata=dict(parent_doc.metadata)
                            )
                            for chunk in semantic_chunks
                        ]
                    else:
                        child_docs = child_splitter.split_documents([parent_doc])

                    for i, child in enumerate(child_docs):
                        child.metadata["doc_id"] = doc_id
                        child.metadata["child_id"] = f"{doc_id}_{i}"
                        child.metadata["source"] = os.path.basename(file_path)
                        child.metadata["ingest_time"] = ingest_time

                        # 步骤 E: Contextual Retrieval - 为子块添加上下文前缀
                        if context_enricher:
                            original_content = child.page_content
                            child.page_content = context_enricher.enrich_chunk(
                                parent_doc.page_content,  # 使用父块作为上下文
                                original_content,
                            )
                            child.metadata["has_context"] = True
                            child.metadata["original_content"] = original_content

                    all_child_docs.extend(child_docs)

                    # 更新进度
                    processed_parents += 1
                    if progress_callback:
                        progress_callback(
                            processed_parents,
                            total_parent_docs,
                            f"正在索引文档块 {processed_parents}/{total_parent_docs}...",
                        )
                    else:
                        # 控制台进度
                        print(
                            f"\r  正在索引: {processed_parents}/{total_parent_docs} 块",
                            end="",
                            flush=True,
                        )

                # 重构：所有父块处理完毕后，统一提交一次事务
                conn.commit()

            print()  # 换行

            print(
                f"  - 处理完成: {len(parent_docs)} 个父块 / {len(all_child_docs)} 个子块"
            )

        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {e}")

    if not all_child_docs:
        return False

    # --- 2. 存入数据库 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={"device": device}
    )

    # BM25 处理
    final_bm25_docs = all_child_docs
    if os.path.exists(paths["bm25_persist_path"]):
        try:
            with open(paths["bm25_persist_path"], "rb") as f:
                existing_docs = pickle.load(f)
            final_bm25_docs = existing_docs + all_child_docs
        except Exception:
            pass

    os.makedirs(os.path.dirname(paths["bm25_persist_path"]), exist_ok=True)
    with open(paths["bm25_persist_path"], "wb") as f:
        pickle.dump(final_bm25_docs, f)

    try:
        print("正在写入向量数据库...")
        vectorstore = Chroma.from_documents(
            documents=all_child_docs,
            embedding=embeddings,
            persist_directory=paths["persist_directory"],
        )
        print("向量数据库写入完成")
    except Exception as e:
        print(f"写入向量数据库时发生错误: {e}")
        return False

    return True
