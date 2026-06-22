import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types.doc.labels import DocItemLabel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.schema import TextNode

from .table_parser import chunk_large_table
from ..utils.text_utils import clean_text, is_valid_text

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def _save_chapter(chap_id, title, content_list, parent_store, all_nodes, text_splitter):
    if not content_list:
        return
    full_text = "\n".join(content_list)
    parent_store[chap_id] = full_text

    split_texts = text_splitter.split_text(full_text)
    for i, chunk_text in enumerate(split_texts):
        if not is_valid_text(chunk_text):
            continue
        cleaned = clean_text(chunk_text)
        if not cleaned:
            continue
        enriched_text = f"章节：{title} 内容：{cleaned}"
        child_node = TextNode(
            text=enriched_text,
            id_=f"{chap_id}_{i}",
            metadata={
                "type": "text",
                "parent_id": chap_id,
                "chapter_title": title,
                "chunk_index": i,
            }
        )
        all_nodes.append(child_node)


def parse_pdf_with_large_chunks(pdf_path):
    """解析 PDF，chunk_size=600，表格用语义化文本 + 大表行级分块"""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    parent_store = {}
    table_store = {}
    all_nodes = []

    current_chapter_title = ""
    current_chapter_content = []
    current_chapter_id = ""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        length_function=len,
        keep_separator=False,
    )

    print(" 正在解析文档结构...")

    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        text = getattr(item, "text", "").strip()
        level = getattr(item, "level", 0)

        if label == DocItemLabel.SECTION_HEADER and level == 1:
            if current_chapter_id:
                _save_chapter(current_chapter_id, current_chapter_title,
                              current_chapter_content, parent_store, all_nodes, text_splitter)
            current_chapter_id = f"chap_{len(parent_store)}"
            current_chapter_title = text
            current_chapter_content = []
            print(f"   新章节: {current_chapter_title}")

        elif label in [DocItemLabel.TEXT, DocItemLabel.LIST_ITEM, DocItemLabel.FOOTNOTE]:
            if text:
                text = clean_text(text)
                if text:
                    current_chapter_content.append(text)

        elif label == DocItemLabel.TABLE:
            df = item.export_to_dataframe(doc=doc)
            if df.empty:
                continue

            table_md_full = df.to_markdown(index=False)
            table_id = f"tbl_{len(table_store)}"
            table_store[table_id] = table_md_full

            chunks = chunk_large_table(table_md_full, current_chapter_title, table_id)

            for ci, chunk in enumerate(chunks):
                sub_id = f"{table_id}_part{ci}" if len(chunks) > 1 else table_id
                node_text = chunk["semantic_text"]

                metadata = {
                    "type": "table",
                    "table_id": table_id,
                    "parent_id": current_chapter_id,
                    "chapter_title": current_chapter_title,
                    "row_range": chunk["row_range"],
                    "row_count": chunk["row_count"],
                    "table_part_index": ci,
                    "table_part_total": len(chunks),
                    "columns": chunk["columns"],
                    "raw_md_fragment": chunk["raw_fragment"],
                }

                table_node = TextNode(
                    text=node_text,
                    id_=f"{current_chapter_id}_{sub_id}",
                    metadata=metadata,
                )
                all_nodes.append(table_node)

            if len(chunks) <= 1:
                print(f"   表格节点: {table_id} ({current_chapter_title}, {chunks[0]['row_count']} 行)" if chunks else f"   表格节点: {table_id} ({current_chapter_title})")
            else:
                print(f"   表格节点: {table_id} ({current_chapter_title}, 拆 {len(chunks)} 块/{sum(c['row_count'] for c in chunks)} 行)")

    if current_chapter_id:
        _save_chapter(current_chapter_id, current_chapter_title,
                      current_chapter_content, parent_store, all_nodes, text_splitter)

    print(f"\n 解析完成：{len(parent_store)} 章节, {len(table_store)} 表格, {len(all_nodes)} 节点")

    print(f"\n{'─'*60}")
    print(" 章节内容量:")
    for chap_id, text in parent_store.items():
        print(f"  {chap_id}: {len(text)} 字符")
    print("节点类型分布:")
    tbl_cnt = sum(1 for n in all_nodes if n.metadata.get("type") == "table")
    txt_cnt = len(all_nodes) - tbl_cnt
    print(f"  文本节点: {txt_cnt}, 表格节点: {tbl_cnt}")
    if all_nodes:
        print("\n前 3 个节点:")
        for n in all_nodes[:3]:
            print(f"  [{n.node_id}] {n.text[:200]}...")

    return parent_store, table_store, all_nodes
