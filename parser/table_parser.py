from .utils.text_utils import clean_text

TABLE_MAX_ROWS = 10
TABLE_OVERLAP_ROWS = 2


def table_to_semantic_text(table_md: str, chapter_title: str) -> str:
    """将表格 markdown 转为语义化描述文本，对 Embedding 和 BM25 都友好"""
    lines = [l for l in table_md.strip().split('\n') if l.strip()]
    if not lines:
        return f"表格标题：{chapter_title}（空表）"

    header_line = lines[0]
    cols = [c.strip() for c in header_line.split('|') if c.strip()]
    data_lines = [l for l in lines[2:] if l.strip()]

    parts = [f"表格标题：{chapter_title}"]
    if cols:
        parts.append(f"列名：{'，'.join(cols)}")
    parts.append(f"数据行数：{len(data_lines)}")

    for i, row in enumerate(data_lines[:15]):
        cells = [c.strip() for c in row.split('|') if c.strip()]
        if not cells:
            continue
        if cols and len(cells) >= len(cols):
            row_desc = "；".join(f"{cols[j]}：{cells[j]}" for j in range(len(cols)))
        else:
            row_desc = "，".join(cells)
        parts.append(f"第{i+1}行：{row_desc}")

    return "\n".join(parts)


def chunk_large_table(table_md: str, chapter_title: str, table_id: str,
                      max_rows=TABLE_MAX_ROWS, overlap_rows=TABLE_OVERLAP_ROWS):
    """将大表格按行拆分为多个子块"""
    lines = [l for l in table_md.strip().split('\n') if l.strip()]
    if not lines:
        return []

    header_line = lines[0]
    cols = [c.strip() for c in header_line.split('|') if c.strip()]
    data_lines = [l for l in lines[2:] if l.strip()]

    if len(data_lines) <= max_rows:
        semantic = table_to_semantic_text(table_md, chapter_title)
        return [{
            "semantic_text": semantic,
            "raw_fragment": table_md,
            "row_range": f"1-{len(data_lines)}",
            "row_count": len(data_lines),
            "columns": cols,
        }]

    sep = "|" + "|".join(["---"] * len(cols)) + "|\n" if cols else "|---|\n"
    step = max_rows - overlap_rows
    chunks = []

    for start in range(0, len(data_lines), step):
        chunk_data = data_lines[start:start + max_rows]
        chunk_md = header_line + "\n" + sep + "\n".join(chunk_data)
        semantic = table_to_semantic_text(chunk_md, chapter_title)
        chunks.append({
            "semantic_text": semantic,
            "raw_fragment": chunk_md,
            "row_range": f"{start+1}-{start+len(chunk_data)}",
            "row_count": len(chunk_data),
            "columns": cols,
        })

    return chunks
