from sentence_transformers import CrossEncoder


class RowLevelReranker:
    """行级重排序器"""

    def __init__(self, model_path: str):
        self._row_ce = CrossEncoder(
            model_path,
            automodel_args={"torch_dtype": "auto"},
        )

    def rerank_table_rows(self, node, query: str, table_md: str):
        """对表格节点做行级 rerank，返回最高分和 top-3 行"""
        lines = [l for l in table_md.strip().split('\n') if l.strip()]
        if len(lines) < 3:
            return None, ""

        cols = [c.strip() for c in lines[0].split('|') if c.strip()]
        data_lines = [l for l in lines[2:] if l.strip()]

        if len(data_lines) <= 2:
            return None, ""

        chapter = node.metadata.get('chapter_title', '')

        row_texts = []
        for row in data_lines:
            cells = [c.strip() for c in row.split('|') if c.strip()]
            if cols and len(cells) >= len(cols):
                row_desc = "；".join(f"{cols[j]}：{cells[j]}" for j in range(len(cols)))
            else:
                row_desc = "，".join(cells)
            row_texts.append(f"表格 {chapter} | {row_desc}")

        pairs = [(query, rt) for rt in row_texts]
        row_scores = self._row_ce.predict(pairs)

        top_k = min(3, len(row_scores))
        top_indices = sorted(range(len(row_scores)), key=lambda i: -row_scores[i])[:top_k]
        top_rows_text = "\n".join(row_texts[i] for i in top_indices)

        best_score = float(max(row_scores))
        return best_score, top_rows_text
