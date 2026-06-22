from llama_index.core.schema import QueryBundle
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

from .bm25 import BM25JiebaRetriever
from .rerank import RowLevelReranker

VECTOR_TOP_K = 15
BM25_TOP_K = 15
FUSION_TOP_K = 10
RERANK_TOP_N = 5


class FixedRetriever:
    """检索器：BM25+jieba + QueryFusion + Rerank + 行级精排"""

    def __init__(self, index, nodes, parent_store, table_store, llm, embed_model, reranker_model_path):
        self.parent_store = parent_store
        self.table_store = table_store

        self.vector_retriever = index.as_retriever(similarity_top_k=VECTOR_TOP_K)

        self.bm25_retriever = BM25JiebaRetriever(
            nodes=nodes,
            similarity_top_k=BM25_TOP_K,
            table_store=table_store,
        )

        self.fusion_retriever = QueryFusionRetriever(
            [self.vector_retriever, self.bm25_retriever],
            similarity_top_k=FUSION_TOP_K,
            llm=llm,
            num_queries=3,
            mode="reciprocal_rerank",
            use_async=True,
        )

        self.reranker = SentenceTransformerRerank(
            model=reranker_model_path,
            top_n=RERANK_TOP_N
        )

        self.row_reranker = RowLevelReranker(reranker_model_path)

    def _apply_row_level_rerank(self, reranked, question):
        """对 rerank 后的结果集做行级精排"""
        final = []
        for nws in reranked:
            node = nws.node
            if node.metadata.get("type") == "table":
                table_md = node.metadata.get("raw_md_fragment", "")
                if not table_md:
                    table_md = self.table_store.get(node.metadata.get("table_id", ""), "")
                row_score, top_rows = self.row_reranker.rerank_table_rows(node, question, table_md)
                if row_score is not None:
                    nws.score = row_score
                    node.metadata["_top_rows"] = top_rows
            final.append(nws)

        final.sort(key=lambda x: x.score or 0, reverse=True)
        return final[:RERANK_TOP_N]

    def retrieve(self, question: str, verbose=True):
        nodes = self.fusion_retriever.retrieve(question)
        reranked = self.reranker.postprocess_nodes(
            nodes, query_bundle=QueryBundle(question)
        )
        reranked = self._apply_row_level_rerank(reranked, question)

        results = []
        for i, nws in enumerate(reranked):
            node = nws.node
            meta = node.metadata
            score = nws.score
            node_type = meta.get("type", "text")
            chapter_title = meta.get("chapter_title", "")
            parent_id = meta.get("parent_id", "")

            parent_ctx = self.parent_store.get(parent_id, "")[:800]

            entry = {
                "rank": i + 1,
                "type": node_type,
                "score": round(score, 4),
                "chapter_title": chapter_title,
                "chunk_text": node.text[:300],
                "parent_context": parent_ctx,
            }

            if node_type == "table":
                top_rows = meta.get("_top_rows", "")
                entry["table_top_rows"] = top_rows
                entry["row_range"] = meta.get("row_range", "")
                entry["table_part"] = f"{meta.get('table_part_index', 0)+1}/{meta.get('table_part_total', 1)}"
            else:
                entry["table_top_rows"] = ""

            results.append(entry)

            if verbose:
                print(f"\n{'='*60}")
                print(f"  [{i+1}] {node_type.upper():<6} | score={score:.4f} | 章节: {chapter_title}")
                if node_type == "table":
                    print(f"      子块: {entry['table_part']} | 行: {entry['row_range']}")
                print(f"{'─'*60}")
                print(f"  匹配片段:\n  {node.text[:200]}...")
                if node_type == "table" and entry["table_top_rows"]:
                    print(f"  命中行(top-3):\n  {entry['table_top_rows'][:300]}")
                print(f"  父块上下文(前200字):\n  {parent_ctx[:200]}...")

        print(f"\n{'='*60}")
        print(f" 共检索到 {len(reranked)} 条结果")
        return results

    def retrieve_contexts(self, question: str) -> list[str]:
        nodes = self.fusion_retriever.retrieve(question)
        reranked = self.reranker.postprocess_nodes(
            nodes, query_bundle=QueryBundle(question)
        )
        reranked = self._apply_row_level_rerank(reranked, question)

        contexts = []
        for nws in reranked:
            node = nws.node
            meta = node.metadata
            if meta.get("type") == "table":
                top_rows = meta.get("_top_rows", "")
                contexts.append(top_rows if top_rows else node.text)
            else:
                contexts.append(node.text)
        return contexts

    def retrieve_ablation(self, question: str):
        print(f"\n{'='*60}")
        print(f" 消融实验: 「{question}」")
        print(f"{'='*60}")

        vec_nodes = self.vector_retriever.retrieve(question)
        print(f"\n 纯向量检索 top-{len(vec_nodes)}:")
        for i, n in enumerate(vec_nodes[:5]):
            ntype = n.node.metadata.get("type", "?")
            print(f"  [{i+1}] score={n.score:.4f} [{ntype}] {n.node.text[:100]}")

        bm25_nodes = self.bm25_retriever.retrieve(question)
        print(f"\n 纯 BM25 (jieba) top-{len(bm25_nodes)}:")
        for i, n in enumerate(bm25_nodes[:5]):
            ntype = n.node.metadata.get("type", "?")
            print(f"  [{i+1}] score={n.score:.4f} [{ntype}] {n.node.text[:100]}")

        print(f"\n 融合 + Rerank top-{RERANK_TOP_N}:")
        return self.retrieve(question, verbose=True)
