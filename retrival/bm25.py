import jieba
from rank_bm25 import BM25Okapi
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore


class BM25JiebaRetriever(BaseRetriever):
    """兼容 llama_index 的 BM25 检索器，表格节点用清洗后的语义文本做分词"""

    def __init__(self, nodes, similarity_top_k=10, table_store=None):
        super().__init__()
        self.similarity_top_k = similarity_top_k
        self._nodes = nodes
        self._table_store = table_store or {}

        tokenized_corpus = []
        for n in nodes:
            bm25_text = self._get_bm25_text(n)
            tokenized_corpus.append(list(jieba.cut(bm25_text)))
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _get_bm25_text(self, node):
        if node.metadata.get("type") == "table":
            meta_text = node.text
            cols = node.metadata.get("columns", [])
            if cols:
                col_prompt = "关键词：" + " ".join(cols)
                if len(meta_text) < 2000:
                    meta_text = meta_text + "\n" + col_prompt
            return meta_text
        return node.text

    def _retrieve(self, query):
        query_tokens = list(jieba.cut(str(query)))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: -scores[i]
        )[:self.similarity_top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(NodeWithScore(
                    node=self._nodes[idx],
                    score=float(scores[idx]),
                ))
        return results
