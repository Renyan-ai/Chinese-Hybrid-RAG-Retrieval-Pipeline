import os
import json
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode

from ..parser.pdf_parser import parse_pdf_with_large_chunks

SAVE_DIR = "./storage_data"
CHROMA_PATH = os.path.join(SAVE_DIR, "chroma_db")
NODES_PATH = os.path.join(SAVE_DIR, "nodes.json")
STORES_PATH = os.path.join(SAVE_DIR, "extra_stores.json")


def build_or_load_index(pdf_path, embed_model, force_rebuild=False):
    os.makedirs(SAVE_DIR, exist_ok=True)

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection("fixed_retrieval")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if not force_rebuild and os.path.exists(NODES_PATH) and collection.count() > 0:
        print(" 加载已有修复索引")
        with open(NODES_PATH, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)
        with open(STORES_PATH, "r", encoding="utf-8") as f:
            stores = json.load(f)

        nodes = [
            TextNode(text=d["text"], id_=d["id_"], metadata=d["metadata"])
            for d in nodes_data
        ]
        parent_store = stores["parent_store"]
        table_store = stores["table_store"]

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        print(f"   → {len(nodes)} 节点, {len(parent_store)} 章节, {len(table_store)} 表格")
    else:
        print(" 重建索引")
        parent_store, table_store, nodes = parse_pdf_with_large_chunks(pdf_path)

        nodes_data = []
        for n in nodes:
            nodes_data.append({
                "id_": n.node_id,
                "text": n.text,
                "metadata": n.metadata,
            })
        with open(NODES_PATH, "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)

        with open(STORES_PATH, "w", encoding="utf-8") as f:
            json.dump({"parent_store": parent_store, "table_store": table_store},
                      f, ensure_ascii=False, indent=2)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes, storage_context=storage_context, embed_model=embed_model
        )
        print(f" 索引构建完成: {len(nodes)} 节点")

    return index, parent_store, table_store, nodes
