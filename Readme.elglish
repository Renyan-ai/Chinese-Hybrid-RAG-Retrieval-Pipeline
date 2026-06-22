# Hybrid RAG for Chinese PDF Documents

> A retrieval-oriented Hybrid RAG framework for Chinese PDF documents, featuring semantic table parsing, hybrid dense–sparse retrieval, query fusion, row-level reranking, and automated evaluation with RAGAS.

## Overview

This project implements a retrieval-oriented Retrieval-Augmented Generation (RAG) pipeline designed for complex Chinese PDF documents, such as financial reports.

Instead of focusing on answer generation, this project focuses on improving retrieval quality by combining dense retrieval, sparse retrieval, semantic table understanding, and reranking techniques.

The pipeline supports structured PDF parsing, semantic table processing, hybrid retrieval, query rewriting, Cross-Encoder reranking, and automatic evaluation.

---

## Features

* 📄 Structured PDF parsing with **Docling**
* ✂️ Recursive text chunking for long documents
* 📊 Semantic conversion of Markdown tables into natural language
* 📑 Large table row-level chunking with overlapping windows
* 🔍 Hybrid Retrieval (Dense Retrieval + BM25)
* 🔄 Multi-query retrieval using Query Fusion
* 🎯 Cross-Encoder reranking (BGE-Reranker)
* 📈 Row-level table reranking for structured data
* 💾 Persistent vector storage with ChromaDB
* 📊 Automatic retrieval evaluation using RAGAS
* 🧪 Retrieval ablation experiments (Vector vs BM25 vs Hybrid)

---

## System Architecture

```text
                    PDF Document
                          │
                          ▼
                 Docling Document Parser
                          │
          ┌───────────────┴───────────────┐
          │                               │
      Text Chunks                 Table Parser
          │                               │
          │                   Semantic Table Conversion
          │                               │
          └───────────────┬───────────────┘
                          ▼
                Embedding Generation
                          │
          ┌───────────────┴───────────────┐
          │                               │
    Dense Retrieval                BM25 Retrieval
          │                               │
          └───────────────┬───────────────┘
                          ▼
                 Query Fusion Retrieval
                          │
                          ▼
               Cross-Encoder Reranking
                          │
                          ▼
              Row-level Table Reranking
                          │
                          ▼
                Retrieved Contexts
```

---

## Retrieval Pipeline

### 1. Document Parsing

The system first parses PDF documents using **Docling**, preserving document hierarchy, text blocks, and tables.

### 2. Chunking

Text content is split using recursive chunking with overlapping windows to preserve semantic continuity.

### 3. Semantic Table Parsing

Instead of embedding raw Markdown tables, each table is transformed into semantic natural language.

Example:

Raw table

```text
| Year | Revenue |
|------|---------|
|2025|7518|
```

Converted representation

```text
Table: Financial Summary

Year: 2025
Revenue: RMB 751.8 Billion
```

This significantly improves both embedding quality and BM25 retrieval.

---

### 4. Hybrid Retrieval

The retrieval system combines

* Dense Retrieval (BGE Embedding)
* Sparse Retrieval (BM25)

using Reciprocal Rank Fusion.

---

### 5. Reranking

Retrieved candidates are reranked using

* BGE Cross-Encoder
* Row-level reranking for table contents

The row-level strategy enables the system to locate the most relevant rows inside large tables instead of ranking the entire table as a single block.

---

### 6. Evaluation

Retrieval quality is automatically evaluated using **RAGAS**, including metrics such as

* Context Precision

The project also provides retrieval ablation experiments for comparing

* Dense Retrieval
* BM25
* Hybrid Retrieval

---

## Project Structure

```text
Hybrid-RAG/
│
├── parser/
│   ├── pdf_parser.py
│   └── table_parser.py
│
├── retrieval/
│   ├── bm25.py
│   ├── hybrid.py
│   ├── rerank.py
│   └── vector.py
│
├── evaluation/
│   └── ragas_eval.py
│
├── storage/
│
├── data/
│   └── sample.pdf
│
├── images/
│
├── main.py
│
├── requirements.txt
│
└── README.md
```

---

## Technologies

| Category         | Technology         |
| ---------------- | ------------------ |
| PDF Parsing      | Docling            |
| Embedding        | BAAI BGE           |
| Vector Database  | ChromaDB           |
| Sparse Retrieval | BM25               |
| Framework        | LlamaIndex         |
| Query Expansion  | DeepSeek           |
| Reranking        | BGE-Reranker-v2-m3 |
| Evaluation       | RAGAS              |

---

## Example

```bash
python main.py --rebuild
```

Example query

```text
Question:
2025年腾讯全年总收入是多少？
```

Retrieved context

```text
Financial Summary

Total Revenue:
RMB 751.8 Billion
```

---

## Future Work

* Multi-document indexing
* GraphRAG integration
* OCR support for scanned PDFs
* Web UI deployment
* Multimodal document retrieval

---

## License

This project is released under the MIT License.
