# RAG Pipeline: Semantic Chunking, Embedding & Cluster Visualization

**Retrieval-Augmented Generation · Vector Embeddings · Unsupervised Clustering · LLM Evaluation**

> End-to-end RAG preprocessing pipeline — from raw document ingestion through vector embedding, unsupervised semantic clustering, and LLM-assisted cluster interpretation. Built for the [Super Data Science AI Challenge](https://www.skool.com/ai-challenge), using the [Netflix Culture Memo](https://jobs.netflix.com/culture) as the target knowledge base.

---

## Overview

This project implements a production-oriented RAG preprocessing pipeline with an emphasis on decisions that are often skipped in tutorial-style implementations: chunking strategy, embedding quality validation, and interpretability of the resulting vector space.

The pipeline covers document ingestion, semantic chunking with overlap, dense vector embedding, unsupervised clustering, dimensionality reduction for visualization, and LLM-assisted cluster labeling — each stage designed to transfer directly to a production retrieval system.

---

## Tech Stack

| Category | Tools & Libraries |
|----------|------------------|
| **LLMs & Embeddings** | OpenAI `text-embedding-3-small`, `gpt-4o-mini` |
| **Vector Operations** | NumPy, in-memory embedding store |
| **Clustering** | KMeans (scikit-learn) |
| **Dimensionality Reduction** | UMAP (`umap-learn`) |
| **Visualization** | Plotly (interactive 3D) |
| **Environment** | Python, Jupyter, python-dotenv |
| **Compatible Vector Stores** | Pinecone, Weaviate, pgvector |

---

## Pipeline Architecture

```
Raw Document
     │
     ▼
[1] Ingest          Load & normalize raw text
     │
     ▼
[2] Semantic Chunk  Boundary-aware splitting with configurable overlap
     │
     ▼
[3] Embed           text-embedding-3-small → 1536-dim dense vectors
     │
     ▼
[4] Cluster         KMeans (k=2) on high-dimensional embeddings
     │
     ▼
[5] Reduce          UMAP → 2D / 3D for visualization & inspection
     │
     ▼
[6] Label           gpt-4o-mini interprets each discovered cluster
```

---

## Chunking Strategy

A common starting point for RAG chunking is a fixed character count — fast to implement, but it frequently splits text mid-sentence and produces inconsistent context windows that degrade retrieval precision.

The custom `chunk_text()` function addresses this by snapping boundaries to natural language breaks in priority order:

| Priority | Boundary | Rationale |
|----------|----------|-----------|
| 1st | Paragraph (`\n\n`) | Preserves topic coherence |
| 2nd | Sentence (`". "`) | Preserves grammatical completeness |
| 3rd | Word (`" "`) | Avoids mid-token splits |
| 4th | Hard cut at `max_size` | Safety fallback only |

Chunks below `min_size` are merged with adjacent chunks rather than emitted standalone, which reduces short-chunk retrieval noise at query time.

```python
chunk_text(text, max_size=500, overlap=50, min_size=250)
```

Overlap (`overlap=50`) ensures that context straddling a chunk boundary is represented in both adjacent chunks — an important property for any retrieval system answering questions that span multiple sentences.

---

## Embedding & Vector Retrieval

Each chunk is encoded with OpenAI's `text-embedding-3-small`, a general-purpose encoder with strong semantic fidelity at relatively low cost (1536 dimensions).

Design decisions relevant to production deployment:

- Embeddings are generated **per-chunk, not per-document**, enabling fine-grained retrieval
- The embedding call structure is compatible with standard **vector databases** (Pinecone, Weaviate, pgvector) without modification
- Vectors are stored in-memory for downstream clustering and visualization in this implementation

---

## Embedding Quality Validation: UMAP + KMeans

Before deploying a RAG system, it is worth confirming that the embeddings reflect meaningful semantic structure. UMAP combined with KMeans provides a practical approach.

**UMAP** reduces the high-dimensional embedding space to 2D or 3D while preserving local neighborhood structure — making the vector space human-inspectable. **KMeans** (k=2) partitions chunks by semantic similarity without requiring labeled data. The resulting visualization reveals whether the document's topics are separable in vector space, which serves as an indirect indicator of retrieval quality.

Applied to the Netflix Culture Memo, the pipeline identifies a clean two-cluster separation:

> **Cluster 0:** Organizational principles &nbsp;|&nbsp; **Cluster 1:** People & culture themes

![UMAP 3D Projection of Chunk Embeddings](3D_Plot_Clean_Separation.png)

---

## LLM-Assisted Cluster Labeling

After clustering, `gpt-4o-mini` is prompted with a sample of chunks from each cluster and asked to produce a short thematic description. This provides human-readable labels for what would otherwise be opaque groupings in vector space.

Using a lightweight model for intermediate annotation is a practical approach in ML pipelines where manual review of every chunk is not feasible — and it connects the labeling step to the same class of model used for downstream generation.

---

## Models & Tool Selection

| Component | Choice | Notes |
|-----------|--------|-------|
| Embeddings | `text-embedding-3-small` | 1536-dim, strong semantic fidelity, cost-efficient |
| Cluster labeling | `gpt-4o-mini` | Low latency and cost for short summarization tasks |
| Dimensionality reduction | UMAP | Preserves local structure better than PCA or t-SNE |
| Clustering | KMeans (scikit-learn) | Interpretable, fast, appropriate for k=2 |
| 3D visualization | Plotly | Interactive and rotatable — surfaces structure that 2D projections miss |

---

## Getting Started

**1. Install dependencies**

```bash
pip install openai python-dotenv numpy matplotlib umap-learn scikit-learn plotly
```

**2. Configure API access**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

**3. Run the notebook**

```bash
jupyter notebook RAG-scratchpad.ipynb
```

Execute all cells top to bottom.

---

## Skills & Concepts Demonstrated

### Retrieval-Augmented Generation (RAG)
End-to-end pipeline design including document ingestion, chunking strategy, embedding generation, and retrieval-oriented architecture.

### Applied Machine Learning
Unsupervised clustering (KMeans), dimensionality reduction (UMAP), and embedding evaluation without labeled training data.

### LLM Integration
Prompt design for structured annotation tasks; cost-aware model selection (GPT-4o-mini for labeling vs. embedding models for retrieval).

### Production Readiness
Chunking heuristics and overlap strategy that reduce retrieval noise; embedding structure compatible with production vector stores; modular pipeline design that generalizes to other document corpora.

### Evaluation & Interpretability
Visual validation of embedding quality prior to deployment; LLM-assisted labeling to make vector space structure accessible to non-technical stakeholders.

---

## Relevance to Engineering Roles

| Role | Applicable Work |
|------|----------------|
| **AI / Applied ML Engineer** | Embedding pipeline, chunking heuristics, clustering, dimensionality reduction, visual evaluation |
| **AI Product Engineer** | End-to-end RAG prototype on a real knowledge base; tradeoff decisions on chunk size, overlap, and retrieval granularity |
| **AI Agent / LLM Engineer** | Structured multi-step pipeline with LLM calls at defined stages; model selection based on task requirements |
| **Forward-Deployed / Solutions Engineer** | Modular, corpus-agnostic pipeline; adaptable embedding and retrieval architecture for customer-specific deployments |

---

## Keywords

`RAG` · `retrieval-augmented generation` · `vector embeddings` · `semantic search` · `chunking` · `OpenAI API` · `text-embedding-3-small` · `gpt-4o-mini` · `KMeans` · `UMAP` · `dimensionality reduction` · `vector database` · `Pinecone` · `Weaviate` · `pgvector` · `LLM` · `NLP` · `information retrieval` · `unsupervised learning` · `Python` · `scikit-learn` · `Plotly` · `Jupyter` · `production ML` · `embedding evaluation`

---

*Built for the [Super Data Science AI Challenge](https://www.skool.com/ai-challenge)*
