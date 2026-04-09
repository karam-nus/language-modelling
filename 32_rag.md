---
title: "Chapter 32 — Retrieval-Augmented Generation (RAG)"
---

[← Back to Table of Contents](./README.md)

# Chapter 32 — Retrieval-Augmented Generation (RAG)

> *"RAG doesn't make the model smarter — it gives the model a library card. The model still has to know how to read."*

## Why RAG?

LLMs have fundamental limitations that RAG addresses:

| Problem | RAG Solution |
|---------|-------------|
| **Knowledge cutoff** — model doesn't know recent events | Retrieve fresh documents at query time |
| **Hallucination** — model confidently makes things up | Ground responses in retrieved evidence |
| **Domain specificity** — model lacks your proprietary data | Index your documents, retrieve relevant chunks |
| **Verifiability** — hard to check where information came from | Cite retrieved source documents |

## The RAG Pipeline

<div class="diagram">
<div class="diagram-title">RAG Pipeline</div>
<div class="flow">
  <div class="flow-step green">📄 Documents — chunk into passages (256–512 tokens each)</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step accent">🧮 Embed — convert each chunk to a dense vector using an embedding model</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">📦 Index — store embeddings in a vector database (FAISS, Chroma, etc.)</div>
  <div class="flow-arrow">↓ At query time:</div>
  <div class="flow-step orange">🔍 Retrieve — embed the query, find top-K most similar chunks</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step accent">📝 Augment — inject retrieved chunks into the LLM prompt as context</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">🤖 Generate — LLM produces answer grounded in the retrieved context</div>
</div>
</div>

## Embedding Models

The embedding model converts text to dense vectors where **similar meaning → nearby vectors**:

| Model | Dimensions | Max Tokens | MTEB Score | Notes |
|-------|-----------|-----------|------------|-------|
| `all-MiniLM-L6-v2` | 384 | 256 | 56.3 | Fast, lightweight |
| `bge-large-en-v1.5` | 1024 | 512 | 64.2 | Strong general-purpose |
| `E5-mistral-7b` | 4096 | 32K | 66.6 | LLM-based, best quality |
| `text-embedding-3-large` | 3072 | 8191 | ~65 | OpenAI, API-only |
| `nomic-embed-text-v1.5` | 768 | 8192 | 62.3 | Open, long context |
| `gte-Qwen2-7B` | 3584 | 32K | 67.2 | Qwen-based, state-of-art |

```python
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Embed documents
doc_texts = ["Transformers use attention...", "RoPE encodes position..."]
doc_embeddings = embed_model.encode(doc_texts, normalize_embeddings=True)

# Embed query
query_embedding = embed_model.encode("How does positional encoding work?",
                                      normalize_embeddings=True)

# Cosine similarity (since normalized, dot product = cosine sim)
similarities = query_embedding @ doc_embeddings.T
```

## Vector Databases

| Database | Type | Key Features |
|----------|------|-------------|
| **FAISS** | Library (Meta) | Fastest for local use, IVF/HNSW indexes, GPU support |
| **Chroma** | Embedded DB | Simple Python API, persistent storage, metadata filtering |
| **Pinecone** | Managed cloud | Serverless, auto-scaling, hybrid search |
| **Weaviate** | Self-hosted/cloud | GraphQL API, hybrid (vector + keyword), modules |
| **Qdrant** | Self-hosted/cloud | Rust-based, filtering, quantization built-in |
| **Milvus** | Distributed | Kubernetes-native, billion-scale, multi-vector |

```python
# Simple RAG with FAISS
import faiss
import numpy as np

# Build index
dimension = 1024  # BGE-large embedding size
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim for normalized)
index.add(np.array(doc_embeddings))

# Search
k = 5  # top-5 results
scores, indices = index.search(np.array([query_embedding]), k)
retrieved_docs = [documents[i] for i in indices[0]]
```

## Building the Prompt

```python
def build_rag_prompt(query, retrieved_docs, max_context_tokens=3000):
    context = "\n\n---\n\n".join([
        f"Source {i+1}:\n{doc['text']}"
        for i, doc in enumerate(retrieved_docs)
    ])
    
    prompt = f"""Answer the question based on the provided context. 
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
    return prompt
```

## Advanced RAG Techniques

<div class="diagram">
<div class="diagram-title">RAG Enhancement Strategies</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card green">
    <div class="card-title">Reranking</div>
    <div class="card-desc">Retrieve top-50 with embeddings (fast), then rerank with a cross-encoder model (accurate) to select top-5. Models: bge-reranker, Cohere Rerank.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Hybrid Search</div>
    <div class="card-desc">Combine dense (semantic) + sparse (BM25 keyword) retrieval. Reciprocal rank fusion merges results. Catches both semantic matches and exact keywords.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Query Expansion</div>
    <div class="card-desc">Use an LLM to rewrite/expand the user query before retrieval. HyDE: generate a hypothetical answer, embed that instead of the query.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Chunking Strategy</div>
    <div class="card-desc">Recursive character splitting, semantic chunking (split at topic boundaries), parent-child (retrieve child, provide parent context).</div>
  </div>
</div>
</div>

### Reranking Example

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")

# First stage: retrieve top-50 with embedding similarity (fast)
candidates = vector_search(query, top_k=50)

# Second stage: rerank with cross-encoder (accurate)
pairs = [(query, doc["text"]) for doc in candidates]
scores = reranker.predict(pairs)

# Sort by reranker score, take top-5
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
```

## RAG with LangChain

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# 1. Load documents
loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Embed + Index
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Create retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# 5. Query
result = qa_chain.invoke({"query": "How does Flash Attention work?"})
print(result["result"])
```

## RAG Evaluation

| Metric | What It Measures |
|--------|-----------------|
| **Context Relevance** | Are retrieved docs relevant to the query? |
| **Faithfulness** | Is the answer grounded in the retrieved context (not hallucinated)? |
| **Answer Relevance** | Does the answer actually address the question? |
| **Context Recall** | Did retrieval find all relevant information? |

Frameworks like **RAGAS** and **DeepEval** automate these evaluations.

## What's Next

RAG retrieves static knowledge. But what if the model needs to **take actions** — search the web, run code, query databases? The next chapter covers **agents and tool use**.

[← Previous: Chapter 31 — Reasoning Models](./31_reasoning_models.md) · **Next: [Chapter 33 — Agents & Tool Use →](./33_agents_and_tool_use.md)**

---

*Last updated: April 2026*
