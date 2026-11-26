---
title: "Chapter 36 — Retrieval-Augmented Generation (RAG)"
description: "Grounding generation in retrieved evidence: chunking, embeddings and the 2025–26 embedding models, vector indexes, hybrid search, reranking, contextual retrieval, late interaction and visual document retrieval, a framework-free pipeline, agentic RAG and deep research, long context vs retrieval, evaluation, and prompt-injection risks."
related:
  - { id: agents, why: "Retrieval as one tool among many." }
---

[← Back to Table of Contents](./README.md)

# Chapter 36 — Retrieval-Augmented Generation (RAG)

A model's weights hold what it learned up to its cutoff, compressed and lossy. **Retrieval-augmented generation** fetches relevant text at query time and puts it in the context, so the model can answer from fresh, private or exact sources — and cite them.

> **The one-sentence version:** RAG turns "answer from memory" into "answer from these documents", and its quality is decided mostly by retrieval — which chunks you find — not by the generator.

## Why RAG?

| Problem | What retrieval adds |
|---------|-------------|
| Knowledge cutoff | Fresh documents fetched at query time |
| Hallucination | Answers grounded in, and checkable against, evidence |
| Private or domain data | Your documents, without retraining |
| Verifiability | Citations to sources |
| Cost | A small model plus good retrieval often beats a large model alone |

## The RAG Pipeline

<div class="diagram">
<div class="diagram-title">RAG Pipeline</div>
<div class="flow">
  <div class="flow-step green">📄 Ingest — parse (PDF, HTML, slides), clean, chunk into passages</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step accent">🧮 Embed — encode each chunk into a vector (and optionally a sparse/BM25 representation)</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">📦 Index — store vectors and text in an approximate-nearest-neighbour index</div>
  <div class="flow-arrow">↓ at query time</div>
  <div class="flow-step orange">🔍 Retrieve — dense + keyword search for candidates (top 50–200)</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step accent">🎯 Rerank — cross-encoder scores query–chunk pairs; keep the best 5–20</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">🤖 Generate — the LLM answers from the retrieved context, with citations</div>
</div>
</div>

## Chunking

Chunks must be small enough to be specific and large enough to be self-contained. Common choices: 200–800 tokens with some overlap; split on document structure (headings, paragraphs, code blocks) before falling back to character counts; keep tables intact; store a pointer to the **parent** section so you can retrieve a small chunk but give the model its surrounding context.

**Contextual retrieval** (Anthropic, 2024) fixes chunks that lose meaning out of context ("the company's revenue grew 3%" — which company?): an LLM writes a one-or-two-sentence context for each chunk from the whole document, prepended before embedding and BM25 indexing. Anthropic reported 49% fewer retrieval failures with contextual embeddings plus contextual BM25, and 67% fewer with reranking added.

## Embedding Models

| Model (year) | Dimensions | Max tokens | Notes |
|-------|-----------|-----------|-------|
| all-MiniLM-L6-v2 (2021) | 384 | 256 | Tiny, CPU-friendly baseline |
| BGE-M3 (2024) | 1,024 | 8,192 | Dense + sparse + multi-vector; 100+ languages |
| text-embedding-3-large (OpenAI, 2024) | up to 3,072 | 8,191 | API; truncatable dimensions |
| EmbeddingGemma (2025) | 768 (Matryoshka to 128) | 2,048 | 308M, on-device |
| Qwen3-Embedding (2025) | up to 4,096 | 32K | 0.6B–8B; topped the multilingual MTEB at release |
| gemini-embedding-001 (2025) | up to 3,072 | 2,048 | API; strong multilingual MTEB |
| jina-embeddings-v4 / ColPali-style (2025) | multi-vector | — | Text and page images (visual documents) |

Leaderboard scores shift with each MTEB revision; check the current MTEB leaderboard *and* evaluate on your own queries. Many models are trained with **Matryoshka** representation learning, so you can truncate vectors (e.g., 1,024 → 256 dimensions) to cut storage with little loss.

```python
from sentence_transformers import SentenceTransformer

emb = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
docs = ["Transformers use attention to mix information across tokens.",
        "RoPE encodes position by rotating query and key vectors."]
D = emb.encode(docs, normalize_embeddings=True)                        # [n_docs, dim]
q = emb.encode(["How does positional encoding work?"], prompt_name="query",
               normalize_embeddings=True)                              # queries use an instruction prompt
print((q @ D.T).round(3))                                              # cosine similarities
```

## Indexes and Vector Databases

Exact search over millions of vectors is too slow; **approximate nearest-neighbour** indexes trade a little recall for speed: HNSW graphs (fast, memory-hungry), IVF clustering (with product quantization for compression), and disk-based indexes (DiskANN) for billions of vectors.

| Option | Type | When |
|----------|------|-------------|
| FAISS | Library | In-process, GPU support, research and batch jobs |
| pgvector | Postgres extension | You already run Postgres; transactional data + vectors |
| Qdrant, Weaviate, Milvus | Vector databases | Filtering, hybrid search, scale-out |
| Elasticsearch / OpenSearch | Search engines | Mature BM25 + vector hybrid |
| LanceDB, Chroma | Embedded | Local apps and prototypes |
| Pinecone, Turbopuffer | Managed | Serverless scale |

## Hybrid Search and Reranking

Dense embeddings capture meaning but miss exact tokens (product codes, function names, rare entities); **BM25** catches exact matches but misses paraphrases. **Hybrid search** runs both and merges rankings, commonly with reciprocal rank fusion: score(d) = Σ 1/(60 + rank_i(d)).

A **cross-encoder reranker** then reads each (query, chunk) pair jointly — far more accurate than comparing two independent vectors, and too slow for the whole corpus, which is why it only sees the top candidates.

## A Minimal Pipeline, No Framework

```python
import faiss, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

emb = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
rerank = CrossEncoder("BAAI/bge-reranker-v2-m3")
llm = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")      # any OpenAI-compatible server

chunks = load_and_chunk("./docs")                                          # your ingestion: list of {"id", "text"}
X = emb.encode([c["text"] for c in chunks], normalize_embeddings=True)
index = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
index.add(X.astype(np.float32))

def answer(question, k_retrieve=50, k_final=6):
    q = emb.encode([question], prompt_name="query", normalize_embeddings=True).astype(np.float32)
    _, ids = index.search(q, k_retrieve)
    cands = [chunks[i] for i in ids[0]]
    scores = rerank.predict([(question, c["text"]) for c in cands])
    top = [c for _, c in sorted(zip(scores, cands), key=lambda t: -t[0])[:k_final]]
    context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in top)
    msgs = [{"role": "system", "content": "Answer only from the sources. Cite them as [id]. "
                                          "If they do not contain the answer, say so."},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}]
    return llm.chat.completions.create(model="Qwen/Qwen3-8B", messages=msgs).choices[0].message.content
```

LangChain, LlamaIndex and Haystack package the same steps with many connectors; the pipeline above is what they do underneath.

## Beyond Basic RAG

<div class="diagram">
<div class="diagram-title">RAG Enhancement Strategies</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card green">
    <div class="card-title">Query rewriting</div>
    <div class="card-desc">Have an LLM rewrite, decompose or expand the query; HyDE embeds a hypothetical answer instead of the question.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Late interaction</div>
    <div class="card-desc">ColBERT keeps one vector per token and scores with max-similarity; ColPali does it over page images, retrieving PDFs and slides without OCR.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Graph-based RAG</div>
    <div class="card-desc">Extract entities and relations (GraphRAG) to answer "global" questions — themes across a corpus — that chunk retrieval misses.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Agentic RAG and deep research</div>
    <div class="card-desc">Retrieval becomes a tool a reasoning model calls repeatedly — search, read, refine, search again — producing cited reports (Chapter 37).</div>
  </div>
</div>
</div>

### RAG vs Long Context

With 1M-token windows, why retrieve at all? Because long context is **expensive** (cost and latency scale with every token, every call), **leaky** (effective context is shorter than advertised, [Chapter 12](./12_mid_training.md)) and **bounded** (corpora are larger than any window). The 2025–26 practice — sometimes called *context engineering* — combines them: retrieve to narrow down, then give a long-context model generous, well-ordered evidence, and let prefix caching ([Chapter 18](./18_kv_cache_optimization.md)) make repeated context cheap.

## Evaluating RAG

| Metric | Question it answers |
|--------|-----------------|
| Retrieval recall@k / nDCG | Did the right chunks appear in the top k? |
| Context precision | How much of what was retrieved was relevant? |
| Faithfulness / groundedness | Is every claim supported by the retrieved text? |
| Answer correctness | Is the answer right (vs a reference or rubric)? |
| Citation accuracy | Do the citations actually support the sentences they are attached to? |

Measure retrieval separately from generation — most failures are retrieval failures. RAGAS, DeepEval and TruLens automate LLM-judged metrics; a small hand-labelled set of real questions keeps the judges honest.

<div class="callout warn">
<span class="callout-title">Retrieved text is untrusted input</span>
<p>Any document in the index can contain instructions ("ignore previous instructions and…"). In RAG and especially in agents, retrieved content must be treated as data, not commands: separate it clearly in the prompt, limit what tools the model can call based on it, and never let it silently change the system's goals. Indirect prompt injection is the most common real-world attack on RAG systems.</p>
</div>

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A reasoning agent that searches, reads and searches again behaves like a researcher; a one-shot retriever behaves like a librarian handing over five books. Which questions in your domain need the researcher — and how would you tell, cheaply, before spending the tokens?</p>
</div>

## Key Takeaways

- RAG = ingest → chunk → embed/index → retrieve (hybrid) → rerank → generate with citations; retrieval quality dominates.
- Use current embedding models (Qwen3-Embedding, BGE-M3, EmbeddingGemma, API models), Matryoshka truncation, and a cross-encoder reranker.
- Contextual retrieval, late interaction (ColBERT/ColPali), GraphRAG and agentic multi-step search address basic RAG's blind spots.
- Long context and RAG are complements: retrieve to narrow, then provide rich context, with prefix caching.
- Evaluate retrieval and generation separately; treat retrieved text as untrusted.

## Check Yourself

<details class="check">
<summary>Why does hybrid search often beat pure dense retrieval on technical documentation?</summary>
<p>Queries contain exact identifiers — function names, error codes, version numbers — that embeddings blur together with similar-looking strings; BM25 matches them exactly. Dense retrieval handles paraphrase. Fusing the two rankings gets both.</p>
</details>

<details class="check">
<summary>Why can't you simply use the cross-encoder for all of retrieval?</summary>
<p>It must read each (query, document) pair through the full model at query time — millions of forward passes for a large corpus. Bi-encoder vectors are precomputed once, so search is a fast nearest-neighbour lookup; the cross-encoder runs only on the shortlist.</p>
</details>

<details class="check">
<summary>A RAG system answers confidently but cites a source that doesn't support the claim. Which metric catches this, and what are two likely causes?</summary>
<p>Faithfulness / citation accuracy. Causes: the right chunk was not retrieved and the model answered from memory, or the prompt did not instruct (and the model was not trained) to answer only from sources and abstain otherwise.</p>
</details>

<details class="check">
<summary>What does contextual retrieval change at indexing time, and why does it help?</summary>
<p>Each chunk gets a short LLM-written description of where it sits in its document, prepended before embedding and BM25 indexing, so chunks that rely on surrounding context ("the company", "this method") become retrievable by the terms a query actually uses.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis et al., 2020 | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) | RAG |
| Dense Passage Retrieval | Karpukhin et al., 2020 | [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) | Bi-encoder retrieval |
| ColBERT | Khattab & Zaharia, 2020 | [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) | Late interaction |
| ColPali | Faysse et al., 2024 | [arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449) | Retrieval over page images |
| Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE) | Gao et al., 2022 | [arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496) | Hypothetical document embeddings |
| From Local to Global: A Graph RAG Approach | Edge et al. (Microsoft), 2024 | [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130) | GraphRAG |
| Introducing Contextual Retrieval | Anthropic, 2024 | [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval) | Contextual embeddings and BM25 |
| Matryoshka Representation Learning | Kusupati et al., 2022 | [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147) | Truncatable embeddings |
| MTEB | Muennighoff et al., 2022 | [arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) | Embedding benchmark |
| RAGAS | Es et al., 2023 | [arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217) | RAG evaluation |

---

**Next:** [Chapter 37 — Agents & Tool Use](./37_agents_and_tool_use.md) — when the model stops answering and starts acting.

[← Back to Table of Contents](./README.md)
