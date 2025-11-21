### RAG functions across the pipeline

You’re right—query construction belongs in the RAG flow. Here’s the complete set of functions extracted and grouped by stage, including the construction pieces.

---

#### Query construction

- **Text-to-query (SQL/Graph):** Converts natural language to structured queries for Retrieval-DBs or GraphDBs.
- **Self-query retriever (filters):** Maps language to metadata filters (e.g., facets, timestamps, authors).
- **Schema-aware parsing:** Aligns entities and relations to known schemas to avoid ambiguous field usage.
- **Constraint lifting:** Extracts hard constraints (dates, IDs, jurisdictions) from text for precise retrieval.
- **Intent classification:** Determines if the query is lookup, exploratory, or synthesis to inform downstream routing.

---

#### Query translation

- **Multi-query:** Generates diverse reformulations to expand recall.
- **RAG-Fusion:** Combines multiple query result sets using fusion (e.g., RRF) for robust ranking.
- **Decomposition:** Splits complex questions into atomic sub-queries.
- **Step-back prompting:** Reformulates to a higher-level abstraction to retrieve grounding context.
- **HyDE (hypothetical docs):** Produces synthetic passages, embeds them, and retrieves by proximity.
- **Term expansion/condensation:** Adds/removes synonyms, acronyms, or domain jargon to normalize retrieval.

---

#### Routing

- **Legal routing:** Directs queries requiring compliance/precedent handling to specialized retrievers.
- **Semantic routing (prompted):** Chooses pipelines (Prompt #1 vs Prompt #2) based on semantics or domain.
- **Index selection:** Picks the best index (dense, sparse, hybrid, graph) for the current query form.
- **Tool selection:** Chooses actions (e.g., DB lookup vs web retrieval vs KG traversal) before retrieval.

---

#### Retrieval

- **Routing-aware retrieval:** Executes retrieval against chosen index/tool path.
- **Rerank (LLM-based):** Uses RankGPT or similar to reorder candidates with instruction-aware scoring.
- **RAG-Fusion rerank:** Applies fusion scoring across multiple result lists to stabilize top-k.
- **Active retrieval (CMKG):** Iteratively expands context via KG hops or feedback-driven query updates.
- **Hybrid retrieval:** Merges dense embeddings, BM25, and lexical constraints for balanced precision/recall.

---

#### Indexing

- **Chunk optimization:** Tunes chunk size/overlap for minimal fragmentation and maximal coherence.
- **Semantic splitter:** Splits by discourse/syntax boundaries rather than fixed tokens.
- **Parent document mapping:** Retrieves parent blocks to preserve context beyond child chunks.
- **Specialized embeddings:** Domain-tuned encoders for legal, code, biomedical, or multilingual corpora.
- **Fine-tuning (e.g., C-AliBERT):** Adapts encoders to task/domain for better similarity alignment.
- **RAPTOR:** Builds hierarchical topic trees for multi-scale retrieval.
- **Metadata enrichment:** Adds entities, citations, and provenance fields to support precise filtering.
- **Sparse signals:** Indexes keywords, headings, and anchors to complement dense vectors.

---

#### Generation

- **Active retrieval during generation:** Pulls additional evidence mid-chain when uncertainty is detected.
- **RAG-Fusion integration:** Merges evidence across queries before synthesis to reduce bias.
- **ReAct-style prompting:** Interleaves thinking and retrieval actions for traceable reasoning.
- **Self-ask decomposition:** Asks directed sub-questions, retrieves, then composes the final answer.
- **Chain-of-thought (structured):** Uses stepwise reasoning with citation and constraint checks.
- **Attribution and provenance:** Aligns claims to sources; surfaces confidence and coverage gaps.
- **Contamination checks:** Flags low-overlap or conflicting sources before finalization.

---

### Quick pipeline slots

- **Construct:** Text-to-query, self-query filters, schema-aware parsing, constraints.
- **Translate:** Multi-query, HyDE, decomposition, step-back, expansion/condensation.
- **Route:** Legal/semantic routing, index/tool selection.
- **Retrieve:** Hybrid search, rerank (LLM/fusion), active KG retrieval.
- **Index:** Chunking, semantic splits, parent-doc, specialized embeddings, RAPTOR, metadata, sparse.


| RAG Function                     | Corresponding Agent          | LLM Model Needed         |
|----------------------------------|------------------------------|--------------------------|
| Query construction                | Intent Parser Agent          | GPT-4                    |
| Query translation                 | Knowledge Retriever Agent     | GPT-4                    |
| Routing                           | Orchestrator Agent           | GPT-4                    |
| Retrieval                         | Knowledge Retriever Agent     | GPT-4                    |
| Indexing                          | Knowledge Retriever Agent     | GPT-4                    |
| Generation                        | Answer Generator Agent        | GPT-4                    |

---
