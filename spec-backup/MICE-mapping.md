# MICE Mapping: RAG Functions to Agents and Infrastructure

## MICE Principle Verification

**Mutually Inclusive**: Each RAG function must be assigned to at least one agent/infrastructure component
**Collectively Exhaustive**: All RAG functions from the pipeline must be covered

---

## Complete RAG Function Mapping

### Stage 1: Query Construction

| RAG Function | Agent | Function Name | Infrastructure | Status |
|--------------|-------|---------------|----------------|--------|
| Text-to-SQL | Intent Parser Agent | `generate_sql_query` | SQLCoder + ClickHouse MCP | ✅ Covered |
| Text-to-Graph | Intent Parser Agent | `generate_graph_query` | (Future: GraphDB MCP) | ⚠️ Not in MVP |
| Self-query retriever (filters) | Intent Parser Agent | `construct_self_query` | PostgreSQL metadata | ✅ Covered |
| Schema-aware parsing | Intent Parser Agent | `get_schema_context` | ClickHouse MCP | ✅ Covered |
| Constraint lifting | Intent Parser Agent | `extract_entities` | LLM (GPT-4o-mini) | ✅ Covered |
| Intent classification | Intent Parser Agent | `classify_query_type` | LLM (GPT-4o-mini) | ✅ Covered |

**Coverage**: 5/6 functions covered (83%) - Graph queries deferred to Phase 2

---

### Stage 2: Query Translation

| RAG Function | Agent | Function Name | Infrastructure | Status |
|--------------|-------|---------------|----------------|--------|
| Multi-query | Intent Parser Agent | `generate_query_variants` | LLM (GPT-4o-mini) | ✅ Covered |
| RAG-Fusion | Knowledge Retriever Agent | `multi_strategy_retrieve` | Parallel execution | ✅ Covered |
| Decomposition | Intent Parser Agent | `decompose_complex_query` | LLM (GPT-4o-mini) | ✅ Covered |
| Step-back prompting | Intent Parser Agent | `step_back_reasoning` | LLM (GPT-4o-mini) | ✅ Covered |
| HyDE (hypothetical docs) | Intent Parser Agent | `generate_hypothetical_document` | LLM (GPT-4) | ✅ Covered |
| Term expansion/condensation | Intent Parser Agent | `augment_query_with_context` | LLM + Context | ✅ Covered |

**Coverage**: 6/6 functions covered (100%)

---

### Stage 3: Routing

| RAG Function | Agent | Function Name | Infrastructure | Status |
|--------------|-------|---------------|----------------|--------|
| Legal routing | Parlant Routing | `route_with_guidelines` | Parlant guidelines | ⚠️ Partial (generic routing) |
| Semantic routing (prompted) | Parlant Routing | `hybrid_route` | LLM (GPT-4) + Parlant | ✅ Covered |
| Index selection | Parlant Routing | `route_with_guidelines` | Parlant guidelines | ✅ Covered |
| Tool selection | Parlant Routing | `route_to_lobs` | Parlant guidelines | ✅ Covered |

**Coverage**: 3.5/4 functions covered (88%) - Legal routing needs domain-specific guidelines

---

### Stage 4: Retrieval

| RAG Function | Agent | Function Name | Infrastructure | Status |
|--------------|-------|---------------|----------------|--------|
| Routing-aware retrieval | Knowledge Retriever Agent | `multi_strategy_retrieve` | pgvector + PostgreSQL | ✅ Covered |
| Rerank (LLM-based) | Knowledge Retriever Agent | `rerank_documents` (llm_judge) | LLM (GPT-4) | ✅ Covered |
| RAG-Fusion rerank | Knowledge Retriever Agent | `rerank_documents` (cohere) | Cohere Rerank API | ✅ Covered |
| Active retrieval (CMKG) | Knowledge Retriever Agent | `contextual_memory_graph_retrieve` | PostgreSQL + Context | ✅ Covered |
| Hybrid retrieval | Knowledge Retriever Agent | `multi_strategy_retrieve` | pgvector + PostgreSQL | ✅ Covered |

**Coverage**: 5/5 functions covered (100%)

---

### Stage 5: Indexing

| RAG Function | Agent | Function Name | Infrastructure | Status |
|--------------|-------|---------------|----------------|--------|
| Chunk optimization | Document Ingestion Agent | `semantic_chunk` + `recursive_chunk` | LlamaIndex | ✅ Covered |
| Semantic splitter | Document Ingestion Agent | `semantic_chunk` | LlamaIndex SemanticSplitter | ✅ Covered |
| Parent document mapping | Document Ingestion Agent | `hierarchical_index` | LlamaIndex + PostgreSQL | ✅ Covered |
| Specialized embeddings | Document Ingestion Agent | `generate_embeddings` | Azure OpenAI / HuggingFace | ✅ Covered |
| Fine-tuning (e.g., C-AliBERT) | Document Ingestion Agent | (Training pipeline) | (Future: Fine-tuning service) | ⚠️ Not in MVP |
| RAPTOR | Document Ingestion Agent | `raptor_index` | LlamaIndex RAPTOR | ✅ Covered |
| Metadata enrichment | Document Ingestion Agent | `extract_metadata` | LLM + NER | ✅ Covered |
| Sparse signals | Document Ingestion Agent | (BM25 indexing) | PostgreSQL full-text search | ⚠️ Partial (basic FTS) |

**Coverage**: 6.5/8 functions covered (81%) - Fine-tuning and advanced sparse signals deferred

---

### Stage 6: Generation

| RAG Function | Agent | Function Name | Infrastructure | Status |
|--------------|-------|---------------|----------------|--------|
| Active retrieval during generation | Orchestrator Agent | ReAct loop with iterative retrieval | ReAct pattern | ✅ Covered |
| RAG-Fusion integration | Answer Generator Agent | `assemble_context` | Context merging | ✅ Covered |
| ReAct-style prompting | Orchestrator Agent | `think()`, `act()` methods | ReAct loop | ✅ Covered |
| Self-ask decomposition | Answer Generator Agent | (Uses Intent Parser) | Intent Parser Agent | ✅ Covered (via Intent Parser) |
| Chain-of-thought (structured) | Answer Generator Agent | `generate_answer` | LLM (GPT-4) with CoT prompts | ✅ Covered |
| Attribution and provenance | Answer Generator Agent | `generate_citations` | Context metadata | ✅ Covered |
| Contamination checks | Evaluator Agent | `check_consistency` | RAGAS consistency | ✅ Covered |

**Coverage**: 7/7 functions covered (100%) - All generation functions now included with ReAct pattern

---

## Overall MICE Analysis

### Coverage Summary

| Stage | Functions Covered | Total Functions | Coverage % | Status |
|-------|-------------------|-----------------|------------|--------|
| Query Construction | 5 | 6 | 83% | ✅ Good |
| Query Translation | 6 | 6 | 100% | ✅ Excellent |
| Routing | 3.5 | 4 | 88% | ✅ Good |
| Retrieval | 5 | 5 | 100% | ✅ Excellent |
| Indexing | 6.5 | 8 | 81% | ✅ Good |
| Generation | 7 | 7 | 100% | ✅ Excellent (with ReAct) |
| **TOTAL** | **33** | **36** | **92%** | ✅ Excellent MVP Coverage |

### Mutually Inclusive Check ✅

**All covered functions are assigned to exactly one primary agent:**
- Intent Parser Agent: Query Construction + Query Translation
- Parlant Routing: Routing
- Knowledge Retriever Agent: Retrieval
- Document Ingestion Agent: Indexing
- Answer Generator Agent: Generation
- Evaluator Agent: Quality checks

**No overlaps or conflicts detected.**

### Collectively Exhaustive Check ✅

**33 out of 36 RAG functions covered (92%)**

**Functions deferred to Phase 2 (3 functions):**
1. Text-to-Graph queries (Query Construction)
2. Fine-tuning specialized embeddings (Indexing)
3. Advanced sparse signals (Indexing)

**Rationale for deferral:**
- Graph queries require GraphDB infrastructure not in MVP
- Fine-tuning requires training pipeline and datasets
- Advanced sparse signals need BM25+ integration

**Added to MVP:**
- ✅ ReAct-style prompting (Generation) - Now core orchestration pattern
- ✅ Active retrieval during generation (Generation) - Enabled by ReAct loop

---

## Infrastructure Mapping

### LLM Model Usage

| Model | Usage | Agents |
|-------|-------|--------|
| GPT-4o-mini | Intent parsing, entity extraction, simple queries | Intent Parser Agent |
| GPT-4 | Complex reasoning, answer generation, evaluation | Answer Generator, Evaluator, Knowledge Retriever |
| o1-preview | Advanced multi-step reasoning, SQL generation | Intent Parser (complex queries) |
| text-embedding-ada-002 | Embedding generation | Document Ingestion Agent |

### Database Infrastructure

| Database | Usage | Agents |
|----------|-------|--------|
| PostgreSQL + pgvector | Vector storage, metadata, structured data | All agents (primary storage) |
| ClickHouse (optional) | Analytics, FinOps data, SQL queries | Intent Parser (via MCP) |
| Redis (optional) | Caching (schema, embeddings, responses) | All agents (performance) |

### External Services

| Service | Usage | Agents |
|---------|-------|--------|
| LlamaIndex | Document chunking, indexing, RAPTOR | Document Ingestion Agent |
| Langfuse | Tracing, observability, cost tracking | All agents |
| Parlant | Routing guidelines, policy management | Parlant Routing |
| Agent Lightning | Prompt optimization, RL feedback | Answer Generator Agent |
| Cohere Rerank | Document reranking | Knowledge Retriever Agent |
| RAGAS | Quality evaluation metrics | Evaluator Agent |

---

## Gaps and Recommendations

### Critical Gaps (Must Address)

None identified. All critical RAG functions are covered.

### Nice-to-Have Gaps (Phase 2)

1. **Graph Query Support**: Add GraphDB MCP for knowledge graph queries
2. **Embedding Fine-tuning**: Add training pipeline for domain-specific embeddings
3. **Advanced Sparse Retrieval**: Integrate BM25+ for hybrid search
4. **ReAct Pattern**: Implement iterative reasoning with tool use
5. **Active Retrieval**: Add mid-generation retrieval for uncertainty handling

### Optimization Opportunities

1. **Legal Routing**: Add domain-specific guidelines for compliance/precedent handling
2. **Sparse Signals**: Enhance PostgreSQL full-text search with BM25 scoring
3. **Contamination Checks**: Add more sophisticated conflict detection

---

## Conclusion

**MICE Principle Status: ✅ SATISFIED**

- **Mutually Inclusive**: ✅ All covered functions have clear agent assignments with no conflicts
- **Collectively Exhaustive**: ✅ 86% coverage (31/36 functions) with clear rationale for deferred items

**The design provides strong MVP coverage of RAG functions with a clear path to 100% coverage in Phase 2.**

**Recommendation**: Proceed with implementation. The 14% gap consists entirely of advanced features that are not critical for MVP functionality.
