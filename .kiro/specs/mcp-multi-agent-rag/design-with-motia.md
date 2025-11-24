# Design Document

## Overview

The MCP-Based Multi-Agent RAG System is an enterprise-grade platform that coordinates multiple specialized agents to handle complex query orchestration, knowledge retrieval, answer generation, quality evaluation, and human-in-the-loop workflows. The system emphasizes three core values: deterministic routing, enforced compliance, and full observability.

The architecture follows a modular design where each agent is an independent component with well-defined responsibilities. Agents communicate via the Model Context Protocol (MCP), enabling horizontal scaling, independent deployment, and clear separation of concerns. The system integrates multiple advanced frameworks including Parlant for routing, Agent Lightning for prompt optimization, LlamaIndex for document indexing, and Langfuse for observability.

## Architecture

### System Overview

The MCP-Based Multi-Agent RAG System is built on a layered architecture that separates concerns while enabling seamless coordination between components. The system leverages Motia Framework for unified backend orchestration (APIs, workflows, and AI agents), CopilotKit for frontend integration, and a comprehensive observability stack for monitoring and continuous improvement.

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  Next.js 16  │  │  CopilotKit  │  │   AG-UI      │               │
│  │   React 19   │──│   Runtime    │──│   Client     │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Backend Layer                               │
│  ┌──────────────────────────────────────────────────────-────────┐  │
│  │                    FastAPI Application                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐               │  │
│  │  │  AG-UI     │  │   CORS     │  │  Endpoint  │               │  │
│  │  │  Endpoint  │  │ Middleware │  │  Routing   │               │  │
│  │  └────────────┘  └────────────┘  └────────────┘               │  │
│  └───────────────────────────────────────────────────────-───────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Agent Orchestration Layer                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Motia Framework                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐               │  │
│  │  │Orchestrator│  │   Intent   │  │ Knowledge  │               │  │
│  │  │   Step     │──│   Parser   │──│ Retriever  │               │  │
│  │  └────────────┘  └────────────┘  └────────────┘               │  │
│  │         │              │                  │                   │  │
│  │         ▼              ▼                  ▼                   │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐               │  │
│  │  │  Answer    │  │ Evaluator  │  │   Human    │               │  │
│  │  │ Generator  │──│   Agent    │──│   Review   │               │  │
│  │  │   Step     │  │   Step     │  │   Step     │               │  │
│  │  └────────────┘  └────────────┘  └────────────┘               │  │
│  └──────────────────────────────────────────────────────-────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Supporting Services Layer                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │  Parlant   │  │   Agent    │  │ LlamaIndex │  │  Langfuse  │     │
│  │  Routing   │  │ Lightning  │  │  Indexing  │  │  Tracing   │     │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Model Layer                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │   Azure    │  │  OpenAI    │  │ Embeddings │  │  SQLCoder  │     │
│  │  OpenAI    │  │   GPT-4    │  │   Models   │  │  (optional)│     │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Storage Layer                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │PostgreSQL  │  │  Weaviate  │  │ ClickHouse │  │   Redis    │     │
│  │  Database  │  │  Vectors   │  │ (optional) │  │   Cache    │     │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent Coordination Pattern

The system uses a **hybrid orchestration approach** combining two complementary patterns, aligned with the **Double Diamond Design Process** and **Six Thinking Hats** framework (see `design-thinking-agent-patterns.md` for detailed integration):

**1. Plan-and-Execute (Orchestrator Level)**
**2. ReAct (Individual Agent Level)**
**3. Chain-of-Thought (Reasoning Agents)**

**Design Thinking Alignment:**
- **Discover Phase**: ReAct for exploration (Green/White/Red Hats)
- **Define Phase**: CoT for validation (Black/Blue Hats)
- **Develop Phase**: ReAct for strategy generation (Green/Yellow/White Hats)
- **Deliver Phase**: CoT for evaluation (Black/Blue Hats)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Plan-and-Execute)               │
│                                                                  │
│  Step 1: PLAN                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Analyze query → Create execution plan                   │    │
│  │ • Parse intent and entities                             │    │
│  │ • Determine agent sequence                              │    │
│  │ • Identify dependencies                                 │    │
│  │ • Allocate resources                                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                          │                                       │
│                          ▼                                       │
│  Step 2: EXECUTE (Parallel where possible)                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                                                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │    │
│  │  │ Intent Parser│  │   Parlant    │  │  Knowledge   │ │    │
│  │  │    Agent     │→ │   Routing    │→ │  Retriever   │ │    │
│  │  │   (ReAct)    │  │ (Rule-based) │  │   (ReAct)    │ │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │    │
│  │                                              │          │    │
│  │                                              ▼          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │    │
│  │  │    Human     │  │  Evaluator   │  │   Answer     │ │    │
│  │  │   Review     │← │    Agent     │← │  Generator   │ │    │
│  │  │   (Manual)   │  │   (ReAct)    │  │   (ReAct)    │ │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │    │
│  │                                                         │    │
│  └────────────────────────────────────────────────────────┘    │
│                          │                                       │
│                          ▼                                       │
│  Step 3: AGGREGATE & RETURN                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Collect results → Validate → Return answer              │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Pattern Selection Rationale:**

| Component | Pattern | Why |
|-----------|---------|-----|
| **Orchestrator** | Plan-and-Execute | • High-level coordination<br>• Clear agent dependencies<br>• Parallel execution where possible<br>• Predictable workflow |
| **Intent Parser** | ReAct | • Iterative entity extraction<br>• Multi-step reasoning<br>• Needs tool use (context lookup)<br>• Confidence refinement through observation |
| **Parlant Routing** | Rule-based | • Deterministic decisions<br>• Guideline matching<br>• No iteration needed |
| **Knowledge Retriever** | ReAct | • Try multiple retrieval strategies<br>• Adapt based on results<br>• Tool use (semantic search, metadata filter)<br>• Iterative reranking |
| **Answer Generator** | Chain-of-Thought (CoT) | • Text generation task<br>• Structured reasoning for grounding<br>• Single-pass with explicit steps<br>• No tool use needed |
| **Evaluator** | Chain-of-Thought (CoT) | • Analytical reasoning task<br>• Multi-metric scoring<br>• Single-pass evaluation<br>• No tool use needed |

**Example: Plan-and-Execute Flow**

```
ORCHESTRATOR PLAN:
1. Parse intent (Intent Parser Agent - ReAct)
2. Route query (Parlant - Rule-based)
3. Retrieve docs (Knowledge Retriever - ReAct)
4. Generate answer (Answer Generator - CoT)
5. Evaluate quality (Evaluator - CoT)
6. [If needed] Human review (Manual)

ORCHESTRATOR EXECUTE:
→ Calls each agent in sequence
→ Each agent uses appropriate pattern for its task:
  • ReAct for tool-using agents (Intent Parser, Retriever)
  • CoT for reasoning agents (Generator, Evaluator)
  • Rule-based for deterministic agents (Parlant)
→ Orchestrator aggregates results
```

**Pattern Selection Guide:**

| Task Type | Pattern | When to Use | Examples |
|-----------|---------|-------------|----------|
| **Coordination** | Plan-and-Execute | Managing multiple agents with dependencies | Orchestrator |
| **Tool Use** | ReAct | Need to iteratively use tools/APIs based on observations | Intent Parser, Knowledge Retriever |
| **Text Generation** | Chain-of-Thought | Generating text with structured reasoning | Answer Generator |
| **Analysis** | Chain-of-Thought | Analytical reasoning without tool use | Evaluator |
| **Decision** | Rule-based | Deterministic decisions with clear rules | Parlant Routing |

**Example: ReAct Within Knowledge Retriever**

```
KNOWLEDGE RETRIEVER (ReAct Loop):

Iteration 1:
  THOUGHT: "Start with semantic search"
  ACTION: Execute semantic search tool
  OBSERVATION: Found 3 docs, confidence 0.65 (low)

Iteration 2:
  THOUGHT: "Low confidence, try metadata filtering"
  ACTION: Execute metadata filter tool on LOB="orders"
  OBSERVATION: Found 8 docs, confidence 0.82 (better)

Iteration 3:
  THOUGHT: "Good results, apply reranking"
  ACTION: Execute rerank tool with Cohere
  OBSERVATION: Top 5 docs, confidence 0.91 (excellent)

Iteration 4:
  THOUGHT: "High confidence, return results"
  ACTION: FINISH
```

**Example: CoT Within Answer Generator**

```
ANSWER GENERATOR (Chain-of-Thought):

Prompt with CoT structure:
"Given the query and retrieved documents, generate an answer following these steps:

Step 1: Identify key information from documents
- Document 1 mentions customer ABC123 has 5 orders
- Document 2 shows total revenue of $50,000
- Document 3 lists top products

Step 2: Determine what information answers the query
- Query asks for customer orders
- Documents 1 and 2 are most relevant
- Document 3 provides supporting context

Step 3: Synthesize answer with proper grounding
- Customer ABC123 has 5 orders [Source: Doc 1]
- Total revenue is $50,000 [Source: Doc 2]
- Top products include... [Source: Doc 3]

Step 4: Add citations and confidence
- All claims backed by retrieved documents
- High confidence (0.92) due to direct evidence

Final Answer: [Generated with citations]"
```

**Example: CoT Within Evaluator**

```
EVALUATOR (Chain-of-Thought):

Prompt with CoT structure:
"Evaluate the answer quality following these steps:

Step 1: Check Faithfulness
- Claim: "Customer has 5 orders"
- Evidence in docs: Yes, Document 1 states this
- Faithfulness score: 1.0

Step 2: Check Relevance
- Query asks about customer orders
- Answer directly addresses this
- Relevance score: 0.95

Step 3: Check Correctness
- All facts match retrieved documents
- No contradictions found
- Correctness score: 0.98

Step 4: Check Coverage
- Answer covers main query aspects
- Missing: order dates (minor)
- Coverage score: 0.85

Step 5: Overall Assessment
- Average score: 0.92
- Threshold: 0.80
- Decision: Quality is good, no review needed

Final Evaluation: [Scores and decision]"
```

**Why This Hybrid Approach?**

- **Plan-and-Execute (Orchestrator)**: Best for high-level coordination with clear dependencies
- **ReAct (Tool-Using Agents)**: Best for agents that need to use tools iteratively (Intent Parser, Knowledge Retriever)
- **Chain-of-Thought (Reasoning Agents)**: Best for agents doing analytical/generative reasoning (Answer Generator, Evaluator)
- **Rule-Based (Deterministic Agents)**: Best for agents with clear decision rules (Parlant Routing)
- **Clear Separation**: Orchestrator plans the workflow, agents execute using appropriate patterns
- **Pattern Fit**: Each agent uses the pattern that matches its task type
- **Observability**: All levels fully logged to Langfuse

### Communication Protocols

**Frontend ↔ Backend (AG-UI Protocol):**
- Protocol: HTTP/HTTPS (REST API, not WebSockets)
- Library: `agent-framework-ag-ui` (Python) + `@ag-ui/client` (TypeScript)
- Format: JSON
- Streaming: Server-Sent Events (SSE) for agent responses
- Integration Flow:
  1. Frontend: CopilotKit → `/api/copilotkit` (Next.js API route)
  2. Next.js: CopilotRuntime wraps `HttpAgent` from `@ag-ui/client`
  3. Backend: `add_agent_framework_fastapi_endpoint()` exposes agent at `/` (root path)
  4. Agent Framework handles SSE streaming automatically
- Base URL: `http://localhost:8880/` (configurable via HttpAgent)
- Authentication: Not implemented in starter (can add JWT tokens)
- State Management: Shared state via `useCoAgent` hook
- Actions: Frontend actions via `useCopilotAction`, backend tools via `@ai_function`

**Agent ↔ Agent:**
- Framework: Microsoft Agent Framework (AutoGen Core)
- Runtime: SingleThreadedAgentRuntime (default) or GrpcWorkerAgentRuntime (distributed)
- Protocol: Message-based communication via RoutedAgent with message_handler decorators
- Format: Pydantic models or dataclasses (serializable)
- Default: SingleThreadedAgentRuntime for high-performance single-process communication
- Optional: GrpcWorkerAgentRuntime for distributed multi-process communication via gRPC
- Coordination: Task dependencies via Plan-and-Execute pattern
- Error Handling: Retry with exponential backoff
- Configuration: Runtime selection via environment variables (no code changes needed)

**Backend ↔ Database:**
- Protocol: PostgreSQL wire protocol
- Driver: asyncpg (async)
- Connection Pooling: Yes
- Transaction Support: Yes

**Backend ↔ LLM:**
- Protocol: HTTPS
- Format: JSON (OpenAI API format)
- Streaming: Yes (token-by-token)
- Rate Limiting: Token bucket algorithm

### Data Flow Pipeline

**Complete Query Processing Flow:**

```
1. User Query
   │
   ├─> Frontend: Capture query + context
   │
   ▼
2. Backend: Validate and route
   │
   ├─> Langfuse: Start trace
   │
   ▼
3. Orchestrator: Decompose task
   │
   ├─> Create agent tasks
   ├─> Determine dependencies
   │
   ▼
4. Intent Parser: Multi-step planning
   │
   ├─> Step-back reasoning
   ├─> Entity extraction
   ├─> Confidence scoring
   ├─> Fallback detection
   │
   ▼
5. Parlant Routing: Select strategy
   │
   ├─> Match guidelines
   ├─> Log decision
   │
   ▼
6. Knowledge Retriever: Multi-strategy retrieval
   │
   ├─> Semantic search (pgvector)
   ├─> Metadata filtering (PostgreSQL)
   ├─> Heuristic matching
   ├─> Merge and deduplicate
   ├─> Optional reranking
   │
   ▼
7. Answer Generator: Synthesize response
   │
   ├─> Select model (complexity-based)
   ├─> Assemble context with provenance
   ├─> Generate answer
   │
   ▼
8. Evaluator: Quality assessment
   │
   ├─> Compute 7 RAG characteristics
   ├─> Calculate overall score
   ├─> Determine if review needed
   │
   ├─── Quality OK ─────────────────┐
   │                                │
   └─── Quality Low ───┐            │
                       ▼            │
9. Human Review: Validate and correct
   │                                │
   ├─> Present to reviewer          │
   ├─> Capture feedback             │
   ├─> Store in LightningStore      │
   │                                │
   └────────────────────────────────┤
                                    ▼
10. Return Answer
    │
    ├─> Stream to frontend
    ├─> Update UI state
    ├─> Log to Langfuse
    │
    ▼
11. Continuous Improvement
    │
    ├─> Agent Lightning: Analyze feedback
    ├─> Update prompts and guidelines
    └─> Deploy improvements
```

## RAG Pipeline Design

This section details how the 6 core RAG stages map to specific agents and functions, ensuring complete coverage of all RAG capabilities.

### RAG Stage 1: Query Construction

**Objective**: Convert user input into structured queries (SQL, semantic, graph-based)

**Agent**: Intent Parser Agent

**Functions**:

1. **Natural Language Understanding**
   ```python
   @ai_function(name="parse_natural_language")
   async def parse_natural_language(query: str) -> IntentResult:
       """
       Extract intent, entities, and query type from natural language.
       Supports: SQL generation, data stories, general QA, visualizations
       """
   ```

2. **Text-to-SQL Generation** (via SQLCoder integration)
   ```python
   @ai_function(name="generate_sql_query")
   async def generate_sql_query(
       query: str,
       schema_context: Dict[str, Any]
   ) -> SQLQuery:
       """
       Generate SQL from natural language using SQLCoder.
       Schema awareness via ClickHouse MCP prevents hallucinations.
       """
   ```

3. **Self-Query Construction**
   ```python
   @ai_function(name="construct_self_query")
   async def construct_self_query(
       query: str,
       metadata_schema: Dict[str, Any]
   ) -> StructuredQuery:
       """
       Build structured query with filters and constraints.
       Extracts: semantic query + metadata filters + temporal constraints
       """
   ```

4. **Schema Awareness**
   ```python
   @ai_function(name="get_schema_context")
   async def get_schema_context(database: str) -> SchemaContext:
       """
       Retrieve schema from ClickHouse MCP with caching.
       Returns: tables, columns, types, relationships, constraints
       """
   ```

**Evaluation**: Spider, UNITE, BIRD datasets; Execution Accuracy; UNITE toolkit

**Langfuse Integration**:
```python
with langfuse.trace("query_construction") as trace:
    trace.span("nl_parsing", input=query, output=intent_result)
    trace.span("sql_generation", input=intent, output=sql_query)
    trace.score("sql_accuracy", accuracy_score)
```

---

### RAG Stage 2: Query Translation

**Objective**: Rewrite, decompose, fuse, or augment queries for better retrieval

**Agent**: Intent Parser Agent (Multi-Step Planning)

**Functions**:

1. **RAG-Fusion** (Multi-Query Generation)
   ```python
   @ai_function(name="generate_query_variants")
   async def generate_query_variants(query: str) -> List[str]:
       """
       Generate multiple query variants for parallel retrieval.
       Techniques: Paraphrasing, perspective shifting, specificity variation
       """
   ```

2. **HyDE** (Hypothetical Document Embeddings)
   ```python
   @ai_function(name="generate_hypothetical_document")
   async def generate_hypothetical_document(query: str) -> str:
       """
       Generate hypothetical answer document for embedding-based retrieval.
       Improves semantic matching by bridging query-document gap.
       """
   ```

3. **Query Decomposition**
   ```python
   @ai_function(name="decompose_complex_query")
   async def decompose_complex_query(query: str) -> List[SubIntent]:
       """
       Break complex queries into sequential sub-queries.
       Returns: ordered sub-intents with dependencies
       """
   ```

4. **Query Augmentation**
   ```python
   @ai_function(name="augment_query_with_context")
   async def augment_query_with_context(
       query: str,
       context: ConversationContext
   ) -> str:
       """
       Enrich query with conversation history and entity resolution.
       Resolves: anaphora, temporal references, implicit entities
       """
   ```

**Evaluation**: BEIR benchmark; Recall@k, MRR, nDCG; DeepEval reranking

**Langfuse Integration**:
```python
with langfuse.trace("query_translation") as trace:
    trace.span("rag_fusion", output=query_variants)
    trace.span("hyde", output=hypothetical_doc)
    trace.span("decomposition", output=sub_intents)
```

---

### RAG Stage 3: Routing

**Objective**: Decide which retriever/tool/agent to use based on query type/context

**Agent**: Parlant Routing Layer

**Functions**:

1. **Guideline-Based Routing**
   ```python
   @ai_function(name="route_with_guidelines")
   def route_with_guidelines(
       query: str,
       intent: IntentResult
   ) -> RoutingDecision:
       """
       Apply Parlant guidelines in priority order.
       Returns: retriever selection + reasoning + confidence
       """
   ```

2. **Hybrid Routing** (Static + Dynamic)
   ```python
   @ai_function(name="hybrid_route")
   async def hybrid_route(
       query: str,
       intent: IntentResult
   ) -> RoutingDecision:
       """
       Combine static rules (SQL, GraphQL) with LLM-assisted routing.
       Static: Predictable patterns (high confidence)
       Dynamic: Ambiguous queries (LLM reasoning)
       """
   ```

3. **Multi-LOB Routing**
   ```python
   @ai_function(name="route_to_lobs")
   def route_to_lobs(intent: IntentResult) -> List[LOB]:
       """
       Determine which Lines of Business to query.
       Supports: Inventory, Orders, Support, Finance
       """
   ```

4. **Fallback Routing**
   ```python
   @ai_function(name="apply_fallback_strategy")
   def apply_fallback_strategy(
       query: str,
       failed_routes: List[str]
   ) -> RoutingDecision:
       """
       Handle routing failures with fallback strategies.
       Strategies: Alternative retrievers, human escalation, clarification
       """
   ```

**Evaluation**: Custom routing accuracy tests; Rule compliance %, Misrouting rate; Parlant logs + Langfuse traces

**Langfuse Integration**:
```python
with langfuse.trace("routing") as trace:
    trace.span("guideline_matching", output=matched_guideline)
    trace.span("lob_selection", output=selected_lobs)
    trace.score("routing_confidence", confidence)
```

---

### RAG Stage 4: Retrieval

**Objective**: Fetch relevant documents from DBs or APIs

**Agent**: Knowledge Retriever Agent

**Functions**:

1. **Semantic Search** (Embedding-Based)
   ```python
   @ai_function(name="semantic_search")
   async def semantic_search(
       query: str,
       top_k: int = 10
   ) -> List[Document]:
       """
       Vector similarity search using pgvector.
       Embedding: text-embedding-ada-002
       Distance: Cosine similarity
       """
   ```

2. **Metadata Filtering** (Structured Constraints)
   ```python
   @ai_function(name="metadata_filter")
   async def metadata_filter(
       constraints: Dict[str, Any],
       lobs: List[LOB]
   ) -> List[Document]:
       """
       Structured filtering via PostgreSQL queries.
       Filters: LOB, timestamp, source, confidence, custom metadata
       """
   ```

3. **Guided Grep** (Heuristic Pattern Matching)
   ```python
   @ai_function(name="guided_grep")
   async def guided_grep(
       pattern: str,
       context: str
   ) -> List[Document]:
       """
       Pattern-based retrieval for exact matches.
       Techniques: Regex, fuzzy matching, keyword extraction
       """
   ```

4. **Multi-Strategy Retrieval**
   ```python
   @ai_function(name="multi_strategy_retrieve")
   async def multi_strategy_retrieve(
       query: str,
       strategies: List[RetrievalStrategy]
   ) -> RetrievalResult:
       """
       Execute multiple retrieval strategies in parallel.
       Strategies: Semantic, Metadata, Heuristic
       Returns: Merged and deduplicated results
       """
   ```

5. **Reranking**
   ```python
   @ai_function(name="rerank_documents")
   async def rerank_documents(
       query: str,
       documents: List[Document],
       method: Literal["cohere", "llm_judge"]
   ) -> List[Document]:
       """
       Rerank candidates for improved relevance.
       Methods: Cohere Rerank API, LLM-as-a-Judge
       """
   ```

6. **Active Retrieval** (CMG - Contextual Memory Graph)
   ```python
   @ai_function(name="contextual_memory_graph_retrieve")
   async def contextual_memory_graph_retrieve(
       query: str,
       conversation_id: UUID
   ) -> List[Document]:
       """
       Retrieve from conversation-specific memory graph.
       Maintains: Entity relationships, temporal context, user preferences
       """
   ```

**Evaluation**: BEIR, MS MARCO, TREC datasets; Precision@k, Recall@k, nDCG; DeepEval + Langfuse retrieval traces

**Langfuse Integration**:
```python
with langfuse.trace("retrieval") as trace:
    trace.span("semantic_search", output=semantic_docs)
    trace.span("metadata_filter", output=filtered_docs)
    trace.span("reranking", output=reranked_docs)
    trace.score("retrieval_precision", precision_at_k)
    trace.score("retrieval_recall", recall_at_k)
```

---

### RAG Stage 5: Indexing

**Objective**: Chunking, embedding, organizing documents for optimal retrieval

**Agent**: Document Ingestion Agent (uses LlamaIndex)

**Functions**:

1. **Semantic Chunking**
   ```python
   @ai_function(name="semantic_chunk")
   async def semantic_chunk(
       document: str,
       chunk_size: int = 512
   ) -> List[str]:
       """
       LlamaIndex SemanticSplitter for NLP-driven chunking.
       Preserves: Sentence boundaries, semantic coherence
       """
   ```

2. **Recursive Chunking**
   ```python
   @ai_function(name="recursive_chunk")
   async def recursive_chunk(
       document: str,
       separators: List[str] = ["\n\n", "\n", ". ", " "]
   ) -> List[str]:
       """
       LlamaIndex RecursiveCharacterTextSplitter.
       Hierarchical splitting by semantic units.
       """
   ```

3. **Hierarchical Indexing**
   ```python
   @ai_function(name="hierarchical_index")
   async def hierarchical_index(
       document: str
   ) -> HierarchicalIndex:
       """
       LlamaIndex parent/child document structures.
       Enables: Multi-level retrieval, context preservation
       """
   ```

4. **RAPTOR Indexing** (Tree-Based)
   ```python
   @ai_function(name="raptor_index")
   async def raptor_index(
       documents: List[str]
   ) -> RAPTORTree:
       """
       LlamaIndex RAPTOR for tree-based multi-level retrieval.
       Builds: Hierarchical summaries, clustered embeddings
       """
   ```

5. **Embedding Generation**
   ```python
   @ai_function(name="generate_embeddings")
   async def generate_embeddings(
       chunks: List[str],
       model: str = "text-embedding-ada-002"
   ) -> List[List[float]]:
       """
       Generate embeddings for chunks.
       Models: Azure OpenAI, HuggingFace, Cohere
       """
   ```

6. **Metadata Extraction**
   ```python
   @ai_function(name="extract_metadata")
   async def extract_metadata(
       document: str,
       chunk: str
   ) -> Dict[str, Any]:
       """
       Extract metadata for filtering and provenance.
       Extracts: Source, timestamp, LOB, entities, keywords
       """
   ```

7. **Storage and Indexing**
   ```python
   @ai_function(name="store_with_index")
   async def store_with_index(
       chunks: List[str],
       embeddings: List[List[float]],
       metadata: List[Dict]
   ) -> None:
       """
       Store in pgvector with IVFFlat index.
       Index: Cosine similarity, 100 lists
       """
   ```

**Evaluation**: BEIR benchmark; Embedding similarity scores, Recall@k, nDCG, latency; LlamaIndex eval suite + Langfuse traces

**Langfuse Integration**:
```python
with langfuse.trace("indexing") as trace:
    trace.span("chunking", output=chunks)
    trace.span("embedding", output=embeddings)
    trace.span("storage", output=storage_result)
    trace.score("indexing_latency_ms", latency)
```

---

### RAG Stage 6: Generation

**Objective**: Generate grounded responses from retrieved docs

**Agent**: Answer Generator Agent

**Functions**:

1. **Context Assembly**
   ```python
   @ai_function(name="assemble_context")
   async def assemble_context(
       documents: List[Document],
       max_tokens: int = 4000
   ) -> Context:
       """
       Assemble context with provenance metadata.
       Includes: Source attribution, timestamps, confidence scores
       """
   ```

2. **Prompt Optimization** (Agent Lightning)
   ```python
   @ai_function(name="get_optimized_prompt")
   async def get_optimized_prompt(
       task_type: str,
       context: Context
   ) -> PromptTemplate:
       """
       Retrieve optimized prompt from Agent Lightning.
       Optimization: RL-based, supervised fine-tuning, A/B tested
       """
   ```

3. **Model Selection**
   ```python
   @ai_function(name="select_generation_model")
   def select_generation_model(
       complexity: TaskComplexity
   ) -> str:
       """
       Choose LLM based on task complexity.
       Simple: GPT-4o-mini, Moderate: GPT-4, Complex: o1-preview
       """
   ```

4. **Answer Generation**
   ```python
   @ai_function(name="generate_answer")
   async def generate_answer(
       query: str,
       context: Context,
       prompt_template: PromptTemplate,
       model: str
   ) -> Answer:
       """
       Generate grounded answer with citations.
       Streaming: Token-by-token for real-time display
       """
   ```

5. **Guardrails Enforcement**
   ```python
   @ai_function(name="apply_guardrails")
   async def apply_guardrails(
       answer: Answer
   ) -> Answer:
       """
       Apply Parlant guardrails and Azure Content Safety.
       Checks: PII, toxicity, factual grounding, policy compliance
       """
   ```

6. **Citation Generation**
   ```python
   @ai_function(name="generate_citations")
   def generate_citations(
       answer: Answer,
       context: Context
   ) -> List[Citation]:
       """
       Generate inline citations for answer.
       Format: [Source: document_id, confidence: 0.92]
       """
   ```

**Evaluation**: TruthfulQA, MT-Bench, Eval Harness; Metrics: Faithfulness, factual accuracy, hallucination rate; Agent Lightning RL feedback loops

**Langfuse Integration**:
```python
with langfuse.trace("generation") as trace:
    trace.span("context_assembly", output=context)
    trace.span("prompt_optimization", output=prompt)
    trace.span("llm_generation", output=answer)
    trace.span("guardrails", output=guardrails_result)
    trace.score("faithfulness", faithfulness_score)
    trace.score("hallucination_rate", hallucination_rate)
    trace.metadata({"tokens_used": tokens, "cost_usd": cost})
```

---

### RAG Pipeline Integration

**Complete Pipeline Execution:**

```python
async def execute_rag_pipeline(query: str, user_id: str) -> Answer:
    """Execute complete RAG pipeline with all 6 stages"""
    
    # Stage 1: Query Construction
    intent = await parse_natural_language(query)
    if intent.query_type == QueryType.SQL_GENERATION:
        sql_query = await generate_sql_query(query, schema_context)
    
    # Stage 2: Query Translation
    query_variants = await generate_query_variants(query)
    sub_intents = await decompose_complex_query(query)
    
    # Stage 3: Routing
    routing_decision = await route_with_guidelines(query, intent)
    lobs = await route_to_lobs(intent)
    
    # Stage 4: Retrieval
    retrieval_result = await multi_strategy_retrieve(
        query,
        strategies=[
            RetrievalStrategy.SEMANTIC,
            RetrievalStrategy.METADATA,
            RetrievalStrategy.HEURISTIC
        ]
    )
    documents = await rerank_documents(query, retrieval_result.documents)
    
    # Stage 5: Indexing (background process for new documents)
    # Runs asynchronously, not blocking query pipeline
    
    # Stage 6: Generation
    context = await assemble_context(documents)
    prompt = await get_optimized_prompt("answer_generation", context)
    model = select_generation_model(intent.abstract_intent.complexity)
    answer = await generate_answer(query, context, prompt, model)
    answer = await apply_guardrails(answer)
    
    return answer
```

### RAG Quality Characteristics Coverage

**Mapping to Agents:**

1. **Coverage**: Knowledge Retriever (multi-strategy retrieval) + Evaluator (coverage scoring)
2. **Consistency**: Knowledge Retriever (deduplication) + Evaluator (consistency scoring)
3. **Freshness**: Document Ingestion (timestamp tracking) + Evaluator (freshness scoring)
4. **Traceability**: Answer Generator (citation generation) + Evaluator (traceability scoring)
5. **Faithfulness**: Answer Generator (grounding) + Evaluator (RAGAS faithfulness)
6. **Relevance**: Knowledge Retriever (reranking) + Evaluator (RAGAS relevance)
7. **Correctness**: Answer Generator (guardrails) + Evaluator (RAGAS correctness)

## Components and Interfaces

### 1. Orchestrator Agent (Plan-and-Execute)

**Responsibility**: High-level coordination using Plan-and-Execute pattern. Creates execution plan, manages agent dependencies, and aggregates results.

**Interface**:
```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class AgentTask(Enum):
    """Agent tasks in execution plan"""
    PARSE_INTENT = "parse_intent"
    ROUTE_QUERY = "route_query"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    GENERATE_ANSWER = "generate_answer"
    EVALUATE_QUALITY = "evaluate_quality"
    REQUEST_REVIEW = "request_review"

@dataclass
class ExecutionPlan:
    """Planned sequence of agent tasks with dependencies"""
    tasks: List[AgentTask]
    dependencies: Dict[AgentTask, List[AgentTask]]  # Task → prerequisite tasks
    parallel_groups: List[List[AgentTask]]  # Tasks that can run in parallel
    estimated_duration_ms: int

@dataclass
class ExecutionContext:
    """Shared context across agent executions"""
    query: str
    user_id: str
    conversation_context: Optional[ConversationContext] = None
    
    # Results from each agent
    intent: Optional[IntentResult] = None
    routing_decision: Optional[RoutingDecision] = None
    retrieved_documents: Optional[List[Document]] = None
    generated_answer: Optional[Answer] = None
    evaluation: Optional[EvaluationResult] = None
    review_result: Optional[ReviewFeedback] = None

class OrchestratorAgent(RoutedAgent):
    """Plan-and-Execute orchestrator for high-level agent coordination using Microsoft Agent Framework"""
    
    def __init__(self, runtime: AgentRuntime) -> None:
        super().__init__("Orchestrator Agent")
        self.runtime = runtime
        # Agent instances are managed by runtime, not directly instantiated
        # Communication happens via message passing through runtime
    
    async def process_query(self, query: str, user_id: str) -> Answer:
        """
        Main Plan-and-Execute workflow:
        1. PLAN: Analyze query and create execution plan
        2. EXECUTE: Run agents according to plan (parallel where possible)
        3. AGGREGATE: Collect and validate results
        """
        # Step 1: PLAN
        context = ExecutionContext(query=query, user_id=user_id)
        plan = await self.create_plan(query, context)
        
        # Log plan to Langfuse
        await self.log_plan(plan, context)
        
        # Step 2: EXECUTE
        await self.execute_plan(plan, context)
        
        # Step 3: AGGREGATE & RETURN
        return await self.aggregate_results(context)
    
    async def create_plan(self, query: str, context: ExecutionContext) -> ExecutionPlan:
        """
        PLAN phase: Analyze query and create execution plan
        
        Determines:
        - Which agents to call
        - In what order
        - Which can run in parallel
        - Dependencies between tasks
        """
        # Standard RAG pipeline plan
        tasks = [
            AgentTask.PARSE_INTENT,
            AgentTask.ROUTE_QUERY,
            AgentTask.RETRIEVE_KNOWLEDGE,
            AgentTask.GENERATE_ANSWER,
            AgentTask.EVALUATE_QUALITY
        ]
        
        # Define dependencies (sequential by default)
        dependencies = {
            AgentTask.PARSE_INTENT: [],
            AgentTask.ROUTE_QUERY: [AgentTask.PARSE_INTENT],
            AgentTask.RETRIEVE_KNOWLEDGE: [AgentTask.ROUTE_QUERY],
            AgentTask.GENERATE_ANSWER: [AgentTask.RETRIEVE_KNOWLEDGE],
            AgentTask.EVALUATE_QUALITY: [AgentTask.GENERATE_ANSWER]
        }
        
        # No parallel groups in standard pipeline (sequential)
        parallel_groups = []
        
        return ExecutionPlan(
            tasks=tasks,
            dependencies=dependencies,
            parallel_groups=parallel_groups,
            estimated_duration_ms=2000
        )
    
    async def execute_plan(self, plan: ExecutionPlan, context: ExecutionContext) -> None:
        """
        EXECUTE phase: Run agents according to plan
        
        Executes tasks respecting dependencies.
        Each agent uses ReAct internally for its specific job.
        """
        for task in plan.tasks:
            # Check dependencies are satisfied
            await self.wait_for_dependencies(task, plan.dependencies, context)
            
            # Execute task (agent uses ReAct internally)
            await self.execute_task(task, context)
            
            # Log task completion
            await self.log_task_completion(task, context)
    
    async def execute_task(self, task: AgentTask, context: ExecutionContext) -> None:
        """
        Execute a single agent task
        
        Each agent uses appropriate pattern internally for its specific job.
        Orchestrator sends messages to agents via Microsoft Agent Framework runtime.
        
        Note: Runtime enables flexible deployment:
        - SingleThreadedAgentRuntime: Direct method calls (high performance, single process)
        - GrpcWorkerAgentRuntime: gRPC calls (distributed, multi-process)
        """
        try:
            if task == AgentTask.PARSE_INTENT:
                # Send message to Intent Parser Agent via runtime
                response = await self.runtime.send_message(
                    IntentParseRequest(
                        query=context.query,
                        user_id=context.user_id,
                        conversation_context=context.conversation_context
                    ),
                    AgentId("intent_parser", "default")
                )
                context.intent = response
            
            elif task == AgentTask.ROUTE_QUERY:
                # Parlant uses rule-based routing (no ReAct needed)
                context.routing_decision = await self.router.route(
                    context.query,
                    context.intent
                )
            
            elif task == AgentTask.RETRIEVE_KNOWLEDGE:
                # Knowledge Retriever uses ReAct internally for multi-strategy retrieval
                context.retrieved_documents = await self.retriever.retrieve(
                    context.query,
                    context.intent,
                    context.routing_decision
                )
            
            elif task == AgentTask.GENERATE_ANSWER:
                # Answer Generator uses ReAct internally for iterative context assembly
                context.generated_answer = await self.generator.generate(
                    context.query,
                    context.retrieved_documents,
                    context.intent
                )
            
            elif task == AgentTask.EVALUATE_QUALITY:
                # Evaluator uses ReAct internally for multi-metric evaluation
                context.evaluation = await self.evaluator.evaluate(
                    context.query,
                    context.retrieved_documents,
                    context.generated_answer
                )
                
                # If quality is low, add review task dynamically
                if context.evaluation.needs_review:
                    context.review_result = await self.reviewer.request_review(
                        context.generated_answer,
                        context.evaluation
                    )
        
        except Exception as e:
            await self.handle_task_failure(task, e, context)
    
    async def wait_for_dependencies(
        self,
        task: AgentTask,
        dependencies: Dict[AgentTask, List[AgentTask]],
        context: ExecutionContext
    ) -> None:
        """Wait for prerequisite tasks to complete"""
        # In sequential execution, this is a no-op
        # In parallel execution, would wait for dependency tasks
        pass
    
    async def aggregate_results(self, context: ExecutionContext) -> Answer:
        """
        AGGREGATE phase: Collect and validate results
        
        Returns final answer with all metadata.
        """
        if not context.generated_answer:
            return self.create_fallback_answer(context)
        
        # Add evaluation metadata to answer
        context.generated_answer.provenance["evaluation"] = {
            "overall_score": context.evaluation.overall_score,
            "faithfulness": context.evaluation.faithfulness,
            "relevance": context.evaluation.relevance
        }
        
        return context.generated_answer
    
    async def handle_task_failure(
        self,
        task: AgentTask,
        error: Exception,
        context: ExecutionContext
    ) -> None:
        """
        Handle task failures with retry logic
        
        Implements exponential backoff and fallback strategies.
        """
        # Log error to Langfuse
        await self.log_error(task, error, context)
        
        # Retry logic or fallback
        # For now, raise to fail fast
        raise
    
    async def log_plan(self, plan: ExecutionPlan, context: ExecutionContext) -> None:
        """Log execution plan to Langfuse"""
        pass
    
    async def log_task_completion(self, task: AgentTask, context: ExecutionContext) -> None:
        """Log task completion to Langfuse"""
        pass
    
    async def log_error(self, task: AgentTask, error: Exception, context: ExecutionContext) -> None:
        """Log errors to Langfuse"""
        pass
    
    def create_fallback_answer(self, context: ExecutionContext) -> Answer:
        """Create fallback answer when execution fails"""
        return Answer(
            text="I apologize, but I couldn't complete processing your query. Please try rephrasing or contact support.",
            sources=[],
            confidence=0.0,
            model_used="fallback",
            generation_time_ms=0,
            provenance={"reason": "execution_failed"}
        )
```

**Plan-and-Execute Benefits:**

1. **Clear Structure**: Explicit execution plan with dependencies
2. **Predictable**: Same workflow for similar queries
3. **Parallelizable**: Can run independent tasks concurrently
4. **Monitorable**: Track progress through plan stages
5. **Composable**: Easy to add/remove agents from pipeline

**Note**: Each agent called by the orchestrator uses **ReAct internally** for its specific job (e.g., Knowledge Retriever uses ReAct to try multiple retrieval strategies).

**Dependencies**: Intent Parser Agent (ReAct), Parlant Router (Rule-based), Knowledge Retriever Agent (ReAct), Answer Generator Agent (ReAct), Evaluator Agent (ReAct), Human Review Agent (Manual)

### 2. Intent Parser Agent

**Responsibility**: Extract intent, entities, and confidence from natural language queries using advanced reasoning techniques.

**Advanced Techniques**:

1. **Multi-Step Planning**: Decompose complex queries into sequential sub-intents
2. **Variable Assignment**: Track entities across multi-turn conversations
3. **Step-Back Reasoning**: Abstract query to higher-level concepts before parsing
4. **Confidence Scoring**: Multi-dimensional confidence with uncertainty quantification
5. **Optional Entities**: Distinguish required vs. optional entities
6. **Fallback & Recovery**: Graceful degradation with clarification strategies

**Interface**:
```python
class IntentParserAgent:
    async def parse_intent(self, query: str, context: ConversationContext) -> IntentResult:
        """
        Extract intent and entities with confidence scores using multi-step reasoning.
        
        Process:
        1. Step-back reasoning: Abstract query to high-level intent
        2. Multi-step planning: Decompose into sub-intents if complex
        3. Entity extraction: Identify required and optional entities
        4. Variable assignment: Resolve entities from conversation context
        5. Confidence scoring: Compute multi-dimensional confidence
        6. Fallback detection: Identify if clarification needed
        """
        
    async def step_back_reasoning(self, query: str) -> AbstractIntent:
        """
        Abstract query to higher-level concepts before detailed parsing.
        
        Example:
        Query: "Show me orders from customer ABC123 in the last week"
        Abstract: "Temporal data retrieval with entity filter"
        
        This helps identify the query pattern before getting into specifics.
        """
        
    async def multi_step_planning(self, query: str, abstract_intent: AbstractIntent) -> List[SubIntent]:
        """
        Decompose complex queries into sequential sub-intents.
        
        Example:
        Query: "Compare revenue between Q1 and Q2, then show top products"
        Sub-intents:
        1. Retrieve Q1 revenue data
        2. Retrieve Q2 revenue data
        3. Compute comparison metrics
        4. Retrieve product data
        5. Rank products by revenue
        
        Each sub-intent has dependencies and execution order.
        """
        
    async def extract_entities(self, query: str, sub_intents: List[SubIntent]) -> EntityExtractionResult:
        """
        Extract entities with required/optional classification.
        
        Returns:
        - required_entities: Must be present for query to succeed
        - optional_entities: Enhance query but not mandatory
        - missing_required: Required entities not found (triggers clarification)
        """
        
    async def assign_variables(self, entities: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """
        Resolve entity values from conversation context.
        
        Example:
        User: "Show me customer ABC123"
        System: [displays customer info]
        User: "What are their recent orders?"
        
        Variable assignment resolves "their" → "ABC123" from context.
        
        Tracks:
        - Entity references across turns
        - Temporal references ("yesterday", "last week")
        - Anaphora resolution ("it", "them", "that")
        """
        
    async def compute_confidence(self, intent: str, entities: Dict, sub_intents: List[SubIntent]) -> ConfidenceScore:
        """
        Multi-dimensional confidence scoring with uncertainty quantification.
        
        Dimensions:
        1. Intent confidence: How certain is the intent classification?
        2. Entity confidence: How certain are entity extractions?
        3. Completeness confidence: Are all required entities present?
        4. Ambiguity score: How ambiguous is the query?
        5. Context confidence: How well does context support interpretation?
        
        Returns confidence vector and overall score.
        """
        
    async def detect_fallback_need(self, confidence: ConfidenceScore, entities: EntityExtractionResult) -> FallbackStrategy:
        """
        Determine if fallback/clarification is needed and select strategy.
        
        Strategies:
        1. Request missing required entities
        2. Disambiguate between multiple interpretations
        3. Confirm low-confidence interpretation
        4. Suggest alternative phrasings
        5. Route to human for complex queries
        """
        
    async def detect_ambiguity(self, query: str) -> AmbiguityScore:
        """
        Identify ambiguous queries requiring clarification.
        
        Ambiguity types:
        - Lexical: Word has multiple meanings
        - Syntactic: Sentence structure unclear
        - Semantic: Intent unclear from context
        - Referential: Unclear what entities refer to
        """
        
    async def classify_query_type(self, query: str, abstract_intent: AbstractIntent) -> QueryType:
        """
        Classify as SQL, data story, general QA, visualization, etc.
        
        Uses abstract intent to determine query category before detailed parsing.
        """
```

**Output**:
```python
@dataclass
class AbstractIntent:
    """High-level query pattern before detailed parsing"""
    pattern: str  # e.g., "temporal_data_retrieval", "comparison_analysis"
    complexity: Literal["simple", "moderate", "complex"]
    requires_multi_step: bool
    estimated_sub_intents: int

@dataclass
class SubIntent:
    """Individual step in multi-step query plan"""
    step_number: int
    intent: str
    entities: Dict[str, Any]
    dependencies: List[int]  # Which steps must complete first
    optional: bool  # Can this step be skipped if it fails?

@dataclass
class EntityExtractionResult:
    """Entity extraction with required/optional classification"""
    required_entities: Dict[str, Any]
    optional_entities: Dict[str, Any]
    missing_required: List[str]
    entity_confidence: Dict[str, float]

@dataclass
class ConfidenceScore:
    """Multi-dimensional confidence with uncertainty quantification"""
    intent_confidence: float  # 0-1
    entity_confidence: float  # 0-1
    completeness_confidence: float  # 0-1
    ambiguity_score: float  # 0-1 (higher = more ambiguous)
    context_confidence: float  # 0-1
    overall_confidence: float  # Weighted average
    uncertainty: float  # Epistemic uncertainty estimate

@dataclass
class FallbackStrategy:
    """Strategy for handling low-confidence or incomplete parses"""
    strategy_type: Literal[
        "request_entities",
        "disambiguate",
        "confirm_interpretation",
        "suggest_rephrase",
        "route_to_human"
    ]
    clarification_questions: List[str]
    suggested_alternatives: List[str]
    can_proceed_with_defaults: bool

@dataclass
class ConversationContext:
    """Context from previous conversation turns"""
    previous_queries: List[str]
    previous_intents: List[IntentResult]
    entity_history: Dict[str, Any]  # Entities mentioned in conversation
    temporal_context: datetime  # When conversation started
    user_preferences: Dict[str, Any]

@dataclass
class IntentResult:
    """Complete intent parsing result with all advanced features"""
    # Core intent
    intent: str
    query_type: QueryType
    
    # Multi-step planning
    abstract_intent: AbstractIntent
    sub_intents: List[SubIntent]
    execution_plan: List[int]  # Ordered list of sub-intent step numbers
    
    # Entity extraction
    entities: Dict[str, Any]  # All entities (required + optional)
    required_entities: Dict[str, Any]
    optional_entities: Dict[str, Any]
    missing_required: List[str]
    
    # Confidence and ambiguity
    confidence: ConfidenceScore
    ambiguity_score: float
    
    # Fallback and recovery
    requires_clarification: bool
    fallback_strategy: Optional[FallbackStrategy]
    
    # Context tracking
    resolved_from_context: Dict[str, Any]  # Entities resolved from history
    conversation_context: ConversationContext
```

**Implementation Example**:

```python
class IntentParserAgent:
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm = LLMClient(model=llm_model)
        self.entity_extractor = EntityExtractor()
        self.confidence_estimator = ConfidenceEstimator()
        
    async def parse_intent(self, query: str, context: ConversationContext) -> IntentResult:
        # Step 1: Step-back reasoning
        abstract_intent = await self.step_back_reasoning(query)
        
        # Step 2: Multi-step planning (if complex)
        if abstract_intent.requires_multi_step:
            sub_intents = await self.multi_step_planning(query, abstract_intent)
        else:
            sub_intents = [SubIntent(
                step_number=1,
                intent=abstract_intent.pattern,
                entities={},
                dependencies=[],
                optional=False
            )]
        
        # Step 3: Entity extraction
        entity_result = await self.extract_entities(query, sub_intents)
        
        # Step 4: Variable assignment from context
        resolved_entities = await self.assign_variables(
            entity_result.required_entities,
            context
        )
        
        # Step 5: Confidence scoring
        confidence = await self.compute_confidence(
            abstract_intent.pattern,
            resolved_entities,
            sub_intents
        )
        
        # Step 6: Fallback detection
        fallback = await self.detect_fallback_need(confidence, entity_result)
        
        # Step 7: Query type classification
        query_type = await self.classify_query_type(query, abstract_intent)
        
        return IntentResult(
            intent=abstract_intent.pattern,
            query_type=query_type,
            abstract_intent=abstract_intent,
            sub_intents=sub_intents,
            execution_plan=[s.step_number for s in sub_intents],
            entities={**entity_result.required_entities, **entity_result.optional_entities},
            required_entities=entity_result.required_entities,
            optional_entities=entity_result.optional_entities,
            missing_required=entity_result.missing_required,
            confidence=confidence,
            ambiguity_score=confidence.ambiguity_score,
            requires_clarification=fallback.strategy_type != "none",
            fallback_strategy=fallback if fallback.strategy_type != "none" else None,
            resolved_from_context=resolved_entities,
            conversation_context=context
        )
```

**Example Execution**:

```python
# Complex query with multi-step planning
query = "Compare revenue between Q1 and Q2 for customer ABC123, then show their top 5 products"
context = ConversationContext(
    previous_queries=[],
    previous_intents=[],
    entity_history={},
    temporal_context=datetime.now(),
    user_preferences={}
)

result = await intent_parser.parse_intent(query, context)

# Result:
# abstract_intent.pattern = "temporal_comparison_with_ranking"
# abstract_intent.complexity = "complex"
# abstract_intent.requires_multi_step = True
# 
# sub_intents = [
#     SubIntent(1, "retrieve_revenue", {"customer": "ABC123", "period": "Q1"}, [], False),
#     SubIntent(2, "retrieve_revenue", {"customer": "ABC123", "period": "Q2"}, [], False),
#     SubIntent(3, "compute_comparison", {"periods": ["Q1", "Q2"]}, [1, 2], False),
#     SubIntent(4, "retrieve_products", {"customer": "ABC123"}, [], False),
#     SubIntent(5, "rank_products", {"limit": 5, "metric": "revenue"}, [4], False)
# ]
# 
# confidence.overall_confidence = 0.92
# requires_clarification = False
```

### 3. Vector Store Adapter Pattern

**Design Philosophy**: Use adapter pattern to enable easy switching between vector databases via configuration.

**Supported Vector Stores**:
- PostgreSQL + pgvector (default, simplest setup)
- Weaviate (production-grade, cloud-native)
- Azure AI Search (Azure-native)

**Adapter Interface**:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    provider: str  # "weaviate", "azure_ai_search"
    connection_string: str
    index_name: str
    embedding_dimension: int = 1536
    distance_metric: str = "cosine"
    additional_config: Dict[str, Any] = {}

class VectorStoreAdapter(ABC):
    """Abstract base class for vector store adapters"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to vector store"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to vector store"""
        pass
    
    @abstractmethod
    async def create_index(self, index_name: str, dimension: int) -> None:
        """Create vector index"""
        pass
    
    @abstractmethod
    async def insert(self, documents: List[Document]) -> None:
        """Insert documents with embeddings"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Semantic search with optional metadata filters"""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,  # Balance between semantic and keyword
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Hybrid search combining semantic and keyword search"""
        pass
    
    @abstractmethod
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    async def update_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """Update document metadata"""
        pass
```

**Concrete Implementations**:

**1. PostgreSQL + pgvector Adapter:**

```python
import asyncpg
from pgvector.asyncpg import register_vector

class PgVectorAdapter(VectorStoreAdapter):
    """PostgreSQL + pgvector adapter for hybrid search"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """Establish connection pool"""
        self.pool = await asyncpg.create_pool(
            self.config.connection_string,
            min_size=5,
            max_size=20
        )
        async with self.pool.acquire() as conn:
            await register_vector(conn)
    
    async def disconnect(self) -> None:
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def create_index(self, index_name: str, dimension: int) -> None:
        """Create IVFFlat index for fast similarity search"""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}_embedding_idx 
                ON {index_name} 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100)
            """)
            # Create GIN index for full-text search (hybrid)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}_content_idx 
                ON {index_name} 
                USING gin(to_tsvector('english', content))
            """)
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Semantic search using cosine similarity"""
        async with self.pool.acquire() as conn:
            where_clause = self._build_where_clause(filters)
            query = f"""
                SELECT id, content, source, lob, confidence, metadata, 
                       embedding <=> $1 AS distance
                FROM {self.config.index_name}
                {where_clause}
                ORDER BY distance
                LIMIT $2
            """
            rows = await conn.fetch(query, query_embedding, top_k)
            return [self._row_to_document(row) for row in rows]
    
    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Hybrid search combining semantic and full-text search"""
        async with self.pool.acquire() as conn:
            where_clause = self._build_where_clause(filters)
            query = f"""
                WITH semantic_results AS (
                    SELECT id, content, source, lob, confidence, metadata,
                           1 - (embedding <=> $1) AS semantic_score
                    FROM {self.config.index_name}
                    {where_clause}
                ),
                keyword_results AS (
                    SELECT id, content, source, lob, confidence, metadata,
                           ts_rank(to_tsvector('english', content), 
                                   plainto_tsquery('english', $2)) AS keyword_score
                    FROM {self.config.index_name}
                    {where_clause}
                )
                SELECT DISTINCT ON (s.id)
                    s.id, s.content, s.source, s.lob, s.confidence, s.metadata,
                    ($3 * s.semantic_score + (1 - $3) * COALESCE(k.keyword_score, 0)) AS combined_score
                FROM semantic_results s
                LEFT JOIN keyword_results k ON s.id = k.id
                ORDER BY combined_score DESC
                LIMIT $4
            """
            rows = await conn.fetch(query, query_embedding, query_text, alpha, top_k)
            return [self._row_to_document(row) for row in rows]
    
    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> str:
        """Build WHERE clause from filters"""
        if not filters:
            return ""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(f"{key} = ANY(${len(conditions) + 3})")
            else:
                conditions.append(f"{key} = ${len(conditions) + 3}")
        return "WHERE " + " AND ".join(conditions) if conditions else ""
    
    def _row_to_document(self, row) -> Document:
        """Convert database row to Document"""
        return Document(
            id=row['id'],
            content=row['content'],
            source=row['source'],
            lob=LOB(row['lob']),
            confidence=row['confidence'],
            metadata=row['metadata'],
            timestamp=row.get('created_at', datetime.now())
        )
```

**2. Weaviate Adapter:**

```python
import weaviate
from weaviate.classes.query import MetadataQuery

class WeaviateAdapter(VectorStoreAdapter):
    """Weaviate adapter for production-grade vector search"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.client: Optional[weaviate.Client] = None
    
    async def connect(self) -> None:
        """Connect to Weaviate cluster"""
        self.client = weaviate.Client(
            url=self.config.connection_string,
            additional_headers=self.config.additional_config.get("headers", {})
        )
    
    async def disconnect(self) -> None:
        """Close Weaviate connection"""
        if self.client:
            self.client.close()
    
    async def create_index(self, index_name: str, dimension: int) -> None:
        """Create Weaviate class (index)"""
        class_obj = {
            "class": index_name,
            "vectorizer": "none",  # We provide embeddings
            "moduleConfig": {
                "text2vec-openai": {"skip": True}
            },
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "source", "dataType": ["string"]},
                {"name": "lob", "dataType": ["string"]},
                {"name": "confidence", "dataType": ["number"]},
                {"name": "metadata", "dataType": ["object"]},
            ]
        }
        self.client.schema.create_class(class_obj)
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Semantic search using Weaviate"""
        query = self.client.query.get(
            self.config.index_name,
            ["content", "source", "lob", "confidence", "metadata"]
        ).with_near_vector({
            "vector": query_embedding
        }).with_limit(top_k)
        
        if filters:
            query = query.with_where(self._build_weaviate_filter(filters))
        
        result = query.do()
        return [self._weaviate_to_document(obj) for obj in result["data"]["Get"][self.config.index_name]]
    
    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Hybrid search using Weaviate's native hybrid search"""
        query = self.client.query.get(
            self.config.index_name,
            ["content", "source", "lob", "confidence", "metadata"]
        ).with_hybrid(
            query=query_text,
            vector=query_embedding,
            alpha=alpha  # 0 = pure keyword, 1 = pure vector
        ).with_limit(top_k)
        
        if filters:
            query = query.with_where(self._build_weaviate_filter(filters))
        
        result = query.do()
        return [self._weaviate_to_document(obj) for obj in result["data"]["Get"][self.config.index_name]]
    
    def _build_weaviate_filter(self, filters: Dict[str, Any]) -> Dict:
        """Build Weaviate filter from dict"""
        # Simplified - expand based on needs
        return {
            "operator": "And",
            "operands": [
                {"path": [key], "operator": "Equal", "valueString": value}
                for key, value in filters.items()
            ]
        }
    
    def _weaviate_to_document(self, obj: Dict) -> Document:
        """Convert Weaviate object to Document"""
        return Document(
            id=obj["_additional"]["id"],
            content=obj["content"],
            source=obj["source"],
            lob=LOB(obj["lob"]),
            confidence=obj["confidence"],
            metadata=obj.get("metadata", {}),
            timestamp=datetime.now()
        )
```

**Vector Store Factory:**

```python
class VectorStoreFactory:
    """Factory for creating vector store adapters"""
    
    _adapters = {
        # "pgvector": PgVectorAdapter,
        "weaviate": WeaviateAdapter,
        # Add more as needed
        # "pinecone": PineconeAdapter,
        # "qdrant": QdrantAdapter,
        # "azure_ai_search": AzureAISearchAdapter,
    }
    
    @classmethod
    def create(cls, config: VectorStoreConfig) -> VectorStoreAdapter:
        """Create vector store adapter based on config"""
        adapter_class = cls._adapters.get(config.provider)
        if not adapter_class:
            raise ValueError(f"Unsupported vector store provider: {config.provider}")
        return adapter_class(config)
    
    @classmethod
    def from_env(cls) -> VectorStoreAdapter:
        """Create adapter from environment variables"""
        config = VectorStoreConfig(
            provider=os.getenv("VECTOR_STORE_PROVIDER", "pgvector"),
            connection_string=os.getenv("VECTOR_STORE_CONNECTION_STRING"),
            index_name=os.getenv("VECTOR_STORE_INDEX_NAME", "documents"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            distance_metric=os.getenv("DISTANCE_METRIC", "cosine")
        )
        return cls.create(config)
```

**Configuration Examples:**

```yaml
# config/vector_store.yaml

# PostgreSQL + pgvector (default)
pgvector:
  provider: pgvector
  connection_string: postgresql://user:pass@localhost:5432/rag_db
  index_name: documents
  embedding_dimension: 1536
  distance_metric: cosine

# Weaviate
weaviate:
  provider: weaviate
  connection_string: http://localhost:8080
  index_name: Documents
  embedding_dimension: 1536
  additional_config:
    headers:
      X-OpenAI-Api-Key: ${OPENAI_API_KEY}

# Easy switching via environment variable
# VECTOR_STORE_PROVIDER=weaviate
# VECTOR_STORE_PROVIDER=pgvector
```

### 3.5. Agent Runtime Configuration (A2A Protocol + Motia Integration)

**Design Philosophy**: Combine **A2A Protocol compliance** with **Motia's native queue system** for agent-to-agent communication - no HTTP needed, full ecosystem interoperability.

**The Magic: A2A + Motia Integration**

```
A2A Protocol           Motia Queue System
─────────────         ──────────────────
AgentCard       ←→    Agent Registry (State)
RequestContext  ←→    Motia Event Input
ExecutionEventBus ←→  Motia context.emit()
AgentExecutor   ←→    Event Step Handler
```

**Key Components**

**1. MotiaA2AAgent Wrapper**
- Wraps any A2A `AgentExecutor`
- Converts Motia events → A2A `RequestContext`
- Converts A2A results → Motia events
- Maintains full A2A protocol compliance

**2. MotiaEventBus**
- Implements A2A's `ExecutionEventBus` interface
- Routes A2A events through Motia's `context.emit()`
- A2A thoughts/actions become Motia events
- All automatically queued by Motia

**3. Event-Driven Communication**
```python
# No HTTP calls - just Motia events
context.emit('agent.task.researcher_a2a', {
    'action': 'execute_task',
    'query': 'AI trends'
})

# Motia automatically queues this
# A2A agent receives and processes
# Results emitted back through Motia queues
```

**Benefits of A2A-Motia Integration**

### ✅ Full A2A Compliance
- Uses real `AgentCard`, `RequestContext`, `ExecutionEventBus`
- A2A agents work exactly as specified
- Can emit thoughts, actions, progress updates
- Compatible with A2A ecosystem

### ✅ Motia's Native Features
- Automatic queuing - no Redis/HTTP setup
- Built-in retry and fault tolerance
- Shared state across agents
- Real-time observability in Workbench
- Multi-language support

### ✅ Better Performance
- No HTTP overhead
- No network latency
- No port conflicts
- No connection pooling needed

### ✅ Simpler Operations
- One runtime (Motia)
- One deployment
- No separate A2A servers
- No service discovery needed

**A2A Events Flow Through Motia**

```
A2A Agent emits:           Motia queues as:
─────────────────         ───────────────────
thought("Planning...") →  'a2a.thought' event
action("web_search") →    'a2a.action' event
progress(0.5, "Working") → 'agent.progress' event
result({...}) →           'task.completed' event
```

All these events:
- Are automatically queued by Motia
- Visible in Workbench
- Can be subscribed to by other Steps
- Have built-in retry logic

**Design Philosophy**: Use Motia's built-in runtime and queue system for all agent communication. No complex runtime abstractions needed - Motia handles distribution, messaging, and orchestration automatically through its event system.

**Key Runtime Features**:
- Automatic event routing based on `subscribes`/`emits` declarations
- Built-in queue management (no external brokers)
- Automatic retry and fault tolerance
- Scalable by default through Motia's architecture

**Agent Implementation Using A2A Protocol + Motia Integration**:

**1. Example Agent with A2A + Motia Integration:**

```python
from motia import step
from typing import Any, Dict, Optional
from enum import Enum

class QueryType(str, Enum):
    """Query classification types"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    COMPARATIVE = "comparative"

@step(
    name="IntentParser",
    emits=["intent.parsed"],  # Emits structured intent data
    subscribes=["query.received"]  # Subscribes to incoming queries
)
async def intent_parser_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    A2A Intent Parser Agent - wrapped by MotiaA2AAgent

    - Receives queries via Motia event queue
    - Uses A2A protocol internally for agent semantics
    - Emits structured intent results to Motia queue
    """
    query = event["query"]
    user_id = event["user_id"]
    conversation_context = event.get("context")

    # Extract A2A RequestContext from Motia event
    a2a_request = context.data.get("a2a_context", {})

    # Parse intent using LLM (A2A compatible)
    intent_result = await parse_intent_with_llm(
        query,
        conversation_context,
        a2a_request
    )

    # Emit structured intent via Motia (automatically queued)
    await context.emit({
        "topic": "intent.parsed",
        "data": {
            "intent": intent_result.intent,
            "entities": intent_result.entities,
            "confidence": intent_result.confidence,
            "query_type": intent_result.query_type,
            "metadata": {
                "a2a_agent_id": "intent_parser_a2a",
                "timestamp": context.timestamp,
                "user_id": user_id
            }
        }
    })

    return {"status": "success"}

async def parse_intent_with_llm(
    query: str,
    context: Optional[Dict],
    a2a_request: Dict
) -> "IntentResult":
    """
    Internal A2A-compatible intent parsing

    Emits A2A thoughts and actions through Motia's event system:
    - context.emit("a2a.thought", {...})
    - context.emit("a2a.action", {...})
    """
    # Emit A2A thought event
    await context.emit({
        "topic": "a2a.thought",
        "data": {
            "agent_id": "intent_parser_a2a",
            "thought": f"Analyzing query: {query[:100]}...",
            "timestamp": context.timestamp
        }
    })

    # Use LLM for intent classification
    llm_response = await llm_client.classify_intent(
        query=query,
        context=context
    )

    # Emit A2A action event
    await context.emit({
        "topic": "a2a.action",
        "data": {
            "agent_id": "intent_parser_a2a",
            "action": "intent_classification",
            "result": llm_response.intent,
            "timestamp": context.timestamp
        }
    })

    return IntentResult(
        intent=llm_response.intent,
        entities=llm_response.entities,
        confidence=llm_response.confidence,
        query_type=llm_response.query_type
    )
```

**2. Motia Runtime Configuration (No Setup Code Needed):**

Motia automatically discovers and registers all `@step` functions - no manual registration required. This is one of the key advantages of the Motia framework:

```python
# In your main.py or app.py
from motia import create_app

# Motia automatically:
# 1. Scans all modules for @step decorators
# 2. Registers them with the runtime
# 3. Sets up event subscriptions
# 4. Configures API endpoints
# 5. Manages workflow execution

app = create_app()

# Start Motia - all agents are already configured
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
```

**3. Agent Communication via Motia Queues:**

```python
# Motia runtime automatically handles:
# - Event routing based on 'subscribes' and 'emits' declarations
# - Queue management (no external broker needed)
# - Retry logic and fault tolerance
# - Progress tracking and observability

# Example: Query received event triggers Intent Parser
{
    "topic": "query.received",
    "data": {
        "query": "What is RAG?",
        "user_id": "user_123",
        "context": {}
    }
}

# Intent Parser emits (automatically routes to Knowledge Retriever):
{
    "topic": "intent.parsed",
    "data": {
        "intent": "information_retrieval",
        "entities": {"topic": "RAG"},
        "metadata": {...}
    }
}

# Knowledge Retriever subscribes to "intent.parsed"
# and emits "knowledge.retrieved" automatically
```

**4. Agent Communication via Motia Queues:**

```python
# Motia automatically routes events based on 'subscribes' and 'emits' declarations
# No manual message sending needed - declare it and Motia handles it!

# Workflow flows naturally:
# 1. API endpoint emits 'query.received' (defined elsewhere)
# 2. IntentParser subscribes → automatically receives event
# 3. IntentParser emits 'intent.parsed' → automatically routed
# 4. KnowledgeRetriever subscribes to 'intent.parsed' → receives it

# For manual orchestration when needed:
async def trigger_workflow(query: str, user_id: str):
    """Manually trigger workflow by emitting initial event"""
    from motia import publish

    await publish({
        "topic": "query.received",
        "data": {
            "query": query,
            "user_id": user_id,
            "context": {},
            "timestamp": "2025-06-05T10:30:00Z"
        }
    })
```

**Motia Runtime Configuration via Environment:**

```python
import os

# Motia configuration is simple and declarative
# No complex runtime factories needed

MOTIA_CONFIG = {
    # Motia runtime mode
    "MOTIA_RUNTIME_MODE": os.getenv("MOTIA_RUNTIME_MODE", "development"),

    # Logging and observability
    "MOTIA_LOG_LEVEL": os.getenv("MOTIA_LOG_LEVEL", "INFO"),

    # Workbench configuration (Motia's built-in UI)
    "MOTIA_WORKBENCH_ENABLED": os.getenv("MOTIA_WORKBENCH_ENABLED", "true"),

    # Python or TypeScript runtime (Motia supports both!)
    "MOTIA_LANGUAGE": os.getenv("MOTIA_LANGUAGE", "python"),

    # Port for HTTP API endpoints
    "PORT": os.getenv("PORT", "3000"),

    # All agent communication happens via Motia queues
    # No separate configuration for gRPC, Redis, etc.
}

# Usage:
# MOTIA_RUNTIME_MODE=production python main.py
# No code changes needed between development and production
```

**Configuration Examples:**

```yaml
# config/motia.yaml

# Development Mode (default)
development:
  log_level: DEBUG
  workbench_enabled: true
  # Motia automatically:
  # - Provides hot reload
  # - Shows detailed logs
  # - Enables Workbench UI

# Production Mode (same code, different config)
production:
  log_level: INFO
  workbench_enabled: true  # Keep for monitoring!
  # Same runtime, same agents, just less verbose logging

# Server Configuration (port, host, etc.)
server:
  port: 3000
  host: 0.0.0.0
  # HTTP API endpoints are automatically generated from @step declarations

# Environment Variables:
# MOTIA_RUNTIME_MODE=development  # Default, hot reload, debug logs
# MOTIA_RUNTIME_MODE=production   # Optimized, info logs
# MOTIA_WORKBENCH_ENABLED=true    # Enable Workbench UI
# MOTIA_LOG_LEVEL=DEBUG           # Log level
```

**A2A-Motia Workflow Implementation:**

```python
from motia import step
from typing import Dict, Any

# The orchestrator subscribes to incoming queries
@step(
    name="Orchestrator",
    subscribes=["query.received"],
    emits=["intent.parse", "knowledge.retrieve", "answer.generate"]
)
async def orchestrator_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    A2A-Motia Orchestrator

    - Receives queries via Motia events
    - Emits tasks to A2A-compatible agents
    - Aggregates results from A2A agents
    - Returns final answer
    """
    query = event["query"]
    user_id = event["user_id"]
    conversation_context = event.get("context", {})

    # Emit A2A workflow start event
    await context.emit({
        "topic": "a2a.workflow.start",
        "data": {
            "workflow_id": context.trace_id,
            "query": query,
            "timestamp": context.timestamp
        }
    })

    # Step 1: Parse intent (emits to Intent Parser agent)
    await context.emit({
        "topic": "intent.parse",
        "data": {
            "query": query,
            "user_id": user_id,
            "context": conversation_context,
            "metadata": {"a2a_agent_id": "intent_parser_a2a"}
        }
    })

    # Intent parser will emit 'intent.parsed' which triggers knowledge retrieval
    # This happens automatically via Motia's event system

    return {"status": "workflow_started", "workflow_id": context.trace_id}

# Alternative: Using A2A agents directly within Motia
@step(
    name="OrchestratorDirect",
    subscribes=["query.received"],
    emits=["answer.generated"]
)
async def orchestrator_direct_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Direct A2A agent orchestration within Motia
    """
    from a2a_protocol import AgentCard, Task, TaskStatus

    # Discover available A2A agents from Motia state
    available_agents = context.state.get("a2a.agents", [])

    # Create A2A task
    task = Task(
        id=context.trace_id,
        input={"query": event["query"]},
        status=TaskStatus.SUBMITTED
    )

    # Send to A2A agent via Motia event
    await context.emit({
        "topic": f"a2a.task.{available_agents[0]['id']}",
        "data": {
            "task": task.to_dict(),
            "callback_topic": "answer.generated"
        }
    })

    return {"status": "task_submitted_to_a2a_agent"}
```

**Benefits of Motia Framework:**

1. **Unified Backend**: Single Step primitive for APIs, background jobs, workflows, and AI agents
2. **Zero Configuration**: Auto-discovery of steps and automatic connection via emits/subscribes
3. **Multi-Language Support**: Write steps in TypeScript, Python, JavaScript, and more in same system
4. **Event-Driven**: Built-in event bus for loosely coupled agent communication
5. **Built-in Observability**: End-to-end tracing, state management, and monitoring
6. **Flexible Deployment**: Single-process or distributed with same code
7. **Developer-Friendly**: Simple file-based configuration, no complex setup
8. **Production-Ready**: Used in ChessArena.ai and other production systems
9. **Scalability**: Built-in state management and horizontal scaling support
10. **Agentic AI**: Native support for AI agent workflows with streaming

**When to Use Different Runtime Modes:**

| Runtime Mode | Use Case | Workers | Logging | Tracing |
|--------------|----------|---------|---------|---------|
| **Development** | Local development, debugging | 1 | DEBUG | Full |
| **Default** | Staging, small production | 1-2 | INFO | Full |
| **Production** | Large-scale production | 4+ | INFO/WARN | Full |

**Motia vs Traditional Approaches:**

| Feature | Traditional (Multiple Frameworks) | Motia (Unified) |
|---------|-----------------------------------|-----------------|
| Workflow | Separate queue + API + Agent systems | Single Step primitive |
| State Management | External Redis/DB | Built-in |
| Language Support | Different runtimes | Unified runtime |
| Observability | Manual integration | Built-in |
| Learning Curve | Multiple frameworks | Single primitive |

**Scalability Path:**

1. **Start**: Single worker, development mode for MVP
2. **Scale**: Increase workers, enable production mode
3. **Distribute**: Use Redis adapters for state/streams across instances
4. **Monitor**: Leverage built-in observability and tracing

### 4. Knowledge Retriever Agent

**Responsibility**: Execute multi-strategy retrieval (embedding, metadata, heuristic) and merge results using pluggable vector store adapter.

**Interface**:
```python
class KnowledgeRetrieverAgent:
    def __init__(self, vector_store: VectorStoreAdapter):
        self.vector_store = vector_store
    
    async def retrieve(self, query: str, intent: IntentResult) -> List[Document]:
        """Execute multi-strategy retrieval"""
        
    async def semantic_search(self, query: str) -> List[Document]:
        """Embedding-based semantic search via adapter"""
        embedding = await self.generate_embedding(query)
        return await self.vector_store.search(embedding, top_k=10)
        
    async def hybrid_search(self, query: str) -> List[Document]:
        """Hybrid search combining semantic and keyword"""
        embedding = await self.generate_embedding(query)
        return await self.vector_store.hybrid_search(
            query_embedding=embedding,
            query_text=query,
            top_k=10,
            alpha=0.7  # Favor semantic over keyword
        )
        
    async def metadata_filter(self, constraints: Dict) -> List[Document]:
        """Structured filtering via adapter with filters"""
        # Use empty embedding for metadata-only search
        return await self.vector_store.search(
            query_embedding=[0.0] * 1536,
            top_k=100,
            filters=constraints
        )
        
    async def guided_grep(self, pattern: str) -> List[Document]:
        """Heuristic pattern matching"""
        
    async def rerank(self, candidates: List[Document]) -> List[Document]:
        """Optional reranking with Cohere or LLM-as-a-Judge"""
```

**Usage Example:**

```python
# Initialize from config
vector_store = VectorStoreFactory.from_env()
await vector_store.connect()

# Create retriever agent
retriever = KnowledgeRetrieverAgent(vector_store=vector_store)

# Use retriever
documents = await retriever.hybrid_search("What is RAG?")

# Easy switching: just change environment variable
# VECTOR_STORE_PROVIDER=weaviate
# No code changes needed!
```

**Benefits of Adapter Pattern:**

1. **Easy Switching**: Change vector store via configuration, no code changes
2. **Consistent Interface**: All agents use same interface regardless of backend
3. **Testability**: Mock adapter for unit tests
4. **Extensibility**: Add new vector stores by implementing adapter interface
5. **Hybrid Search**: Unified interface for semantic + keyword search
6. **Production Ready**: Start with pgvector, scale to Weaviate when needed

**Output**:
```python
@dataclass
class Document:
    content: str
    source: str
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any]
    lob: str  # Line of Business
```

### 4. Answer Generator Agent (Chain-of-Thought)

**Responsibility**: Synthesize answers from retrieved documents using Chain-of-Thought reasoning for grounded generation.

**Pattern**: Chain-of-Thought (CoT) - Structured reasoning in a single pass, no tool use needed.

**Interface**:
```python
class AnswerGeneratorAgent:
    """Uses Chain-of-Thought prompting for structured answer generation"""
    
    def __init__(self, llm_client: LLMClient, prompt_optimizer: AgentLightningOptimizer):
        self.llm = llm_client
        self.prompt_optimizer = prompt_optimizer
    
    async def generate(
        self,
        query: str,
        documents: List[Document],
        intent: IntentResult
    ) -> Answer:
        """
        Generate answer using Chain-of-Thought reasoning
        
        CoT Structure:
        1. Identify key information from documents
        2. Determine what information answers the query
        3. Synthesize answer with proper grounding
        4. Add citations and confidence assessment
        """
        # Select appropriate model based on complexity
        model = self.select_model(intent.abstract_intent.complexity)
        
        # Assemble context with provenance
        context = await self.assemble_context(documents)
        
        # Get optimized CoT prompt from Agent Lightning
        cot_prompt = await self.prompt_optimizer.get_optimized_prompt(
            task_type="answer_generation_cot",
            context=context
        )
        
        # Generate answer with CoT reasoning
        answer = await self.llm.generate(
            prompt=cot_prompt,
            query=query,
            context=context,
            model=model,
            temperature=0.3,  # Lower for more grounded responses
            include_reasoning=True  # Return CoT steps
        )
        
        # Extract citations from reasoning steps
        citations = self.extract_citations(answer.reasoning_steps, documents)
        
        return Answer(
            text=answer.text,
            sources=[doc.source for doc in documents],
            confidence=self.estimate_confidence(answer.reasoning_steps),
            model_used=model,
            generation_time_ms=answer.generation_time_ms,
            provenance={
                "reasoning_steps": answer.reasoning_steps,
                "citations": citations,
                "context_used": len(documents)
            }
        )
    
    def select_model(self, complexity: TaskComplexity) -> str:
        """
        Choose appropriate LLM based on task complexity
        
        Simple: GPT-4o-mini (fast, cost-effective)
        Moderate: GPT-4 (balanced)
        Complex: o1-preview (advanced reasoning)
        """
        if complexity == TaskComplexity.SIMPLE:
            return "gpt-4o-mini"
        elif complexity == TaskComplexity.MODERATE:
            return "gpt-4"
        else:
            return "o1-preview"
    
    async def assemble_context(self, documents: List[Document]) -> Context:
        """
        Assemble context with provenance metadata
        
        Includes:
        - Document content
        - Source attribution
        - Confidence scores
        - Timestamps for freshness
        """
        return Context(
            documents=documents,
            total_tokens=sum(len(doc.content.split()) for doc in documents),
            sources=[doc.source for doc in documents],
            confidence_scores=[doc.confidence for doc in documents],
            timestamps=[doc.timestamp for doc in documents]
        )
    
    def extract_citations(
        self,
        reasoning_steps: List[str],
        documents: List[Document]
    ) -> List[Citation]:
        """Extract citations from CoT reasoning steps"""
        citations = []
        for step in reasoning_steps:
            # Parse citations from reasoning (e.g., "[Source: Doc 1]")
            # Implementation details...
            pass
        return citations
    
    def estimate_confidence(self, reasoning_steps: List[str]) -> float:
        """
        Estimate confidence based on CoT reasoning quality
        
        Factors:
        - Number of supporting documents cited
        - Consistency across reasoning steps
        - Presence of hedging language
        """
        # Implementation details...
        return 0.85
```

**CoT Prompt Template Example**:
```
Given the user query and retrieved documents, generate a grounded answer following these steps:

Step 1: Identify Key Information
Review each document and extract information relevant to the query.
- Document 1: [key points]
- Document 2: [key points]
- Document 3: [key points]

Step 2: Determine Relevance
Identify which information directly answers the query.
- Most relevant: [information]
- Supporting context: [information]
- Not relevant: [information]

Step 3: Synthesize Answer
Combine relevant information into a coherent answer.
- Main point: [answer with citation]
- Supporting details: [details with citations]
- Caveats: [any limitations or uncertainties]

Step 4: Add Citations and Assess Confidence
Ensure all claims are backed by retrieved documents.
- Citations: [list of sources used]
- Confidence: [high/medium/low based on evidence quality]

Final Answer: [Generated answer with inline citations]
```

**Output**:
```python
@dataclass
class Answer:
    text: str
    sources: List[str]
    confidence: float
    model_used: str
    generation_time_ms: int
    provenance: Dict[str, Any]  # Includes reasoning_steps, citations
```

**Why CoT for Answer Generator?**
- **Text generation task**: Producing natural language, not taking actions
- **Structured reasoning**: Explicit steps ensure grounding in documents
- **Single-pass**: No need for iterative tool use
- **Explainable**: Reasoning steps provide transparency
- **Grounded**: Citations tied to specific reasoning steps

### 5. Evaluator Agent (Chain-of-Thought)

**Responsibility**: Assess answer quality using 7 RAG characteristics (faithfulness, relevance, correctness, coverage, consistency, freshness, traceability).

**Pattern**: Chain-of-Thought (CoT) - Structured analytical reasoning, no tool use needed.

**Interface**:
```python
class EvaluatorAgent:
    """Uses Chain-of-Thought prompting for structured quality evaluation"""
    
    def __init__(self, llm_client: LLMClient, ragas_evaluator: RAGASEvaluator):
        self.llm = llm_client
        self.ragas = ragas_evaluator
    
    async def evaluate(
        self,
        query: str,
        documents: List[Document],
        answer: Answer
    ) -> EvaluationResult:
        """
        Comprehensive quality evaluation using Chain-of-Thought
        
        CoT Structure:
        1. Check Faithfulness (grounding in documents)
        2. Check Relevance (addresses query)
        3. Check Correctness (factual accuracy)
        4. Check Coverage (completeness)
        5. Check Consistency (internal coherence)
        6. Check Freshness (recency)
        7. Check Traceability (source attribution)
        8. Overall assessment and decision
        """
        # Use CoT prompt for structured evaluation
        cot_prompt = self.build_evaluation_cot_prompt(query, documents, answer)
        
        evaluation_reasoning = await self.llm.generate(
            prompt=cot_prompt,
            temperature=0.1,  # Very low for consistent evaluation
            model="gpt-4"
        )
        
        # Parse scores from CoT reasoning
        scores = self.parse_scores_from_reasoning(evaluation_reasoning)
        
        # Also run RAGAS metrics for validation
        ragas_scores = await self.ragas.evaluate(query, documents, answer)
        
        # Combine CoT and RAGAS scores
        final_scores = self.combine_scores(scores, ragas_scores)
        
        # Determine if human review needed
        overall_score = self.calculate_overall_score(final_scores)
        needs_review = overall_score < 0.80  # Threshold
        review_reasons = self.identify_review_reasons(final_scores) if needs_review else []
        
        return EvaluationResult(
            faithfulness=final_scores["faithfulness"],
            relevance=final_scores["relevance"],
            correctness=final_scores["correctness"],
            coverage=final_scores["coverage"],
            consistency=final_scores["consistency"],
            freshness=final_scores["freshness"],
            traceability=final_scores["traceability"],
            overall_score=overall_score,
            needs_review=needs_review,
            review_reasons=review_reasons,
            reasoning_steps=evaluation_reasoning.steps
        )
    
    def build_evaluation_cot_prompt(
        self,
        query: str,
        documents: List[Document],
        answer: Answer
    ) -> str:
        """Build CoT prompt for evaluation"""
        return f"""
        Evaluate the answer quality following these steps:
        
        Query: {query}
        Answer: {answer.text}
        Documents: {[doc.content for doc in documents]}
        
        Step 1: Check Faithfulness
        - Examine each claim in the answer
        - Verify if it's supported by the documents
        - Score: 0.0 (not faithful) to 1.0 (fully faithful)
        
        Step 2: Check Relevance
        - Does the answer address the query?
        - Is it on-topic and useful?
        - Score: 0.0 (irrelevant) to 1.0 (highly relevant)
        
        Step 3: Check Correctness
        - Are the facts accurate?
        - Any contradictions with documents?
        - Score: 0.0 (incorrect) to 1.0 (correct)
        
        Step 4: Check Coverage
        - Does it cover all query aspects?
        - Any important information missing?
        - Score: 0.0 (incomplete) to 1.0 (comprehensive)
        
        Step 5: Check Consistency
        - Is the answer internally consistent?
        - Any contradictions within the answer?
        - Score: 0.0 (inconsistent) to 1.0 (consistent)
        
        Step 6: Check Freshness
        - How recent is the information?
        - Are documents up-to-date?
        - Score: 0.0 (stale) to 1.0 (fresh)
        
        Step 7: Check Traceability
        - Are sources properly cited?
        - Can claims be traced to documents?
        - Score: 0.0 (no citations) to 1.0 (fully cited)
        
        Step 8: Overall Assessment
        - Calculate average score
        - Determine if human review needed (threshold: 0.80)
        - Provide reasoning for decision
        
        Provide scores and reasoning for each step.
        """
    
    def parse_scores_from_reasoning(self, reasoning: Any) -> Dict[str, float]:
        """Extract scores from CoT reasoning steps"""
        # Implementation: Parse scores from structured reasoning
        return {
            "faithfulness": 0.95,
            "relevance": 0.92,
            "correctness": 0.98,
            "coverage": 0.85,
            "consistency": 0.90,
            "freshness": 0.88,
            "traceability": 0.93
        }
    
    def combine_scores(
        self,
        cot_scores: Dict[str, float],
        ragas_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine CoT and RAGAS scores with weighted average"""
        combined = {}
        for metric in cot_scores:
            # Weight: 60% CoT (more explainable), 40% RAGAS (more rigorous)
            combined[metric] = 0.6 * cot_scores[metric] + 0.4 * ragas_scores.get(metric, cot_scores[metric])
        return combined
    
    def calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        weights = {
            "faithfulness": 0.25,  # Most important
            "relevance": 0.20,
            "correctness": 0.20,
            "coverage": 0.15,
            "consistency": 0.10,
            "freshness": 0.05,
            "traceability": 0.05
        }
        return sum(scores[metric] * weights[metric] for metric in scores)
    
    def identify_review_reasons(self, scores: Dict[str, float]) -> List[str]:
        """Identify which metrics failed threshold"""
        reasons = []
        threshold = 0.80
        for metric, score in scores.items():
            if score < threshold:
                reasons.append(f"{metric}: {score:.2f} < {threshold}")
        return reasons
```

**CoT Evaluation Prompt Example**:
```
Step 1: Check Faithfulness
Claim: "Customer ABC123 has 5 orders"
Evidence: Document 1 states "Customer ABC123: 5 orders"
Assessment: Fully supported by evidence
Score: 1.0

Step 2: Check Relevance
Query asks: "How many orders does customer ABC123 have?"
Answer provides: Number of orders for ABC123
Assessment: Directly addresses query
Score: 0.95

[... continues for all 7 characteristics ...]

Step 8: Overall Assessment
Average score: 0.92
Threshold: 0.80
Decision: Quality is good, no review needed
Reasoning: All metrics above threshold, answer is well-grounded and relevant
```

**Output**:
```python
@dataclass
class EvaluationResult:
    faithfulness: float
    relevance: float
    correctness: float
    coverage: float
    consistency: float
    freshness: float
    traceability: float
    overall_score: float
    needs_review: bool
    review_reasons: List[str]
    reasoning_steps: List[str]  # CoT steps for transparency
```

**Why CoT for Evaluator?**
- **Analytical reasoning task**: Assessing quality, not taking actions
- **Multi-metric evaluation**: Structured steps for each characteristic
- **Single-pass**: No need for iterative tool use
- **Explainable**: Reasoning shows why scores were assigned
- **Consistent**: Low temperature ensures reproducible evaluations
    consistency: float
    freshness: float
    traceability: float
    overall_score: float
    needs_review: bool


### 6. Human Review Agent

**Responsibility**: Present low-quality answers to human reviewers and capture feedback.

**Interface**:
```python
class HumanReviewAgent:
    async def request_review(self, answer: Answer, evaluation: EvaluationResult) -> ReviewRequest:
        """Create review request with context"""
        
    async def present_to_reviewer(self, request: ReviewRequest) -> None:
        """Display in review UI"""
        
    async def capture_feedback(self, review_id: str) -> ReviewFeedback:
        """Capture reviewer decision and corrections"""
        
    async def store_feedback(self, feedback: ReviewFeedback) -> None:
        """Store in LightningStore for optimization"""
```


**Output**:
```python
@dataclass
class ReviewFeedback:
    review_id: str
    decision: Literal["approve", "reject"]
    corrected_routing: Optional[str]
    corrected_prompts: Optional[str]
    feedback_text: str
    category: Literal["routing", "answer_quality", "relevance"]
    timestamp: datetime
```

### 7. Parlant Behavioral Guidelines Layer

**Responsibility**: Ensure reliable agent behavior through contextually-matched guidelines with explainability. Implements Six Thinking Hats patterns across all RAG pipeline phases.

**Core Capabilities**:
1. **Behavioral Guidelines**: Natural language rules that are reliably enforced
2. **Journeys**: Structured progression through Double Diamond phases
3. **Canned Responses**: Eliminate hallucinations in critical scenarios
4. **Domain Adaptation**: Glossary terms and context variables
5. **Guardrails**: Quality thresholds and risk assessment gates

**Interface**:
```python
class ParlantRouter:
    async def route(self, query: str, intent: IntentResult) -> RoutingDecision:
        """Apply guidelines in priority order with Six Thinking Hats"""
        
    async def match_guideline(self, query: str, guideline: Guideline) -> bool:
        """Check if guideline condition matches"""
        
    async def apply_thinking_hat(self, hat: ThinkingHat, context: Dict) -> Any:
        """Apply specific thinking hat pattern (Green, White, Red, Black, Yellow, Blue)"""
        
    async def log_decision(self, decision: RoutingDecision) -> None:
        """Log to Langfuse for observability with guideline provenance"""
        
    async def update_guidelines(self, optimized: List[Guideline]) -> None:
        """Deploy updated guidelines from Agent Lightning"""
        
    async def create_journey_step(self, name: str, criteria: str) -> None:
        """Define journey step with success criteria"""
        
    async def create_canned_response(self, condition: str, response: str) -> None:
        """Create canned response for specific conditions"""
```

**Guideline Structure**:
```python
@dataclass
class Guideline:
    name: str
    condition: str  # Natural language condition
    action: str  # Natural language action
    tools: List[str]  # Tools to use
    priority: int  # Execution priority
    thinking_hat: ThinkingHat  # Which hat this guideline represents
    phase: DesignPhase  # Discover, Define, Develop, or Deliver
```

**Output**:
```python
@dataclass
class RoutingDecision:
    retriever: str
    guideline_matched: str
    confidence: float
    reasoning: str
    thinking_hat_used: ThinkingHat
    phase: DesignPhase
    explainability: Dict[str, Any]  # Why this guideline was matched
```

**Example Guidelines by Phase**:

```python
# Discover Phase (Green Hat)
await agent.create_guideline(
    condition="User query is ambiguous or has multiple interpretations",
    action="Generate at least 3 diverse interpretations using Green Hat thinking",
    tools=[generate_interpretations, retrieve_evidence],
    thinking_hat=ThinkingHat.GREEN,
    phase=DesignPhase.DISCOVER
)

# Define Phase (Black Hat)
await agent.create_guideline(
    condition="Validating interpretation confidence",
    action="Apply Black Hat critical thinking: check factual grounding, identify risks",
    tools=[validate_interpretation, assess_risks],
    thinking_hat=ThinkingHat.BLACK,
    phase=DesignPhase.DEFINE
)

# Develop Phase (Yellow Hat)
await agent.create_guideline(
    condition="Evaluating retrieval strategy performance",
    action="Score strategies using Yellow Hat optimism: identify benefits of each",
    tools=[evaluate_benefits, score_strategies],
    thinking_hat=ThinkingHat.YELLOW,
    phase=DesignPhase.DEVELOP
)

# Deliver Phase (Blue Hat)
await agent.create_guideline(
    condition="Answer quality score is 0.8 or above",
    action="Use Blue Hat orchestration to deliver answer with citations",
    tools=[format_answer, add_citations, deliver_to_user],
    thinking_hat=ThinkingHat.BLUE,
    phase=DesignPhase.DELIVER
)
```

### 8. Agent Lightning Optimizer

**Responsibility**: Optimize prompts and routing policies using reinforcement learning and human feedback.

**Interface**:
```python
class AgentLightningOptimizer:
    async def analyze_feedback(self, traces: List[Trace]) -> PatternAnalysis:
        """Identify patterns in human corrections"""
        
    async def optimize_prompts(self, analysis: PatternAnalysis) -> List[PromptTemplate]:
        """Generate improved prompt templates"""
        
    async def optimize_routing(self, analysis: PatternAnalysis) -> List[Guideline]:
        """Update routing guidelines"""
        
    async def run_ab_test(self, old_policy: Policy, new_policy: Policy) -> ABTestResult:
        """Compare policies with statistical significance"""
        
    async def promote_to_production(self, policy: Policy) -> None:
        """Deploy validated improvements"""
```

## Data Models

### Core Data Structures

All data models use Pydantic for validation and serialization, compatible with Microsoft Agent Framework and FastAPI.

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

# Enumerations

class QueryType(str, Enum):
    """Types of queries the system can handle"""
    SQL_GENERATION = "sql_generation"
    DATA_STORY = "data_story"
    GENERAL_QA = "general_qa"
    VISUALIZATION = "visualization"
    MULTI_STEP = "multi_step"

class TaskComplexity(str, Enum):
    """Complexity levels for model selection"""
    SIMPLE = "simple"  # GPT-4o-mini
    MODERATE = "moderate"  # GPT-4
    COMPLEX = "complex"  # o1-preview

class LOB(str, Enum):
    """Lines of Business for data sources"""
    INVENTORY = "inventory"
    ORDERS = "orders"
    SUPPORT = "support"
    FINANCE = "finance"
    GENERAL = "general"

class RetrievalStrategy(str, Enum):
    """Retrieval strategies for knowledge retrieval"""
    SEMANTIC = "semantic"  # Embedding-based
    METADATA = "metadata"  # Structured filtering
    HEURISTIC = "heuristic"  # Pattern matching
    HYBRID = "hybrid"  # Multiple strategies

class RoutingDecisionType(str, Enum):
    """Types of routing decisions"""
    NORMAL = "normal"
    FALLBACK = "fallback"
    HUMAN_REVIEW = "human_review"
    CLARIFICATION_NEEDED = "clarification_needed"

class ThinkingHat(str, Enum):
    """Six Thinking Hats for design thinking"""
    WHITE = "white"  # Facts, data, information
    RED = "red"  # Emotions, feelings, intuition
    BLACK = "black"  # Caution, risks, critical judgment
    YELLOW = "yellow"  # Benefits, optimism, positive thinking
    GREEN = "green"  # Creativity, alternatives, new ideas
    BLUE = "blue"  # Process control, meta-thinking, orchestration

class DesignPhase(str, Enum):
    """Double Diamond Design Process phases"""
    DISCOVER = "discover"  # Diverge: Explore problem space
    DEFINE = "define"  # Converge: Synthesize insights
    DEVELOP = "develop"  # Diverge: Generate solutions
    DELIVER = "deliver"  # Converge: Test and deliver

# Query and Intent Models

class Query(BaseModel):
    """User query with metadata"""
    id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., description="User's natural language query")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    conversation_id: Optional[UUID] = None

class ConversationContext(BaseModel):
    """Context from previous conversation turns"""
    conversation_id: UUID
    previous_queries: List[str] = Field(default_factory=list)
    previous_intents: List[str] = Field(default_factory=list)
    entity_history: Dict[str, Any] = Field(default_factory=dict)
    temporal_context: datetime = Field(default_factory=datetime.now)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

class AbstractIntent(BaseModel):
    """High-level query pattern before detailed parsing"""
    pattern: str = Field(..., description="Query pattern (e.g., 'temporal_data_retrieval')")
    complexity: TaskComplexity
    requires_multi_step: bool
    estimated_sub_intents: int

class SubIntent(BaseModel):
    """Individual step in multi-step query plan"""
    step_number: int
    intent: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[int] = Field(default_factory=list)
    optional: bool = False

class EntityExtractionResult(BaseModel):
    """Entity extraction with required/optional classification"""
    required_entities: Dict[str, Any] = Field(default_factory=dict)
    optional_entities: Dict[str, Any] = Field(default_factory=dict)
    missing_required: List[str] = Field(default_factory=list)
    entity_confidence: Dict[str, float] = Field(default_factory=dict)

class ConfidenceScore(BaseModel):
    """Multi-dimensional confidence with uncertainty quantification"""
    intent_confidence: float = Field(..., ge=0.0, le=1.0)
    entity_confidence: float = Field(..., ge=0.0, le=1.0)
    completeness_confidence: float = Field(..., ge=0.0, le=1.0)
    ambiguity_score: float = Field(..., ge=0.0, le=1.0)
    context_confidence: float = Field(..., ge=0.0, le=1.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)

class FallbackStrategy(BaseModel):
    """Strategy for handling low-confidence or incomplete parses"""
    strategy_type: Literal[
        "request_entities",
        "disambiguate",
        "confirm_interpretation",
        "suggest_rephrase",
        "route_to_human",
        "none"
    ]
    clarification_questions: List[str] = Field(default_factory=list)
    suggested_alternatives: List[str] = Field(default_factory=list)
    can_proceed_with_defaults: bool = False

class IntentResult(BaseModel):
    """Complete intent parsing result with all advanced features"""
    # Core intent
    intent: str
    query_type: QueryType
    
    # Multi-step planning
    abstract_intent: AbstractIntent
    sub_intents: List[SubIntent] = Field(default_factory=list)
    execution_plan: List[int] = Field(default_factory=list)
    
    # Entity extraction
    entities: Dict[str, Any] = Field(default_factory=dict)
    required_entities: Dict[str, Any] = Field(default_factory=dict)
    optional_entities: Dict[str, Any] = Field(default_factory=dict)
    missing_required: List[str] = Field(default_factory=list)
    
    # Confidence and ambiguity
    confidence: ConfidenceScore
    ambiguity_score: float = Field(..., ge=0.0, le=1.0)
    
    # Fallback and recovery
    requires_clarification: bool
    fallback_strategy: Optional[FallbackStrategy] = None
    
    # Context tracking
    resolved_from_context: Dict[str, Any] = Field(default_factory=dict)
    conversation_context: ConversationContext

# Retrieval Models

class Document(BaseModel):
    """Retrieved document with metadata and provenance"""
    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    lob: LOB
    embedding: Optional[List[float]] = None
    chunk_index: Optional[int] = None
    parent_document_id: Optional[UUID] = None

class RetrievalResult(BaseModel):
    """Result from knowledge retrieval"""
    documents: List[Document]
    strategy_used: RetrievalStrategy
    total_candidates: int
    reranked: bool
    retrieval_time_ms: int

# Answer Generation Models

class Answer(BaseModel):
    """Generated answer with provenance"""
    id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    text: str = Field(..., description="Generated answer text")
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str
    generation_time_ms: int
    provenance: Dict[str, Any] = Field(default_factory=dict)
    prompt_template_id: Optional[str] = None

# Evaluation Models

class EvaluationResult(BaseModel):
    """Comprehensive quality evaluation with 7 RAG characteristics"""
    answer_id: UUID
    query_id: UUID
    
    # 7 RAG Characteristics
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    correctness: float = Field(..., ge=0.0, le=1.0)
    coverage: float = Field(..., ge=0.0, le=1.0)
    consistency: float = Field(..., ge=0.0, le=1.0)
    freshness: float = Field(..., ge=0.0, le=1.0)
    traceability: float = Field(..., ge=0.0, le=1.0)
    
    # Overall assessment
    overall_score: float = Field(..., ge=0.0, le=1.0)
    needs_review: bool
    review_reasons: List[str] = Field(default_factory=list)
    evaluation_time_ms: int

# Human Review Models

class ReviewFeedback(BaseModel):
    """Human reviewer feedback and corrections"""
    review_id: UUID = Field(default_factory=uuid4)
    answer_id: UUID
    query_id: UUID
    decision: Literal["approve", "reject"]
    corrected_routing: Optional[str] = None
    corrected_prompts: Optional[str] = None
    corrected_answer: Optional[str] = None
    feedback_text: str
    category: Literal["routing", "answer_quality", "relevance", "other"]
    reviewer_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    time_to_review_seconds: Optional[int] = None

# Routing Models

class RoutingDecision(BaseModel):
    """Routing decision with explainability and design thinking context"""
    retriever: str
    guideline_matched: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    decision_type: RoutingDecisionType
    fallback_used: bool = False
    thinking_hat_used: Optional[ThinkingHat] = None
    phase: Optional[DesignPhase] = None
    explainability: Dict[str, Any] = Field(default_factory=dict)

# Tracing Models

class Trace(BaseModel):
    """Complete trace for observability"""
    trace_id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    user_id: str
    
    # Pipeline stages
    intent_parsing_ms: Optional[int] = None
    routing_decision: Optional[RoutingDecision] = None
    routing_ms: Optional[int] = None
    retrieval_strategy: Optional[RetrievalStrategy] = None
    retrieval_ms: Optional[int] = None
    documents_retrieved: int = 0
    generation_ms: Optional[int] = None
    evaluation_ms: Optional[int] = None
    
    # Results
    answer_generated: bool
    evaluation_scores: Optional[Dict[str, float]] = None
    human_feedback: Optional[ReviewFeedback] = None
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    total_time_ms: int
    llm_tokens_used: int = 0
    llm_cost_usd: float = 0.0

# Agent Task Models

class AgentTask(BaseModel):
    """Task for agent execution"""
    task_id: UUID = Field(default_factory=uuid4)
    agent_name: str
    task_type: str
    input_data: Dict[str, Any]
    dependencies: List[UUID] = Field(default_factory=list)
    priority: int = 0
    timeout_seconds: int = 30

class AgentResult(BaseModel):
    """Result from agent execution"""
    task_id: UUID
    agent_name: str
    success: bool
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: int
```

### Database Schema

**PostgreSQL Tables with pgvector Extension**:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Queries table
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id UUID,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_queries_user_id ON queries(user_id);
CREATE INDEX idx_queries_conversation_id ON queries(conversation_id);
CREATE INDEX idx_queries_created_at ON queries(created_at DESC);

-- Conversation context table
CREATE TABLE conversation_contexts (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    previous_queries TEXT[] DEFAULT '{}',
    previous_intents TEXT[] DEFAULT '{}',
    entity_history JSONB DEFAULT '{}',
    user_preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_conversation_contexts_user_id ON conversation_contexts(user_id);

-- Intent results table
CREATE TABLE intent_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID NOT NULL REFERENCES queries(id),
    intent TEXT NOT NULL,
    query_type TEXT NOT NULL,
    abstract_intent JSONB NOT NULL,
    sub_intents JSONB DEFAULT '[]',
    execution_plan INTEGER[] DEFAULT '{}',
    entities JSONB DEFAULT '{}',
    required_entities JSONB DEFAULT '{}',
    optional_entities JSONB DEFAULT '{}',
    missing_required TEXT[] DEFAULT '{}',
    confidence JSONB NOT NULL,
    ambiguity_score FLOAT NOT NULL,
    requires_clarification BOOLEAN DEFAULT false,
    fallback_strategy JSONB,
    resolved_from_context JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_intent_results_query_id ON intent_results(query_id);
CREATE INDEX idx_intent_results_query_type ON intent_results(query_type);

-- Documents table with vector embeddings
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    lob TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),  -- OpenAI ada-002 dimension
    chunk_index INTEGER,
    parent_document_id UUID REFERENCES documents(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_documents_source ON documents(source);
CREATE INDEX idx_documents_lob ON documents(lob);
CREATE INDEX idx_documents_parent ON documents(parent_document_id);
CREATE INDEX idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Answers table
CREATE TABLE answers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID NOT NULL REFERENCES queries(id),
    text TEXT NOT NULL,
    sources TEXT[] DEFAULT '{}',
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_used TEXT NOT NULL,
    generation_time_ms INTEGER NOT NULL,
    provenance JSONB DEFAULT '{}',
    prompt_template_id TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_answers_query_id ON answers(query_id);
CREATE INDEX idx_answers_model_used ON answers(model_used);
CREATE INDEX idx_answers_created_at ON answers(created_at DESC);

-- Evaluation results table
CREATE TABLE evaluation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    answer_id UUID NOT NULL REFERENCES answers(id),
    query_id UUID NOT NULL REFERENCES queries(id),
    
    -- 7 RAG Characteristics
    faithfulness FLOAT NOT NULL CHECK (faithfulness >= 0 AND faithfulness <= 1),
    relevance FLOAT NOT NULL CHECK (relevance >= 0 AND relevance <= 1),
    correctness FLOAT NOT NULL CHECK (correctness >= 0 AND correctness <= 1),
    coverage FLOAT NOT NULL CHECK (coverage >= 0 AND coverage <= 1),
    consistency FLOAT NOT NULL CHECK (consistency >= 0 AND consistency <= 1),
    freshness FLOAT NOT NULL CHECK (freshness >= 0 AND freshness <= 1),
    traceability FLOAT NOT NULL CHECK (traceability >= 0 AND traceability <= 1),
    
    overall_score FLOAT NOT NULL CHECK (overall_score >= 0 AND overall_score <= 1),
    needs_review BOOLEAN DEFAULT false,
    review_reasons TEXT[] DEFAULT '{}',
    evaluation_time_ms INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_evaluation_results_answer_id ON evaluation_results(answer_id);
CREATE INDEX idx_evaluation_results_query_id ON evaluation_results(query_id);
CREATE INDEX idx_evaluation_results_needs_review ON evaluation_results(needs_review);
CREATE INDEX idx_evaluation_results_overall_score ON evaluation_results(overall_score DESC);

-- Review feedback table
CREATE TABLE review_feedback (
    review_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    answer_id UUID NOT NULL REFERENCES answers(id),
    query_id UUID NOT NULL REFERENCES queries(id),
    decision TEXT NOT NULL CHECK (decision IN ('approve', 'reject')),
    corrected_routing TEXT,
    corrected_prompts TEXT,
    corrected_answer TEXT,
    feedback_text TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('routing', 'answer_quality', 'relevance', 'other')),
    reviewer_id TEXT NOT NULL,
    time_to_review_seconds INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_review_feedback_answer_id ON review_feedback(answer_id);
CREATE INDEX idx_review_feedback_query_id ON review_feedback(query_id);
CREATE INDEX idx_review_feedback_decision ON review_feedback(decision);
CREATE INDEX idx_review_feedback_category ON review_feedback(category);
CREATE INDEX idx_review_feedback_reviewer_id ON review_feedback(reviewer_id);

-- Traces table for observability
CREATE TABLE traces (
    trace_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID NOT NULL REFERENCES queries(id),
    user_id TEXT NOT NULL,
    
    -- Pipeline stage timings
    intent_parsing_ms INTEGER,
    routing_decision JSONB,
    routing_ms INTEGER,
    retrieval_strategy TEXT,
    retrieval_ms INTEGER,
    documents_retrieved INTEGER DEFAULT 0,
    generation_ms INTEGER,
    evaluation_ms INTEGER,
    
    -- Results
    answer_generated BOOLEAN DEFAULT false,
    evaluation_scores JSONB,
    human_feedback_id UUID REFERENCES review_feedback(review_id),
    
    -- Metadata
    total_time_ms INTEGER NOT NULL,
    llm_tokens_used INTEGER DEFAULT 0,
    llm_cost_usd FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_traces_query_id ON traces(query_id);
CREATE INDEX idx_traces_user_id ON traces(user_id);
CREATE INDEX idx_traces_created_at ON traces(created_at DESC);
CREATE INDEX idx_traces_total_time_ms ON traces(total_time_ms);

-- Routing decisions table
CREATE TABLE routing_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id UUID NOT NULL REFERENCES traces(trace_id),
    retriever TEXT NOT NULL,
    guideline_matched TEXT,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    reasoning TEXT NOT NULL,
    decision_type TEXT NOT NULL CHECK (decision_type IN ('normal', 'fallback', 'human_review', 'clarification_needed')),
    fallback_used BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_routing_decisions_trace_id ON routing_decisions(trace_id);
CREATE INDEX idx_routing_decisions_decision_type ON routing_decisions(decision_type);

-- Agent tasks table
CREATE TABLE agent_tasks (
    task_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    input_data JSONB NOT NULL,
    dependencies UUID[] DEFAULT '{}',
    priority INTEGER DEFAULT 0,
    timeout_seconds INTEGER DEFAULT 30,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_agent_tasks_agent_name ON agent_tasks(agent_name);
CREATE INDEX idx_agent_tasks_status ON agent_tasks(status);
CREATE INDEX idx_agent_tasks_created_at ON agent_tasks(created_at DESC);

-- Agent results table
CREATE TABLE agent_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES agent_tasks(task_id),
    agent_name TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    output_data JSONB DEFAULT '{}',
    error_message TEXT,
    execution_time_ms INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_agent_results_task_id ON agent_results(task_id);
CREATE INDEX idx_agent_results_success ON agent_results(success);

-- Prompt templates table (for Agent Lightning)
CREATE TABLE prompt_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    template TEXT NOT NULL,
    version INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    performance_metrics JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_prompt_templates_agent_name ON prompt_templates(agent_name);
CREATE INDEX idx_prompt_templates_is_active ON prompt_templates(is_active);

-- Guidelines table (for Parlant routing)
CREATE TABLE guidelines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    condition TEXT NOT NULL,
    action TEXT NOT NULL,
    priority INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT true,
    performance_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_guidelines_priority ON guidelines(priority);
CREATE INDEX idx_guidelines_is_active ON guidelines(is_active);

CREATE TABLE feedback_traces (
    id UUID PRIMARY KEY,
    query_id UUID NOT NULL,
    query_text TEXT NOT NULL,
    routing_decision TEXT,
    retrieval_strategy TEXT,
    answer_text TEXT,
    evaluation_scores JSONB,
    human_feedback JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON feedback_traces (created_at DESC);
CREATE INDEX ON feedback_traces ((human_feedback->>'decision'));
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Orchestrator Task Decomposition
*For any* user query, the Orchestrator Agent should decompose it into a sequence of agent-specific tasks that collectively address the query intent.
**Validates: Requirements 1.1**

### Property 2: Agent Result Routing
*For any* completed agent task, the system should route results to the next appropriate agent according to workflow dependencies.
**Validates: Requirements 1.2**

### Property 3: Parallel Execution Aggregation
*For any* set of agents executing in parallel, the system should aggregate all results before proceeding to the next workflow stage.
**Validates: Requirements 1.3**

### Property 4: Failure Handling
*For any* agent failure, the system should either retry the operation or escalate to Human Review based on failure type and retry count.
**Validates: Requirements 1.4**

### Property 5: Microsoft Agent Framework Runtime Configurability
*For any* agent communication, the system should support both SingleThreadedAgentRuntime and GrpcWorkerAgentRuntime modes via configurable runtime.
**Validates: Requirements 16.1, 16.2, 16.3**

### Property 5a: Message Type Safety
*For any* agent communication, messages should be transmitted with validated Pydantic/dataclass schemas via Microsoft Agent Framework.
**Validates: Requirements 1.5**

### Property 6: Intent Extraction Completeness
*For any* natural language query, the Intent Parser should return intent, entities, and confidence score.
**Validates: Requirements 2.1**

### Property 7: Ambiguity Detection
*For any* query with ambiguous intent, the system should request user clarification before proceeding with retrieval.
**Validates: Requirements 2.2**

### Property 8: Low Confidence Routing
*For any* intent with confidence below threshold, the system should route to fallback handling or human review.
**Validates: Requirements 2.3**

### Property 9: SQL Intent Detection
*For any* query requiring data analysis, the system should detect SQL-generation intent and route to SQLCoder.
**Validates: Requirements 2.4**

### Property 10: Data Story Event Emission
*For any* query matching data story patterns, the system should emit a data story suggestion event.
**Validates: Requirements 2.5**

### Property 11: Semantic Search Execution
*For any* retrieval operation, the Knowledge Retriever should execute embedding-based semantic search.
**Validates: Requirements 3.1**

### Property 12: Metadata Filtering Application
*For any* query with structured constraints, the system should apply metadata filtering using ClickHouse schema.
**Validates: Requirements 3.2**

### Property 13: Pattern Matching Strategy
*For any* query requiring exact pattern matching, the system should use guided grep for heuristic refinement.
**Validates: Requirements 3.3**

### Property 14: Result Deduplication
*For any* set of retrieval results from multiple strategies, the system should merge and deduplicate candidate chunks.
**Validates: Requirements 3.4**

### Property 15: Optional Reranking
*For any* merged candidate set, the system should support optional reranking using Cohere or LLM-as-a-Judge.
**Validates: Requirements 3.5**

### Property 16: Guideline Priority Ordering
*For any* query evaluated by Parlant, guidelines should be applied in priority order with deterministic matching.
**Validates: Requirements 4.1**

### Property 17: Routing Decision Logging
*For any* matched guideline, the system should log the guideline name and routing decision to Langfuse.
**Validates: Requirements 4.2**

### Property 18: Fallback Routing
*For any* query with no matching guidelines, the system should route to a fallback retriever.
**Validates: Requirements 4.3**

### Property 19: Routing Traceability
*For any* routing decision, the system should create a Langfuse trace with decision reasoning.
**Validates: Requirements 4.4**

### Property 20: Guideline Version Control
*For any* guideline update, the system should version control changes and support A/B testing.
**Validates: Requirements 4.5**

### Property 21: Optimized Prompt Usage
*For any* answer generation, the Answer Generator should use optimized prompts from Agent Lightning.
**Validates: Requirements 5.1**

### Property 22: Provenance Metadata Inclusion
*For any* assembled context, the system should include source, timestamp, and confidence metadata.
**Validates: Requirements 5.2**

### Property 23: Adaptive Model Selection
*For any* answer generation task, the system should select LLM model based on task complexity.
**Validates: Requirements 5.3**

### Property 24: Feedback-Driven Prompt Updates
*For any* feedback received by Agent Lightning, the system should update prompt templates using reinforcement learning.
**Validates: Requirements 5.4**

### Property 25: Prompt Deployment
*For any* optimized prompt template, the system should deploy updates to Parlant for production use.
**Validates: Requirements 5.5**

### Property 26: Faithfulness Scoring
*For any* generated answer, the Evaluator Agent should compute a faithfulness score using RAGAS metrics.
**Validates: Requirements 6.1**

### Property 27: Comprehensive Quality Assessment
*For any* answer evaluation, the system should assess all seven RAG characteristics (faithfulness, relevance, correctness, coverage, consistency, freshness, traceability).
**Validates: Requirements 6.2**

### Property 28: Quality-Based Review Routing
*For any* answer with quality scores below threshold, the system should route to Human Review.
**Validates: Requirements 6.3**

### Property 29: High-Quality Answer Delivery
*For any* answer with quality scores above threshold, the system should return the answer with confidence indicators.
**Validates: Requirements 6.4**

### Property 30: Evaluation Metrics Logging
*For any* completed evaluation, the system should log all quality metrics to Langfuse.
**Validates: Requirements 6.5**

### Property 31: Review Request Presentation
*For any* answer requiring review, the Human Review Agent should present answer, context, and quality scores.
**Validates: Requirements 7.1**

### Property 32: Approval Handling
*For any* reviewer approval, the system should deliver the answer to the user and log the approval decision.
**Validates: Requirements 7.2**

### Property 33: Rejection Feedback Capture
*For any* reviewer rejection, the system should capture corrected routing, prompts, and detailed feedback.
**Validates: Requirements 7.3**

### Property 34: Feedback Persistence
*For any* captured human feedback, the system should store complete traces in LightningStore.
**Validates: Requirements 7.4**

### Property 35: Pattern-Based Policy Updates
*For any* feedback processed by Agent Lightning, the system should identify patterns and update routing policies.
**Validates: Requirements 7.5**

### Property 36: Message Type Discrimination
*For any* frontend message, the system should correctly discriminate between AI messages, direct UI updates, and database operations.
**Validates: Requirements 8.1**

### Property 37: SSE Streaming for AI Messages
*For any* AI message processing, the system should stream events via Server-Sent Events (SSE).
**Validates: Requirements 8.2**

### Property 38: Direct UI Update Optimization
*For any* direct UI update, the system should execute immediately without LLM invocation.
**Validates: Requirements 8.3**

### Property 39: Chart Highlighting Events
*For any* chart highlighting request, the system should emit events with chartIds for frontend rendering.
**Validates: Requirements 8.4**

### Property 40: Data Story Coordination
*For any* generated data story, the system should coordinate step-by-step playback with audio and chart highlighting.
**Validates: Requirements 8.5**

### Property 41: Data Story Intent Event
*For any* detected data story intent, the system should emit a data story suggestion event.
**Validates: Requirements 9.1**

### Property 42: Strategic Commentary Generation
*For any* accepted data story suggestion, the system should generate strategic commentary with chart references.
**Validates: Requirements 9.2**

### Property 43: Data Story Structure Completeness
*For any* generated data story, steps should include talking points and chart IDs.
**Validates: Requirements 9.3**

### Property 44: Audio Generation
*For any* audio request, the system should generate narration using Azure OpenAI TTS or local TTS.
**Validates: Requirements 9.4**

### Property 45: TTS Fallback
*For any* audio generation failure, the system should fallback to browser TTS for accessibility.
**Validates: Requirements 9.5**

### Property 46: Operation Tracing
*For any* agent operation, the system should create a Langfuse trace with nested spans.
**Validates: Requirements 10.1**

### Property 47: LLM Metrics Logging
*For any* LLM call, the system should log token usage, latency, and cost.
**Validates: Requirements 10.2**

### Property 48: Retrieval Performance Tracking
*For any* retrieval operation, the system should track semantic scores, metadata filtering effectiveness, and reranker performance.
**Validates: Requirements 10.3**

### Property 49: Evaluation Metrics Logging
*For any* quality evaluation, the system should log faithfulness, relevance, and correctness scores.
**Validates: Requirements 10.4**

### Property 50: Trace Export
*For any* completed trace, the system should export to OpenTelemetry Collector for unified dashboards.
**Validates: Requirements 10.5**

### Property 51: LlamaIndex Chunking
*For any* ingested document, the system should use LlamaIndex for chunking with semantic splitters.
**Validates: Requirements 11.1**

### Property 52: Embedding Generation
*For any* created chunk, the system should generate embeddings using Azure OpenAI or HuggingFace models.
**Validates: Requirements 11.2**

### Property 53: Vector Storage with Metadata
*For any* generated embedding, the system should store vectors in pgvector with complete metadata.
**Validates: Requirements 11.3**

### Property 54: Post-Indexing Coverage Evaluation
*For any* completed indexing operation, the system should trigger coverage evaluation to identify knowledge gaps.
**Validates: Requirements 11.4**

### Property 55: Schema Change Re-indexing
*For any* detected schema change, the system should invalidate cache and re-index affected documents.
**Validates: Requirements 11.5**

### Property 56: SQL Routing with Schema Context
*For any* data analysis request, the system should route to SQLCoder with ClickHouse MCP schema context.
**Validates: Requirements 12.1**

### Property 57: SQL Schema Validation
*For any* generated SQL, the system should validate against real schema to prevent hallucinations.
**Validates: Requirements 12.2**

### Property 58: Read-Only SQL Execution
*For any* SQL execution, the system should enforce read-only mode via MCP for safety.
**Validates: Requirements 12.3**

### Property 59: Structured Result Formatting
*For any* SQL execution completion, the system should return structured results with metadata.
**Validates: Requirements 12.4**

### Property 60: Schema Caching
*For any* schema access, the system should use in-memory cache (refreshed every 60 seconds) for performance.
**Validates: Requirements 12.5**

### Property 61: Persona-Aware Visualization
*For any* visualization request, the system should consider user persona (analyst, executive, manager, stakeholder).
**Validates: Requirements 13.1**

### Property 62: ECharts Persona Recommendations
*For any* chart generation, the system should use ECharts MCP with persona-aware recommendations.
**Validates: Requirements 13.2**

### Property 63: Analyst Visualization Preferences
*For any* analyst persona, the system should prefer detailed visualizations (scatter, heatmap, detailed tables).
**Validates: Requirements 13.3**

### Property 64: Executive Visualization Preferences
*For any* executive persona, the system should prefer summary visualizations (KPI cards, line, bar).
**Validates: Requirements 13.4**

### Property 65: Preference Learning
*For any* user feedback on visualizations, the system should learn preferences and adjust future recommendations.
**Validates: Requirements 13.5**

### Property 66: MCP Message Encryption
*For any* agent communication, the system should encrypt all MCP channel messages.
**Validates: Requirements 14.1**

### Property 67: Provenance Logging
*For any* operation, the system should log with complete provenance (user, timestamp, agent, version).
**Validates: Requirements 14.2**

### Property 68: PII Masking
*For any* detected PII, the system should apply masking or redaction policies.
**Validates: Requirements 14.3**

### Property 69: Role-Based Access Control
*For any* human reviewer data access, the system should enforce role-based access control.
**Validates: Requirements 14.4**

### Property 70: Audit Trail Completeness
*For any* audit log query, the system should provide complete traceability for compliance reporting.
**Validates: Requirements 14.5**

### Property 71: Feedback Trace Storage
*For any* collected human feedback, the system should store traces with corrections in LightningStore.
**Validates: Requirements 15.1**

### Property 72: Error Pattern Identification
*For any* Agent Lightning pattern analysis, the system should identify routing errors and prompt weaknesses.
**Validates: Requirements 15.2**

### Property 73: Optimization Deployment
*For any* generated optimization, the system should update Parlant guidelines and prompt templates.
**Validates: Requirements 15.3**

### Property 74: Statistical A/B Testing
*For any* A/B test, the system should compare old and new policies with statistical significance.
**Validates: Requirements 15.4**

### Property 75: Automated Policy Promotion
*For any* validated improvement, the system should promote optimized policies to production automatically.
**Validates: Requirements 15.5**

### Property 76: Microsoft Agent Framework A2A Support
*For any* agent-to-agent communication, the system should support both SingleThreadedAgentRuntime and GrpcWorkerAgentRuntime communication channels.
**Validates: Requirements 16.1**

### Property 77: SingleThreadedAgentRuntime Performance
*For any* single-threaded communication, the system should provide high-performance message passing without serialization overhead via SingleThreadedAgentRuntime.
**Validates: Requirements 16.2**

### Property 78: GrpcWorkerAgentRuntime Distributed Communication
*For any* distributed communication, the system should support inter-process and distributed agent communication via GrpcWorkerAgentRuntime with gRPC protocol.
**Validates: Requirements 16.3**

### Property 79: AG-UI Protocol Implementation
*For any* human interaction requirement, the system should implement ag-ui protocol for agent-to-human integration.
**Validates: Requirements 16.4**

### Property 80: Factory Pattern for Adapters
*For any* new integration module, the system should provide Factory Pattern for creating different types of integration adapters.
**Validates: Requirements 16.5**

### Property 81: Adapter Pattern for External Systems
*For any* connection to diverse external systems, the system should implement Adapter Pattern for seamless integration.
**Validates: Requirements 16.6**

### Property 82: Plug-and-Play Module Loading
*For any* integration module development, the system should support plug-and-play loading without system restart.
**Validates: Requirements 16.7**

### Property 83: External Configuration Support
*For any* configuration change, the system should allow integration modules to be configured via external configuration files.
**Validates: Requirements 16.8**

### Property 84: Extension Points
*For any* extensibility requirement, the system should provide clear extension points for adding new integration capabilities.
**Validates: Requirements 16.9**

### Property 85: Agent Registry for Discovery
*For any* agent discovery need, the system should maintain an agent registry for dynamic discovery of available agents.
**Validates: Requirements 16.10**


## Error Handling

### Error Categories

1. **Agent Failures**: Timeout, crash, invalid output
2. **Retrieval Failures**: No documents found, vector DB unavailable
3. **Generation Failures**: LLM timeout, rate limit, invalid response
4. **Evaluation Failures**: Metric computation error, threshold violation
5. **System Failures**: Database connection, MCP channel error, network issue

### Error Handling Strategies

**Retry Logic**:
- Exponential backoff for transient failures (3 retries max)
- Circuit breaker pattern for persistent failures
- Fallback to alternative agents/models when available

**Escalation Paths**:

```python
class ErrorHandler:
    async def handle_agent_failure(self, agent: str, error: Exception) -> Response:
        if is_transient(error) and retry_count < 3:
            return await retry_with_backoff(agent)
        elif has_fallback(agent):
            return await execute_fallback(agent)
        else:
            return await escalate_to_human_review(agent, error)
    
    async def handle_retrieval_failure(self, error: Exception) -> Response:
        # Try alternative retrieval strategies
        if semantic_search_failed:
            return await try_metadata_filter()
        elif all_strategies_failed:
            return await return_empty_with_explanation()
    
    async def handle_generation_failure(self, error: Exception) -> Response:
        # Fallback to simpler model or cached response
        if rate_limit_error:
            return await use_fallback_model()
        elif timeout_error:
            return await use_cached_similar_response()
```

**Logging and Alerting**:
- All errors logged to Langfuse with full context
- Critical errors trigger alerts to on-call team
- Error patterns analyzed for system improvements

## Testing Strategy

### Unit Testing

**Framework**: pytest with async support

**Coverage Areas**:
- Agent interface contracts
- Data model validation
- MCP message serialization/deserialization
- Routing logic (Parlant guidelines)
- Evaluation metrics computation

**Example Unit Tests**:
```python
@pytest.mark.asyncio
async def test_intent_parser_extracts_entities():
    """Test that intent parser extracts entities from query"""
    parser = IntentParserAgent()
    result = await parser.parse_intent("Show me orders from customer ABC123")
    
    assert result.intent == "order_lookup"
    assert "customer_id" in result.entities
    assert result.entities["customer_id"] == "ABC123"
    assert result.confidence > 0.8

@pytest.mark.asyncio
async def test_evaluator_computes_all_metrics():
    """Test that evaluator computes all 7 RAG characteristics"""
    evaluator = EvaluatorAgent()
    result = await evaluator.evaluate(query, context, answer)
    
    assert result.faithfulness is not None
    assert result.relevance is not None
    assert result.correctness is not None
    assert result.coverage is not None
    assert result.consistency is not None
    assert result.freshness is not None
    assert result.traceability is not None
```

### Property-Based Testing

**Framework**: Hypothesis (Python property-based testing library)

**Configuration**: Each property test runs 100 iterations minimum

**Test Tagging**: Each property test tagged with format `**Feature: mcp-multi-agent-rag, Property {number}: {property_text}**`

**Example Property Tests**:

```python
from hypothesis import given, strategies as st

@given(query=st.text(min_size=1, max_size=500))
@pytest.mark.asyncio
async def test_property_1_orchestrator_task_decomposition(query):
    """
    **Feature: mcp-multi-agent-rag, Property 1: Orchestrator Task Decomposition**
    For any user query, the Orchestrator Agent should decompose it into 
    a sequence of agent-specific tasks.
    """
    orchestrator = OrchestratorAgent()
    tasks = await orchestrator.decompose_task(query)
    
    # Property: decomposition always produces at least one task
    assert len(tasks) > 0
    # Property: all tasks have valid agent assignments
    assert all(task.agent in VALID_AGENTS for task in tasks)
    # Property: tasks have execution order
    assert all(hasattr(task, 'order') for task in tasks)

@given(
    intent=st.text(min_size=1),
    confidence=st.floats(min_value=0.0, max_value=1.0)
)
@pytest.mark.asyncio
async def test_property_8_low_confidence_routing(intent, confidence):
    """
    **Feature: mcp-multi-agent-rag, Property 8: Low Confidence Routing**
    For any intent with confidence below threshold, the system should 
    route to fallback handling or human review.
    """
    threshold = 0.7
    router = ParlantRouter()
    
    intent_result = IntentResult(
        intent=intent,
        entities={},
        confidence=confidence,
        query_type=QueryType.GENERAL_QA,
        requires_clarification=False
    )
    
    decision = router.route_based_on_confidence(intent_result)
    
    # Property: low confidence always triggers fallback or review
    if confidence < threshold:
        assert decision.route_type in ["fallback", "human_review"]
    else:
        assert decision.route_type == "normal"

@given(
    documents=st.lists(
        st.builds(Document, 
            id=st.uuids(),
            content=st.text(min_size=10),
            source=st.text(min_size=1),
            confidence=st.floats(min_value=0.0, max_value=1.0)
        ),
        min_size=1,
        max_size=20
    )
)
@pytest.mark.asyncio
async def test_property_14_result_deduplication(documents):
    """
    **Feature: mcp-multi-agent-rag, Property 14: Result Deduplication**
    For any set of retrieval results from multiple strategies, the system 
    should merge and deduplicate candidate chunks.
    """
    retriever = KnowledgeRetrieverAgent()
    
    # Introduce duplicates
    documents_with_dupes = documents + documents[:len(documents)//2]
    
    merged = await retriever.merge_and_deduplicate(documents_with_dupes)
    
    # Property: no duplicate document IDs
    doc_ids = [doc.id for doc in merged]
    assert len(doc_ids) == len(set(doc_ids))
    
    # Property: all original documents preserved
    original_ids = set(doc.id for doc in documents)
    merged_ids = set(doc.id for doc in merged)
    assert original_ids.issubset(merged_ids)

@given(
    answer=st.builds(Answer,
        id=st.uuids(),
        text=st.text(min_size=10),
        sources=st.lists(st.text(min_size=1), min_size=1),
        confidence=st.floats(min_value=0.0, max_value=1.0)
    ),
    context=st.lists(
        st.builds(Document,
            id=st.uuids(),
            content=st.text(min_size=10),
            source=st.text(min_size=1)
        ),
        min_size=1
    )
)
@pytest.mark.asyncio
async def test_property_27_comprehensive_quality_assessment(answer, context):
    """
    **Feature: mcp-multi-agent-rag, Property 27: Comprehensive Quality Assessment**
    For any answer evaluation, the system should assess all seven RAG 
    characteristics.
    """
    evaluator = EvaluatorAgent()
    query = "test query"
    
    result = await evaluator.evaluate(query, context, answer)
    
    # Property: all 7 characteristics are computed
    assert result.faithfulness is not None
    assert result.relevance is not None
    assert result.correctness is not None
    assert result.coverage is not None
    assert result.consistency is not None
    assert result.freshness is not None
    assert result.traceability is not None
    
    # Property: all scores are in valid range [0, 1]
    scores = [
        result.faithfulness, result.relevance, result.correctness,
        result.coverage, result.consistency, result.freshness, result.traceability
    ]
    assert all(0.0 <= score <= 1.0 for score in scores)
```

### Integration Testing

**Scope**: End-to-end workflows across multiple agents

**Test Scenarios**:
1. Complete query → answer pipeline with high-quality response
2. Low-quality answer → human review → feedback loop
3. SQL generation → ClickHouse execution → result formatting
4. Data story generation → audio narration → chart highlighting
5. Multi-LOB retrieval → result merging → reranking

**Example Integration Test**:
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_query_pipeline():
    """Test complete pipeline from query to answer"""
    # Setup
    orchestrator = OrchestratorAgent()
    query = "What are the top 10 customers by revenue this quarter?"
    
    # Execute
    answer = await orchestrator.process_query(query, user_id="test_user")
    
    # Verify
    assert answer.text is not None
    assert len(answer.sources) > 0
    assert answer.confidence > 0.7
    assert answer.provenance["routing_decision"] is not None
    assert answer.provenance["retrieval_strategy"] is not None
    
    # Verify tracing
    traces = await langfuse_client.get_traces(query_id=answer.id)
    assert len(traces) > 0
    assert "orchestrator" in traces[0].spans
    assert "intent_parser" in traces[0].spans
    assert "knowledge_retriever" in traces[0].spans
    assert "answer_generator" in traces[0].spans
    assert "evaluator" in traces[0].spans
```

### Performance Testing

**Load Testing**:
- Simulate 100 concurrent users
- Target: 95th percentile latency < 3 seconds
- Target: Throughput > 50 queries/second

**Stress Testing**:
- Gradually increase load until system degradation
- Identify bottlenecks (LLM API, vector DB, database)
- Verify graceful degradation and error handling

**Tools**: Locust for load generation, Grafana for monitoring

## Deployment Strategy

### Infrastructure

**Cloud Platform**: Azure

**Components**:
- **Frontend**: Next.js app on Azure Static Web Apps
- **Backend**: FastAPI on Azure Container Apps
- **Database**: Azure Database for PostgreSQL with pgvector
- **Vector Store**: Azure AI Search or self-hosted pgvector
- **Analytics DB**: ClickHouse on Azure VMs
- **Observability**: Langfuse Cloud or self-hosted
- **LLM**: Azure OpenAI Service

### Containerization

**Docker Images**:
```dockerfile
# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8880"]
```

**Docker Compose** (for local development):
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8880:8880"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/rag_system
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
    depends_on:
      - db
      - clickhouse
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8880
  
  db:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=rag_system
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports:
      - "8123:8123"
    volumes:
      - clickhouse_data:/var/lib/clickhouse

volumes:
  postgres_data:
  clickhouse_data:
```

### CI/CD Pipeline

**GitHub Actions Workflow**:
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio hypothesis
      - name: Run unit tests
        run: pytest tests/unit -v
      - name: Run property tests
        run: pytest tests/property -v --hypothesis-profile=ci
      - name: Run integration tests
        run: pytest tests/integration -v
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build -t rag-backend:${{ github.sha }} ./backend
          docker build -t rag-frontend:${{ github.sha }} ./frontend
      - name: Push to Azure Container Registry
        run: |
          az acr login --name myregistry
          docker push rag-backend:${{ github.sha }}
          docker push rag-frontend:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Azure Container Apps
        run: |
          az containerapp update \
            --name rag-backend \
            --resource-group rag-system-rg \
            --image myregistry.azurecr.io/rag-backend:${{ github.sha }}
```

### Monitoring and Alerting

**Metrics to Monitor**:
- Query latency (p50, p95, p99)
- Agent execution time
- LLM token usage and cost
- Retrieval quality scores
- Evaluation scores (faithfulness, relevance, correctness)
- Error rates by agent
- Human review queue length

**Alerts**:
- Latency > 5 seconds for 5 minutes
- Error rate > 5% for 5 minutes
- Quality scores < 0.7 for 10 consecutive queries
- Human review queue > 50 items
- LLM cost > $100/hour

**Dashboards**:
- Real-time query pipeline visualization
- Agent performance metrics
- Cost tracking and optimization
- Quality trends over time
- Human feedback analysis

## Complete Integration Workflow: Parlant + CopilotKit + Microsoft Agent Framework

### End-to-End Query Processing with Design Thinking

```python
async def process_query_with_design_thinking(query: str, user_id: str) -> Answer:
    """
    Complete RAG pipeline with Parlant guidelines and CopilotKit HIL
    Follows Double Diamond: Discover → Define → Develop → Deliver
    """
    
    # ===== DISCOVER PHASE (Diverge) =====
    # Green Hat: Creative exploration
    # White Hat: Fact gathering
    # Red Hat: Empathy and intent
    
    # Parlant ensures diverse exploration
    interpretations = await parlant_agent.process({
        query,
        phase: "discover",
        # Guidelines automatically enforce:
        # - Green Hat: Generate 3+ diverse interpretations
        # - White Hat: Retrieve supporting evidence
        # - Red Hat: Assess user intent
    })
    
    # CopilotKit HIL: Request clarification if needed
    if interpretations.needsClarification:
        user_selection = await copilotkit.requestClarification({
            interpretations: interpretations.candidates
        })
        interpretations.selected = user_selection
    
    # ===== DEFINE PHASE (Converge) =====
    # Black Hat: Critical validation
    # Blue Hat: Process control
    
    # Parlant ensures critical validation
    validation = await parlant_agent.process({
        interpretations,
        phase: "define",
        # Guidelines automatically enforce:
        # - Black Hat: Critical validation
        # - Blue Hat: Structured selection
    })
    
    # CopilotKit HIL: Validate intent if borderline
    if validation.confidence >= 0.6 and validation.confidence < 0.75:
        user_approval = await copilotkit.validateIntent({
            formal_intent: validation.intent,
            validation_concerns: validation.concerns
        })
        if not user_approval.approved:
            return process_query_with_design_thinking(query, user_id)  # Restart
    
    # ===== DEVELOP PHASE (Diverge) =====
    # Green Hat: Creative alternatives
    # Yellow Hat: Benefit analysis
    # White Hat: Data gathering
    
    # Parlant ensures creative strategy generation
    strategies = await parlant_agent.process({
        formal_intent: validation.intent,
        phase: "develop",
        # Guidelines automatically enforce:
        # - Green Hat: Generate alternative strategies
        # - Yellow Hat: Evaluate benefits
    })
    
    # CopilotKit HIL: Let user select strategy if similar performance
    if strategies.needsUserSelection:
        user_choice = await copilotkit.selectStrategy({
            strategies: strategies.candidates
        })
        strategies.selected = user_choice
    
    # Execute retrieval and generate answers
    answers = await parlant_agent.process({
        strategy: strategies.selected,
        phase: "develop_answers"
    })
    
    # ===== DELIVER PHASE (Converge) =====
    # Black Hat: Quality evaluation
    # Blue Hat: Delivery orchestration
    
    # Parlant ensures quality evaluation
    evaluation = await parlant_agent.process({
        answers,
        phase: "deliver",
        # Guidelines automatically enforce:
        # - Black Hat: Quality evaluation
        # - Blue Hat: Delivery orchestration
    })
    
    # CopilotKit HIL: Request approval if medium quality
    if evaluation.quality_score >= 0.6 and evaluation.quality_score < 0.8:
        user_approval = await copilotkit.approveAnswer({
            answer: evaluation.best_answer,
            quality_metrics: evaluation.metrics,
            concerns: evaluation.concerns
        })
        
        if not user_approval.approved:
            return process_query_with_design_thinking(query, user_id)  # Restart
        
        if user_approval.edited_answer:
            evaluation.best_answer = user_approval.edited_answer
    
    # Deliver final answer
    return evaluation.best_answer
```

### Integration Flow Diagram

```mermaid
graph TD
    subgraph "Discover Phase (Diverge)"
        A[User Query] --> B{Parlant: Ambiguity Check}
        B -->|Clear| C[ReAct: Generate Interpretations]
        B -->|Ambiguous| D[CopilotKit HIL: Request Clarification]
        D --> C
        C --> E{Parlant: Diversity Check}
        E -->|Pass| F[Multiple Interpretations]
        E -->|Fail| C
        
        style B fill:#FFE4B5
        style D fill:#87CEEB
        style E fill:#FFE4B5
    end
    
    subgraph "Define Phase (Converge)"
        F --> G[CoT: Validate Each]
        G --> H{Parlant: Confidence Check}
        H -->|High 0.75+| I[Select Best Intent]
        H -->|Borderline 0.6-0.75| J[CopilotKit HIL: Validate Intent]
        H -->|Low <0.6| K[Parlant: Canned Response]
        J --> I
        K --> A
        I --> L[Formal Intent]
        
        style H fill:#FFE4B5
        style J fill:#87CEEB
        style K fill:#FFE4B5
    end
    
    subgraph "Develop Phase (Diverge)"
        L --> M[ReAct: Generate Strategies]
        M --> N[Execute Strategies]
        N --> O{Parlant: Performance Check}
        O -->|Clear Winner| P[Select Strategy]
        O -->|Similar Scores| Q[CopilotKit HIL: User Selects]
        Q --> P
        P --> R[ToT: Generate Answers]
        R --> S[Answer Candidates]
        
        style O fill:#FFE4B5
        style Q fill:#87CEEB
    end
    
    subgraph "Deliver Phase (Converge)"
        S --> T[CoT: Evaluate Quality]
        T --> U{Parlant: Quality Gate}
        U -->|High 0.8+| V[Direct Delivery]
        U -->|Medium 0.6-0.8| W[CopilotKit HIL: Approve Answer]
        U -->|Low <0.6| X[Parlant: Route to Review]
        W -->|Approved| V
        W -->|Rejected| Y[Regenerate]
        X --> Z[Human Review]
        V --> AA[Final Answer]
        
        style U fill:#FFE4B5
        style W fill:#87CEEB
        style X fill:#FFE4B5
    end
```

**Legend:**
- 🟡 **Parlant (Orange)**: Behavioral guidelines, quality gates, routing rules
- 🔵 **CopilotKit HIL (Blue)**: Human intervention points, approval workflows
- ⚪ **Agent Patterns (White)**: ReAct, CoT, ToT, Plan-and-Execute

### Benefits of Complete Integration

**1. Reliability (Parlant)**
- Ensured compliance with design thinking principles
- No prompt engineering required
- Guaranteed Six Thinking Hats coverage
- Explainable guideline matching

**2. Control (CopilotKit)**
- Strategic human oversight at critical points
- Quality gates prevent low-quality outputs
- Ambiguity resolution improves accuracy
- User preferences guide agent behavior

**3. Transparency**
- Parlant explains which guidelines were matched
- CopilotKit shows intermediate agent state
- Users understand agent reasoning
- Full visibility into decision-making

**4. Efficiency**
- Agents handle routine exploration/validation
- Humans intervene only when needed
- Optimal balance of speed and quality
- Reduced cognitive load on users

**5. Scalability (Microsoft Agent Framework)**
- SingleThreadedAgentRuntime for single-process deployment
- GrpcWorkerAgentRuntime for distributed scaling
- Configuration-driven runtime selection
- No code changes needed to scale

## Security Considerations

### Authentication and Authorization

- **User Authentication**: OAuth 2.0 / OpenID Connect
- **API Authentication**: JWT tokens with short expiration
- **Service-to-Service**: Managed identities (Azure AD)
- **RBAC**: Role-based access for human reviewers

### Data Protection

- **Encryption in Transit**: TLS 1.3 for all communications
- **Encryption at Rest**: Azure Storage encryption
- **PII Detection**: Automated scanning and masking
- **Data Retention**: Configurable retention policies (30/90/365 days)

### Compliance

- **Audit Logging**: Complete provenance for all operations
- **GDPR**: Right to deletion, data portability
- **SOC 2**: Access controls, change management, monitoring
- **HIPAA** (if applicable): PHI handling, BAA agreements

## Scalability Considerations

### Horizontal Scaling

- **Stateless Agents**: All agents designed for horizontal scaling
- **Load Balancing**: Azure Load Balancer for backend instances
- **Database Scaling**: Read replicas for PostgreSQL
- **Vector DB Scaling**: Sharding for large document collections

### Caching Strategy

- **Schema Cache**: In-memory cache for ClickHouse schema (60s TTL)
- **Embedding Cache**: Redis for frequently accessed embeddings
- **Response Cache**: Cache similar queries with semantic matching
- **Prompt Cache**: Cache optimized prompts from Agent Lightning

### Performance Optimization

- **Batch Processing**: Batch embedding generation for ingestion
- **Parallel Execution**: Concurrent retrieval strategies
- **Connection Pooling**: Database connection pools
- **Async I/O**: Full async/await for non-blocking operations

## Future Enhancements

### Phase 2 Features

1. **Multi-Modal RAG**: Support for images, videos, audio in retrieval
2. **Graph RAG**: Knowledge graph integration for relationship queries
3. **Streaming Generation**: Token-by-token streaming for long answers
4. **Multi-Language Support**: Internationalization for queries and answers
5. **Advanced Personas**: Custom persona creation and management

### Phase 3 Features

1. **Federated Learning**: Privacy-preserving model updates
2. **Active Learning**: Intelligent sample selection for human review
3. **Explainable AI**: Detailed reasoning traces for all decisions
4. **Multi-Tenant**: Isolated environments for different organizations
5. **Edge Deployment**: On-premise deployment for sensitive data


## Design Thinking Integration: Parlant + CopilotKit HIL

### Overview

The system integrates **Parlant behavioral guidelines** and **CopilotKit Human-in-the-Loop (HIL)** patterns to ensure reliable agent behavior and strategic human oversight. This integration maps to the **Double Diamond Design Process** (Discover → Define → Develop → Deliver) and **Six Thinking Hats** framework.

### Parlant's Role: Ensured Compliance

Parlant provides behavioral guidelines that agents reliably follow across all phases:

**Key Parlant Features:**
1. **Behavioral Guidelines**: Natural language rules contextually matched and enforced
2. **Journeys**: Structured progression through workflow phases
3. **Canned Responses**: Eliminate hallucinations in critical scenarios
4. **Domain Adaptation**: Glossary terms and context variables
5. **Guardrails**: Quality thresholds and risk assessment gates

### CopilotKit HIL's Role: Strategic Human Control

CopilotKit provides human intervention points where judgment adds value:

**Key CopilotKit Features:**
1. **renderAndWaitForResponse**: Synchronous human approval workflows
2. **Generative UI (render)**: Asynchronous information display
3. **Agentic State Streaming**: Real-time agent state visibility
4. **Shared State (useCoAgent)**: Bidirectional app-agent communication

### Integration by RAG Pipeline Phase

#### Phase 1: Intent Parsing (Discover)

**Parlant Guidelines:**
```python
# Green Hat: Creative exploration
await agent.create_guideline(
    condition="User query is ambiguous or has multiple interpretations",
    action="Generate at least 3 diverse interpretations covering different domains (business, technical, emotional). Ensure diversity score above 0.6",
    tools=[generate_interpretations, retrieve_evidence, assess_user_intent]
)

# White Hat: Fact gathering
await agent.create_guideline(
    condition="Exploring query interpretations",
    action="Retrieve supporting evidence from knowledge base for each interpretation",
    tools=[retrieve_evidence]
)

# Domain terminology
await agent.create_glossary_term(
    term="business performance",
    definition="Metrics including revenue, growth, customer satisfaction, and operational efficiency"
)
```

**CopilotKit HIL:**
```typescript
// Clarification request when interpretations have similar confidence
useCopilotAction({
  name: "request_clarification",
  description: "Request user clarification when interpretations have similar confidence scores",
  parameters: [
    {
      name: "interpretations",
      type: "object[]",
      description: "Top interpretations with similar confidence",
      required: true,
    },
  ],
  renderAndWaitForResponse: ({ args, respond }) => {
    return (
      <InterpretationSelector
        interpretations={args.interpretations}
        onSelect={(selected) => respond?.({ 
          selectedInterpretation: selected,
          userConfirmed: true 
        })}
        onProvideMore={() => respond?.({ 
          needsMoreContext: true 
        })}
      />
    );
  },
});

// Show exploration progress
useCoAgentStateRender({
  name: "intent_parser_agent",
  render: ({ state }) => (
    <ExplorationProgress
      interpretations={state.interpretations}
      currentIteration={state.iteration}
      diversityScore={state.diversity_score}
    />
  ),
});
```

#### Phase 2: Routing (Define)

**Parlant Guidelines:**
```python
# Black Hat: Critical validation
await agent.create_guideline(
    condition="Validating interpretation confidence",
    action="Apply critical thinking: check factual grounding, identify risks and ambiguities, assess completeness. Reject interpretations with confidence below 0.6",
    tools=[validate_interpretation, assess_risks]
)

# Blue Hat: Process control
await agent.create_guideline(
    condition="Selecting best interpretation from validated candidates",
    action="Apply weighted scoring (confidence 40%, evidence quality 30%, completeness 20%, risk level 10%). Select highest scoring interpretation",
    tools=[calculate_weighted_score, formalize_intent]
)

# Canned response for low confidence
await agent.create_canned_response(
    condition="All interpretations have confidence below 0.7",
    response="I found multiple possible interpretations, but I'm not confident enough to proceed. Could you provide more context about: {missing_context_areas}?"
)
```

**CopilotKit HIL:**
```typescript
// Intent validation for borderline confidence
useCopilotAction({
  name: "validate_intent",
  description: "Request user validation when interpretation confidence is borderline (0.6-0.75)",
  parameters: [
    {
      name: "formal_intent",
      type: "object",
      description: "The formalized intent to validate",
      required: true,
    },
    {
      name: "validation_concerns",
      type: "string[]",
      description: "Specific concerns identified by Black Hat analysis",
      required: true,
    },
  ],
  renderAndWaitForResponse: ({ args, respond }) => {
    return (
      <IntentValidationCard
        intent={args.formal_intent}
        concerns={args.validation_concerns}
        onApprove={() => respond?.({ 
          approved: true,
          confidence_boost: 0.15 
        })}
        onReject={() => respond?.({ 
          approved: false,
          feedback: "User rejected interpretation" 
        })}
        onModify={(modifications) => respond?.({ 
          approved: true,
          modified_intent: modifications 
        })}
      />
    );
  },
});
```

#### Phase 3: Retrieval (Develop)

**Parlant Guidelines:**
```python
# Green Hat: Creative alternatives
await agent.create_guideline(
    condition="Generating retrieval strategies for formal intent",
    action="Generate at least 3 alternative retrieval strategies (semantic, metadata-based, hybrid, graph-based)",
    tools=[generate_retrieval_strategies]
)

# Yellow Hat: Benefit analysis
await agent.create_guideline(
    condition="Evaluating retrieval strategy performance",
    action="Score strategies on recall (40%), precision (30%), latency (20%), cost (10%). Identify benefits of each approach",
    tools=[evaluate_benefits, score_strategies]
)

# Context variable for real-time metrics
await agent.create_variable(
    name="retrieval_metrics",
    tool=get_current_retrieval_metrics,
    update_interval="on_strategy_execution"
)
```

**CopilotKit HIL:**
```typescript
// Strategy selection when multiple strategies have similar performance
useCopilotAction({
  name: "select_retrieval_strategy",
  description: "Let user choose retrieval strategy when multiple strategies have similar performance",
  parameters: [
    {
      name: "strategies",
      type: "object[]",
      description: "Retrieval strategies with performance metrics",
      required: true,
    },
  ],
  renderAndWaitForResponse: ({ args, respond }) => {
    return (
      <StrategyComparisonTable
        strategies={args.strategies}
        metrics={["recall", "precision", "latency", "cost"]}
        onSelect={(strategy) => respond?.({ 
          selected_strategy: strategy,
          user_preference: strategy.name 
        })}
        onAutoSelect={() => respond?.({ 
          auto_select: true,
          use_highest_score: true 
        })}
      />
    );
  },
});

// Show retrieval progress
useCoAgentStateRender({
  name: "knowledge_retriever_agent",
  render: ({ state }) => (
    <RetrievalDashboard
      strategies={state.strategy_results}
      showMetrics={true}
      showDocuments={true}
    />
  ),
});
```

#### Phase 4: Evaluation & Delivery (Deliver)

**Parlant Guidelines:**
```python
# Black Hat: Quality evaluation
await agent.create_guideline(
    condition="Evaluating answer candidate quality",
    action="Check faithfulness, relevance, correctness, completeness, consistency. Calculate overall score as average",
    tools=[evaluate_answer_quality]
)

# Blue Hat: Delivery orchestration
await agent.create_guideline(
    condition="Answer quality score is 0.8 or above",
    action="Deliver answer directly with citations",
    tools=[format_answer, add_citations, deliver_to_user]
)

await agent.create_guideline(
    condition="Answer quality score is between 0.6 and 0.8",
    action="Route to human approval before delivery",
    tools=[route_to_human_approval]
)

await agent.create_guideline(
    condition="Answer quality score is below 0.6",
    action="Route to human review with quality concerns",
    tools=[route_to_human_review]
)

# Canned response for low quality
await agent.create_canned_response(
    condition="All answer candidates score below 0.6",
    response="I couldn't generate a high-quality answer with the available information. The main issues are: {quality_issues}. Would you like me to try a different approach?"
)
```

**CopilotKit HIL:**
```typescript
// Answer approval for medium quality or Black Hat concerns
useCopilotAction({
  name: "approve_answer",
  description: "Request human approval for answers with quality score 0.6-0.8 or when Black Hat identifies risks",
  parameters: [
    {
      name: "answer",
      type: "string",
      description: "The generated answer",
      required: true,
    },
    {
      name: "quality_metrics",
      type: "object",
      description: "Detailed quality scores",
      required: true,
    },
    {
      name: "concerns",
      type: "string[]",
      description: "Issues identified by Black Hat analysis",
      required: true,
    },
  ],
  renderAndWaitForResponse: ({ args, status, respond }) => {
    return (
      <AnswerApprovalCard
        answer={args.answer}
        qualityMetrics={args.quality_metrics}
        concerns={args.concerns}
        isExecuting={status === "executing"}
        onApprove={() => respond?.({ 
          approved: true,
          delivery_authorized: true,
          metadata: { approvedAt: new Date().toISOString() }
        })}
        onReject={() => respond?.({ 
          approved: false,
          feedback: "Answer quality insufficient"
        })}
        onEdit={(editedAnswer) => respond?.({ 
          approved: true,
          edited_answer: editedAnswer,
          human_edited: true
        })}
      />
    );
  },
});

// Quality dashboard
useCopilotAction({
  name: "show_quality_metrics",
  parameters: [
    {
      name: "evaluation_results",
      type: "object",
    },
  ],
  render: ({ args }) => (
    <QualityDashboard
      faithfulness={args.evaluation_results.faithfulness}
      relevance={args.evaluation_results.relevance}
      correctness={args.evaluation_results.correctness}
      completeness={args.evaluation_results.completeness}
      consistency={args.evaluation_results.consistency}
      overallScore={args.evaluation_results.overall_score}
      threshold={0.8}
    />
  ),
});
```

### Decision Matrix: When to Use Parlant vs CopilotKit HIL

| Scenario | Use Parlant | Use CopilotKit HIL | Rationale |
|----------|-------------|-------------------|-----------|
| Routine exploration | ✅ | ❌ | Parlant ensures consistent behavior |
| Ambiguous query | ✅ | ✅ | Parlant explores, HIL resolves |
| Validation checks | ✅ | ❌ | Parlant enforces quality rules |
| Borderline quality | ✅ | ✅ | Parlant detects, HIL decides |
| Strategy selection | ✅ | ✅ (optional) | Parlant scores, HIL overrides if needed |
| Critical approval | ✅ | ✅ | Parlant routes, HIL approves |
| Progress visibility | ❌ | ✅ | CopilotKit shows intermediate state |
| Domain terminology | ✅ | ❌ | Parlant glossary ensures consistency |

### Integration Benefits

**Reliability (Parlant):**
- Ensured compliance with design thinking principles
- No prompt engineering required
- Guaranteed Six Thinking Hats coverage
- Explainable guideline matching

**Control (CopilotKit):**
- Strategic human oversight at critical points
- Quality gates prevent low-quality outputs
- Ambiguity resolution improves accuracy
- User preferences guide agent behavior

**Transparency:**
- Parlant explains which guidelines were matched
- CopilotKit shows intermediate agent state
- Users understand agent reasoning
- Full visibility into decision-making

**Efficiency:**
- Agents handle routine exploration/validation
- Humans intervene only when needed
- Optimal balance of speed and quality
- Reduced cognitive load on users

### Implementation Example

```python
# Backend: Parlant agent setup
async def setup_rag_agent_with_parlant():
    async with parlant.Server() as server:
        agent = await server.create_agent(
            name="RAG_Agent",
            description="Multi-agent RAG with design thinking"
        )
        
        # Intent Parser guidelines (Discover phase)
        await agent.create_guideline(
            condition="User query is ambiguous",
            action="Generate diverse interpretations using Green Hat thinking",
            tools=[generate_interpretations]
        )
        
        # Routing guidelines (Define phase)
        await agent.create_guideline(
            condition="Validating interpretation",
            action="Apply Black Hat critical thinking to assess risks",
            tools=[validate_interpretation]
        )
        
        # Retrieval guidelines (Develop phase)
        await agent.create_guideline(
            condition="Generating retrieval strategies",
            action="Create alternatives using Green Hat, evaluate with Yellow Hat",
            tools=[generate_strategies, evaluate_benefits]
        )
        
        # Evaluation guidelines (Deliver phase)
        await agent.create_guideline(
            condition="Answer quality score below 0.8",
            action="Route to human review with quality concerns",
            tools=[route_to_human_review]
        )
        
        return agent
```

```typescript
// Frontend: CopilotKit HIL setup
function RAGWithDesignThinking() {
  // Intent clarification (Discover)
  useCopilotAction({
    name: "request_clarification",
    renderAndWaitForResponse: ({ args, respond }) => (
      <InterpretationSelector onSelect={respond} />
    ),
  });
  
  // Intent validation (Define)
  useCopilotAction({
    name: "validate_intent",
    renderAndWaitForResponse: ({ args, respond }) => (
      <IntentValidationCard onApprove={respond} />
    ),
  });
  
  // Strategy selection (Develop)
  useCopilotAction({
    name: "select_strategy",
    renderAndWaitForResponse: ({ args, respond }) => (
      <StrategyComparisonTable onSelect={respond} />
    ),
  });
  
  // Answer approval (Deliver)
  useCopilotAction({
    name: "approve_answer",
    renderAndWaitForResponse: ({ args, respond }) => (
      <AnswerApprovalCard onApprove={respond} />
    ),
  });
  
  return <div className="rag-interface">{/* UI components */}</div>;
}
```

## Technology Stack

### Current Implementation (Unified with Motia)

**Backend:**
- **Motia Framework** - Unified backend for APIs, workflows, and AI agents
- **FastAPI** - HTTP server and API endpoints (or use Motia's native API steps)
- **Python 3.12+** - Runtime environment (with TypeScript/JavaScript multi-lang support)
- **Azure OpenAI** or **OpenAI** - LLM providers (configurable via environment)
- **Pydantic** - Data validation and serialization
- **python-dotenv** - Environment configuration
- **uvicorn** - ASGI server

**Frontend:**
- **Next.js 16** (App Router) with **React 19** and **TypeScript**
- **CopilotKit** (`@copilotkit/react-core`, `@copilotkit/react-ui`, `@copilotkit/runtime`) - Agent UI framework
- **Tailwind CSS 4** - Styling
- **Zod** - Schema validation

**Development Tools:**
- **pnpm** - Monorepo management (Motia uses pnpm workspaces)
- **Biome** - Modern linting/formatting (replaces ESLint/Prettier)
- **TypeScript 5** - Type safety
- **Husky** - Git hooks
- **Concurrently** - Run frontend and backend simultaneously

### Planned Additions for RAG System

**Backend Extensions:**
- **LlamaIndex** - Document indexing, chunking, and retrieval engines
- **pgvector** with **PostgreSQL** - Vector database for embeddings
- **ClickHouse** - Analytics database for FinOps data (optional)
- **Langfuse SDK** - Tracing and observability
- **RAGAS** - RAG evaluation metrics (faithfulness, relevance, correctness)
- **Hypothesis** - Property-based testing framework
- **asyncpg** - Async PostgreSQL driver
- **Parlant** - Behavioral guidelines and routing with ensured compliance
- **Agent Lightning** - Prompt optimization with RL-based tuning (optional)
- **Motia Adapters** - Redis for state/streams, RabbitMQ for events (if distributing)

**Optional Components:**
- **Cohere Rerank** - Document reranking (if quality improvement needed)
- **DeepEval** - Automated quality evaluation (alternative to RAGAS)

**Vector Database Options:**
- **Primary:** PostgreSQL with pgvector extension (simplest integration)
- **Alternatives:**  Weaviate, FAISS, Azure AI Search

**Infrastructure:**
- **Docker & Docker Compose** - Containerization
- **Azure** - Cloud platform (already using Azure OpenAI)
- **GitHub Actions** - CI/CD pipeline

**Observability:**
- **Langfuse** - Distributed tracing, metrics, cost tracking
- **Motia Workbench** - Visual debugger and tracing UI (built-in)
- **OpenTelemetry** (optional) - Unified observability
- **Custom dashboards** - Real-time agent monitoring

**Evaluation Datasets:**
- **BEIR** - Retrieval quality benchmarks
- **Spider, UNITE, BIRD** - SQL generation benchmarks
- **TruthfulQA, MT-Bench** - Generation quality benchmarks

### Architecture Alignment with Motia Framework

The design leverages Motia's unified Step-based architecture:

**1. Step Definition Pattern:**
```python
from typing import Dict, Any
from motia import StepFactory

# Define as API step that emits events for workflow orchestration
@StepFactory.api_step(
    name="intent_parser",
    path="/parse-intent",
    method="POST",
    emits=["intent.parsed"]
)
async def intent_parser_step(request: Dict[str, Any], context: Dict[str, Any]):
    """
    Intent Parser - Receives query via API, parses intent, emits event for next step
    """
    query = request.get("query")
    user_id = request.get("user_id")

    # Parse intent using LLM
    intent_result = await parse_intent_with_llm(query)

    # Emit event for knowledge retriever step
    await context["emit"]({
        "topic": "intent.parsed",
        "data": {
            "query": query,
            "user_id": user_id,
            "intent": intent_result.intent,
            "entities": intent_result.entities,
            "confidence": intent_result.confidence.overall_confidence
        }
    })

    return {"status": 200, "body": intent_result}

# Event-driven step that automatically executes when intent.parsed is emitted
@StepFactory.event_step(name="knowledge_retriever", subscribes="intent.parsed")
async def knowledge_retriever_step(event_data: Dict[str, Any], context: Dict[str, Any]):
    """
    Knowledge Retriever - Automatically triggered by intent.parsed event
    """
    query = event_data["query"]
    intent = event_data["intent"]

    # Retrieve documents from vector store
    documents = await retrieve_documents(query, intent)

    # Emit event for answer generation
    await context["emit"]({
        "topic": "documents.retrieved",
        "data": {
            "query": query,
            "documents": documents,
            "intent": intent
        }
    })

    context["logger"].info(f"Retrieved {len(documents)} documents")
```

**2. Tool Functions:**
```python
# Standard Python functions used as tools within steps
from typing import Annotated
from pydantic import Field

async def parse_intent_with_llm(
    query: Annotated[str, Field(description="User's natural language query")],
    context: Annotated[dict, Field(description="Conversation context")]
) -> IntentResult:
    """Parse intent using multi-step reasoning with LLM"""
    # Implementation using Azure OpenAI or OpenAI
    llm_client = get_llm_client()

    prompt = f"""
    Extract intent and entities from this query: {query}

    Return structured output with:
    - intent: main user intention
    - entities: extracted parameters
    - confidence: confidence scores
    - query_type: classification
    """

    response = await llm_client.complete(prompt)
    return IntentResult.parse_from_llm(response)

async def retrieve_documents(
    query: str,
    intent: str,
    vector_store: VectorStoreAdapter,
    top_k: int = 10
) -> List[Document]:
    """Retrieve relevant documents from vector store"""
    embedding = await generate_embedding(query)
    return await vector_store.search(embedding, top_k=top_k)
```

**3. State Management Across Steps:**
```python
# State is automatically managed by Motia across step executions
# Steps can access and modify shared state via context["state"]

@StepFactory.event_step(name="answer_generator", subscribes="documents.retrieved")
async def answer_generator_step(event_data: Dict[str, Any], context: Dict[str, Any]):
    """Generate answer with access to shared state"""
    # Access shared state from previous steps
    state = context["state"]

    query = event_data["query"]
    documents = event_data["documents"]

    # Store intermediate results in state
    state["query"] = query
    state["documents"] = documents
    state["generation_start_time"] = datetime.now()

    # Generate answer using LLM
    answer = await generate_answer(query, documents)

    # Update state with result
    state["answer"] = answer
    state["generation_end_time"] = datetime.now()

    # Emit event for evaluator
    await context["emit"]({
        "topic": "answer.generated",
        "data": {"query": query, "answer": answer, "documents": documents}
    })

    return answer
```

**4. Step Configuration and Registration:**
```python
# steps/intent_parser.py
from motia import StepFactory
from typing import Dict, Any

@StepFactory.api_step(
    name="intent_parser",
    path="/api/intent",
    method="POST",
    emits=["intent.parsed"]
)
async def intent_parser_step(request: Dict[str, Any], context: Dict[str, Any]):
    """Extract intent from user query"""
    # Implementation here
    pass

# steps/knowledge_retriever.py
@StepFactory.event_step(
    name="knowledge_retriever",
    subscribes="intent.parsed"
)
async def knowledge_retriever_step(event_data: Dict[str, Any], context: Dict[str, Any]):
    """Retrieve relevant documents"""
    # Implementation here
    pass

# In main.py or setup.py
def register_all_steps():
    """Auto-discover and register all step modules"""
    import steps.intent_parser
    import steps.knowledge_retriever
    import steps.answer_generator
    import steps.evaluator_step
    import steps.human_review_step
    # Steps are registered automatically via decorators

async def initialize_motia():
    """Initialize Motia runtime"""
    register_all_steps()
    runtime = await StepFactory.create_runtime()
    return runtime
```

**5. FastAPI Integration (Optional):**
```python
# If using FastAPI alongside Motia
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP Multi-Agent RAG System with Motia")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Motia steps can also be FastAPI endpoints via the api step type
# Or integrate with existing FastAPI routes

@app.post("/query")
async def process_query(query_request: QueryRequest):
    """Legacy API endpoint - can still use Motia steps"""
    runtime = await get_motia_runtime()
    result = await runtime.execute_step(
        "intent_parser",
        query_request.dict()
    )
    return result
```

**6. LLM Client Configuration:**
```python
import os
from typing import Union

# Abstract LLM client - works with both Azure OpenAI and OpenAI
class LLMClient:
    def __init__(self, provider: str = "azure"):
        self.provider = provider
        if provider == "azure":
            self.client = self._setup_azure_client()
        elif provider == "openai":
            self.client = self._setup_openai_client()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _setup_azure_client(self):
        from openai import AsyncAzureOpenAI
        return AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def _setup_openai_client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def complete(self, prompt: str, **kwargs):
        """Unified completion interface"""
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.3)
        )
        return response.choices[0].message.content
)
```

### Migration Path to Motia Implementation

**Phase 1: Bootstrap Motia Framework (Weeks 1-2)**
- Add Motia as dependency: `motia` to `pyproject.toml`
- Create step directory structure: `steps/intent_parser.py`, `steps/knowledge_retriever.py`, etc.
- Convert existing agents to Motia Step pattern with `@StepFactory` decorators
- Set up PostgreSQL with pgvector extension for vector storage
- Initialize LlamaIndex for document processing

**Phase 2: Core RAG Pipeline Steps (Weeks 3-4)**
- Implement Intent Parser as API step emitting `intent.parsed` events
- Implement Knowledge Retriever as event step subscribing to `intent.parsed`
- Implement Answer Generator for synthesis with Chain-of-Thought prompting
- Implement Evaluator step with RAGAS metrics for quality assessment
- Test end-to-end pipeline with event-driven workflow

**Phase 3: Parlant + CopilotKit HIL Integration (Weeks 5-6)**
- Integrate Parlant SDK for behavioral guidelines across all steps
- Define Phase-specific guidelines: Discover, Define, Develop, Deliver
- Implement CopilotKit intervention points for human oversight
- Add Six Thinking Hats patterns to step implementations
- Create glossary terms and canned responses for consistency

**Phase 4: Quality & Observability (Weeks 7-8)**
- Integrate Langfuse SDK for comprehensive tracing across steps
- Add RAGAS evaluation with custom metrics for RAG-specific quality
- Implement Human Review workflow with approval/rejection flows
- Add property-based tests with Hypothesis for all steps
- Set up Motia Workbench for visual debugging and monitoring

**Phase 5: Advanced Features & Optimization (Weeks 9-10)**
- Add optional ClickHouse for analytics and SQL generation
- Implement data story generation with audio narration
- Integrate Agent Lightning for prompt optimization
- Add Redis adapters for distributed deployment (if needed)
- Performance optimization: caching, batching, parallel execution

### Key Design Decisions with Motia

**1. Embrace Motia's Step Primitive**
- Convert all agents to Motia steps (API, event, or cron types)
- Use `@StepFactory` decorators for automatic registration
- Leverage event-driven workflow via emits/subscribes
- Maintain CopilotKit integration for human-in-the-loop

**2. Unified but Extensible**
- Keep Motia as single source of truth for orchestration
- Add LlamaIndex, Parlant, Langfuse as integrated services
- Reuse existing LLM client configuration
- Maintain environment variable patterns from current setup

**3. Gradual Migration Strategy**
- Start with Intent Parser and Knowledge Retriever steps
- Gradually migrate remaining agents to step pattern
- Keep FastAPI for legacy compatibility if needed
- Use Motia's built-in observability to monitor migration

**4. Best of Both Worlds**
- Motia for unified orchestration and agent communication
- CopilotKit for frontend integration and human oversight
- Parlant for behavioral guidelines and quality gates
- Langfuse for comprehensive tracing and evaluation

**3. Incremental Complexity**
- Start with simple retrieval (Phase 1)
- Add multi-agent coordination (Phase 2)
- Layer in quality evaluation (Phase 3)
- Add advanced features last (Phase 4)

**4. Maintain CopilotKit Compatibility**
- All agents must work with CopilotKit UI
- Use state schemas for UI synchronization
- Support approval workflows for human-in-the-loop
- Maintain AG-UI protocol compatibility

