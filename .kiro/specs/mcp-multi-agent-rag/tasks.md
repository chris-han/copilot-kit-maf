# Implementation Plan

## Phase 1: Foundation & Infrastructure

- [ ] 1. Set up project structure and core infrastructure
  - Create Python backend structure with FastAPI
  - Set up Next.js 16 frontend with React 19
  - Configure TypeScript and Python type checking
  - Set up development environment (uvicorn, hot reload)
  - _Requirements: 1.1, 1.2, 1.5_

- [ ] 2. Configure AG-UI protocol integration
  - Install `agent-framework-ag-ui` (Python) and `@ag-ui/client` (TypeScript)
  - Update `src/app/api/copilotkit/route.ts` to use port 8880
  - Configure CopilotKit Runtime with HttpAgent
  - Test basic agent communication
  - _Requirements: 8.1, 8.2_

- [ ] 3. Set up observability infrastructure
  - Install and configure Langfuse SDK
  - Create Langfuse project and get API keys
  - Implement trace context propagation
  - Set up basic logging for all agent interactions
  - _Requirements: 1.4, 4.4, 5.5, 6.5_

- [ ] 4. Configure database connections
  - Set up PostgreSQL with pgvector extension
  - Create database schema for documents and metadata
  - Configure asyncpg connection pool
  - Test vector similarity queries
  - _Requirements: 3.1, 3.4_

---

## Phase 2: Agent Registry & Discovery

- [ ] 5. Implement agent metadata schema
  - Create `AgentMetadata` Pydantic model with all fields
  - Define `AgentPattern`, `AgentCapability`, `MessageType` enums
  - Add input/output schema validation
  - _Requirements: 1.1, 1.2_

- [ ] 6. Build agent registry system
  - Implement `AgentRegistry` class with indexing
  - Add capability-based lookup (O(1) access)
  - Add pattern-based lookup
  - Implement dependency resolution (DFS)
  - Add health monitoring
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 7. Create discovery API
  - Implement `AgentDiscoveryAPI` class
  - Add `discover_for_task()` method
  - Add `get_agent_chain()` method
  - Add `recommend_agent()` method
  - _Requirements: 1.2, 1.3_

- [ ] 8. Integrate Langfuse with registry
  - Create `ObservableAgentRegistry` wrapper
  - Add tracing for agent registration
  - Add tracing for discovery operations
  - Log discovery results and performance
  - _Requirements: 1.4, 4.4_

---

## Phase 3: Intent Parser Agent (ReAct Pattern)

- [ ] 9. Implement Intent Parser Agent core
  - Create `IntentParserAgent` class extending `ChatAgent`
  - Register agent metadata with capabilities
  - Define input/output schemas
  - _Requirements: 2.1_

- [ ] 10. Implement ReAct loop for intent parsing
  - Create `parse_intent()` method with ReAct pattern
  - Implement thought generation
  - Implement action execution (tool use)
  - Implement observation processing
  - Add iteration limit (max 5 iterations)
  - _Requirements: 2.1, 2.3_

- [ ] 11. Add entity extraction tools
  - Create `@ai_function` for context lookup
  - Create `@ai_function` for entity extraction
  - Add confidence scoring
  - Handle missing entities
  - _Requirements: 2.1, 2.3_

- [ ] 12. Implement fallback handling
  - Detect low confidence scenarios
  - Generate clarification questions
  - Handle ambiguous queries
  - Route to human review when needed
  - _Requirements: 2.2, 2.3_

- [ ] 12.1 Write property tests for Intent Parser
  - **Property 1: Intent extraction consistency**
  - **Validates: Requirements 2.1**
  - Test that similar queries produce similar intents
  - Use 100+ test iterations

---

## Phase 4: Parlant Routing Layer (Rule-Based)

- [ ] 13. Set up Parlant framework
  - Install Parlant SDK
  - Configure Parlant project
  - Define routing guidelines schema
  - _Requirements: 4.1, 4.2_

- [ ] 14. Implement routing guidelines
  - Create guidelines for SQL generation queries
  - Create guidelines for semantic search queries
  - Create guidelines for data story queries
  - Define priority order and fallback rules
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 15. Build Parlant router integration
  - Create `ParlantRouter` class
  - Implement `route()` method with guideline matching
  - Add logging for matched guidelines
  - Implement fallback routing
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 16. Add routing observability
  - Log all routing decisions to Langfuse
  - Track guideline match rates
  - Monitor fallback frequency
  - Enable A/B testing support
  - _Requirements: 4.4, 4.5_

---

## Phase 5: Knowledge Retriever Agent (ReAct Pattern)

- [ ] 17. Implement vector store adapter pattern
  - Create `VectorStoreAdapter` abstract base class
  - Implement `PgVectorAdapter` for PostgreSQL + pgvector
  - Add `VectorStoreFactory` for adapter creation
  - Configure from environment variables
  - _Requirements: 3.1, 3.4_

- [ ] 18. Create Knowledge Retriever Agent
  - Create `KnowledgeRetrieverAgent` class
  - Register agent metadata with capabilities
  - Initialize with vector store adapter
  - _Requirements: 3.1_

- [ ] 19. Implement ReAct loop for retrieval
  - Create `retrieve()` method with ReAct pattern
  - Implement semantic search action
  - Implement metadata filter action
  - Implement reranking action
  - Add iteration logic based on confidence
  - _Requirements: 3.1, 3.2, 3.3, 3.5_

- [ ] 20. Add multi-strategy retrieval
  - Implement parallel strategy execution
  - Merge and deduplicate results
  - Score and rank candidates
  - Return top-k documents
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 20.1 Write property tests for retrieval
  - **Property 2: Retrieval recall consistency**
  - **Validates: Requirements 3.1, 3.4**
  - Test that relevant documents are always retrieved
  - Use 100+ test iterations

---

## Phase 6: Answer Generator Agent (Chain-of-Thought)

- [ ] 21. Implement Answer Generator Agent
  - Create `AnswerGeneratorAgent` class
  - Register agent metadata with CoT pattern
  - Define input/output schemas
  - _Requirements: 5.1, 5.3_

- [ ] 22. Build Chain-of-Thought generation
  - Create CoT prompt template
  - Implement `generate()` method with CoT reasoning
  - Add structured reasoning steps (identify, determine, synthesize, cite)
  - Extract citations from reasoning
  - _Requirements: 5.1, 5.2_

- [ ] 23. Add model selection logic
  - Implement `select_model()` based on complexity
  - Map simple → GPT-4o-mini
  - Map moderate → GPT-4
  - Map complex → o1-preview
  - _Requirements: 5.3_

- [ ] 24. Implement context assembly
  - Create `assemble_context()` method
  - Add provenance metadata (source, timestamp, confidence)
  - Manage token limits (max 4000 tokens)
  - _Requirements: 5.2_

- [ ] 24.1 Write property tests for answer generation
  - **Property 3: Citation completeness**
  - **Validates: Requirements 5.2**
  - Test that all claims have citations
  - Use 100+ test iterations

---

## Phase 7: Evaluator Agent (Chain-of-Thought)

- [ ] 25. Set up RAGAS framework
  - Install RAGAS SDK
  - Configure RAGAS evaluator
  - Test faithfulness, relevance, correctness metrics
  - _Requirements: 6.1, 6.2_

- [ ] 26. Implement Evaluator Agent
  - Create `EvaluatorAgent` class
  - Register agent metadata with CoT pattern
  - Initialize with RAGAS evaluator
  - _Requirements: 6.1, 6.2_

- [ ] 27. Build CoT evaluation logic
  - Create CoT evaluation prompt template
  - Implement `evaluate()` method with structured steps
  - Score all 7 RAG characteristics (faithfulness, relevance, correctness, coverage, consistency, freshness, traceability)
  - Combine CoT scores with RAGAS scores (60/40 weight)
  - _Requirements: 6.1, 6.2_

- [ ] 28. Add quality thresholds and routing
  - Calculate overall quality score
  - Implement threshold check (0.80)
  - Identify failing metrics
  - Route to human review when needed
  - _Requirements: 6.3, 6.4, 6.5_

- [ ] 28.1 Write property tests for evaluation
  - **Property 4: Evaluation consistency**
  - **Validates: Requirements 6.1, 6.2**
  - Test that similar answers get similar scores
  - Use 100+ test iterations

---

## Phase 8: Orchestrator Agent (Plan-and-Execute)

- [ ] 29. Implement Orchestrator Agent core
  - Create `OrchestratorAgent` class
  - Register agent metadata with Plan-and-Execute pattern
  - Initialize with agent registry and discovery API
  - _Requirements: 1.1, 1.2_

- [ ] 30. Build Plan-and-Execute workflow
  - Implement `process_query()` main entry point
  - Create `create_plan()` method for execution planning
  - Define task dependencies
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 31. Implement task execution
  - Create `execute_plan()` method
  - Implement `execute_task()` for individual tasks
  - Use agent discovery to find agents dynamically
  - Handle task failures with retry logic
  - _Requirements: 1.2, 1.3, 1.4_

- [ ] 32. Add result aggregation
  - Implement `aggregate_results()` method
  - Combine results from all agents
  - Add evaluation metadata to final answer
  - Create fallback answer for failures
  - _Requirements: 1.3, 1.4_

- [ ] 32.1 Write property tests for orchestration
  - **Property 5: Pipeline completeness**
  - **Validates: Requirements 1.1, 1.2, 1.3**
  - Test that all required agents are called
  - Use 100+ test iterations

---

## Phase 9: Human Review Agent

- [ ] 33. Implement Human Review Agent
  - Create `HumanReviewAgent` class
  - Register agent metadata
  - Define review request schema
  - _Requirements: 7.1, 7.2_

- [ ] 34. Build review request workflow
  - Implement `request_review()` method
  - Create review UI component
  - Present answer with quality scores
  - Show failing metrics
  - _Requirements: 7.1, 7.2_

- [ ] 35. Add feedback capture
  - Implement `capture_feedback()` method
  - Store feedback with traceability
  - Link to original query, routing, and prompts
  - Send to Agent Lightning for learning
  - _Requirements: 7.3, 7.4_

- [ ] 36. Integrate with Agent Lightning
  - Configure Agent Lightning SDK
  - Send feedback to LightningStore
  - Enable prompt optimization pipeline
  - _Requirements: 7.4, 5.4_

---

## Phase 10: Frontend Integration

- [ ] 37. Update CopilotKit UI components
  - Customize CopilotSidebar appearance
  - Add theme color support
  - Configure suggestions
  - _Requirements: 8.1_

- [ ] 38. Implement shared state management
  - Use `useCoAgent` hook for agent state
  - Define `AgentState` TypeScript interface
  - Sync state between frontend and backend
  - _Requirements: 8.2_

- [ ] 39. Add generative UI components
  - Create result cards for answers
  - Add citation display
  - Show quality scores
  - Display reasoning steps (CoT)
  - _Requirements: 8.3_

- [ ] 40. Implement human review UI
  - Create review modal component
  - Add approve/reject buttons
  - Add feedback text input
  - Show quality metrics
  - _Requirements: 7.1, 7.2, 8.4_

---

## Phase 11: Testing & Quality Assurance

- [ ] 41. Set up testing infrastructure
  - Configure pytest for Python
  - Configure Jest/Vitest for TypeScript
  - Set up property-based testing library (Hypothesis for Python)
  - Create test fixtures and utilities
  - _Requirements: All_

- [ ] 42. Write integration tests
  - Test complete RAG pipeline end-to-end
  - Test agent discovery and coordination
  - Test error handling and retries
  - Test human review workflow
  - _Requirements: 1.1-1.5, 2.1-2.5, 3.1-3.5, 4.1-4.5, 5.1-5.5, 6.1-6.5, 7.1-7.4_

- [ ] 43. Write unit tests for core components
  - Test vector store adapters
  - Test agent registry operations
  - Test discovery API methods
  - Test routing logic
  - _Requirements: All_

- [ ] 44. Run property-based tests
  - Execute all property tests (100+ iterations each)
  - Fix any failing properties
  - Document test coverage
  - _Requirements: All_

---

## Phase 12: Observability & Monitoring

- [ ] 45. Configure Langfuse dashboards
  - Create dashboard for agent performance
  - Create dashboard for routing decisions
  - Create dashboard for quality metrics
  - Set up alerts for low quality scores
  - _Requirements: 1.4, 4.4, 5.5, 6.5_

- [ ] 46. Add performance monitoring
  - Track agent latency
  - Monitor token usage and costs
  - Track retrieval recall and precision
  - Monitor quality score distributions
  - _Requirements: 1.4, 6.5_

- [ ] 47. Implement error tracking
  - Log all agent failures
  - Track retry attempts
  - Monitor fallback usage
  - Alert on high error rates
  - _Requirements: 1.4_

---

## Phase 13: Deployment & Documentation

- [ ] 48. Create Docker configuration
  - Write Dockerfile for backend (port 8880)
  - Write Dockerfile for frontend
  - Create docker-compose.yml
  - Configure environment variables
  - _Requirements: All_

- [ ] 49. Write deployment documentation
  - Document environment setup
  - Document configuration options
  - Create deployment guide
  - Document monitoring and troubleshooting
  - _Requirements: All_

- [ ] 50. Create API documentation
  - Document AG-UI protocol endpoints
  - Document agent interfaces
  - Document data models
  - Create usage examples
  - _Requirements: 8.1, 8.2_

---

## Phase 14: Optimization & Continuous Improvement

- [ ] 51. Enable Agent Lightning optimization
  - Configure RL training pipeline
  - Set up A/B testing for prompts
  - Monitor prompt performance
  - Deploy optimized prompts
  - _Requirements: 5.4, 7.4_

- [ ] 52. Optimize retrieval performance
  - Tune pgvector index parameters
  - Optimize reranking strategy
  - Cache frequent queries
  - Monitor and improve recall
  - _Requirements: 3.1-3.5_

- [ ] 53. Performance tuning
  - Profile agent latency
  - Optimize database queries
  - Tune LLM parameters (temperature, max_tokens)
  - Implement caching where appropriate
  - _Requirements: All_

---

## Notes

- Tasks marked with `*` are optional and can be skipped for MVP
- Property-based tests should run 100+ iterations minimum
- Each task should be completed and tested before moving to the next
- Langfuse tracing should be added to all agent operations
- All code should follow type safety best practices (Pydantic, TypeScript)
