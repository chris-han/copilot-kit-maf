# Framework Leverage Analysis

## Objective
Identify what Microsoft Agent Framework and CopilotKit already provide to avoid reinventing wheels.

---

## Microsoft Agent Framework (AutoGen) - What's Already Built

### ✅ Core Features We Should Use

| Feature | MAF Provides | Our Current Design | Recommendation |
|---------|--------------|-------------------|----------------|
| **Agent Lifecycle** | ✅ Runtime manages creation/destruction | ✅ Using RoutedAgent | **KEEP** - Already leveraging |
| **Message Routing** | ✅ @message_handler decorators | ✅ Using message handlers | **KEEP** - Already leveraging |
| **Runtime Modes** | ✅ SingleThreaded / GrpcWorker | ✅ Using both | **KEEP** - Already leveraging |
| **Team Orchestration** | ✅ RoundRobinGroupChat, SelectorGroupChat, Swarm | ❌ Custom Plan-and-Execute | **REPLACE** - Use MAF teams |
| **Agent Registry** | ✅ Built-in via runtime.register() | ❌ Custom AgentRegistry class | **REPLACE** - Use MAF registry |
| **Termination Conditions** | ✅ TextMentionTermination, ExternalTermination | ❌ Custom logic | **ADD** - Use MAF conditions |
| **Handoff Pattern** | ✅ HandoffMessage for agent transitions | ❌ Custom routing | **ADD** - Use MAF handoffs |
| **Group Chat** | ✅ Multiple team patterns | ❌ Not using | **CONSIDER** - For multi-agent collaboration |

### 🔧 MAF Team Patterns We Should Leverage

**1. Swarm Pattern** (Best fit for our RAG pipeline)
```python
from autogen_agentchat.teams import Swarm
from autogen_agentchat.messages import HandoffMessage

# Swarm uses HandoffMessage to transition between agents
# Perfect for: Intent Parser → Router → Retriever → Generator → Evaluator
team = Swarm([intent_parser, router, retriever, generator, evaluator])
```

**Benefits:**
- ✅ Built-in agent transitions via HandoffMessage
- ✅ Clear handoff logic between specialized agents
- ✅ No custom orchestration code needed
- ✅ Production-tested by Microsoft

**2. SelectorGroupChat** (For dynamic routing)
```python
from autogen_agentchat.teams import SelectorGroupChat

# Uses LLM to select next speaker
# Perfect for: Dynamic routing based on query complexity
team = SelectorGroupChat([agent1, agent2, agent3], model_client=model)
```

**Benefits:**
- ✅ LLM-based agent selection
- ✅ Handles complex routing logic
- ✅ No custom Parlant integration needed (can still use Parlant for guidelines)

---

## CopilotKit - What's Already Built

### ✅ Frontend Features We Should Use

| Feature | CopilotKit Provides | Our Current Design | Recommendation |
|---------|---------------------|-------------------|----------------|
| **AG-UI Protocol** | ✅ Built-in via CopilotRuntime | ✅ Using it | **KEEP** - Already leveraging |
| **SSE Streaming** | ✅ Automatic streaming | ✅ Using it | **KEEP** - Already leveraging |
| **State Management** | ✅ useCoAgent hook | ✅ Using it | **KEEP** - Already leveraging |
| **Action Handling** | ✅ useCopilotAction | ✅ Using it | **KEEP** - Already leveraging |
| **UI Components** | ✅ CopilotSidebar, CopilotChat | ✅ Using them | **KEEP** - Already leveraging |
| **Generative UI** | ✅ Built-in support | ❌ Custom components | **ADD** - Use CopilotKit's generative UI |
| **Human-in-Loop** | ✅ Built-in approval flows | ❌ Custom HumanReviewAgent | **REPLACE** - Use CopilotKit's approval |

---

## What We Should STOP Building (Reinventing Wheels)

### ❌ 1. Custom Orchestrator Agent

**Current Design:**
```python
class OrchestratorAgent(RoutedAgent):
    async def create_plan(self, query: str) -> ExecutionPlan:
        # Custom plan-and-execute logic
        pass
    
    async def execute_plan(self, plan: ExecutionPlan) -> None:
        # Custom execution logic
        pass
```

**Replace With MAF Swarm:**
```python
from autogen_agentchat.teams import Swarm
from autogen_agentchat.messages import HandoffMessage

# Define handoff logic in each agent
class IntentParserAgent(AssistantAgent):
    async def on_messages(self, messages, cancellation_token):
        result = await self.parse_intent(messages[-1].content)
        # Handoff to next agent
        return HandoffMessage(target="knowledge_retriever", content=result)

# Create swarm (no custom orchestrator needed)
team = Swarm([intent_parser, retriever, generator, evaluator])
result = await team.run(task="What is RAG?")
```

**Benefits:**
- ✅ No custom orchestration code
- ✅ Built-in error handling
- ✅ Production-tested
- ✅ Simpler to maintain

---

### ❌ 2. Custom Agent Registry

**Current Design:**
```python
class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
    
    def register_agent(self, name: str, agent: Agent):
        self.agents[name] = agent
    
    def discover_agent(self, capability: str) -> Agent:
        # Custom discovery logic
        pass
```

**Replace With MAF Built-in Registry:**
```python
# MAF handles registration automatically
runtime = SingleThreadedAgentRuntime()
await IntentParserAgent.register(runtime, "intent_parser", lambda: IntentParserAgent())
await KnowledgeRetrieverAgent.register(runtime, "retriever", lambda: KnowledgeRetrieverAgent())

# Discovery via AgentId
agent_id = AgentId("intent_parser", "default")
response = await runtime.send_message(message, agent_id)
```

**Benefits:**
- ✅ No custom registry code
- ✅ Built-in lifecycle management
- ✅ Type-safe agent IDs

---

### ❌ 3. Custom Human Review Agent

**Current Design:**
```python
class HumanReviewAgent:
    async def request_review(self, answer: Answer) -> ReviewFeedback:
        # Custom review UI and logic
        pass
```

**Replace With CopilotKit Approval Flow:**
```python
from copilotkit import useCopilotAction

# Frontend: Built-in approval UI
const { executeAction } = useCopilotAction({
  name: "reviewAnswer",
  requiresApproval: true,  // Built-in approval flow
  handler: async (answer) => {
    // Handle approved answer
  }
});

# Backend: No custom agent needed
@ai_function(name="generate_answer")
async def generate_answer(query: str) -> Answer:
    # CopilotKit handles approval automatically
    return answer
```

**Benefits:**
- ✅ No custom review agent
- ✅ Built-in approval UI
- ✅ Automatic feedback capture

---

## What We Should KEEP Building (Custom Logic)

### ✅ 1. Domain-Specific Agents

**Keep these custom implementations:**
- Intent Parser Agent (domain-specific NLP logic)
- Knowledge Retriever Agent (RAG-specific retrieval strategies)
- Answer Generator Agent (domain-specific generation)
- Evaluator Agent (7 RAG characteristics evaluation)

**Why:** These contain domain-specific business logic that frameworks can't provide.

---

### ✅ 2. Integration Adapters

**Keep these custom implementations:**
- Vector Store Adapter (pgvector, Weaviate, etc.)
- Parlant Router (guideline-based routing)
- Agent Lightning Optimizer (prompt optimization)
- RAGAS Evaluator (quality metrics)

**Why:** These integrate external services specific to our RAG pipeline.

---

### ✅ 3. Data Models

**Keep these custom implementations:**
- IntentResult, Document, Answer, EvaluationResult
- Query, ConversationContext
- All Pydantic models

**Why:** These are domain-specific data structures.

---

## Recommended Architecture Changes

### Before (Custom Orchestration):
```
Orchestrator Agent (Custom)
  ├─> Intent Parser Agent
  ├─> Parlant Router
  ├─> Knowledge Retriever Agent
  ├─> Answer Generator Agent
  ├─> Evaluator Agent
  └─> Human Review Agent (Custom)
```

### After (Leveraging MAF):
```
MAF Swarm Team
  ├─> Intent Parser Agent (HandoffMessage → retriever)
  ├─> Knowledge Retriever Agent (HandoffMessage → generator)
  ├─> Answer Generator Agent (HandoffMessage → evaluator)
  └─> Evaluator Agent (HandoffMessage → approval or retry)

CopilotKit Approval Flow (replaces Human Review Agent)
```

---

## Implementation Priorities

### Phase 1: Replace Custom Orchestration
1. ✅ Remove custom OrchestratorAgent class
2. ✅ Implement MAF Swarm pattern
3. ✅ Add HandoffMessage logic to each agent
4. ✅ Test end-to-end pipeline

### Phase 2: Replace Custom Registry
1. ✅ Remove custom AgentRegistry class
2. ✅ Use MAF's built-in registration
3. ✅ Update agent discovery to use AgentId

### Phase 3: Replace Custom Human Review
1. ✅ Remove HumanReviewAgent class
2. ✅ Implement CopilotKit approval flow
3. ✅ Add requiresApproval to actions

### Phase 4: Add MAF Features
1. ✅ Add termination conditions (TextMentionTermination)
2. ✅ Add error handling via MAF patterns
3. ✅ Add observability via MAF tracing

---

## Benefits of Leveraging Frameworks

### Reduced Code
- **Before:** ~2000 lines of custom orchestration code
- **After:** ~500 lines (75% reduction)

### Improved Reliability
- ✅ Production-tested by Microsoft
- ✅ Built-in error handling
- ✅ Community support

### Faster Development
- ✅ No need to build orchestration from scratch
- ✅ Focus on domain-specific logic
- ✅ Faster time to market

### Better Maintainability
- ✅ Less custom code to maintain
- ✅ Framework updates handled by Microsoft
- ✅ Standard patterns easier for new developers

---

## Conclusion

**Stop Reinventing:**
1. ❌ Custom Orchestrator → Use MAF Swarm
2. ❌ Custom Agent Registry → Use MAF built-in
3. ❌ Custom Human Review → Use CopilotKit approval

**Keep Building:**
1. ✅ Domain-specific agents (Intent Parser, Retriever, Generator, Evaluator)
2. ✅ Integration adapters (Vector Store, Parlant, RAGAS)
3. ✅ Data models (Pydantic schemas)

**Result:** Simpler, more reliable, faster to build, easier to maintain.
