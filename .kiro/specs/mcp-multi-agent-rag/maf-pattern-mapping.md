# MAF Orchestration Pattern Mapping

## Objective
Map Microsoft Agent Framework's built-in patterns to AI agent patterns (Plan-and-Execute, ReAct, CoT, ToT) and identify what needs custom implementation.

---

## AI Agent Patterns Overview

| Pattern | Description | Use Case | Complexity |
|---------|-------------|----------|------------|
| **Plan-and-Execute** | Create plan upfront, execute sequentially | Multi-step workflows with clear dependencies | Medium |
| **ReAct** | Iterative Reasoning + Acting with tool use | Dynamic tool selection, exploratory tasks | High |
| **Chain-of-Thought (CoT)** | Step-by-step reasoning in single pass | Analytical tasks, no tool use | Low |
| **Tree-of-Thoughts (ToT)** | Explore multiple reasoning paths, backtrack | Complex problem-solving, optimization | Very High |

---

## MAF Built-in Patterns

### 1. RoundRobinGroupChat
**Description:** Agents take turns in fixed order, all share context

**Maps To:** ❌ None directly
- Not Plan-and-Execute (no planning phase)
- Not ReAct (no iterative tool use)
- Not CoT (multi-agent, not single reasoning)
- Not ToT (no branching/backtracking)

**Use Case:** Simple turn-taking conversations

---

### 2. SelectorGroupChat
**Description:** LLM selects next speaker after each message

**Maps To:** ⚠️ Partial Plan-and-Execute
- ✅ Dynamic agent selection (like planning)
- ❌ No explicit planning phase
- ❌ No dependency management
- ❌ No parallel execution

**Use Case:** Dynamic routing based on context

**Gap:** Needs explicit planning and dependency tracking

---

### 3. Swarm
**Description:** Agents use HandoffMessage to transition between agents

**Maps To:** ✅ Plan-and-Execute (Best Match)
- ✅ Explicit handoffs (like task dependencies)
- ✅ Sequential execution
- ✅ Clear agent transitions
- ⚠️ No explicit planning phase (agents decide handoffs)
- ❌ No parallel execution

**Use Case:** Sequential workflows with clear handoffs

**Gap:** Needs explicit planning phase and parallel execution support

---

### 4. MagenticOneGroupChat
**Description:** Generalist multi-agent system for web/file tasks

**Maps To:** ⚠️ Partial ReAct
- ✅ Tool use (web browsing, file operations)
- ✅ Iterative execution
- ⚠️ Specialized for web/file tasks
- ❌ Not general-purpose ReAct

**Use Case:** Web scraping, file processing

**Gap:** Needs general-purpose ReAct implementation

---

## Pattern Mapping Matrix

| AI Pattern | MAF Pattern | Match Quality | Gaps | Custom Implementation Needed? |
|------------|-------------|---------------|------|-------------------------------|
| **Plan-and-Execute** | Swarm | 70% | Planning phase, parallel execution | ✅ YES - Add planning layer |
| **ReAct** | MagenticOne | 40% | General-purpose tool use | ✅ YES - Implement ReAct loop |
| **CoT** | None | 0% | Single-agent reasoning | ✅ YES - Implement in prompts |
| **ToT** | None | 0% | Branching, backtracking | ✅ YES - Full custom implementation |

---

## Recommended Implementation Strategy

### 1. Plan-and-Execute: Build on MAF Swarm

**MAF Provides:**
- ✅ Agent handoffs via HandoffMessage
- ✅ Sequential execution
- ✅ Context sharing

**We Need to Add:**
- ✅ Explicit planning phase (Orchestrator creates plan)
- ✅ Dependency tracking
- ❌ Parallel execution (NOT needed for MVP - sequential is sufficient)

**Implementation:**
```python
from autogen_agentchat.teams import Swarm
from autogen_agentchat.messages import HandoffMessage
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ExecutionPlan:
    """Plan created by Orchestrator (sequential only for MVP)"""
    tasks: List[AgentTask]
    dependencies: Dict[AgentTask, List[AgentTask]]

class PlanAndExecuteOrchestrator(AssistantAgent):
    """Custom orchestrator that adds planning to MAF Swarm"""
    
    def __init__(self, swarm: Swarm):
        super().__init__("orchestrator")
        self.swarm = swarm
    
    async def on_messages(self, messages, cancellation_token):
        # PLAN PHASE (Custom)
        plan = await self.create_plan(messages[-1].content)
        
        # EXECUTE PHASE (Use MAF Swarm - Sequential)
        results = []
        for task in plan.tasks:
            # Check dependencies are satisfied
            await self.wait_for_dependencies(task, plan.dependencies, results)
            
            # Execute task via Swarm
            result = await self.execute_task(task)
            results.append(result)
        
        return results
    
    async def create_plan(self, query: str) -> ExecutionPlan:
        """Custom planning logic"""
        # Analyze query
        # Determine agent sequence
        # Identify dependencies
        # Return sequential plan
        return ExecutionPlan(
            tasks=[
                AgentTask(agent="intent_parser", ...),
                AgentTask(agent="knowledge_retriever", ...),
                AgentTask(agent="answer_generator", ...),
                AgentTask(agent="evaluator", ...)
            ],
            dependencies={
                # Task dependencies for validation
            }
        )
    
    async def wait_for_dependencies(
        self, 
        task: AgentTask, 
        dependencies: Dict, 
        completed_results: List
    ):
        """Validate dependencies are satisfied (sequential execution)"""
        # In sequential execution, this just validates order
        required_tasks = dependencies.get(task, [])
        for req_task in required_tasks:
            if req_task not in [r.task for r in completed_results]:
                raise ValueError(f"Dependency {req_task} not satisfied")
    
    async def execute_task(self, task: AgentTask):
        """Execute task via MAF Swarm"""
        # Create handoff message
        handoff = HandoffMessage(
            target=task.agent_name,
            content=task.input_data
        )
        # Swarm handles execution
        return await self.swarm.run(task=handoff)
```

**Result:** ✅ Full Plan-and-Execute on top of MAF Swarm (sequential execution)

---

### 2. ReAct: Custom Implementation on MAF Infrastructure

**MAF Provides:**
- ✅ Agent runtime
- ✅ Message passing
- ✅ Tool registration

**We Need to Add:**
- ✅ ReAct loop (Thought → Action → Observation)
- ✅ Tool selection logic
- ✅ Iteration control
- ✅ Termination conditions

**Implementation:**
```python
from autogen_core import RoutedAgent, MessageContext, message_handler

class ReActAgent(RoutedAgent):
    """Custom ReAct implementation on MAF infrastructure"""
    
    def __init__(self, tools: List[Tool]):
        super().__init__("react_agent")
        self.tools = tools
        self.max_iterations = 10
    
    @message_handler
    async def handle_query(self, message: QueryMessage, ctx: MessageContext):
        """ReAct loop"""
        observations = []
        
        for iteration in range(self.max_iterations):
            # THOUGHT: Reason about next action
            thought = await self.generate_thought(
                query=message.content,
                observations=observations
            )
            
            # Check if done
            if thought.is_final_answer:
                return thought.answer
            
            # ACTION: Select and execute tool
            action = await self.select_action(thought)
            result = await self.execute_tool(action)
            
            # OBSERVATION: Record result
            observation = Observation(
                action=action,
                result=result,
                iteration=iteration
            )
            observations.append(observation)
        
        # Max iterations reached
        return self.generate_fallback_answer(observations)
    
    async def generate_thought(self, query: str, observations: List) -> Thought:
        """Generate reasoning step"""
        prompt = f"""
        Query: {query}
        Previous observations: {observations}
        
        Think step-by-step:
        1. What do I know so far?
        2. What do I need to find out?
        3. What tool should I use next?
        4. Or can I answer now?
        """
        return await self.llm.generate(prompt)
    
    async def select_action(self, thought: Thought) -> Action:
        """Select tool based on thought"""
        # Tool selection logic
        pass
    
    async def execute_tool(self, action: Action):
        """Execute selected tool"""
        tool = self.tools[action.tool_name]
        return await tool.execute(action.parameters)
```

**Result:** ✅ Full ReAct on MAF infrastructure

---

### 3. Chain-of-Thought (CoT): Prompt-Based Implementation

**MAF Provides:**
- ✅ Agent runtime
- ✅ LLM integration

**We Need to Add:**
- ✅ CoT prompt templates
- ✅ Structured reasoning steps

**Implementation:**
```python
class CoTAgent(AssistantAgent):
    """Chain-of-Thought via structured prompts"""
    
    def __init__(self):
        super().__init__("cot_agent")
        self.cot_prompt_template = """
        Given the query and context, reason step-by-step:
        
        Step 1: Identify key information
        {step1_instructions}
        
        Step 2: Determine relevance
        {step2_instructions}
        
        Step 3: Synthesize answer
        {step3_instructions}
        
        Step 4: Add citations
        {step4_instructions}
        
        Final Answer: [Your answer here]
        """
    
    async def on_messages(self, messages, cancellation_token):
        """Generate answer using CoT"""
        query = messages[-1].content
        
        # Single LLM call with CoT prompt
        cot_prompt = self.cot_prompt_template.format(
            step1_instructions="...",
            step2_instructions="...",
            step3_instructions="...",
            step4_instructions="..."
        )
        
        response = await self.model_client.create([
            {"role": "system", "content": cot_prompt},
            {"role": "user", "content": query}
        ])
        
        return response
```

**Result:** ✅ CoT via prompts (no custom loop needed)

---

### 4. Tree-of-Thoughts (ToT): Full Custom Implementation

**MAF Provides:**
- ✅ Agent runtime
- ✅ Message passing

**We Need to Add:**
- ✅ Tree structure for reasoning paths
- ✅ Branching logic
- ✅ Backtracking mechanism
- ✅ Path evaluation
- ✅ Best path selection

**Implementation:**
```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ThoughtNode:
    """Node in reasoning tree"""
    thought: str
    parent: Optional['ThoughtNode']
    children: List['ThoughtNode']
    score: float
    depth: int

class ToTAgent(RoutedAgent):
    """Tree-of-Thoughts implementation"""
    
    def __init__(self, max_depth: int = 5, branching_factor: int = 3):
        super().__init__("tot_agent")
        self.max_depth = max_depth
        self.branching_factor = branching_factor
    
    @message_handler
    async def handle_query(self, message: QueryMessage, ctx: MessageContext):
        """ToT search"""
        # Initialize root
        root = ThoughtNode(
            thought="Initial query analysis",
            parent=None,
            children=[],
            score=0.0,
            depth=0
        )
        
        # Build tree via BFS/DFS
        best_path = await self.search_tree(root, message.content)
        
        # Generate final answer from best path
        return await self.synthesize_answer(best_path)
    
    async def search_tree(self, root: ThoughtNode, query: str) -> List[ThoughtNode]:
        """Search reasoning tree"""
        frontier = [root]
        best_path = None
        best_score = float('-inf')
        
        while frontier:
            node = frontier.pop(0)
            
            # Check if max depth reached
            if node.depth >= self.max_depth:
                if node.score > best_score:
                    best_score = node.score
                    best_path = self.get_path(node)
                continue
            
            # Generate child thoughts
            children = await self.generate_children(node, query)
            
            # Evaluate children
            for child in children:
                child.score = await self.evaluate_thought(child, query)
                node.children.append(child)
                frontier.append(child)
            
            # Prune low-scoring branches
            frontier = self.prune_frontier(frontier)
        
        return best_path
    
    async def generate_children(self, node: ThoughtNode, query: str) -> List[ThoughtNode]:
        """Generate branching thoughts"""
        prompt = f"""
        Query: {query}
        Current thought: {node.thought}
        
        Generate {self.branching_factor} different next reasoning steps.
        """
        # Generate multiple thoughts
        pass
    
    async def evaluate_thought(self, node: ThoughtNode, query: str) -> float:
        """Score thought quality"""
        # Evaluate how promising this reasoning path is
        pass
    
    def prune_frontier(self, frontier: List[ThoughtNode]) -> List[ThoughtNode]:
        """Keep only top-k nodes"""
        return sorted(frontier, key=lambda n: n.score, reverse=True)[:10]
    
    def get_path(self, node: ThoughtNode) -> List[ThoughtNode]:
        """Get path from root to node"""
        path = []
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))
```

**Result:** ✅ Full ToT implementation (most complex)

**Agent-Level ToT Use Cases:**

1. **Intent Parser with ToT** (Phase 2+)
   - Explore multiple query interpretations
   - Backtrack if entity extraction fails
   - Select best interpretation path

2. **Knowledge Retriever with ToT** (Phase 2+)
   - Try multiple retrieval strategies
   - Backtrack if results are poor
   - Explore different query reformulations

3. **Answer Generator with ToT** (Phase 2+)
   - Generate multiple answer candidates
   - Evaluate each for quality
   - Select best answer or combine

**When to Use ToT at Agent Level:**
- ✅ High-stakes decisions (medical, legal, financial)
- ✅ Complex ambiguous queries
- ✅ When single-path reasoning insufficient
- ❌ NOT for MVP (adds significant complexity)
- ❌ NOT for simple queries (overkill)

---

## Summary: What to Use vs What to Build

| Pattern | Use MAF | Build Custom | Complexity | Priority | Notes |
|---------|---------|--------------|------------|----------|-------|
| **Plan-and-Execute** | ✅ Swarm (70%) | ✅ Planning layer (30%) | Medium | **HIGH** - Core orchestration | Sequential only (no parallel) |
| **ReAct** | ✅ Infrastructure | ✅ Full loop | High | **HIGH** - Tool-using agents | Intent Parser, Retriever |
| **CoT** | ✅ Infrastructure | ✅ Prompts only | Low | **HIGH** - Reasoning agents | Generator, Evaluator |
| **ToT** | ✅ Infrastructure | ✅ Full tree search | Very High | **PHASE 2** - Advanced optimization | Agent-level for complex queries |

---

## Recommended Architecture

### Orchestrator Level: Plan-and-Execute (MAF Swarm + Custom Planning)
```python
# Use MAF Swarm for execution
swarm = Swarm([intent_parser, retriever, generator, evaluator])

# Add custom planning layer
class PlanAndExecuteOrchestrator:
    def __init__(self, swarm: Swarm):
        self.swarm = swarm
    
    async def process_query(self, query: str):
        # PLAN (Custom)
        plan = await self.create_plan(query)
        
        # EXECUTE (MAF Swarm)
        return await self.swarm.run_with_plan(plan)
```

### Agent Level: ReAct for Tool-Using Agents
```python
# Custom ReAct implementation on MAF
class IntentParserAgent(ReActAgent):
    def __init__(self):
        super().__init__(tools=[
            ContextLookupTool(),
            EntityExtractionTool(),
            ConfidenceScoringTool()
        ])
```

### Agent Level: CoT for Reasoning Agents
```python
# CoT via prompts (no custom loop)
class AnswerGeneratorAgent(CoTAgent):
    def __init__(self):
        super().__init__(cot_template=ANSWER_GENERATION_COT_PROMPT)
```

---

## Conclusion

**Leverage MAF:**
- ✅ Swarm for agent handoffs (70% of Plan-and-Execute)
- ✅ Runtime infrastructure for all patterns
- ✅ Message passing and lifecycle management

**Build Custom:**
- ✅ Planning layer for Plan-and-Execute (30%)
- ✅ ReAct loop for tool-using agents (100%)
- ✅ CoT prompts for reasoning agents (minimal)
- ⚠️ ToT only if needed (defer to Phase 2)

**Result:** Best of both worlds - leverage MAF infrastructure while implementing AI patterns on top.
