# Agent Registration & Discovery Design

## Overview

This document defines a comprehensive metadata-driven agent discovery system for the Microsoft Agent Framework. The design enables dynamic agent registration, capability-based discovery, and runtime coordination without requiring external service registries.

## Design Principles

1. **Metadata-Driven**: Agents self-describe their capabilities, patterns, and interfaces
2. **Local-First**: Discovery happens in-process without external dependencies
3. **Type-Safe**: Leverage Pydantic for schema validation
4. **Observable**: Full integration with Langfuse for discovery tracing
5. **Extensible**: Easy to add new metadata fields and discovery criteria

---

## 1. Agent Metadata Schema

### 1.1 Core Metadata Structure

```python
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum

class AgentPattern(str, Enum):
    """Orchestration patterns supported by agents"""
    REACT = "react"
    CHAIN_OF_THOUGHT = "cot"
    PLAN_AND_EXECUTE = "plan_and_execute"
    RULE_BASED = "rule_based"

class AgentCapability(str, Enum):
    """Standard agent capabilities"""
    # RAG Capabilities
    INTENT_PARSING = "intent_parsing"
    QUERY_ROUTING = "query_routing"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    ANSWER_GENERATION = "answer_generation"
    QUALITY_EVALUATION = "quality_evaluation"
    
    # General Capabilities
    TEXT_GENERATION = "text_generation"
    TEXT_ANALYSIS = "text_analysis"
    SUMMARIZATION = "summarization"
    ENTITY_EXTRACTION = "entity_extraction"
    CLASSIFICATION = "classification"
    
    # Tool Use
    TOOL_USE = "tool_use"
    API_INTEGRATION = "api_integration"
    DATABASE_ACCESS = "database_access"

class MessageType(str, Enum):
    """Supported message types"""
    AGENT_MESSAGE = "AgentMessage"
    AGENT_TASK = "AgentTask"
    AGENT_RESULT = "AgentResult"
    AGENT_ERROR = "AgentError"

class AgentMetadata(BaseModel):
    """Comprehensive agent metadata for discovery"""
    
    # Identity
    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    version: str = Field(default="1.0.0", description="Agent version")
    description: str = Field(..., description="Agent purpose and functionality")
    
    # Capabilities
    capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="List of agent capabilities"
    )
    pattern: AgentPattern = Field(
        ...,
        description="Orchestration pattern used by agent"
    )
    
    # Communication
    supported_messages: List[MessageType] = Field(
        default_factory=list,
        description="Message types this agent can handle"
    )
    input_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="JSON schema for expected inputs"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="JSON schema for outputs"
    )
    
    # Dependencies
    required_agents: List[str] = Field(
        default_factory=list,
        description="Agent IDs this agent depends on"
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="External tools/APIs required"
    )
    
    # Performance
    avg_latency_ms: Optional[int] = Field(
        None,
        description="Average response latency"
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum concurrent requests"
    )
    
    # Observability
    langfuse_tags: List[str] = Field(
        default_factory=list,
        description="Tags for Langfuse tracing"
    )
    
    # Custom metadata
    custom: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata fields"
    )
    
    class Config:
        use_enum_values = True
```

### 1.2 Agent Implementation with Metadata

```python
from agent_framework import ChatAgent, ai_function
from agent_framework._clients import ChatClientProtocol

class IntentParserAgent(ChatAgent):
    """Intent Parser Agent with comprehensive metadata"""
    
    def __init__(self, chat_client: ChatClientProtocol):
        super().__init__(
            id="intent-parser",
            chat_client=chat_client
        )
        
        # Register metadata
        self.metadata = AgentMetadata(
            agent_id="intent-parser",
            name="Intent Parser Agent",
            version="1.0.0",
            description="Extracts user intent and entities using ReAct pattern",
            capabilities=[
                AgentCapability.INTENT_PARSING,
                AgentCapability.ENTITY_EXTRACTION,
                AgentCapability.TOOL_USE
            ],
            pattern=AgentPattern.REACT,
            supported_messages=[
                MessageType.AGENT_MESSAGE,
                MessageType.AGENT_TASK
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "context": {"type": "object"}
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "entities": {"type": "object"},
                    "confidence": {"type": "number"}
                }
            },
            required_agents=[],
            required_tools=["context_lookup", "entity_extractor"],
            avg_latency_ms=500,
            max_concurrent_requests=20,
            langfuse_tags=["intent-parsing", "react-agent"],
            custom={
                "model": "gpt-4",
                "max_iterations": 5
            }
        )
    
    @ai_function
    async def parse_intent(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Parse user intent with ReAct loop"""
        # Implementation...
        pass
```

---

## 2. Enhanced Runtime Registry

### 2.1 Agent Registry Implementation

```python
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import asyncio

class AgentRegistry:
    """
    Centralized registry for agent discovery and coordination
    
    Features:
    - Metadata-driven discovery
    - Capability-based queries
    - Dependency resolution
    - Health monitoring
    """
    
    def __init__(self):
        self._agents: Dict[str, ChatAgent] = {}
        self._metadata: Dict[str, AgentMetadata] = {}
        self._capability_index: Dict[AgentCapability, List[str]] = defaultdict(list)
        self._pattern_index: Dict[AgentPattern, List[str]] = defaultdict(list)
        self._health_status: Dict[str, bool] = {}
    
    def register_agent(
        self,
        agent: ChatAgent,
        metadata: AgentMetadata
    ) -> None:
        """
        Register agent with metadata
        
        Automatically indexes by capabilities and patterns for fast lookup.
        """
        agent_id = metadata.agent_id
        
        # Store agent and metadata
        self._agents[agent_id] = agent
        self._metadata[agent_id] = metadata
        self._health_status[agent_id] = True
        
        # Index by capabilities
        for capability in metadata.capabilities:
            self._capability_index[capability].append(agent_id)
        
        # Index by pattern
        self._pattern_index[metadata.pattern].append(agent_id)
        
        print(f"✓ Registered agent: {metadata.name} ({agent_id})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent and clean up indexes"""
        if agent_id not in self._agents:
            return
        
        metadata = self._metadata[agent_id]
        
        # Remove from capability index
        for capability in metadata.capabilities:
            if agent_id in self._capability_index[capability]:
                self._capability_index[capability].remove(agent_id)
        
        # Remove from pattern index
        if agent_id in self._pattern_index[metadata.pattern]:
            self._pattern_index[metadata.pattern].remove(agent_id)
        
        # Remove from storage
        del self._agents[agent_id]
        del self._metadata[agent_id]
        del self._health_status[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[ChatAgent]:
        """Get agent by ID"""
        return self._agents.get(agent_id)
    
    def get_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        """Get agent metadata by ID"""
        return self._metadata.get(agent_id)
    
    def find_agents_by_capability(
        self,
        capability: AgentCapability
    ) -> List[ChatAgent]:
        """Find all agents with specific capability"""
        agent_ids = self._capability_index.get(capability, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def find_agents_by_pattern(
        self,
        pattern: AgentPattern
    ) -> List[ChatAgent]:
        """Find all agents using specific pattern"""
        agent_ids = self._pattern_index.get(pattern, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def find_agents_by_message_type(
        self,
        message_type: MessageType
    ) -> List[ChatAgent]:
        """Find agents that support specific message type"""
        matching_agents = []
        for agent_id, metadata in self._metadata.items():
            if message_type in metadata.supported_messages:
                matching_agents.append(self._agents[agent_id])
        return matching_agents
    
    def list_all_agents(self) -> List[AgentMetadata]:
        """List metadata for all registered agents"""
        return list(self._metadata.values())
    
    def resolve_dependencies(self, agent_id: str) -> List[str]:
        """
        Resolve agent dependencies recursively
        
        Returns ordered list of agent IDs (dependencies first)
        """
        if agent_id not in self._metadata:
            return []
        
        visited = set()
        result = []
        
        def dfs(aid: str):
            if aid in visited:
                return
            visited.add(aid)
            
            metadata = self._metadata.get(aid)
            if not metadata:
                return
            
            # Visit dependencies first
            for dep_id in metadata.required_agents:
                dfs(dep_id)
            
            result.append(aid)
        
        dfs(agent_id)
        return result
    
    async def health_check(self, agent_id: str) -> bool:
        """Check if agent is healthy"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Simple ping test
            agent = self._agents[agent_id]
            # Could implement actual health check method
            self._health_status[agent_id] = True
            return True
        except Exception:
            self._health_status[agent_id] = False
            return False
    
    def get_healthy_agents(self) -> List[str]:
        """Get list of healthy agent IDs"""
        return [aid for aid, healthy in self._health_status.items() if healthy]
```

### 2.2 Discovery API

```python
class AgentDiscoveryAPI:
    """
    High-level API for agent discovery
    
    Provides convenient methods for common discovery patterns.
    """
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
    
    def discover_for_task(
        self,
        required_capabilities: List[AgentCapability],
        preferred_pattern: Optional[AgentPattern] = None
    ) -> List[ChatAgent]:
        """
        Discover agents suitable for a task
        
        Args:
            required_capabilities: Capabilities needed for task
            preferred_pattern: Preferred orchestration pattern
        
        Returns:
            List of matching agents, sorted by relevance
        """
        # Find agents with all required capabilities
        candidates = set()
        for capability in required_capabilities:
            agents = self.registry.find_agents_by_capability(capability)
            if not candidates:
                candidates = set(a.id for a in agents)
            else:
                candidates &= set(a.id for a in agents)
        
        # Filter by pattern if specified
        if preferred_pattern:
            pattern_agents = set(
                a.id for a in self.registry.find_agents_by_pattern(preferred_pattern)
            )
            candidates &= pattern_agents
        
        # Return agents sorted by health and latency
        result = []
        for agent_id in candidates:
            agent = self.registry.get_agent(agent_id)
            metadata = self.registry.get_metadata(agent_id)
            if agent and metadata and self.registry._health_status.get(agent_id):
                result.append((agent, metadata.avg_latency_ms or 999999))
        
        result.sort(key=lambda x: x[1])
        return [agent for agent, _ in result]
    
    def get_agent_chain(
        self,
        start_capability: AgentCapability,
        end_capability: AgentCapability
    ) -> List[ChatAgent]:
        """
        Find agent chain from start to end capability
        
        Useful for building pipelines.
        """
        # Simple implementation - could use graph search
        start_agents = self.registry.find_agents_by_capability(start_capability)
        end_agents = self.registry.find_agents_by_capability(end_capability)
        
        # For now, return both sets
        # In production, would build dependency graph
        return start_agents + end_agents
    
    def recommend_agent(
        self,
        query_description: str,
        available_capabilities: List[AgentCapability]
    ) -> Optional[ChatAgent]:
        """
        Recommend best agent for a query using LLM
        
        Uses GPT-4 to analyze query and recommend agent.
        """
        # Could use LLM to analyze query and recommend agent
        # For now, simple heuristic
        for capability in available_capabilities:
            agents = self.registry.find_agents_by_capability(capability)
            if agents:
                return agents[0]
        return None
```

---

## 3. Integration with Multi-Agent RAG System

### 3.1 Register All RAG Agents

```python
from agent_framework._clients import ChatClientProtocol

def setup_rag_agent_registry(chat_client: ChatClientProtocol) -> AgentRegistry:
    """
    Setup complete agent registry for RAG system
    
    Registers all specialized agents with metadata.
    """
    registry = AgentRegistry()
    
    # 1. Intent Parser Agent (ReAct)
    intent_parser = IntentParserAgent(chat_client)
    registry.register_agent(
        agent=intent_parser,
        metadata=intent_parser.metadata
    )
    
    # 2. Knowledge Retriever Agent (ReAct)
    retriever_metadata = AgentMetadata(
        agent_id="knowledge-retriever",
        name="Knowledge Retriever Agent",
        version="1.0.0",
        description="Multi-strategy document retrieval using ReAct",
        capabilities=[
            AgentCapability.KNOWLEDGE_RETRIEVAL,
            AgentCapability.TOOL_USE
        ],
        pattern=AgentPattern.REACT,
        supported_messages=[MessageType.AGENT_TASK],
        required_tools=["semantic_search", "metadata_filter", "rerank"],
        avg_latency_ms=800,
        langfuse_tags=["retrieval", "react-agent"]
    )
    # Create and register retriever...
    
    # 3. Answer Generator Agent (CoT)
    generator_metadata = AgentMetadata(
        agent_id="answer-generator",
        name="Answer Generator Agent",
        version="1.0.0",
        description="Generates grounded answers using Chain-of-Thought",
        capabilities=[
            AgentCapability.ANSWER_GENERATION,
            AgentCapability.TEXT_GENERATION
        ],
        pattern=AgentPattern.CHAIN_OF_THOUGHT,
        supported_messages=[MessageType.AGENT_TASK],
        required_agents=["knowledge-retriever"],
        avg_latency_ms=1200,
        langfuse_tags=["generation", "cot-agent"]
    )
    # Create and register generator...
    
    # 4. Evaluator Agent (CoT)
    evaluator_metadata = AgentMetadata(
        agent_id="evaluator",
        name="Evaluator Agent",
        version="1.0.0",
        description="Evaluates answer quality using Chain-of-Thought",
        capabilities=[
            AgentCapability.QUALITY_EVALUATION,
            AgentCapability.TEXT_ANALYSIS
        ],
        pattern=AgentPattern.CHAIN_OF_THOUGHT,
        supported_messages=[MessageType.AGENT_TASK],
        required_agents=["answer-generator"],
        avg_latency_ms=600,
        langfuse_tags=["evaluation", "cot-agent"]
    )
    # Create and register evaluator...
    
    # 5. Orchestrator Agent (Plan-and-Execute)
    orchestrator_metadata = AgentMetadata(
        agent_id="orchestrator",
        name="Orchestrator Agent",
        version="1.0.0",
        description="Coordinates RAG pipeline using Plan-and-Execute",
        capabilities=[],  # Orchestrator doesn't have domain capabilities
        pattern=AgentPattern.PLAN_AND_EXECUTE,
        supported_messages=[MessageType.AGENT_MESSAGE],
        required_agents=[
            "intent-parser",
            "knowledge-retriever",
            "answer-generator",
            "evaluator"
        ],
        avg_latency_ms=3000,
        langfuse_tags=["orchestration", "plan-execute"]
    )
    # Create and register orchestrator...
    
    return registry
```

### 3.2 Use Discovery in Orchestrator

```python
class OrchestratorAgent:
    """Orchestrator using agent discovery"""
    
    def __init__(
        self,
        registry: AgentRegistry,
        discovery: AgentDiscoveryAPI
    ):
        self.registry = registry
        self.discovery = discovery
    
    async def process_query(self, query: str, user_id: str) -> Answer:
        """
        Process query using discovered agents
        
        Dynamically discovers and coordinates agents.
        """
        # 1. Discover intent parser
        intent_agents = self.registry.find_agents_by_capability(
            AgentCapability.INTENT_PARSING
        )
        if not intent_agents:
            raise RuntimeError("No intent parser available")
        
        intent_parser = intent_agents[0]
        intent = await intent_parser.parse_intent(query)
        
        # 2. Discover retriever
        retriever_agents = self.registry.find_agents_by_capability(
            AgentCapability.KNOWLEDGE_RETRIEVAL
        )
        if not retriever_agents:
            raise RuntimeError("No retriever available")
        
        retriever = retriever_agents[0]
        documents = await retriever.retrieve(query, intent)
        
        # 3. Discover generator
        generator_agents = self.registry.find_agents_by_capability(
            AgentCapability.ANSWER_GENERATION
        )
        if not generator_agents:
            raise RuntimeError("No generator available")
        
        generator = generator_agents[0]
        answer = await generator.generate(query, documents)
        
        # 4. Discover evaluator
        evaluator_agents = self.registry.find_agents_by_capability(
            AgentCapability.QUALITY_EVALUATION
        )
        if evaluator_agents:
            evaluator = evaluator_agents[0]
            evaluation = await evaluator.evaluate(query, documents, answer)
            answer.evaluation = evaluation
        
        return answer
```

---

## 4. Observability Integration

### 4.1 Langfuse Tracing for Discovery

```python
from langfuse import Langfuse
from datetime import datetime

class ObservableAgentRegistry(AgentRegistry):
    """Agent registry with Langfuse tracing"""
    
    def __init__(self, langfuse_client: Langfuse):
        super().__init__()
        self.langfuse = langfuse_client
    
    def register_agent(self, agent: ChatAgent, metadata: AgentMetadata) -> None:
        """Register agent with Langfuse event"""
        with self.langfuse.trace("agent_registration") as trace:
            trace.span("register", metadata={
                "agent_id": metadata.agent_id,
                "capabilities": [c.value for c in metadata.capabilities],
                "pattern": metadata.pattern.value
            })
            super().register_agent(agent, metadata)
    
    def find_agents_by_capability(
        self,
        capability: AgentCapability
    ) -> List[ChatAgent]:
        """Find agents with Langfuse tracing"""
        with self.langfuse.trace("agent_discovery") as trace:
            trace.span("capability_search", metadata={
                "capability": capability.value,
                "timestamp": datetime.now().isoformat()
            })
            
            agents = super().find_agents_by_capability(capability)
            
            trace.span("discovery_result", metadata={
                "found_count": len(agents),
                "agent_ids": [a.id for a in agents]
            })
            
            return agents
```

---

## 5. Usage Examples

### 5.1 Basic Registration and Discovery

```python
# Setup
chat_client = AzureOpenAIChatClient(...)
registry = AgentRegistry()
discovery = AgentDiscoveryAPI(registry)

# Register agents
intent_parser = IntentParserAgent(chat_client)
registry.register_agent(intent_parser, intent_parser.metadata)

# Discover by capability
parsers = registry.find_agents_by_capability(AgentCapability.INTENT_PARSING)
print(f"Found {len(parsers)} intent parsers")

# Discover by pattern
react_agents = registry.find_agents_by_pattern(AgentPattern.REACT)
print(f"Found {len(react_agents)} ReAct agents")

# Resolve dependencies
deps = registry.resolve_dependencies("orchestrator")
print(f"Orchestrator depends on: {deps}")
```

### 5.2 Task-Based Discovery

```python
# Find agents for specific task
agents = discovery.discover_for_task(
    required_capabilities=[
        AgentCapability.INTENT_PARSING,
        AgentCapability.ENTITY_EXTRACTION
    ],
    preferred_pattern=AgentPattern.REACT
)

if agents:
    best_agent = agents[0]  # Sorted by latency
    result = await best_agent.parse_intent("What is RAG?")
```

### 5.3 Health Monitoring

```python
# Check agent health
healthy = await registry.health_check("intent-parser")
print(f"Intent parser healthy: {healthy}")

# Get all healthy agents
healthy_agents = registry.get_healthy_agents()
print(f"Healthy agents: {healthy_agents}")
```

---

## 6. Benefits

1. **Dynamic Discovery**: Agents can be added/removed at runtime
2. **Type Safety**: Pydantic ensures metadata validity
3. **Fast Lookup**: Indexed by capabilities and patterns
4. **Dependency Resolution**: Automatic dependency ordering
5. **Observable**: Full Langfuse integration
6. **Extensible**: Easy to add new metadata fields
7. **No External Dependencies**: Pure in-process discovery

---

## 7. Future Enhancements

1. **Distributed Discovery**: Extend to multi-process/multi-host
2. **Load Balancing**: Route to least-loaded agent
3. **Version Management**: Support multiple agent versions
4. **A/B Testing**: Route percentage of traffic to new versions
5. **Circuit Breaker**: Automatic failover for unhealthy agents
6. **Metrics Collection**: Track discovery performance
7. **LLM-Based Matching**: Use GPT-4 to match queries to agents
