combines **A2A Protocol compliance** with **Motia's native queue system** - no HTTP needed!

## How It Works

### The Magic: A2A + Motia Integration

The `MotiaA2AAgent` wrapper bridges the two systems:

```
A2A Protocol           Motia Queue System
─────────────         ──────────────────
AgentCard       ←→    Agent Registry (State)
RequestContext  ←→    Motia Event Input
ExecutionEventBus ←→  Motia context.emit()
AgentExecutor   ←→    Event Step Handler
```

### Key Components

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

## Benefits of This Approach

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

## A2A Events Flow Through Motia

```
A2A Agent emits:                 Motia queues as:
─────────────────               ───────────────────
thought("Planning...")     →    'a2a.thought' event
action("web_search")       →    'a2a.action' event  
progress(0.5, "Working")   →    'agent.progress' event
result({...})              →    'task.completed' event
```

All these events:
- Are automatically queued by Motia
- Visible in Workbench
- Can be subscribed to by other Steps
- Have built-in retry logic

## Example: Multi-Agent A2A Workflow

```
1. API receives request
   ↓ (Motia queue)
2. Orchestrator discovers A2A researcher
   ↓ (Motia queue)
3. A2A Researcher executes
   - Emits A2A thoughts
   - Emits A2A actions
   - Returns A2A result
   ↓ (All via Motia queues)
4. Orchestrator discovers A2A analyzer
   ↓ (Motia queue)
5. A2A Analyzer executes
   - Uses A2A protocol
   - Emits progress updates
   - Returns structured result
   ↓ (Motia queue)
6. Workflow completes
```

## Observability

In Motia Workbench, you'll see:

```
📊 Workflow Diagram:
API → Orchestrator → Researcher (A2A) → Analyzer (A2A)

📝 Event Log:
• agent.task.researcher_a2a queued
• a2a.thought: "Planning research strategy..."
• a2a.action: web_search
• agent.progress: 30% - Starting research
• a2a.action: synthesize_results
• agent.progress: 100% - Research complete
• task.completed: {findings...}

🔍 Trace View:
Shows full execution path with A2A protocol events
```

## Comparison: HTTP A2A vs Motia A2A

| Aspect | HTTP-based A2A | Motia-based A2A |
|--------|----------------|-----------------|
| **Setup** | Multiple servers + ports | Single Motia runtime |
| **Communication** | HTTP requests | Motia events |
| **Queuing** | External (Redis/etc) | Built-in (automatic) |
| **Discovery** | Service registry | Motia state |
| **Retry** | Manual | Automatic |
| **Observability** | Separate tools | Built-in Workbench |
| **Latency** | Network overhead | In-process events |
| **Deployment** | Multiple services | Single deployment |


Add more advanced A2A features like streaming responses.