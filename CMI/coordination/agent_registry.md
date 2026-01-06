# 🤖 Agent Registry

**Last Updated**: 2026-01-05  
**Purpose**: Fast agent capability lookup for efficient task assignment

---

## Overview

This registry maintains a searchable index of all available agents and their capabilities to enable O(1) agent selection for tasks instead of linear scanning.

---

## Active Agents

### Agent Capability Index

| Agent ID | Roles | Skills | Availability | Load | Last Active |
|----------|-------|--------|--------------|------|-------------|
| `dsp-001` | DSP Developer | Audio processing, Real-time optimization, SIMD, JUCE | Available | Low (0/3) | 2026-01-05 |
| `ai-001` | AI/ML Specialist | ONNX, PyTorch, Model optimization, Quantization | Available | Low (0/3) | 2026-01-05 |
| `gui-001` | GUI Developer | JUCE GUI, React, UX design, Accessibility | Available | Low (0/3) | 2026-01-05 |
| `qa-001` | QA/Testing | Unit testing, Integration testing, Performance testing | Available | Low (0/3) | 2026-01-05 |
| `integ-001` | Integration | Cross-module integration, API design, Build systems | Available | Low (0/3) | 2026-01-05 |
| `devops-001` | DevOps | CI/CD, CMake, Build automation, Deployment | Available | Low (0/3) | 2026-01-05 |
| `doc-001` | Documentation | Technical writing, API docs, Tutorials | Available | Low (0/3) | 2026-01-05 |
| `arch-001` | Architect | System design, Architecture, API specifications | Available | Low (0/3) | 2026-01-05 |

### Skill Tag Index

Fast lookup table for skill-based agent matching:

```yaml
audio_processing: [dsp-001]
real_time_optimization: [dsp-001]
simd: [dsp-001]
juce: [dsp-001, gui-001]
onnx: [ai-001]
pytorch: [ai-001]
model_optimization: [ai-001]
quantization: [ai-001]
gui_design: [gui-001]
react: [gui-001]
ux_design: [gui-001]
accessibility: [gui-001]
testing: [qa-001]
unit_testing: [qa-001]
integration_testing: [qa-001, integ-001]
performance_testing: [qa-001]
api_design: [integ-001, arch-001]
build_systems: [integ-001, devops-001]
ci_cd: [devops-001]
cmake: [devops-001]
deployment: [devops-001]
technical_writing: [doc-001]
api_documentation: [doc-001]
system_design: [arch-001]
architecture: [arch-001]
```

---

## Agent Selection Algorithm (Optimized)

### Fast Path O(1) - Single Skill Match
```python
def select_agent_fast(required_skill: str) -> str:
    """O(1) lookup for single-skill tasks"""
    candidates = SKILL_INDEX.get(required_skill, [])
    if not candidates:
        return None
    
    # Filter by availability
    available = [a for a in candidates if AGENTS[a].is_available()]
    if not available:
        return None
    
    # Return least loaded agent
    return min(available, key=lambda a: AGENTS[a].current_load)
```

### Complex Path O(n) - Multi-Skill Match
```python
def select_agent_complex(required_skills: list[str], 
                         required_modules: list[str]) -> str:
    """O(n) lookup for multi-skill tasks with constraints"""
    # Pre-filter by any required skill (intersection)
    candidates = set()
    for skill in required_skills:
        skill_agents = set(SKILL_INDEX.get(skill, []))
        if not candidates:
            candidates = skill_agents
        else:
            candidates &= skill_agents
    
    # Score remaining candidates
    scored = []
    for agent_id in candidates:
        agent = AGENTS[agent_id]
        
        # Skip if module conflict
        if agent.has_module_conflict(required_modules):
            continue
        
        # Calculate match score
        score = (
            agent.skill_match_score(required_skills) * 0.40 +
            agent.availability_score() * 0.25 +
            agent.context_score(required_modules) * 0.20 +
            agent.history_score() * 0.15
        )
        
        scored.append((score, agent_id))
    
    # Return best match
    if not scored:
        return None
    
    return max(scored, key=lambda x: x[0])[1]
```

---

## Agent Status Management

### Availability States

- `available`: Ready for new tasks (load < 3)
- `busy`: Working on tasks (load = 3)
- `offline`: Not currently active
- `blocked`: Waiting on dependencies

### Load Tracking

Each agent can handle up to 3 concurrent tasks:
- **Low**: 0-1 tasks (preferred)
- **Medium**: 2 tasks (acceptable)
- **High**: 3 tasks (at capacity)

### Auto-Update Protocol

Agents should update their status when:
1. Claiming a new task (increment load)
2. Completing a task (decrement load)
3. Going offline/online
4. Changing skill set

**Format**:
```bash
# Update availability
@macf status agent-id available

# Update load
@macf load agent-id 2

# Update skills
@macf skills agent-id add "new_skill"
```

---

## Module Ownership Index

Fast lookup for module conflicts:

```yaml
OnnxEngine.cpp: [ai-001]
OnnxEngine.h: [ai-001]
PluginProcessor.cpp: []
PluginProcessor.h: []
AIFXEngine.cpp: [integ-001]
AIFXEngine.h: [integ-001]
PatternEngine.cpp: []
PatternEngine.h: []
PluginEditor.cpp: [gui-001]
PluginEditor.h: [gui-001]
```

**Empty list `[]`** = Module is unlocked and available

---

## Performance Metrics

### Target Performance
- Agent lookup: < 1ms (single skill)
- Agent matching: < 10ms (multi-skill with constraints)
- Status update: < 5ms
- Module conflict check: < 1ms

### Monitoring
Track these metrics:
- Average time to assign task
- Task reassignment rate
- Module conflict rate
- Agent utilization rate

---

## Usage Examples

### Example 1: Quick Single-Skill Assignment
```python
# Need DSP work
agent = select_agent_fast("audio_processing")
# Returns: dsp-001 (instant lookup)
```

### Example 2: Complex Multi-Skill Assignment
```python
# Need AI model that integrates with GUI
agent = select_agent_complex(
    required_skills=["onnx", "juce"],
    required_modules=["OnnxEngine.cpp", "PluginEditor.cpp"]
)
# Returns: None (module conflict - both locked)
# Need to coordinate or wait
```

### Example 3: Finding Available Integration Agent
```python
# Need someone to merge DSP and AI work
agent = select_agent_complex(
    required_skills=["integration_testing", "api_design"],
    required_modules=["AIFXEngine.cpp"]
)
# Returns: integ-001 (already owns the module)
```

---

## Registry Maintenance

### Adding New Agent
1. Add entry to "Active Agents" table
2. Update skill tag index with agent's skills
3. Set initial load to 0
4. Set status to "available"

### Removing Agent
1. Verify agent has no active tasks
2. Release all module locks
3. Remove from Active Agents table
4. Remove from skill tag index

### Updating Agent Skills
1. Modify agent entry in Active Agents table
2. Update skill tag index:
   - Add agent to new skill entries
   - Remove agent from old skill entries

---

## Best Practices

1. **Keep registry updated**: Update immediately when claiming/completing tasks
2. **Accurate skill tags**: Only list skills you're proficient in
3. **Honest availability**: Mark offline if you can't respond within 2 hours
4. **Release locks promptly**: Don't hold module locks longer than needed
5. **Distribute load**: Don't claim new tasks if already at capacity

---

## Integration with MACF

The agent registry enhances MACF efficiency:

- **Before**: Linear scan through mission logs to find qualified agents
- **After**: O(1) or O(n) lookup using indexed registry
- **Speedup**: 10-100x faster for large agent pools

### MACF Commands Enhanced

```bash
# Query agent capabilities
@macf query-agent agent-id

# Find agent for task
@macf find-agent --skills "onnx,real_time" --modules "OnnxEngine.cpp"

# Update agent status
@macf update-agent agent-id --load 2 --status busy

# List available agents
@macf list-agents --available --skills "testing"
```

---

## Version History

- **v1.0** (2026-01-05): Initial registry with 8 agent roles, skill indexing, and O(1) lookup

---

**Maintained By**: All Agents (self-service updates)  
**Review Frequency**: Weekly  
**Performance SLA**: <10ms for any agent query
