# 🔗 Module Dependency Graph

**Last Updated**: 2026-01-05  
**Purpose**: Fast dependency resolution and conflict detection

---

## Overview

This graph maps dependencies between code modules to enable intelligent task assignment and prevent cascading conflicts. Uses graph algorithms for O(log n) conflict detection.

---

## Module Nodes

Core modules in the MAEVN codebase:

```yaml
modules:
  # Audio Processing Core
  PluginProcessor:
    path: "Source/PluginProcessor.{cpp,h}"
    type: "core"
    complexity: "high"
    dependencies: [OnnxEngine, AIFXEngine, PatternEngine]
    dependents: [PluginEditor]
    real_time_critical: true
    
  # AI/ML Engine
  OnnxEngine:
    path: "Source/OnnxEngine.{cpp,h}"
    type: "ml"
    complexity: "high"
    dependencies: [Models]
    dependents: [AIFXEngine, PluginProcessor]
    real_time_critical: true
    
  # AI Effects Engine
  AIFXEngine:
    path: "Source/AIFXEngine.{cpp,h}"
    type: "processing"
    complexity: "medium"
    dependencies: [OnnxEngine]
    dependents: [PluginProcessor]
    real_time_critical: true
    
  # Pattern Engine
  PatternEngine:
    path: "Source/PatternEngine.{cpp,h}"
    type: "processing"
    complexity: "medium"
    dependencies: []
    dependents: [PluginProcessor]
    real_time_critical: true
    
  # GUI
  PluginEditor:
    path: "Source/PluginEditor.{cpp,h}"
    type: "gui"
    complexity: "medium"
    dependencies: [PluginProcessor]
    dependents: []
    real_time_critical: false
    
  # Models
  Models:
    path: "Models/"
    type: "data"
    complexity: "low"
    dependencies: []
    dependents: [OnnxEngine]
    real_time_critical: false
    
  # Tests
  Tests:
    path: "Tests/"
    type: "testing"
    complexity: "medium"
    dependencies: [PluginProcessor, OnnxEngine, AIFXEngine, PatternEngine, PluginEditor]
    dependents: []
    real_time_critical: false
```

---

## Dependency Graph Visualization

```
Models
  └─> OnnxEngine
        └─> AIFXEngine ──┐
              └──────────┼──> PluginProcessor ──> PluginEditor
                         │
PatternEngine ───────────┘
                         
                         └──> Tests
```

---

## Dependency Types

### Direct Dependencies
Module A depends on module B if A includes B's header files or links to B's implementation.

### Transitive Dependencies
Module A transitively depends on module C if A → B → C.

### Circular Dependencies
**Not allowed** - Will cause compilation errors and design issues.

---

## Conflict Detection Algorithms

### Simple Conflict (O(1))
```python
def has_direct_conflict(module_a: str, module_b: str) -> bool:
    """Check if two modules can be modified simultaneously"""
    # Same module = conflict
    if module_a == module_b:
        return True
    
    # Check if one depends on the other
    if module_b in MODULES[module_a].dependencies:
        return True
    if module_a in MODULES[module_b].dependencies:
        return True
    
    return False
```

### Transitive Conflict (O(log n) with caching)
```python
def has_transitive_conflict(modules_set: set[str]) -> list[tuple]:
    """Check if any modules in set have dependency relationships"""
    conflicts = []
    
    for module_a in modules_set:
        for module_b in modules_set:
            if module_a == module_b:
                continue
            
            # Check if module_a depends on module_b (directly or transitively)
            if is_dependent(module_a, module_b):
                conflicts.append((module_a, module_b, "depends_on"))
    
    return conflicts

def is_dependent(module_a: str, module_b: str, visited=None) -> bool:
    """DFS to check transitive dependency"""
    if visited is None:
        visited = set()
    
    if module_a in visited:
        return False
    visited.add(module_a)
    
    # Check direct dependency
    deps = MODULES[module_a].dependencies
    if module_b in deps:
        return True
    
    # Check transitive
    for dep in deps:
        if is_dependent(dep, module_b, visited):
            return True
    
    return False
```

### Build Order Conflict (O(n))
```python
def check_build_order_safe(new_locks: set[str], existing_locks: set[str]) -> bool:
    """Ensure new locks won't break existing work"""
    # Get all modules in dependency chain
    all_affected = set()
    
    for module in new_locks:
        all_affected.update(get_all_dependents(module))
    
    # Check if any existing locks are affected
    return all_affected.isdisjoint(existing_locks)

def get_all_dependents(module: str) -> set[str]:
    """Get all modules that depend on this one (transitively)"""
    dependents = set(MODULES[module].dependents)
    
    for dep in MODULES[module].dependents:
        dependents.update(get_all_dependents(dep))
    
    return dependents
```

---

## Safe Parallelization Rules

### Rule 1: Independent Modules
Modules with no dependency relationship can be modified in parallel.

**Example**: `PatternEngine` + `Models` ✅

### Rule 2: Leaf Modules
Multiple leaf modules (no dependents) can be modified in parallel.

**Example**: `PluginEditor` + `Tests` ✅

### Rule 3: Dependency Chain
Modules in a dependency chain must be modified sequentially.

**Example**: `Models` → `OnnxEngine` → `AIFXEngine` ❌ (sequential only)

### Rule 4: Shared Dependencies
Modules sharing a common dependency can be parallelized if dependency is stable.

**Example**: `AIFXEngine` + `PatternEngine` ✅ (both depend on stable interfaces)

---

## Critical Path Analysis

### Longest Dependency Chain
```
Models → OnnxEngine → AIFXEngine → PluginProcessor → PluginEditor
(5 modules deep)
```

**Impact**: Changes to `Models` can affect all downstream modules.

### Most Depended-On Module
```
PluginProcessor
├─ Depended on by: PluginEditor, Tests
└─ Depends on: OnnxEngine, AIFXEngine, PatternEngine
```

**Impact**: Changes to `PluginProcessor` interface affect many modules.

### Bottleneck Modules
1. **PluginProcessor** - Central hub, high complexity
2. **OnnxEngine** - Critical ML infrastructure
3. **AIFXEngine** - Core effects processing

**Strategy**: Minimize changes to these; use stable interfaces.

---

## Parallel Work Strategies

### Strategy 1: Vertical Slicing
Work on complete features across dependency chain.

```
Agent A: Models → OnnxEngine → AIFXEngine (Feature X)
Agent B: PatternEngine → PluginProcessor (Feature Y)
```

### Strategy 2: Horizontal Slicing
Work on same layer across modules.

```
Agent A: PluginProcessor interface
Agent B: AIFXEngine + PatternEngine implementations
```

### Strategy 3: Leaf-First Development
Start with leaf modules, work backwards.

```
Phase 1: PluginEditor, Models (parallel)
Phase 2: OnnxEngine, PatternEngine (parallel)
Phase 3: AIFXEngine
Phase 4: PluginProcessor
```

---

## Lock Granularity

### Coarse-Grained (Module-Level)
Lock entire module: `OnnxEngine.*`

**Pros**: Simple, no conflicts  
**Cons**: Limits parallelization

### Fine-Grained (File-Level)
Lock specific files: `OnnxEngine.cpp` vs `OnnxEngine.h`

**Pros**: More parallelization  
**Cons**: Complex, need coordination

**Recommended**: Module-level for most cases, file-level for large modules.

---

## Impact Analysis

### Change Impact Score
```python
def calculate_impact_score(module: str) -> float:
    """Estimate impact of changes to this module"""
    score = 0.0
    
    # Direct dependents
    score += len(MODULES[module].dependents) * 2.0
    
    # Transitive dependents
    all_deps = get_all_dependents(module)
    score += len(all_deps) * 1.0
    
    # Complexity multiplier
    complexity_weight = {
        "low": 0.5,
        "medium": 1.0,
        "high": 2.0
    }
    score *= complexity_weight[MODULES[module].complexity]
    
    # Real-time critical multiplier
    if MODULES[module].real_time_critical:
        score *= 1.5
    
    return score
```

### Impact Rankings
1. **PluginProcessor**: 12.0 (high impact)
2. **OnnxEngine**: 9.0 (high impact)
3. **AIFXEngine**: 6.0 (medium impact)
4. **PatternEngine**: 4.5 (medium impact)
5. **Models**: 3.0 (low impact)
6. **PluginEditor**: 1.0 (low impact)
7. **Tests**: 0.0 (no impact on production)

---

## Recommended Workflow

### Before Starting Work

1. **Identify required modules**
   ```python
   modules = get_modules_for_task(task_id)
   ```

2. **Check impact score**
   ```python
   impact = sum(calculate_impact_score(m) for m in modules)
   if impact > 10.0:
       # High impact - needs careful coordination
       require_architect_review = True
   ```

3. **Detect conflicts**
   ```python
   existing_locks = get_current_locks()
   conflicts = has_transitive_conflict(modules | existing_locks)
   if conflicts:
       # Cannot proceed - coordinate with conflicting agents
       coordinate_with_agents(conflicts)
   ```

4. **Lock modules**
   ```python
   if safe_to_lock(modules):
       lock_modules(modules, agent_id)
   ```

### During Work

1. **Monitor dependent modules**
   - Watch for changes to dependencies
   - Rebuild if upstream changes detected

2. **Minimize interface changes**
   - Keep public APIs stable
   - Use deprecation for breaking changes

3. **Test integration frequently**
   - Build full project, not just your module
   - Run dependent module tests

### After Completion

1. **Release locks promptly**
   ```python
   unlock_modules(modules, agent_id)
   ```

2. **Notify dependent work**
   ```python
   notify_dependents(modules)
   ```

3. **Update dependency graph if needed**
   - New module added?
   - Dependencies changed?

---

## Graph Maintenance

### Auto-Detection
Use build system to extract dependencies:

```bash
# Extract #include directives
grep -r "^#include" Source/ | \
  grep -v "^//" | \
  sed 's/.*#include.*[<"]\(.*\)[>"]/\1/' | \
  sort | uniq

# Parse CMakeLists.txt for target dependencies
grep target_link_libraries CMakeLists.txt
```

### Manual Updates
When adding new module:
1. Add to `modules` section
2. List dependencies and dependents
3. Mark complexity and criticality
4. Update visualization

### Validation
```bash
# Check for circular dependencies
@macf validate-deps --check-cycles

# Check consistency with actual code
@macf validate-deps --check-includes
```

---

## Performance Metrics

| Operation | Complexity | Target Time |
|-----------|------------|-------------|
| Direct conflict check | O(1) | < 0.1ms |
| Transitive conflict check | O(log n) | < 1ms |
| Build order validation | O(n) | < 5ms |
| Impact score calculation | O(n) | < 2ms |
| Full graph validation | O(n²) | < 50ms |

**n** = number of modules (currently 7, expected max 50)

---

## Integration with MACF

Enhanced commands:

```bash
# Check if modules can be locked together
@macf check-modules OnnxEngine AIFXEngine

# Get module impact score
@macf impact PluginProcessor

# Find safe parallel work
@macf suggest-parallel --exclude PluginProcessor

# Validate dependency graph
@macf validate-deps
```

---

## Future Enhancements

1. **Automated Lock Suggestions**
   - AI suggests minimal lock set for task
   
2. **Predictive Conflict Detection**
   - Warn before conflicts occur
   
3. **Smart Work Sequencing**
   - Auto-schedule tasks to minimize conflicts
   
4. **Dependency Visualization**
   - Interactive graph UI for coordinators

---

**Version**: 1.0  
**Last Updated**: 2026-01-05  
**Maintained By**: Architect Agent  
**Review Frequency**: Monthly or when modules added/removed
