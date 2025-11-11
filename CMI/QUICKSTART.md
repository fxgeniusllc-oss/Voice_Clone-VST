# 🚀 Quick Start Guide for AI Agents

## Purpose

This guide helps AI agents quickly get up to speed with the MAEVN Multi-Agent Development System.

---

## ⚡ 5-Minute Quick Start

### 1. Understand Your Role

**Q**: What kind of agent are you?
- **DSP/Audio Expert** → DSP Developer Agent
- **Machine Learning/AI** → AI/ML Agent
- **UI/Frontend** → GUI Developer Agent
- **Testing/QA** → QA/Testing Agent
- **Documentation** → Documentation Agent
- **System Design** → Architect Agent
- **Integration** → Integration Agent
- **DevOps/Build** → DevOps Agent

See `/CMI/agent_roles.md` for detailed role descriptions.

---

### 2. Read the Mission Logs

**Before starting ANY work**:

```bash
# Check active missions
ls -la CMI/active_missions/

# Read the latest mission logs
cat CMI/active_missions/*.md

# Review task assignments
cat CMI/coordination/task_assignments.md
```

This tells you:
- What's currently being worked on
- Which modules are locked
- What tasks are available
- Context from previous work

---

### 3. Check for Available Tasks

Look in `/CMI/coordination/task_assignments.md` for:
- ✅ Tasks matching your expertise
- ✅ Tasks with no dependencies
- ✅ Modules that aren't locked

**Claim a task**:
1. Update `task_assignments.md` with your name
2. Lock any modules you'll modify
3. Set status to "In Progress"

---

### 4. Create Your Mission Log

```bash
# Copy the template
cp CMI/mission_logs/mission_log_template.md \
   CMI/active_missions/mission_XXX_your_feature.md

# Edit with your details
nano CMI/active_missions/mission_XXX_your_feature.md
```

**Fill in**:
- Mission ID (e.g., `mission_010`)
- Objective (what you're building)
- Context (why it's needed)
- Approach (how you'll do it)

---

### 5. Do the Work

**Follow these principles**:

✅ **DO**:
- Make small, incremental changes
- Test frequently
- Update mission log with progress
- Commit with mission ID in message: `feat: Add reverb effect (mission_010)`
- Document decisions and reasoning

❌ **DON'T**:
- Commit `.onnx` binary files
- Break existing tests
- Allocate memory in audio thread
- Make changes without updating mission log

---

### 6. Handoff or Complete

**When handing off**:
- Document current state completely
- List what's done and what remains
- Note any blockers
- Update mission log
- Release module locks (if appropriate)

**When completing**:
- Run all tests
- Update mission log with final status
- Create PR with mission ID
- Release all module locks
- Archive mission log to `CMI/mission_logs/`

---

## 📚 Essential Reading

Read these files **in order**:

1. **`/CMI/README.md`** (5 min)
   - Understand CMI purpose and structure

2. **`/CMI/agent_roles.md`** (10 min)
   - Find your role and responsibilities

3. **`/CMI/operational_ethics.md`** (15 min)
   - Learn required practices and prohibitions

4. **`/CMI/MACF.md`** (20 min)
   - Understand coordination workflows

5. **`/Models/metadata.json`** (5 min)
   - See available models and their metadata

6. **`/Models/LayerMap.md`** (10 min)
   - Understand model architectures

**Total**: ~65 minutes for complete onboarding

---

## 🎯 Common Scenarios

### Scenario: "I'm a DSP agent asked to add a new effect"

1. ✅ Read active missions for context
2. ✅ Create mission log: `mission_XXX_new_effect.md`
3. ✅ Lock `AIFXEngine.*` in task_assignments.md
4. ✅ Implement the effect:
   - Create `NewEffectModule.cpp/.h`
   - Add to `AIFXEngine::initModules()`
   - Ensure real-time safety (< 1ms)
5. ✅ Test thoroughly
6. ✅ Update mission log
7. ✅ Handoff to Integration Agent for GUI

---

### Scenario: "I'm an AI agent asked to create a model"

1. ✅ Read architecture from mission log or Architect Agent
2. ✅ Create mission log: `mission_XXX_model_name.md`
3. ✅ Train and export model:
   - Create export script in `scripts/`
   - Export to ONNX format
   - Optimize for inference
4. ✅ Update `Models/metadata.json`:
   - Add model entry with all metadata
   - Include training details
5. ✅ Update `Models/LayerMap.md`:
   - Document each layer's purpose
   - Explain architecture choices
6. ✅ **DO NOT commit .onnx file**
7. ✅ Test inference performance
8. ✅ Handoff to DSP Agent with model specs

---

### Scenario: "I'm a QA agent asked to review code"

1. ✅ Read the mission log for context
2. ✅ Review code changes:
   - Check for real-time safety
   - Verify no memory allocations in audio thread
   - Look for potential buffer overflows
   - Check numerical stability
3. ✅ Run tests:
   ```bash
   cmake --build Build --config Release
   ctest --test-dir Build
   ```
4. ✅ Profile performance if needed
5. ✅ Document findings in mission log
6. ✅ Approve or request changes

---

### Scenario: "I found a blocker"

1. ✅ Document in mission log with `@macf escalate`
2. ✅ Update task_assignments.md with blocker status
3. ✅ Notify relevant agents via mission log
4. ✅ Suggest potential solutions
5. ✅ Work on different task while blocked

---

## 🚫 Common Mistakes to Avoid

### ❌ Mistake 1: Not reading mission logs first
**Result**: Duplicate work, conflicts, wasted time

**Fix**: Always read CMI before starting

---

### ❌ Mistake 2: Committing .onnx files
**Result**: Git repo bloat, version control issues

**Fix**: Add to .gitignore, provide export script instead

---

### ❌ Mistake 3: Breaking real-time safety
**Result**: Audio dropouts, unusable plugin

**Fix**: 
- Pre-allocate all buffers
- No file I/O in audio thread
- No locks in audio thread
- Profile everything

---

### ❌ Mistake 4: Not updating mission logs
**Result**: Lost context, coordination failures

**Fix**: Update mission log after every significant change

---

### ❌ Mistake 5: Working on locked modules
**Result**: Merge conflicts, duplicate work

**Fix**: Check task_assignments.md first

---

## 🔧 Quick Commands

### Check What's Happening
```bash
# See active missions
ls CMI/active_missions/

# Check task assignments
cat CMI/coordination/task_assignments.md

# See recent commits
git log --oneline -10

# Check git status
git status
```

### Start New Work
```bash
# Create branch
git checkout -b feature/my-feature

# Copy mission template
cp CMI/mission_logs/mission_log_template.md \
   CMI/active_missions/mission_XXX_my_feature.md

# Edit mission log
nano CMI/active_missions/mission_XXX_my_feature.md
```

### During Work
```bash
# Build
cmake --build Build --config Release

# Test
ctest --test-dir Build

# Commit
git add .
git commit -m "feat: Add feature X (mission_XXX)"
```

### Complete Work
```bash
# Final test
cmake --build Build --config Release
ctest --test-dir Build

# Archive mission log
mv CMI/active_missions/mission_XXX_*.md CMI/mission_logs/

# Update task assignments
nano CMI/coordination/task_assignments.md

# Create PR
# (Use GitHub UI or gh CLI)
```

---

## 📋 Pre-Flight Checklist

Before starting work:

- [ ] Read all active mission logs
- [ ] Checked task_assignments.md
- [ ] Identified an available task matching my skills
- [ ] Verified no module locks conflict with my work
- [ ] Created my mission log
- [ ] Updated task_assignments.md with my assignment
- [ ] Read operational_ethics.md

Before committing:

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Real-time safety verified (if DSP code)
- [ ] Mission log updated with progress
- [ ] No .onnx files being committed
- [ ] No credentials or secrets
- [ ] Commit message includes mission ID

Before completing:

- [ ] All success criteria met
- [ ] Documentation updated
- [ ] Mission log updated with final status
- [ ] Handoff notes written (if applicable)
- [ ] Module locks released
- [ ] Task assignments updated
- [ ] Mission log archived

---

## 🆘 Need Help?

### Questions About...

**Your Role**: Read `/CMI/agent_roles.md`

**How to Coordinate**: Read `/CMI/MACF.md`

**What's Allowed**: Read `/CMI/operational_ethics.md`

**Model Architecture**: Read `/Models/LayerMap.md`

**Build Issues**: Check README.md build instructions

**Real-Time Audio**: See operational_ethics.md "Real-Time Audio Processing"

### Escalation Path

1. Document issue in mission log
2. Tag with `@macf escalate`
3. Notify project coordinator
4. Wait for guidance or work on different task

---

## 🎓 Example Mission Log Flow

### 1. Agent receives task: "Add chorus effect"

### 2. Agent creates mission log:
```markdown
## Mission ID: mission_011

### Objective
Add chorus effect to AIFXEngine

### Status
in-progress

### Assigned Agent
DSP Developer Agent
```

### 3. Agent works and updates:
```markdown
### [2025-01-15 10:30 UTC] - DSP Developer Agent
**Work Completed**:
- Created ChorusModule class
- Implemented delay line with LFO modulation
- Added to AIFXEngine::initModules()

**Next Steps**:
- Test with different settings
- Profile CPU usage
```

### 4. Agent completes:
```markdown
### [2025-01-15 14:00 UTC] - DSP Developer Agent
**Status**: completed

**Final Outcome**:
Chorus effect implemented successfully.
CPU usage: 2.1% (acceptable)
Tests passing.

**Handoff to**: Integration Agent for GUI controls
```

### 5. Mission archived to mission_logs/

---

## 💡 Pro Tips

### Tip 1: Use Command Shortcuts
In mission logs, use MACF commands:
- `@macf lock AIFXEngine` - Lock a module
- `@macf handoff mission_011 to Integration-Agent` - Request handoff
- `@macf escalate mission_011` - Escalate a blocker

### Tip 2: Learn from History
Read completed mission logs in `CMI/mission_logs/` to see:
- How others approached similar tasks
- Common pitfalls and solutions
- Best practices

### Tip 3: Communicate Proactively
If you see a potential issue, document it even if it's not your responsibility. Someone else will appreciate the heads-up.

### Tip 4: Test Early, Test Often
Don't wait until the end to test. Build and test after every significant change.

### Tip 5: Profile Performance
For any DSP code, always profile:
```cpp
auto start = std::chrono::high_resolution_clock::now();
// ... your code ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
// Log duration
```

---

## ✅ You're Ready!

Now you understand:
- ✅ Your role in the multi-agent system
- ✅ How to find and claim tasks
- ✅ How to create and update mission logs
- ✅ Required practices and prohibitions
- ✅ How to coordinate with other agents
- ✅ How to complete and handoff work

**Welcome to the Vocal Cloning Quantum Collective!**

Start by reading the active missions and finding a task that matches your expertise.

---

**Version**: 1.0  
**Last Updated**: 2025-01-15
