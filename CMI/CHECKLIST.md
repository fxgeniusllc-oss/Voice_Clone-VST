# 📋 Multi-Agent Development Checklist

## Purpose
This checklist ensures all agents follow the proper workflow when contributing to MAEVN.

---

## 🆕 First-Time Setup (One Time Only)

### For AI Agents
- [ ] Read `/CMI/README.md` (5 min)
- [ ] Read `/CMI/agent_roles.md` (10 min)
- [ ] Read `/CMI/operational_ethics.md` (15 min)
- [ ] Read `/CMI/MACF.md` (20 min)
- [ ] Read `/CMI/QUICKSTART.md` (10 min)
- [ ] Review example mission log: `/CMI/mission_logs/mission_009_spectral_ghost_choir.md` (10 min)
- [ ] Understand your specific agent role
- [ ] Review Models documentation if working with AI/ML

### For Human Developers
- [ ] Same as above, plus:
- [ ] Set up development environment (JUCE, ONNX Runtime, CMake)
- [ ] Build the project successfully
- [ ] Run existing tests
- [ ] Review existing codebase structure

**Total Onboarding Time**: ~70 minutes

---

## 🚀 Starting New Work (Every Task)

### 1. Pre-Work Review
- [ ] Read all files in `/CMI/active_missions/`
- [ ] Review `/CMI/coordination/task_assignments.md`
- [ ] Check for module locks that might conflict
- [ ] Identify dependencies on other ongoing work
- [ ] Verify no one else is working on the same task

### 2. Task Selection
- [ ] Found a task matching my agent role
- [ ] Task has no unresolved dependencies
- [ ] Required modules are not locked by others
- [ ] Task priority aligns with project needs

### 3. Task Claiming
- [ ] Updated `/CMI/coordination/task_assignments.md`:
  - [ ] Added task to "Currently Active Assignments"
  - [ ] Set status to "In Progress"
  - [ ] Added my agent name
  - [ ] Set start date and ETA
  - [ ] Locked any modules I'll be modifying

### 4. Mission Log Creation
- [ ] Copied `/CMI/mission_logs/mission_log_template.md` to `/CMI/active_missions/`
- [ ] Renamed to `mission_XXX_descriptive_name.md` (using next available number)
- [ ] Filled in all metadata:
  - [ ] Mission ID
  - [ ] Status (set to "in-progress")
  - [ ] Created timestamp
  - [ ] My agent role
  - [ ] Related modules
- [ ] Filled in objective section:
  - [ ] Primary goal (clear, concise)
  - [ ] Success criteria (specific, measurable)
- [ ] Filled in context section:
  - [ ] Background/rationale
  - [ ] Dependencies
  - [ ] Related issues/PRs
- [ ] Filled in technical details:
  - [ ] Approach
  - [ ] Files to modify
  - [ ] New files to create
- [ ] Created first progress log entry

---

## 💻 During Development (Continuous)

### Every Significant Change
- [ ] Made small, incremental change
- [ ] Tested the change:
  - [ ] Code compiles without errors or warnings
  - [ ] Existing tests still pass
  - [ ] New functionality works as expected
- [ ] Updated mission log with progress entry:
  - [ ] Timestamp
  - [ ] Work completed
  - [ ] Decisions made (with rationale)
  - [ ] Issues encountered
  - [ ] Next steps
- [ ] Committed with proper message format:
  ```
  <type>: <description> (mission_XXX)
  
  Examples:
  feat: Add chorus effect (mission_011)
  fix: Resolve buffer underrun in OnnxEngine (mission_012)
  docs: Update LayerMap for new model (mission_013)
  ```

### Code Quality Checks (Every Few Changes)
- [ ] No compiler warnings
- [ ] Code follows JUCE style guide (braces on new lines, 4-space indent)
- [ ] Doxygen comments for public APIs
- [ ] No magic numbers (use named constants)
- [ ] Descriptive variable and function names

### Real-Time Safety Checks (If DSP Code)
- [ ] No memory allocation in audio thread
- [ ] No file I/O in audio thread
- [ ] No locks/mutexes in audio thread
- [ ] All buffers pre-allocated
- [ ] Processing time < 1ms per buffer (profiled)

### Model Development Checks (If AI/ML Work)
- [ ] Export script created in `/scripts/`
- [ ] Model optimized for inference
- [ ] NOT committing .onnx binary file
- [ ] Updated `/Models/metadata.json` with complete metadata:
  - [ ] Model info (name, version, description)
  - [ ] Architecture details
  - [ ] Input/output formats
  - [ ] Performance metrics
  - [ ] Training metadata
  - [ ] Checksum
- [ ] Updated `/Models/LayerMap.md` with layer-by-layer explanation
- [ ] Tested model inference:
  - [ ] No NaN or Inf outputs
  - [ ] Inference time acceptable
  - [ ] Memory usage acceptable

---

## 🔄 Handoff to Another Agent (When Needed)

### Preparation
- [ ] Updated mission log with complete handoff notes:
  - [ ] Current state description
  - [ ] What's done (list all completed work)
  - [ ] What remains (list remaining work)
  - [ ] Important context for next agent
  - [ ] Recommended next steps
  - [ ] Any blockers or issues
- [ ] Updated `/CMI/coordination/task_assignments.md`:
  - [ ] Changed assigned agent to next agent
  - [ ] Added handoff note
  - [ ] Released module locks if appropriate (or noted they remain)
- [ ] Committed all work in progress
- [ ] Notified next agent (if known) via mission log

---

## ✅ Completing Work (Task Finished)

### Pre-Completion Validation
- [ ] All success criteria met (from mission log)
- [ ] All tests pass:
  ```bash
  cmake --build Build --config Release
  ctest --test-dir Build
  ```
- [ ] Code compiles without warnings
- [ ] Documentation updated:
  - [ ] README.md (if user-facing change)
  - [ ] Code comments (API documentation)
  - [ ] LayerMap.md (if model work)
  - [ ] metadata.json (if model work)
- [ ] Performance validated:
  - [ ] CPU usage acceptable
  - [ ] Memory usage acceptable
  - [ ] Real-time constraints met (if DSP)
- [ ] Security check:
  - [ ] No vulnerabilities introduced
  - [ ] No secrets committed
  - [ ] Input validation present

### Mission Log Completion
- [ ] Added final progress log entry
- [ ] Filled in "Final Status" section:
  - [ ] Completion date
  - [ ] Final outcome summary
  - [ ] Lessons learned
  - [ ] Follow-up actions
- [ ] Updated all checklist items
- [ ] Set status to "completed"
- [ ] Filled in metrics section:
  - [ ] Performance impact
  - [ ] Code changes (lines added/removed/modified)
  - [ ] Tests added

### Cleanup
- [ ] Updated `/CMI/coordination/task_assignments.md`:
  - [ ] Moved task from "Currently Active" to "Recently Completed"
  - [ ] Released all module locks
  - [ ] Updated completion date
  - [ ] Added link to mission log
- [ ] Moved mission log from `/CMI/active_missions/` to `/CMI/mission_logs/`
- [ ] Created Pull Request:
  - [ ] Title includes mission ID
  - [ ] Description references mission log
  - [ ] All commits included
  - [ ] Requested appropriate reviewer(s)

---

## 🔍 Code Review (For QA/Review Agents)

### Initial Review
- [ ] Read the mission log for full context
- [ ] Understand the objective and approach
- [ ] Review all changed files

### Code Quality Review
- [ ] Code compiles without warnings
- [ ] Follows project style guide
- [ ] Proper error handling
- [ ] No code duplication
- [ ] Appropriate use of comments
- [ ] Public APIs documented

### Functional Review
- [ ] Logic is correct
- [ ] Edge cases handled
- [ ] No obvious bugs
- [ ] Meets success criteria from mission log

### Performance Review (If DSP Code)
- [ ] No memory allocation in audio thread
- [ ] No blocking operations in audio thread
- [ ] Buffer sizes appropriate
- [ ] Processing time within budget
- [ ] Profiling results acceptable

### Security Review
- [ ] Input validation present
- [ ] No buffer overflows
- [ ] No injection vulnerabilities
- [ ] No hardcoded secrets

### Testing Review
- [ ] Adequate test coverage
- [ ] Tests actually test the functionality
- [ ] Edge cases tested
- [ ] Tests pass consistently

### Documentation Review
- [ ] README updated (if needed)
- [ ] API documentation complete
- [ ] Mission log complete and accurate
- [ ] Model documentation (if applicable)

### Review Completion
- [ ] Documented findings in mission log or PR comments
- [ ] Requested changes if needed, or approved
- [ ] Updated mission log with review notes

---

## 🚨 Blocked / Issue Escalation

### When Blocked
- [ ] Documented blocker in mission log with details
- [ ] Updated status to "blocked"
- [ ] Added to "Active Blockers" section with:
  - [ ] Description of blocker
  - [ ] What's needed to unblock
  - [ ] Suggested solutions (if any)
- [ ] Updated `/CMI/coordination/task_assignments.md` status
- [ ] Tagged with `@macf escalate mission_XXX` in mission log
- [ ] Switched to different task (if possible)

### When Discovering Issues
- [ ] Documented issue in mission log
- [ ] If security issue: followed security escalation procedure
- [ ] If blocking others: notified via task_assignments.md
- [ ] If critical: escalated immediately

---

## 📊 Periodic Maintenance (Weekly)

### For Project Coordinators
- [ ] Review all active missions for stalled work
- [ ] Check module locks for forgotten releases
- [ ] Archive old completed mission logs (> 30 days)
- [ ] Update health metrics
- [ ] Identify and resolve systemic blockers

---

## ❌ What NOT to Do

### Never Do These (Violations)
- [ ] ❌ Commit .onnx binary files
- [ ] ❌ Work on locked modules without coordination
- [ ] ❌ Make changes without updating mission log
- [ ] ❌ Break existing tests
- [ ] ❌ Commit code that doesn't compile
- [ ] ❌ Allocate memory in audio thread
- [ ] ❌ Commit secrets or credentials
- [ ] ❌ Skip code review
- [ ] ❌ Delete or comment out tests
- [ ] ❌ Ignore blockers without documentation

---

## 🎯 Quick Reference

### Daily Workflow Summary
1. ✅ Read active missions
2. ✅ Check task assignments
3. ✅ Claim a task
4. ✅ Create mission log
5. ✅ Work incrementally
6. ✅ Test frequently
7. ✅ Update mission log
8. ✅ Commit regularly
9. ✅ Complete or handoff
10. ✅ Update task assignments

### Key Files
- `/CMI/active_missions/` - Current work
- `/CMI/coordination/task_assignments.md` - Task tracking
- `/CMI/operational_ethics.md` - Rules and guidelines
- `/Models/metadata.json` - Model registry
- `.gitignore` - What not to commit

### Key Commands
```bash
# Build
cmake --build Build --config Release

# Test
ctest --test-dir Build

# Commit
git commit -m "feat: Description (mission_XXX)"

# Status
git status
```

---

## ✨ Excellence Indicators

You're doing great if:
- ✅ Mission logs are detailed and up-to-date
- ✅ Code is clean and well-documented
- ✅ Tests pass consistently
- ✅ Other agents can easily understand your work
- ✅ No merge conflicts
- ✅ Real-time performance maintained
- ✅ Proactive communication about blockers
- ✅ Helping other agents succeed

---

**Print this checklist and refer to it for every task!**

**Version**: 1.0  
**Last Updated**: 2025-01-15
