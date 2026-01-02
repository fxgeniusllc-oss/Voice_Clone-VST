# ❓ Multi-Agent Development FAQ

## Frequently Asked Questions about the MAEVN Multi-Agent System

---

## 🎛️ DAW Compatibility

### Q: Can I use MAEVN with Audacity?

**A:** No, unfortunately MAEVN is not currently compatible with Audacity. Here's why:

- **MAEVN** is built as a **VST3** plugin
- **Audacity** only supports **VST2** (legacy), **LV2**, and **AU** (macOS) plugin formats
- Audacity **does not support VST3** plugins

**Workaround Options:**
1. **Use the Standalone version** - MAEVN can be built as a standalone application that runs independently without a DAW
2. **Use a VST3-compatible DAW** - See the list of compatible DAWs below

**Compatible DAWs (VST3 Support):**
- ✅ Ableton Live 10+
- ✅ FL Studio 20+
- ✅ Reaper 5.0+
- ✅ Bitwig Studio 3.0+
- ✅ Steinberg Cubase 10.5+
- ✅ Steinberg Nuendo
- ✅ PreSonus Studio One 4+
- ✅ Tracktion Waveform

**Why not add VST2/LV2 support?**
- VST2 SDK was deprecated by Steinberg and removed from JUCE framework
- Adding LV2 support would require significant development effort
- VST3 is the modern standard supported by most DAWs

**Future Plans:**
- LV2 format support may be considered in future releases
- Community contributions for additional plugin formats are welcome

---

## 🤖 General Questions

### Q: What is the Multi-Agent Development System?

**A:** The Multi-Agent Development System is a framework that enables multiple AI agents and human developers to collaborate on the MAEVN codebase simultaneously. Each agent has a specialized role (DSP, AI/ML, GUI, Testing, etc.) and coordinates through the Cognitive Mesh Interface (CMI).

---

### Q: Why use multiple agents instead of a single developer?

**A:**
- **Parallelization**: Multiple agents work simultaneously on different modules
- **Specialization**: Each agent has deep expertise in their domain
- **Quality**: Multiple perspectives catch more issues
- **Speed**: Parallel development is faster than sequential
- **Knowledge Sharing**: Agents learn from each other's work

---

### Q: What is the Cognitive Mesh Interface (CMI)?

**A:** CMI is a shared conversation state repository located in the `/CMI/` directory. It contains:
- Mission logs (work tracking)
- Task assignments (coordination)
- Agent role definitions
- Operational guidelines
- Documentation

It ensures all agents have shared context and can coordinate effectively.

---

## 👥 Agent Roles

### Q: Which agent role should I take?

**A:** Choose based on your expertise:
- **DSP/Audio expert** → DSP Developer Agent
- **Machine Learning** → AI/ML Agent
- **Frontend/UI** → GUI Developer Agent
- **Testing/QA** → QA/Testing Agent
- **System design** → Architect Agent
- **Documentation** → Documentation Agent
- **Build systems** → DevOps Agent
- **Integration** → Integration Agent

See `/CMI/agent_roles.md` for detailed role descriptions.

---

### Q: Can I switch agent roles?

**A:** Yes! If a task requires different expertise, you can:
1. Complete your current mission or handoff to another agent
2. Update your role in task_assignments.md
3. Start working in the new role

However, consistency helps build expertise, so switching too often isn't recommended.

---

### Q: Can multiple agents have the same role?

**A:** Absolutely! Multiple DSP agents, multiple AI agents, etc. can work simultaneously on different tasks. Just make sure you're not working on the same modules (check module locks).

---

## 📝 Mission Logs

### Q: Do I really need to create a mission log for every task?

**A:** Yes, for any significant work. Mission logs provide:
- Context for future developers
- Decision history and rationale
- Progress tracking
- Handoff information
- Debugging breadcrumbs

Small fixes (typos, formatting) may not need a full mission log, but use judgment.

---

### Q: How detailed should mission logs be?

**A:** Include:
- ✅ What you're building and why
- ✅ Decisions made and rationale
- ✅ Issues encountered and solutions
- ✅ What's done and what remains
- ❌ Don't include: Step-by-step code changes (that's what Git is for)

Aim for a developer to understand your work in 5-10 minutes of reading.

---

### Q: What if I forget to update the mission log?

**A:** Update it as soon as you remember! Better late than never. If you've already committed code:
1. Review your commits
2. Reconstruct what you did
3. Update the mission log retroactively
4. Include a note that it was updated after the fact

---

## 🔄 Coordination

### Q: What if someone is already working on the module I need?

**A:**
1. Check `/CMI/coordination/task_assignments.md` for who has the lock
2. Read their active mission log to understand their work
3. Options:
   - Wait for them to finish
   - Coordinate with them via mission log
   - Work on the interface while they work on implementation
   - Choose a different task

Never override a module lock without coordination.

---

### Q: How do I coordinate with another agent?

**A:**
- **Mission logs**: Leave notes in mission logs for specific agents
- **Task assignments**: Use the notes section
- **MACF commands**: Use `@macf` tags in mission logs:
  - `@macf handoff mission_XXX to [Agent]`
  - `@macf review mission_XXX`
  - `@macf escalate mission_XXX`

---

### Q: What if two agents need to modify the same file?

**A:**
1. One agent takes the lead (locks the module)
2. Other agent waits or works on a different module
3. OR: Use Git branches and coordinate merge timing
4. OR: Divide the file (one works on interface, one on implementation)

Communication is key - document the plan in both mission logs.

---

## 🔧 Technical Questions

### Q: Why can't I commit .onnx files?

**A:** Several reasons:
- **Size**: .onnx files are large (1-50MB+), bloating the Git repo
- **Binary**: Git doesn't handle binary diffs well
- **Reproducibility**: Export scripts ensure reproducible builds
- **Versioning**: Model versions tracked in metadata.json

Instead: Commit the export script and document in metadata.json.

---

### Q: What is "real-time safety" and why does it matter?

**A:** Real-time safety means audio processing code:
- Completes in < 1ms per buffer (typically ~10ms of audio)
- Never allocates memory
- Never does file I/O
- Never uses locks that might block

Violations cause audio dropouts (crackles, pops) which ruin user experience.

---

### Q: How do I test for real-time safety?

**A:**
1. **Profile**: Measure execution time
   ```cpp
   auto start = std::chrono::high_resolution_clock::now();
   // ... audio processing ...
   auto end = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
   // Ensure duration < 1000 microseconds (1ms)
   ```

2. **Static Analysis**: Check for:
   - `new`, `malloc`, `std::vector::push_back()`
   - `File::`, `fopen`, `std::ifstream`
   - `std::mutex`, `std::lock_guard`

3. **Runtime Tools**: Use JUCE's `BlockscopedRenderingLock` or similar

---

### Q: What if my inference takes longer than 10ms?

**A:** Options:
1. **Optimize the model**: Quantization, pruning, distillation
2. **Run on background thread**: Process ahead of time, use results later
3. **Reduce quality**: Fewer layers, smaller dimensions
4. **GPU acceleration**: If available (but maintain CPU fallback)
5. **Cache results**: If inputs are repetitive

Document your optimization approach in the mission log.

---

## 🐛 Common Issues

### Q: My build is failing. What should I check?

**A:**
1. **CMake configuration**: 
   ```bash
   cmake -B Build -S . -DJUCE_PATH="path" -DONNXRUNTIME_PATH="path"
   ```
2. **Dependencies**: JUCE 7+, ONNX Runtime, CMake 3.20+
3. **Recent changes**: Check if someone broke the build
4. **Clean build**: 
   ```bash
   rm -rf Build && cmake -B Build -S .
   ```

If still failing, document in mission log and escalate.

---

### Q: Tests are failing. What do I do?

**A:**
1. **Check if tests were passing before your changes**:
   ```bash
   git stash
   cmake --build Build --config Release
   ctest --test-dir Build
   git stash pop
   ```

2. **If they were passing**: Your changes broke something. Fix it.

3. **If they weren't passing**: You're not responsible for pre-existing failures (unless your task is to fix them)

Document in mission log either way.

---

### Q: I found a security vulnerability. What do I do?

**A:**
1. **Don't disclose publicly** (don't commit description in visible place)
2. **Document in mission log** with severity level
3. **Tag with** `@macf escalate`
4. **Notify project coordinator** immediately
5. **If in your modified code**: Fix it as part of your mission
6. **If elsewhere**: Create separate high-priority mission

See `/CMI/operational_ethics.md` for full security guidelines.

---

## 📊 Models & AI

### Q: How do I add a new ONNX model?

**A:**
1. **Create the model**: Train and export to .onnx
2. **Create export script**: `scripts/export_model_name.py`
3. **Update metadata.json**: Add full model entry
4. **Update LayerMap.md**: Document architecture
5. **Update config.json**: Add model role mapping
6. **Test inference**: Verify performance
7. **DO NOT commit .onnx file**

See example mission: `mission_009_spectral_ghost_choir.md`

---

### Q: What should be in metadata.json?

**A:** For each model:
- ID, name, version, description
- File path (relative)
- Category and type
- Architecture name
- Input/output format specifications
- Performance metrics (inference time, memory, CPU)
- Training metadata (dataset, epochs, optimizer, loss, date, author)
- Explainability reference (LayerMap.md)
- Checksum (for verification)
- Status (development/production)

---

### Q: What goes in LayerMap.md?

**A:** Layer-by-layer explanation:
- Layer name and type
- Input/output shapes
- Purpose of each layer
- Why this architecture choice
- Special considerations (stability, performance)
- Any known limitations

Think: "If I had to debug this model in 6 months, what would I need to know?"

---

## 🚀 Workflow Questions

### Q: Do I need to create a branch for every task?

**A:** Recommended but not required. Options:
- **Feature branches**: `feature/mission-XXX-description`
- **Direct to main/dev**: If you're confident and changes are small
- **Agent branches**: `agent/dsp-agent-work`

Use branches for:
- Experimental work
- Long-running features
- When coordinating with other agents

---

### Q: How often should I commit?

**A:** 
- **Minimum**: After every completed subtask
- **Recommended**: Every 1-2 hours of work
- **Maximum**: Daily (at least)

Small, frequent commits are better than large, infrequent ones.

Commit message format:
```
<type>: <description> (mission_XXX)

Examples:
feat: Add reverb effect (mission_010)
fix: Resolve buffer underrun (mission_011)
docs: Update API documentation (mission_012)
test: Add unit tests for PatternEngine (mission_013)
```

---

### Q: When should I handoff vs. complete a mission?

**A:**

**Handoff when**:
- You've completed your part but another agent needs to continue
- You're blocked and another agent can unblock
- Task requires expertise you don't have
- You're going offline for extended time

**Complete when**:
- All success criteria met
- All tests passing
- Documentation updated
- No more work needed on this feature

---

## 🎓 Best Practices

### Q: What makes a good mission log?

**A:**
- ✅ Clear objective
- ✅ Specific success criteria
- ✅ Sufficient context
- ✅ Regular progress updates
- ✅ Decisions documented with rationale
- ✅ Handoff notes (if applicable)
- ✅ Lessons learned

**Example**: See `/CMI/mission_logs/mission_009_spectral_ghost_choir.md`

---

### Q: How do I write good handoff notes?

**A:** Include:
1. **Current state**: What's the code doing now?
2. **What's done**: List completed work
3. **What remains**: List remaining work with specifics
4. **Context**: Anything the next agent should know
5. **Blockers**: Any issues they'll face
6. **Suggestions**: Recommended approach for remaining work

Aim for: Next agent can continue without asking questions.

---

### Q: What if I disagree with another agent's approach?

**A:**
1. **Document your concerns** in mission log or PR comments
2. **Explain your reasoning** with specifics
3. **Suggest alternatives** with pros/cons
4. **Be respectful** - assume good intent
5. **Escalate if needed**: Use `@macf escalate`
6. **Accept decision**: Once made, move forward

See `/CMI/operational_ethics.md` → "Conflict Resolution"

---

## 🔮 Advanced Topics

### Q: Can I use AI tools to help write code?

**A:** Yes! Use GitHub Copilot, ChatGPT, Claude, etc. But:
- ✅ Review all generated code
- ✅ Ensure it meets quality standards
- ✅ Test thoroughly
- ✅ Adapt to MAEVN conventions
- ✅ Document as if you wrote it

You're responsible for the code, regardless of who/what wrote it.

---

### Q: What is the "Vocal Cloning Quantum Collective"?

**A:** The collective name for all agents (human and AI) working on MAEVN. It represents:
- Mesh of intelligent agents
- Co-authoring an evolving system
- Each node contributes deterministically
- Maintains transparency and quality
- Builds next-gen AI audio tools

It's both practical (the system we use) and philosophical (how we think about multi-agent development).

---

### Q: Can agents work asynchronously (different time zones)?

**A:** Absolutely! That's a key benefit. Mission logs enable async coordination:
1. Agent A works during their time
2. Updates mission log with progress
3. Agent B reads log during their time
4. Continues work with full context

No real-time meetings required (though helpful occasionally).

---

## 🆘 Getting Help

### Q: I'm stuck. How do I get help?

**A:**
1. **Document in mission log**: Describe the issue
2. **Tag for escalation**: `@macf escalate mission_XXX`
3. **Update task_assignments.md**: Set status to "blocked"
4. **Work on something else**: Don't waste time being stuck
5. **Wait for response**: Another agent or coordinator will help

---

### Q: Where can I learn more?

**A:**
- `/CMI/README.md` - CMI overview
- `/CMI/QUICKSTART.md` - Quick start guide
- `/CMI/agent_roles.md` - Role details
- `/CMI/MACF.md` - Coordination framework
- `/CMI/operational_ethics.md` - Guidelines
- `/CMI/CHECKLIST.md` - Workflow checklist
- `/Models/LayerMap.md` - Model architectures
- `README.md` - Project overview

---

### Q: Who maintains this FAQ?

**A:** The Documentation Agent primarily, but any agent can suggest updates. If you encounter a question not covered here, add it!

---

## 💡 Pro Tips

### Tip 1: Read mission logs regularly
Even missions you're not involved in. You'll learn patterns, avoid mistakes, and understand the system better.

### Tip 2: Over-communicate rather than under-communicate
When in doubt, add more detail to mission logs. Future you (and other agents) will thank you.

### Tip 3: Test early and often
Don't wait until the end. Test after every significant change.

### Tip 4: Use the example mission as a template
`mission_009_spectral_ghost_choir.md` shows an ideal multi-agent workflow.

### Tip 5: Build incrementally
Small, working changes beat large, broken changes.

---

## 🎉 Success Stories

### Q: Has this system been used successfully?

**A:** Yes! See `/CMI/mission_logs/mission_009_spectral_ghost_choir.md` for a complete example of:
- 5 agents coordinating
- Clear handoffs
- Comprehensive documentation
- Successful completion

This demonstrates the system works in practice, not just theory.

---

**Have a question not answered here? Add it by creating a PR or mission log!**

**Version**: 1.0  
**Last Updated**: 2025-01-15  
**Maintained By**: Documentation Agent
