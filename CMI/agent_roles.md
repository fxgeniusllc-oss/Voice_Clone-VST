# 🤖 Agent Roles & Responsibilities

## Overview

This document defines the roles and responsibilities of different agents in the MAEVN Multi-Agent Development System. Each role represents a specialized domain of expertise.

---

## Core Agent Roles

### 1. **Architect Agent** 🏗️

**Primary Responsibility**: System design and architectural decisions

**Capabilities**:
- Design high-level system architecture
- Define module interfaces and contracts
- Plan integration strategies
- Review architectural consistency
- Document design decisions

**Typical Tasks**:
- Creating design documents for new features
- Defining API contracts between modules
- Planning refactoring strategies
- Reviewing PRs for architectural compliance

**Tools/Platforms**: ChatGPT, Claude (design-focused prompts)

---

### 2. **DSP Developer Agent** 🎚️

**Primary Responsibility**: Audio DSP and JUCE engine logic

**Capabilities**:
- Implement real-time audio processing
- Optimize DSP algorithms for performance
- Integrate JUCE framework components
- Debug audio buffer issues
- Ensure thread-safe audio processing

**Typical Tasks**:
- Implementing audio effects (compression, EQ, reverb)
- Optimizing `processBlock()` performance
- Managing audio buffer lifecycles
- Integrating with DAW hosts

**Tools/Platforms**: GitHub Copilot, specialized DSP agents

**Key Files**:
- `PluginProcessor.cpp/.h`
- `AIFXEngine.cpp/.h`

---

### 3. **AI/ML Agent** 🧠

**Primary Responsibility**: ONNX model design, training, and export

**Capabilities**:
- Design neural network architectures
- Train models for specific tasks
- Export models to ONNX format
- Optimize inference performance
- Document model behavior

**Typical Tasks**:
- Creating TTS/vocoder models
- Exporting instrument generation models
- Optimizing model sizes for real-time use
- Writing model metadata and layer maps

**Tools/Platforms**: Python-based agents, ML-specialized LLMs

**Key Files**:
- `OnnxEngine.cpp/.h`
- `Models/metadata.json`
- `Models/LayerMap.md`

---

### 4. **GUI Developer Agent** 🎨

**Primary Responsibility**: User interface and user experience

**Capabilities**:
- Implement JUCE GUI components
- Design intuitive user interactions
- Integrate with plugin editor
- Ensure cross-platform compatibility
- Implement responsive layouts

**Typical Tasks**:
- Creating preset browser UI
- Implementing timeline visualization
- Designing FX parameter controls
- Building undo history components

**Tools/Platforms**: GitHub Copilot, UI-focused agents

**Key Files**:
- `PluginEditor.cpp/.h`
- `PresetBrowserComponent.cpp/.h`
- `TimelineLane.cpp/.h`

---

### 5. **Integration Agent** 🔗

**Primary Responsibility**: Module integration and system testing

**Capabilities**:
- Integrate disparate modules
- Write integration tests
- Debug cross-module issues
- Ensure API consistency
- Coordinate between agents

**Typical Tasks**:
- Connecting OnnxEngine to PluginProcessor
- Integrating preset system with GUI
- Writing end-to-end tests
- Resolving merge conflicts

**Tools/Platforms**: GitHub Copilot, multi-purpose agents

---

### 6. **QA/Testing Agent** 🧪

**Primary Responsibility**: Quality assurance and testing

**Capabilities**:
- Write unit tests
- Perform code reviews
- Test numerical stability
- Verify real-time performance
- Document bugs

**Typical Tasks**:
- Writing GoogleTest test cases
- Reviewing PRs for correctness
- Testing buffer handling edge cases
- Profiling CPU usage

**Tools/Platforms**: Claude (analysis-focused), testing-specialized agents

---

### 7. **Documentation Agent** 📝

**Primary Responsibility**: Documentation and knowledge management

**Capabilities**:
- Write clear documentation
- Create tutorials and guides
- Document APIs and interfaces
- Maintain README files
- Create mission logs

**Typical Tasks**:
- Writing developer guides
- Documenting new features
- Creating API reference docs
- Updating CMI logs

**Tools/Platforms**: Documentation-focused LLMs

---

### 8. **DevOps Agent** ⚙️

**Primary Responsibility**: Build systems, CI/CD, and tooling

**Capabilities**:
- Configure CMake builds
- Set up CI/CD pipelines
- Manage dependencies
- Automate workflows
- Handle deployment

**Typical Tasks**:
- Updating CMakeLists.txt
- Configuring build scripts
- Setting up automated testing
- Managing ONNX Runtime integration

**Tools/Platforms**: Automation-focused agents

---

## Agent Coordination Protocol

### Task Assignment
1. Review `CMI/coordination/task_assignments.md`
2. Claim a task by updating the assignment file
3. Create or update the relevant mission log
4. Begin work with full context from CMI

### Handoffs
- When completing work, document:
  - What was accomplished
  - What remains to be done
  - Any blockers or issues
  - Next recommended steps
  - Files modified

### Conflict Resolution
- If two agents need to work on the same module:
  - Coordinate via mission logs
  - Use feature branches
  - One agent focuses on interface, another on implementation
  - Schedule integration points

---

## Role Selection Guidelines

When starting a new task:
1. Identify the primary domain (DSP, AI, GUI, etc.)
2. Select the appropriate agent role
3. Read that role's documentation and constraints
4. Follow the role's best practices
5. Document work in the role's context

---

## Multi-Agent Workflows

### Example: Adding a New AI Effect

1. **Architect Agent**: Design the effect's interface and integration points
2. **AI/ML Agent**: Create and export the ONNX model
3. **DSP Developer Agent**: Implement the effect wrapper in C++
4. **GUI Developer Agent**: Add UI controls for the effect
5. **Integration Agent**: Connect all components
6. **QA/Testing Agent**: Write tests and verify performance
7. **Documentation Agent**: Document the new effect

Each agent updates the shared mission log at each step.

---

## Conclusion

This role system ensures specialized expertise is applied to each part of the MAEVN codebase while maintaining coordination through the CMI. Agents should respect role boundaries but communicate freely through mission logs.
