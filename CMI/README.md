# 🧠 Cognitive Mesh Interface (CMI)

## Overview

The **Cognitive Mesh Interface (CMI)** is a shared conversation state repository that enables multiple AI agents (ChatGPT, Claude, GitHub Copilot, etc.) and human developers to collaborate seamlessly on the MAEVN codebase.

## Purpose

CMI ensures:
- **Shared memory** of the repository structure
- **Unified coding conventions** across all contributors
- **Conflict-free merges** through coordinated development
- **Transparent communication** between agents and developers

## Directory Structure

```
CMI/
├── README.md                    # This file
├── agent_roles.md               # Definitions of agent roles and responsibilities
├── mission_logs/                # Historical mission logs
│   └── mission_log_template.md  # Template for new mission logs
├── active_missions/             # Currently active mission logs
└── coordination/                # Agent coordination artifacts
    └── task_assignments.md      # Current task assignments
```

## How to Use CMI

### For AI Agents

1. **Before starting work**: Read the latest mission log in `active_missions/`
2. **During work**: Document your progress and decisions
3. **After completing work**: Update the mission log with results and handoff notes
4. **When blocked**: Document blockers in the mission log for other agents to address

### For Human Developers

1. **Creating new tasks**: Use the mission log template to define clear objectives
2. **Reviewing agent work**: Check mission logs for context and reasoning
3. **Coordinating multiple agents**: Update `task_assignments.md` to prevent conflicts

## Mission Log Format

Each mission log should include:
- **Mission ID**: Unique identifier (e.g., `mission_001`)
- **Objective**: Clear statement of what needs to be accomplished
- **Assigned Agent(s)**: Which agent(s) are responsible
- **Status**: `planned`, `in-progress`, `completed`, `blocked`
- **Context**: Background information and dependencies
- **Progress Updates**: Timestamped updates on work completed
- **Handoff Notes**: Information for the next agent or developer

## Best Practices

1. **Always sync before starting**: Read the latest mission logs
2. **Document assumptions**: Make implicit knowledge explicit
3. **Use clear language**: Write for both humans and AI agents
4. **Keep logs updated**: Update status frequently
5. **Preserve history**: Never delete old mission logs, move them to `mission_logs/`

## Integration with Development Workflow

- Mission logs complement Git commits with higher-level context
- Use mission IDs in commit messages for traceability
- Reference mission logs in pull requests
- Update mission logs before and after significant code changes
