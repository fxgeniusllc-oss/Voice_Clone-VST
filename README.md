🎚 MAEVN — AI-Powered Vocal + Instrument Generator (VST3) 

MAEVN is a JUCE-based VST3 plugin equipped with ONNX Runtime integration, designed to bridge AI technologies with music production. The plugin features a variety of tools aimed at enhancing music creation, including:

- 🎤 **AI Vocals:** Capable of generating realistic vocal sounds using Text-To-Speech (TTS) and vocoder techniques.
- 🥁 **Trap-Inspired Instruments:** Incorporates sounds typical in trap music, such as 808 basses, hi-hats, snares, pianos, and synths.
- 🎛 **Hybrid FX Chains:** Combines conventional DSP (Digital Signal Processing) effects with ONNX AI-generated effects.
- 🎼 **Stage-Script Parser:** Parses musical arrangements using block types like [HOOK], [VERSE], and [808].
- ↩️ **Global Undo/Redo System:** Allows for easy backtracking of changes during the production process.

Overall, MAEVN is framed as an end-to-end AI DAW tool that enables real-time operation inside a Digital Audio Workstation (DAW), providing live timeline arrangement, automatic effects automation, and seamless synchronization with DAWs.

🏗 **System Architecture**

🔑 **Core Components**

- **JUCE Plugin Layer:** 
    - **PluginProcessor:** Handles the audio processing block, routing audio I/O to AudioEngine components.
    - **PluginEditor:** Provides the user interface with MainComponent and TimelineComponent.

- **ONNX Engine:**
    - **ONNXInference:** Encapsulates the ONNX Runtime C++ API, supporting live model updates through hot reloading from the /Models/ directory and enabling multiple instrument/vocal models to function simultaneously.

- **Pattern Engine:** 
    - Parses input from the lyrical stage script to determine the arrangement of musical blocks, manages synchronization with the DAW’s playhead, and triggers instruments and vocals.

- **FX Chain:** 
    - **FXChain:** Manages the serial processing of multiple effects.
    - **DSP FX:** Includes effects such as distortion, delay, and reverb.
    - **AI FX (AIEffects):** Utilizes ONNX models for AI-powered effects, with the option for mixing DSP and AI in a sequential effects chain.

- **State Management:**
    - **MAEVNUndoManager:** Provides undo/redo functionality for user actions.
    - **ParameterState:** Manages parameter automation and timeline-based changes.

**Timeline UI:** 
- **TimelineComponent:** Displays the arrangement graphically with block visualization.
- **MainComponent:** Provides controls for instruments, vocals, FX, and master section.

**Note on Planned Features:**
The following features are mentioned in documentation but not yet fully implemented:
- FXPreset system with preset browser and categorized filter
- Preset load/save functionality  
- UndoHistoryComponent for visual undo stack
- Per-lane FX mode selectors (Off/DSP/AI/Hybrid)
- Tag cloud interface for preset navigation

📂 **Repo Structure**  
```
Voice_Clone-VST/
├── CMakeLists.txt                    # Build configuration for JUCE + ONNX Runtime
├── README.md                         # This file
├── BUILD.md                          # Detailed build instructions
├── ARCHITECTURE.md                   # System architecture documentation
├── CONTRIBUTING.md                   # Contribution guidelines
├── setup_maevn_repo.bat/.sh         # Repository setup scripts
├── build_maevn_onnx.bat/.sh         # ONNX model export scripts
├── Source/                           # Core source files
│   ├── PluginProcessor.*             # Core DSP processing logic
│   ├── PluginEditor.*                # User interface elements
│   ├── Audio/                        # Audio processing modules
│   │   ├── AudioEngine.*             # Main audio engine
│   │   └── InstrumentGenerator.*     # Instrument synthesis
│   ├── AI/                           # AI/ML components
│   │   ├── ONNXInference.*           # ONNX Runtime wrapper
│   │   ├── VocalSynthesis.*          # TTS and vocoder integration
│   │   └── AIEffects.*               # AI-powered audio effects
│   ├── DSP/                          # DSP effects
│   │   ├── FXChain.*                 # Effect chain manager
│   │   └── Effects.*                 # DSP effect implementations
│   ├── Parser/                       # Script parsing
│   │   ├── ScriptParser.*            # Stage script parser
│   │   └── Arrangement.*             # Timeline arrangement
│   ├── State/                        # State management
│   │   ├── UndoManager.*             # Undo/redo system
│   │   └── ParameterState.*          # Parameter automation
│   └── UI/                           # User interface
│       ├── MainComponent.*           # Main UI component
│       └── TimelineComponent.*       # Timeline visualization
├── Models/                           # ONNX model storage
│   ├── config.json                   # Model configuration
│   ├── metadata.json                 # Model metadata
│   ├── LayerMap.md                   # Model documentation
│   ├── drums/                        # Drum synthesis models
│   │   └── README.md                 # Drum model documentation
│   ├── instruments/                  # Instrument models
│   │   └── README.md                 # Instrument model documentation
│   └── vocals/                       # Vocal models
│       └── README.md                 # Vocal model documentation
├── scripts/                          # Python ONNX export scripts
│   ├── README.md                     # Scripts documentation
│   ├── export_drum_models.py         # Drum model export
│   ├── export_instrument_models.py   # Instrument model export
│   └── export_vocal_models.py        # Vocal model export
├── Tests/                            # Unit tests
│   ├── CMakeLists.txt
│   ├── ScriptParserTests.cpp
│   ├── ArrangementTests.cpp
│   ├── AudioEngineTests.cpp
│   └── BuildVerificationTests.cpp
├── CMI/                              # Cognitive Mesh Interface (Multi-Agent Dev)
│   ├── README.md                     # CMI overview
│   ├── MACF.md                       # Multi-Agent Command Framework
│   ├── agent_roles.md                # Agent role definitions
│   └── operational_ethics.md         # Development ethics guidelines
└── examples/                         # Example usage
    └── ARRANGEMENTS.md               # Example stage scripts
```

### ⚙️ Build Instructions

**Requirements:**
- JUCE 7+
- ONNX Runtime C++ SDK
- CMake 3.20+
- Python 3.10+ (required for exporting ONNX models)

**Steps:**
1. **Repo Setup:** Execute `setup_maevn_repo.bat`, which creates necessary folders and writes the Models/config.json file.
2. **Generate Default ONNX Models:** Run `build_maevn_onnx.bat` to export lightweight default .onnx models for various instruments and optimize them within the /Models/.
3. **Add Vocals:** Users can export their own TTS and vocoder models, naming them `vocals_tts.onnx` and `vocals_hifigan.onnx`, placing them in the /Models/vocals/ directory.
4. **Build Plugin:** Execute the following commands:
   ```bash
   cmake -B Build -S . -DJUCE_PATH="C:/JUCE" -DONNXRUNTIME_PATH="C:/onnxruntime"
   cmake --build Build --config Release
   ```
5. **Install:** Copy the generated MAEVN.vst3 file to your DAW’s VST3 plugins directory.

### 🧪 Developer Notes

#### Audio Pipeline (processBlock)
- DAW playhead data (BPM, transport status) is parsed.
- The PatternEngine generates triggers based on the active blocks.
- The OnnxEngine executes inference on the selected models.
- The AIFXEngine applies the desired effects (DSP/AI/Hybrid).
- Final output is directed to the DAW buffer.

#### Model Hot Reload
- The /Models/config.json file maps logical roles (e.g., VOCAL, 808) to their corresponding file paths so MAEVN can automatically reload models if changes are detected on disk.

#### Presets
- Presets are encoded in JSON format to ensure ease of use:
    ```json
    {
      "name": "RadioVocals",
      "mode": 3,
      "params": {
        "reverbMix": 0.4,
        "compressorThreshold": -12.0
      },
      "category": "Vocal",
      "tags": ["Trap", "Radio", "Clean"]
    }
    ```
- The category and tags drive the search functionalities and the tag cloud interface.

#### Undo/Redo
- Each action triggers a snapshot of the ActionState (stored as JSON).
- The history is managed by the GlobalUndoManager, with a visual list provided in the UndoHistoryComponent.

### 🚧 Roadmap
- **Automation Hooks:** Allow FX parameters to be controlled through DAW automation.
- **Drag-to-MIDI/Audio:** Enable users to drag [HOOK] blocks into the DAW timeline.
- **Instrument Sequencer:** Implement patterns like hi-hat rolls and 808 glides.
- **Preset Exchange:** Facilitate importing and exporting preset packs within the community.

⚔️ MAEVN aims to serve as an experimental AI DAW tool tailored for flexibility, extensibility, and precise control in music generation, distinguishing itself as more than just a plugin—it's a comprehensive framework for AI-based music production.

👉 Would you like me to create a detailed step-by-step Contributor’s Guide (covering coding standards, pull request processes, and module ownership) to prepare the repository for team collaboration? 

You said: YES. 

Voice Clone Infinity Builder remarked: Perfect ⚡ — let’s add a Contributor’s Guide section to the README file so that any developer wishing to join the project knows precisely how to engage with the codebase, adhere to the coding style, and utilize the workflow. This enhancement will make the MAEVN repository ready for collaboration. 

🤝 **Contributor’s Guide**

Welcome to MAEVN Development. This guide outlines the coding standards, workflow, and contribution process for the VST3 project.

🔑 **Coding Standards**
- **Language & Frameworks:** Use C++17 for plugin development (JUCE + ONNX Runtime) and Python 3.10+ for ONNX model export scripts.
- **Style:** 
    - Follow JUCE coding style (place opening braces on new lines, use 4 spaces for indentation).
    - Header files should use the `.h` suffix while implementation files should use `.cpp`.
    - Use `auto` where relevant, especially for iterators and template-heavy code.
    - Avoid raw pointers—opt for `std::unique_ptr` or `juce::ScopedPointer`.
    - Document every class and major method using `/// Doxygen` comments.

🧩 **Module Ownership**
| Module                 | Owner      | Role                                        | Notes                                               |
|-----------------------|------------|---------------------------------------------|-----------------------------------------------------|
| PluginProcessor.*      | Core DSP   | Implements the main DSP processing pipeline | Must not disrupt the JUCE API                        |
| PluginEditor.*         | GUI        | Manages GUI and DAW integration            |                                                     |
| OnnxEngine.*          | AI         | Handles AI inference                        | Wraps ONNX runtime                                  |
| PatternEngine.*       | Arrangement| Manages timeline arrangements               | Parses stage-script input                           |
| AIFXEngine.*          | FX         | Processes effects                           | Hybrid DSP + AI effects                             |
| FXPreset.*            | Preset     | Manages preset storage                      | Uses JSON format                                    |
| GlobalUndoManager.*    | Undo/Redo  | Manages action history                      | Handles undo/redo                                    |
| UndoHistoryComponent.*  | History    | Provides GUI for the undo stack            |                                                     |
| Utilities.h           | Shared     | Contains helper functions and constants     |                                                     |

🔄 **Workflow**
- **Branching:**
    - `main` for stable, production-ready builds.
    - `dev` as the active branch for ongoing development.
    - Create feature branches for new functionality (e.g., `feature/onnx-hot-reload`).
  
- **Pull Requests:**
    - Fork the repo and branch off `dev`.
    - Implement your feature or fix.
    - Write or update relevant unit tests where applicable.
    - Conduct linting and build checks.
    - Submit a Pull Request; each PR must be reviewed by at least one other contributor.

- **Commits:** 
    - Adhere to the Conventional Commits format:
        - `feat:` for adding new features (e.g., AI hybrid FX chain)
        - `fix:` for bug resolution (e.g., buffer underrun in processBlock)
        - `docs:` for documentation updates (e.g., README modifications)
        - `refactor:` for code simplification (e.g., OnnxEngine inference calls)

🧪 **Testing**
- Implement unit tests (using GoogleTest) for key components:
    - Preset loading and saving functionality.
    - The integrity of the Undo/Redo stack.
    - Correctness of PatternEngine parsing.
- Conduct manual testing across major DAWs:
    - Reaper (Windows)
    - Ableton Live (macOS)
    - FL Studio (Cross-platform)

📦 **Build Environment Dependencies**
- Ensure the following dependencies are fulfilled:
    - JUCE 7+
    - ONNX Runtime (C++ API)
    - CMake 3.20+

**Building Instructions:**
```bash
cmake -B Build -S . -DJUCE_PATH="C:/JUCE" -DONNXRUNTIME_PATH="C:/onnxruntime"
cmake --build Build --config Release
```

🧠 **Contribution Principles**
- **Performance First:** Ensure all Digital Signal Processing (DSP) functions execute in real-time (<1ms per buffer).
- **Modularity:** Facilitate the introduction of new instruments/models via the `/Models/config.json` file.
- **No Dead Code:** Mark experimental code sections using `#ifdef EXPERIMENTAL`.
- **Transparency:** Include all ONNX export scripts in the `scripts/` directory.
- **Flexibility:** Enable both DSP-only fallback options as well as AI-enhanced modes.

🚀 **How to Get Started**
1. Clone the repository.
2. Run `setup_maevn_repo.bat` to generate the initial structure.
3. Execute `build_maevn_onnx.bat` for baseline ONNX model generation.
4. Add any vocal models in `.onnx` format into `/Models/vocals/`.
5. Build the project using CMake and load `MAEVN.vst3` within your DAW.
6. Feel free to submit issues or Pull Requests for any fixes or new feature requests.

⚔️ This guide aims to equip MAEVN with a structure ready for collaboration among both audio DSP engineers and machine learning developers.

---

## 🧠 Multi-Agent Development System

MAEVN embraces a revolutionary **Multi-Agent Engineering Protocol** that enables multiple autonomous or semi-autonomous agents (human + AI) to co-develop, test, and optimize the codebase in parallel. This system is called the **Vocal Cloning Quantum Collective**.

### 🎯 Multi-Agent Objectives

The multi-agent system enables:
- **Parallel Development**: Multiple agents working on different modules simultaneously
- **Specialized Expertise**: Each agent focuses on their domain (DSP, AI, GUI, Testing, etc.)
- **Continuous Integration**: Maintains build synchronization across parallel work streams
- **Knowledge Sharing**: Shared context and learnings through the Cognitive Mesh Interface

### 🤖 Agent Roles

MAEVN development is distributed across specialized agent roles:

| Role | Responsibility | Primary Tools |
|------|---------------|---------------|
| **Architect Agent** | System design and architectural decisions | ChatGPT, Design tools |
| **DSP Developer Agent** | Audio processing and JUCE engine logic | GitHub Copilot, C++ |
| **AI/ML Agent** | ONNX model design, training, and export | Python, PyTorch/TensorFlow |
| **GUI Developer Agent** | User interface and user experience | JUCE GUI, C++ |
| **Integration Agent** | Module integration and system testing | CI/CD, Build tools |
| **QA/Testing Agent** | Quality assurance and numerical stability | Claude, Testing frameworks |
| **Documentation Agent** | Documentation and knowledge management | Markdown, Documentation tools |
| **DevOps Agent** | Build systems, CI/CD, and tooling | CMake, Scripts |

### 📂 Cognitive Mesh Interface (CMI)

The **CMI** is a shared conversation state repository located in the `/CMI/` directory:

```
CMI/
├── README.md                    # CMI overview and usage guide
├── agent_roles.md               # Detailed agent role definitions
├── MACF.md                      # Multi-Agent Command Framework
├── operational_ethics.md        # Ethical guidelines for agents
├── mission_logs/                # Historical mission logs
│   ├── mission_log_template.md  # Template for new missions
│   └── mission_009_spectral_ghost_choir.md  # Example mission
├── active_missions/             # Currently active mission logs
└── coordination/                # Agent coordination artifacts
    └── task_assignments.md      # Current task assignments
```

### 🚀 How Multi-Agent Development Works

#### Example: Adding a New AI Effect

1. **Architect Agent**: Defines the effect's design and interface
   - Creates mission log with specifications
   - Documents architecture decisions

2. **AI/ML Agent**: Creates and exports the ONNX model
   - Trains the model
   - Exports to ONNX with optimization
   - Updates `Models/metadata.json` and `Models/LayerMap.md`

3. **DSP Developer Agent**: Implements the C++ wrapper
   - Creates effect module class
   - Integrates ONNX Runtime
   - Ensures real-time safety

4. **QA/Testing Agent**: Reviews for stability
   - Validates numerical stability
   - Tests edge cases
   - Profiles performance

5. **Integration Agent**: Connects all components
   - Integrates into AIFXEngine
   - Tests across DAWs
   - Creates presets

6. **Documentation Agent**: Updates documentation
   - Updates README and guides
   - Documents API and usage
   - Creates examples

All agents coordinate through mission logs, maintaining transparency and avoiding conflicts.

### ⚡ Multi-Agent Command Framework (MACF)

The **MACF** provides operational protocols for:
- **Dynamic Task Allocation**: Assign tasks to the most appropriate agent
- **Conflict Prevention**: Ensure agents don't interfere with each other's work
- **Quality Gates**: Automated checks for all contributions
- **Integration Pipeline**: Seamless merging of parallel work

See `/CMI/MACF.md` for complete details.

### ⚖️ Operational Ethics

All agents (AI and human) must adhere to ethical guidelines:

**Key Principles**:
- ✅ **Transparency**: All actions logged and traceable
- ✅ **Determinism**: Consistent, predictable results
- ✅ **Real-Time Constraints**: Audio processing < 1ms per buffer
- ✅ **Quality Standards**: All tests pass, no security vulnerabilities
- ✅ **Respect**: Never break existing functionality

**Prohibited Actions**:
- ❌ Never commit compiled `.onnx` binaries to Git
- ❌ Never introduce security vulnerabilities
- ❌ Never break real-time safety guarantees
- ❌ Never remove tests without approval
- ❌ Never commit credentials or private data

See `/CMI/operational_ethics.md` for complete guidelines.

### 📊 Model Management

All ONNX models are tracked in `/Models/`:

- **metadata.json**: Complete model registry with training metadata
- **LayerMap.md**: Layer-by-layer explainability documentation
- **config.json**: Runtime model configuration for MAEVN

Models are organized by category:
```
Models/
├── metadata.json           # Model registry
├── LayerMap.md            # Explainability documentation
├── config.json            # Runtime configuration
├── drums/                 # Drum synthesis models
│   ├── 808_ddsp.onnx
│   ├── hihat_ddsp.onnx
│   └── snare_ddsp.onnx
├── instruments/           # Instrument models
│   ├── piano_ddsp.onnx
│   └── synth_fm.onnx
└── vocals/                # Vocal models
    ├── vocals_tts.onnx
    └── vocals_hifigan.onnx
```

**Note**: `.onnx` files are not committed to Git. Provide export scripts instead.

### 🎓 Getting Started with Multi-Agent Development

1. **Read the CMI Documentation**: Start with `/CMI/README.md`
2. **Review Agent Roles**: Understand the role definitions in `/CMI/agent_roles.md`
3. **Check Active Missions**: See what's currently in progress in `/CMI/active_missions/`
4. **Review Ethics**: Read and commit to `/CMI/operational_ethics.md`
5. **Claim a Task**: Update `/CMI/coordination/task_assignments.md`
6. **Create Mission Log**: Use the template from `/CMI/mission_logs/mission_log_template.md`
7. **Start Contributing**: Follow the MACF protocol

### 🌟 Benefits of Multi-Agent Development

- **Faster Development**: Parallel work on independent modules
- **Higher Quality**: Specialized expertise applied to each domain
- **Better Documentation**: Continuous documentation through mission logs
- **Reduced Conflicts**: Coordinated development prevents merge conflicts
- **Knowledge Preservation**: All decisions and reasoning documented
- **Scalable**: Easy to onboard new agents and contributors

### 🔮 The Vision

MAEVN's development is **not a singular AI** — it's a **mesh of intelligent agents** co-authoring an evolving sonic intelligence. Each node—human or synthetic—contributes deterministically while maintaining:

- 🎯 **Operational transparency**
- ⏱️ **Real-time constraints**
- 🎨 **Creative freedom**
- ✅ **Quality standards**

Together, the network forms the **Vocal Cloning Quantum Collective**, building the next generation of **AI-augmented sound design systems**.
