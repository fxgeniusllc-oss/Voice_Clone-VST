# 🎚 MAEVN — AI-Powered Vocal + Instrument Generator (VST3) 

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

MAEVN is a **JUCE-based VST3 plugin** with **ONNX Runtime integration**, bridging AI technologies with professional music production. It's an experimental AI DAW tool that enables real-time operation inside Digital Audio Workstations (DAWs), providing live timeline arrangement, automatic effects automation, and seamless DAW synchronization.

## ✨ Key Features

- 🎤 **AI Vocals** — Generate realistic vocal sounds using Text-To-Speech (TTS) and neural vocoder techniques
- 🥁 **Trap Instruments** — 5 synthesis engines: 808 bass, hi-hats, snares, piano, and synth
- 🎛 **Hybrid FX Chain** — Combines DSP effects (distortion, delay, reverb) with AI-powered effects via ONNX
- 🎼 **Stage-Script Parser** — Musical arrangement system using blocks like `[HOOK]`, `[VERSE]`, `[808]`
- ↩️ **Global Undo/Redo** — 100-level history for parameter changes
- 🔄 **Hot Model Reload** — Update ONNX models on disk and reload without restarting
- 🎯 **Real-time Safe** — Audio processing optimized for <1ms per buffer
- 🔌 **VST3 + Standalone** — Use in DAWs or run as standalone application

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Build Architecture](#-build-architecture)
- [Repository Structure](#-repository-structure)
- [DAW Compatibility](#%EF%B8%8F-daw-compatibility)
- [Build Instructions](#%EF%B8%8F-build-instructions)
- [Usage Guide](#-usage-guide)
- [Contributing](#-contributing)
- [Multi-Agent Development](#-multi-agent-development-system)
- [Documentation](#-documentation)
- [Roadmap](#-roadmap)
- [License](#-license)

## 🚀 Quick Start

### For Users (Pre-built Binary)

1. Download the latest release from [Releases](../../releases) (when available)
2. Copy `MAEVN.vst3` to your plugin folder:
   - **Windows:** `C:\Program Files\Common Files\VST3\`
   - **macOS:** `~/Library/Audio/Plug-Ins/VST3/`
   - **Linux:** `~/.vst3/`
3. Rescan plugins in your DAW
4. Load MAEVN as an instrument

### For Developers (Build from Source)

```bash
# 1. Clone repository
git clone https://github.com/fxgeniusllc-oss/Voice_Clone-VST.git
cd Voice_Clone-VST

# 2. Setup repository (creates Models directories)
./setup_maevn_repo.sh  # or setup_maevn_repo.bat on Windows

# 3. Export default ONNX models (optional, requires Python 3.10+)
./build_maevn_onnx.sh  # or build_maevn_onnx.bat on Windows

# 4. Build plugin and standalone
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# 5. Install standalone app and VST3 plugin
cd ..
./install_maevn.sh  # or install_maevn.bat on Windows

# 6. Launch standalone (or use from DAW as VST3)
./launch_maevn.sh  # or launch_maevn.bat on Windows
```

See [BUILD.md](BUILD.md) for detailed build instructions.

## 🏗 Build Architecture

MAEVN uses a modern, modular build system designed for cross-platform compatibility.

### Build System Stack

```
CMake 3.15+ (Build Configuration)
    ↓
JUCE 7.0.9 (Audio Framework)
    ↓
C++17 Compiler (GCC/Clang/MSVC)
    ↓
ONNX Runtime (Optional - AI Features)
```

### Build Process Flow

1. **Repository Setup** → `setup_maevn_repo.sh/bat` creates Models directory structure
2. **Model Export** → `build_maevn_onnx.sh/bat` exports placeholder ONNX models (requires Python 3.10+)
3. **CMake Configuration** → Fetches JUCE, configures build system
4. **Compilation** → Builds VST3 plugin and Standalone application
5. **Installation** → Copies artifacts to system plugin directories

### Key Build Files

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Main build configuration, JUCE integration, source file definitions |
| `setup_maevn_repo.sh/bat` | Creates Models folder structure, prepares repository |
| `build_maevn_onnx.sh/bat` | Exports default ONNX models using Python scripts |
| `scripts/export_*.py` | Python scripts for ONNX model generation |
| `Models/config.json` | Runtime model configuration mapping |

### Component Wiring Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    PluginProcessor                      │
│  (Main VST3 Interface - manages lifecycle & state)     │
└─────────────────────┬───────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ↓                         ↓
┌─────────────────┐      ┌─────────────────┐
│  PluginEditor   │      │   AudioEngine   │
│  (UI Layer)     │      │  (DSP Core)     │
└─────────────────┘      └────────┬────────┘
         │                         │
    ┌────┴────┐          ┌────────┴─────────┬──────────┬──────────┐
    ↓         ↓          ↓                  ↓          ↓          ↓
┌─────┐  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌─────┐  ┌──────────┐
│Main │  │Timeline│  │Instrument│  │  Vocal   │  │ FX  │  │Arrangement│
│ UI  │  │  View  │  │Generator │  │Synthesis │  │Chain│  │  Parser  │
└─────┘  └────────┘  └─────┬────┘  └────┬─────┘  └──┬──┘  └────┬─────┘
                           │                │         │          │
                           └────────┬───────┴─────────┘          │
                                    ↓                             │
                           ┌──────────────┐                       │
                           │ONNXInference │←──────────────────────┘
                           │(AI Engine)   │
                           └──────────────┘
```

### Core Components

| Component | Responsibility | Key Files |
|-----------|---------------|-----------|
| **PluginProcessor** | VST3 interface, parameter management, state serialization | `Source/PluginProcessor.*` |
| **PluginEditor** | User interface, visual timeline, controls | `Source/PluginEditor.*` |
| **AudioEngine** | Audio routing, transport sync, DSP coordination | `Source/Audio/AudioEngine.*` |
| **InstrumentGenerator** | Synthesize 5 trap instruments (808, hi-hat, snare, piano, synth) | `Source/Audio/InstrumentGenerator.*` |
| **VocalSynthesis** | TTS + vocoder pipeline, formant synthesis fallback | `Source/AI/VocalSynthesis.*` |
| **FXChain** | Serial effects processing (distortion, delay, reverb, AI FX) | `Source/DSP/FXChain.*` |
| **ONNXInference** | ONNX Runtime wrapper, model loading, tensor operations | `Source/AI/ONNXInference.*` |
| **ScriptParser** | Parse stage scripts (`[SECTION:START:DURATION]`) | `Source/Parser/ScriptParser.*` |
| **Arrangement** | Timeline management, DAW transport sync | `Source/Parser/Arrangement.*` |
| **UndoManager** | 100-level undo/redo history | `Source/State/UndoManager.*` |

### Dependencies

**Required:**
- CMake 3.15+
- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- JUCE 7+ (auto-fetched by CMake via FetchContent)

**Optional (for AI features):**
- ONNX Runtime C++ SDK
- Python 3.10+ (for model export scripts)
- PyTorch/TensorFlow (for custom model training)

**Platform-Specific:**
- **Linux:** ALSA, JACK, X11, Freetype, WebKit2GTK
- **macOS:** Xcode 12+, macOS 10.13+
- **Windows:** Visual Studio 2019+, Windows 10 SDK

## 📂 Repository Structure

```
Voice_Clone-VST/
├── CMakeLists.txt                    # Build configuration for JUCE + ONNX Runtime
├── README.md                         # This file
├── BUILD.md                          # Detailed build instructions
├── ARCHITECTURE.md                   # System architecture documentation
├── QUICKSTART.md                     # Quick start guide for users
├── LAUNCHER_GUIDE.md                 # Launcher and installation guide
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
│
├── setup_maevn_repo.sh/.bat         # Repository setup scripts
├── build_maevn_onnx.sh/.bat         # ONNX model export scripts
├── install_maevn.sh/.bat            # Installation scripts (VST3 + Standalone)
├── launch_maevn.sh/.bat             # Standalone launcher scripts
├── maevn.desktop                    # Linux desktop entry file
│
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
│
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
│
├── scripts/                          # Python ONNX export scripts
│   ├── README.md                     # Scripts documentation
│   ├── export_drum_models.py         # Drum model export
│   ├── export_instrument_models.py   # Instrument model export
│   └── export_vocal_models.py        # Vocal model export
│
├── Tests/                            # Unit tests
│   ├── CMakeLists.txt
│   ├── ScriptParserTests.cpp
│   ├── ArrangementTests.cpp
│   ├── AudioEngineTests.cpp
│   └── BuildVerificationTests.cpp
│
├── CMI/                              # Cognitive Mesh Interface (Multi-Agent Dev)
│   ├── README.md                     # CMI overview
│   ├── MACF.md                       # Multi-Agent Command Framework
│   ├── agent_roles.md                # Agent role definitions
│   └── operational_ethics.md         # Development ethics guidelines
│
└── examples/                         # Example usage
    └── ARRANGEMENTS.md               # Example stage scripts
```

## 🎛️ DAW Compatibility

**Plugin Format:** VST3 and Standalone

### ✅ Compatible DAWs (VST3 Support Required)

- Ableton Live 10+
- Steinberg Cubase 10.5+
- FL Studio 20+
- Reaper 5.0+
- Bitwig Studio 3.0+
- PreSonus Studio One 4+
- Tracktion Waveform
- Steinberg Nuendo

### ❌ Currently NOT Compatible

- **Audacity** — Does not support VST3 format (only VST2/LV2/AU)
- Pro Tools — AAX format required
- Older DAWs without VST3 support

**Note:** MAEVN is built as a VST3 plugin. To use MAEVN, your DAW must support the VST3 plugin format. For Audacity users, use the **Standalone build** which can run independently without a DAW.

## ⚙️ Build Instructions

### Requirements

- **CMake** 3.15 or later
- **JUCE** 7+ (auto-fetched by CMake)
- **ONNX Runtime** C++ SDK (optional)
- **Python** 3.10+ (optional, for exporting ONNX models)
- **C++17** compatible compiler

### Quick Build

```bash
# 1. Setup repository
./setup_maevn_repo.sh  # Creates Models directories

# 2. Export models (optional)
./build_maevn_onnx.sh  # Requires Python 3.10+ and PyTorch

# 3. Configure and build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# 4. Install plugin
# Copy from build/MAEVN_artefacts/Release/VST3/MAEVN.vst3
# to your system's VST3 folder
```

### Platform-Specific Instructions

See [BUILD.md](BUILD.md) for detailed platform-specific build instructions including:
- Windows (Visual Studio)
- macOS (Xcode)
- Linux (GCC/Clang)
- ONNX Runtime integration
- Troubleshooting

### Building with ONNX Runtime

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXRUNTIME_LIB=/path/to/onnxruntime/lib/libonnxruntime.so \
  -DCMAKE_CXX_FLAGS="-DONNXRUNTIME_AVAILABLE"
```

**Note:** Without ONNX Runtime, MAEVN uses high-quality DSP fallbacks for all AI features.

## 📖 Usage Guide

### Interface Overview

The MAEVN interface is divided into 4 sections:

#### 1. Instruments (Left Panel)
Toggle trap instruments:
- **808 Bass** — Sub bass (responds to any MIDI note)
- **Hi-Hat** — High frequency percussion (MIDI notes 42-46)
- **Snare** — Snare drum (MIDI notes 38-40)
- **Piano** — Melodic piano (any MIDI note)
- **Synth** — Lead synthesizer (any MIDI note)

#### 2. Vocals (Center-Left Panel)
- **Enable Vocals** — Turn on/off vocal synthesis
- **Pitch** — Adjust vocal pitch (-12 to +12 semitones)
- **Formant** — Change vocal character (0.5 to 2.0)

#### 3. Effects (Center-Right Panel)
- **Reverb** — Space and ambience
- **Delay** — Echo effect with feedback
- **Distortion** — Saturation and drive
- **AI FX** — Neural effects (requires ONNX model)

#### 4. Master (Right Panel)
- **Gain** — Overall volume
- **Pan** — Stereo positioning

### Stage Script System

Define song arrangements using the stage script editor:

```
[INTRO:0:8] [VERSE:8:16] [HOOK:24:16] [808:40:8] [OUTRO:48:8]
```

**Format:** `[SECTION:START:DURATION]`
- **START:** Position in quarter notes (beats)
- **DURATION:** Length in quarter notes

**Section Types:**
- `HOOK` — Full instrumentation + vocals
- `VERSE` — Selected instruments + vocals
- `808` — Only 808 bass
- `INTRO` — Intro section with reduced instruments
- `OUTRO` — Outro section

### Audio Pipeline

```
MIDI Input → Instruments → Vocals → FX Chain → Master → Output
                ↓            ↓         ↓
           ONNX Models  ONNX Models  ONNX Models
           (optional)   (optional)   (optional)
```

### Model Hot Reload

The `/Models/config.json` file maps logical roles (e.g., `VOCAL`, `808`) to file paths. MAEVN automatically reloads models if changes are detected on disk.

See [QUICKSTART.md](QUICKSTART.md) for detailed usage instructions.

## 🤝 Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Coding standards (C++17, JUCE style)
- Module ownership
- Workflow (branching, PRs, commits)
- Testing requirements
- Build environment setup

### Coding Standards

- **Language:** C++17 for plugin, Python 3.10+ for model scripts
- **Style:** JUCE coding style (4 spaces, braces on new lines)
- **Documentation:** Doxygen comments for classes and methods
- **Memory:** Use smart pointers, avoid raw pointers
- **Real-time:** No allocations in audio thread

### Workflow

1. Fork the repository
2. Create feature branch from `dev`
3. Implement changes
4. Write/update tests
5. Submit Pull Request
6. Code review by maintainers

### Commit Convention

Use Conventional Commits format:
- `feat:` — New features
- `fix:` — Bug fixes
- `docs:` — Documentation updates
- `refactor:` — Code refactoring
- `test:` — Test additions/changes

## 🧠 Multi-Agent Development System

MAEVN embraces a revolutionary **Multi-Agent Engineering Protocol** enabling multiple autonomous or semi-autonomous agents (human + AI) to co-develop, test, and optimize the codebase in parallel.

### Agent Roles

| Role | Responsibility | Primary Tools |
|------|---------------|---------------|
| **Architect Agent** | System design and architectural decisions | Design tools, documentation |
| **DSP Developer Agent** | Audio processing and JUCE engine logic | C++, JUCE framework |
| **AI/ML Agent** | ONNX model design, training, and export | Python, PyTorch/TensorFlow |
| **GUI Developer Agent** | User interface and user experience | JUCE GUI, C++ |
| **Integration Agent** | Module integration and system testing | CI/CD, Build tools |
| **QA/Testing Agent** | Quality assurance and numerical stability | Testing frameworks |
| **Documentation Agent** | Documentation and knowledge management | Markdown, docs |
| **DevOps Agent** | Build systems, CI/CD, and tooling | CMake, Scripts |

### Cognitive Mesh Interface (CMI)

The `/CMI/` directory contains:
- Agent role definitions
- Multi-Agent Command Framework (MACF)
- Operational ethics guidelines
- Mission logs and task coordination

See [CMI/README.md](CMI/README.md) for complete multi-agent development documentation.

### Operational Ethics

All agents (AI and human) must adhere to:

**Key Principles:**
- ✅ Transparency — All actions logged and traceable
- ✅ Determinism — Consistent, predictable results
- ✅ Real-Time Constraints — Audio processing < 1ms per buffer
- ✅ Quality Standards — All tests pass, no security vulnerabilities
- ✅ Respect — Never break existing functionality

**Prohibited Actions:**
- ❌ Never commit compiled `.onnx` binaries to Git
- ❌ Never introduce security vulnerabilities
- ❌ Never break real-time safety guarantees
- ❌ Never remove tests without approval
- ❌ Never commit credentials or private data

See [CMI/operational_ethics.md](CMI/operational_ethics.md) for complete guidelines.

## 📚 Documentation

- **[README.md](README.md)** — This file (overview and quick start)
- **[BUILD.md](BUILD.md)** — Detailed build instructions for all platforms
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — System architecture and design
- **[QUICKSTART.md](QUICKSTART.md)** — User quick start guide
- **[LAUNCHER_GUIDE.md](LAUNCHER_GUIDE.md)** — Launcher and installation guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines
- **[TESTING.md](TESTING.md)** — Testing guidelines and practices
- **[CMI/README.md](CMI/README.md)** — Multi-agent development system
- **[examples/ARRANGEMENTS.md](examples/ARRANGEMENTS.md)** — Example stage scripts

## 🚧 Roadmap

### Current Status

MAEVN is in active development. The core functionality is implemented and stable:

✅ **Complete:**
- VST3 plugin architecture
- 5 trap instruments (808, hi-hat, snare, piano, synth)
- Basic vocal synthesis with formant fallback
- DSP effects chain (distortion, delay, reverb)
- Stage script parser and timeline
- ONNX Runtime integration
- Undo/redo system
- Standalone application

🚧 **In Progress:**
- FXPreset system with browser
- Visual undo history component
- Per-lane FX mode selectors
- Tag cloud preset navigation
- Enhanced AI model library
- Preset exchange system

### Future Features

- **Automation Hooks** — DAW automation for all parameters
- **Drag-to-MIDI/Audio** — Export blocks to DAW timeline
- **Instrument Sequencer** — Built-in pattern editor (hi-hat rolls, 808 glides)
- **Preset Exchange** — Community preset packs
- **Sample Playback** — Import audio samples
- **More Instruments** — Expand synthesis library
- **Modulation System** — LFOs and envelopes
- **MIDI CC Mapping** — Map controllers to parameters
- **Recording** — Internal audio bounce

## ⚡ Performance

### CPU Usage (Typical)

- Instruments: ~5% per active instrument
- Vocals: ~10% (DSP) or variable (ONNX)
- Effects: ~5-10% per effect
- AI FX: Variable based on model complexity

### Optimization Tips

1. Disable unused instruments
2. Reduce effect mix when not needed
3. Use DSP fallbacks instead of ONNX
4. Increase DAW buffer size
5. Build in Release mode (not Debug)

## 🐛 Troubleshooting

### Plugin Won't Load

1. Ensure VST3 is in correct system folder
2. Rescan plugins in DAW
3. Check DAW supports VST3 format
4. Verify 64-bit plugin for 64-bit DAW

### No Sound

1. Check master gain is not at 0
2. Enable at least one instrument
3. Verify MIDI is being received
4. Check audio routing in DAW

### Build Errors

1. Ensure all dependencies installed
2. Use Release build configuration
3. Check CMake version (3.15+)
4. See [BUILD.md](BUILD.md) troubleshooting section

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **JUCE Framework** — Audio plugin framework
- **ONNX Runtime** — AI model inference
- **Contributors** — All developers who contribute to MAEVN
- **Community** — Users and testers providing feedback

## 📞 Support

- **GitHub Issues:** [Report bugs or request features](../../issues)
- **GitHub Discussions:** [Ask questions and share ideas](../../discussions)
- **Documentation:** See docs in this repository

## 🌟 Star History

If you find MAEVN useful, please consider giving it a star ⭐ on GitHub!

---

**MAEVN** — Where AI meets music production. An experimental framework for AI-augmented sound design.

**Built with** ❤️ **by the Vocal Cloning Quantum Collective**
