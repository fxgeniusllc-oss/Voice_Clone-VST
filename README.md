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
    - **PluginProcessor:** Handles the audio processing block, routing audio I/O to the PatternEngine, OnnxEngine, and AIFXEngine.
    - **PluginEditor:** Provides the user interface, including timeline lanes, preset browser, and undo history.

- **ONNX Engine:**
    - **OnnxEngine:** Encapsulates the ONNX Runtime C++ API, supporting live model updates through hot reloading from the /Models/ directory and enabling multiple instrument/vocal models to function simultaneously.

- **Pattern Engine:** 
    - Parses input from the lyrical stage script to determine the arrangement of musical blocks, manages synchronization with the DAW’s playhead, and triggers instruments and vocals.

- **AI FX Engine:** 
    - **DSP FX:** Includes effects such as compression, equalization, reverb, and limiting.
    - **AI FX:** Utilizes ONNX models for effects like autotuning and AI mastering, with the option for mixing DSP and AI in a sequential effects chain.

**Timeline Lanes:** Each track lane (VOCAL, 808, HiHat, etc.) offers:
- **FX Mode Selector:** Options to select Off, DSP, AI, or Hybrid modes.
- **Preset Load/Save:** Simplifies the management of effect settings.
- **Block Visualization:** Displays the arrangement graphically.

**Preset System:**
- **FXPreset Struct:** Maintains information such as category, tags, and parameters related to presets, with JSON serialization for seamless data handling.
- **Preset Browser Component:** Features a categorized filter, search functionality, and a tag cloud for easy navigation of presets.

📂 **Repo Structure**  
MAEVN/
- ├── CMakeLists.txt          # Build configuration for JUCE + ONNX Runtime
- ├── README.md               # Documentation
- ├── Source/                 # Core source files
- │   ├── PluginProcessor.*   # Core DSP processing logic
- │   ├── PluginEditor.*      # User interface elements
- │   ├── OnnxEngine.*        # AI inference handling module
- │   ├── PatternEngine.*     # Script parsing and arrangement logic
- │   ├── AIFXEngine.*        # Hybrid effects processing
- │   ├── TimelineLane.*      # GUI elements for track lanes
- │   ├── FXPreset.*          # Preset management schema
- │   ├── FXPresetManager.*   # Handling of preset I/O operations
- │   ├── PresetBrowserComponent.* # User interface for preset navigation
- │   ├── GlobalUndoManager.* # Management of action history
- │   ├── UndoHistoryComponent.* # User interface for undo list
- │   └── Utilities.h         # Shared utility functions and constants
- ├── Models/                 # Directory for ONNX models
- │   ├── drums/              # Drum instrument models
- │   │   ├── 808_ddsp.onnx
- │   │   ├── hihat_ddsp.onnx
- │   │   └── snare_ddsp.onnx
- │   ├── instruments/        # Instrument models
- │   │   ├── piano_ddsp.onnx
- │   │   └── synth_fm.onnx
- │   └── vocals/             # Vocal models
- │       ├── vocals_tts.onnx
- │       └── vocals_hifigan.onnx
- └── config.json             # Maps model roles to file paths

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
