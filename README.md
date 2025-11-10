# 🎚 MAEVN — AI-Powered Vocal + Instrument Generator

MAEVN is a comprehensive JUCE-based VST3 plugin with ONNX Runtime integration that combines AI-powered vocal synthesis with trap-inspired instruments and hybrid effects processing.

## Features

### 🎤 AI Vocals
- **TTS + Vocoder**: Text-to-speech synthesis with neural vocoder support
- **ONNX Integration**: Load custom TTS and vocoder models
- **Pitch & Formant Control**: Real-time vocal manipulation
- **Simple Fallback**: Built-in formant synthesis when AI models aren't available

### 🥁 Trap-Inspired Instruments
- **808 Bass**: Classic sub-bass with pitch sweep envelope
- **Hi-Hat**: Noise-based percussion with fast decay
- **Snare**: Hybrid tone + noise synthesis
- **Piano**: Multi-harmonic melodic instrument
- **Synth**: LFO-modulated sawtooth synthesizer

### 🎛 Hybrid FX Chains
- **DSP Effects**:
  - Reverb (JUCE DSP)
  - Delay with feedback
  - Soft-clipping distortion
- **AI Effects**: ONNX-based neural audio effects (customizable)

### 🎼 Stage-Script Parser
Parse arrangement scripts with timeline sections:
```
[INTRO:0:4] [VERSE:4:12] [HOOK:12:20] [VERSE:20:28] [HOOK:28:36] [OUTRO:36:40]
```

Each section automatically configures:
- Which instruments are active
- Vocal enable/disable
- Duration and timing

Supported section types:
- `[HOOK:start:duration]` - Full arrangement with all instruments
- `[VERSE:start:duration]` - Verse section with selected instruments
- `[808:start:duration]` - 808-only section
- `[INTRO:start:duration]` - Intro with reduced instrumentation
- `[OUTRO:start:duration]` - Outro section

### ↩️ Global Undo/Redo
- Full parameter change history
- Up to 100 undo levels
- Transaction-based state management

### 🎯 Real-Time Features
- **Timeline Arrangement**: Visual arrangement with color-coded sections
- **FX Automation**: Time-based parameter automation with interpolation
- **DAW Sync**: Tempo, transport, and position synchronization
- **Live Performance**: Real-time MIDI input and audio processing

## Building the Plugin

### Prerequisites

1. **CMake** (3.15 or later)
2. **C++17 compatible compiler**:
   - Windows: Visual Studio 2019 or later
   - macOS: Xcode 12 or later
   - Linux: GCC 9+ or Clang 10+
3. **JUCE Framework** (automatically fetched via CMake)
4. **ONNX Runtime** (optional, for AI features)

### Quick Build

```bash
# Clone the repository
git clone https://github.com/fxgeniusllc-oss/Voice_Clone-VST.git
cd Voice_Clone-VST

# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# The VST3 plugin will be in:
# - Windows: build/MAEVN_artefacts/Release/VST3/
# - macOS: build/MAEVN_artefacts/Release/VST3/
# - Linux: build/MAEVN_artefacts/Release/VST3/
```

### Building with ONNX Runtime

If you have ONNX Runtime installed:

```bash
# Option 1: Using find_package (if ONNX Runtime is installed system-wide)
cmake ..

# Option 2: Manual paths
cmake .. \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXRUNTIME_LIB=/path/to/onnxruntime/lib/libonnxruntime.so

# Then build
cmake --build . --config Release
```

To enable ONNX features, define `ONNXRUNTIME_AVAILABLE` during compilation:
```bash
cmake .. -DCMAKE_CXX_FLAGS="-DONNXRUNTIME_AVAILABLE"
```

### Platform-Specific Notes

#### Windows
```bash
# Use Visual Studio generator
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

#### macOS
```bash
# Use Xcode generator
cmake .. -G Xcode
cmake --build . --config Release

# Or use Unix Makefiles
cmake .. -G "Unix Makefiles"
make -j8
```

#### Linux
```bash
# Install dependencies
sudo apt-get install libasound2-dev libjack-jackd2-dev \
    libcurl4-openssl-dev libfreetype6-dev libx11-dev \
    libxcomposite-dev libxcursor-dev libxinerama-dev \
    libxrandr-dev libxrender-dev libwebkit2gtk-4.0-dev \
    libglu1-mesa-dev mesa-common-dev

# Build
cmake ..
make -j$(nproc)
```

## Installation

### VST3 Installation Paths

Copy the built VST3 bundle to your system's VST3 folder:

- **Windows**: `C:\Program Files\Common Files\VST3\`
- **macOS**: `~/Library/Audio/Plug-Ins/VST3/` or `/Library/Audio/Plug-Ins/VST3/`
- **Linux**: `~/.vst3/` or `/usr/lib/vst3/`

## Usage

### Basic Workflow

1. **Load the Plugin**: Open MAEVN in your DAW as a VST3 instrument
2. **Configure Instruments**: Enable/disable trap instruments (808, Hi-Hat, Snare, Piano, Synth)
3. **Set Up Vocals**: Enable vocals and adjust pitch/formant parameters
4. **Apply Effects**: Mix DSP and AI effects (Reverb, Delay, Distortion, AI FX)
5. **Define Arrangement**: Edit the stage script to create timeline sections
6. **Play**: MAEVN responds to MIDI input and syncs with your DAW

### Stage Script Format

```
[SECTION_NAME:START_TIME:DURATION]
```

- **SECTION_NAME**: INTRO, VERSE, HOOK, 808, OUTRO, or custom
- **START_TIME**: Start position in quarter notes (PPQ)
- **DURATION**: Length in quarter notes

Example:
```
[INTRO:0:4] [VERSE:4:8] [HOOK:12:8] [808:20:4] [OUTRO:24:4]
```

### MIDI Mapping

- **808 Bass**: Any MIDI note (responds to note pitch)
- **Hi-Hat**: MIDI notes 42-46 (typical hi-hat range)
- **Snare**: MIDI notes 38-40 (typical snare range)
- **Piano**: Any MIDI note (melodic)
- **Synth**: Any MIDI note (melodic)

### Loading AI Models

To use AI features, load ONNX models:

1. Place your `.onnx` model files in a known location
2. Use the audio processor API to load models:
   ```cpp
   // TTS Model
   audioProcessor.getAudioEngine().getVocalSynthesis().loadTTSModel("/path/to/tts.onnx");
   
   // Vocoder Model
   audioProcessor.getAudioEngine().getVocalSynthesis().loadVocoderModel("/path/to/vocoder.onnx");
   
   // AI FX Model
   audioProcessor.getAudioEngine().getFXChain().getAIEffects().loadModel("/path/to/fx.onnx");
   ```

Note: AI features gracefully degrade to DSP-based synthesis when models aren't available.

## Architecture

### Component Overview

```
MAEVN
├── Audio/
│   ├── AudioEngine - Main audio processing coordinator
│   └── InstrumentGenerator - Trap instrument synthesis
├── DSP/
│   ├── FXChain - Effect chain coordinator
│   └── Effects - DSP effect implementations
├── AI/
│   ├── ONNXInference - ONNX Runtime wrapper
│   ├── VocalSynthesis - TTS + vocoder pipeline
│   └── AIEffects - Neural audio effects
├── Parser/
│   ├── ScriptParser - Stage script parser
│   └── Arrangement - Timeline arrangement manager
├── State/
│   ├── UndoManager - Global undo/redo system
│   └── ParameterState - Automation and state management
└── UI/
    ├── MainComponent - Main plugin interface
    └── TimelineComponent - Visual timeline display
```

### Processing Pipeline

1. **MIDI Input** → Instrument Generator
2. **Instrument Audio** → Vocal Synthesis → FX Chain → Master Output
3. **Timeline** → Arrangement → Section Configuration
4. **Parameters** → Automation → Real-time Control

## Parameters

### Instruments (Boolean)
- `enable808` - Enable 808 bass
- `enableHiHat` - Enable hi-hat
- `enableSnare` - Enable snare
- `enablePiano` - Enable piano
- `enableSynth` - Enable synth

### Vocals
- `enableVocals` (Boolean) - Enable vocal synthesis
- `vocalPitch` (-12 to +12 semitones) - Pitch shift
- `vocalFormant` (0.5 to 2.0) - Formant shift

### Effects
- `reverbMix` (0.0 to 1.0) - Reverb wet/dry mix
- `delayMix` (0.0 to 1.0) - Delay wet/dry mix
- `distortion` (0.0 to 1.0) - Distortion amount
- `aiFxMix` (0.0 to 1.0) - AI effects mix

### Master
- `masterGain` (0.0 to 1.0) - Master output gain
- `masterPan` (-1.0 to 1.0) - Master stereo pan

## Development

### Project Structure

```
Voice_Clone-VST/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
└── Source/
    ├── PluginProcessor.cpp/h    # Main plugin processor
    ├── PluginEditor.cpp/h       # Plugin editor
    ├── Audio/                   # Audio engine components
    ├── DSP/                     # DSP effects
    ├── AI/                      # AI/ML components
    ├── Parser/                  # Script parser
    ├── State/                   # State management
    └── UI/                      # User interface
```

### Adding Custom AI Models

MAEVN supports custom ONNX models for:

1. **TTS (Text-to-Speech)**: Convert text to mel-spectrogram
2. **Vocoder**: Convert mel-spectrogram to audio waveform
3. **AI Effects**: Neural audio processing

Model requirements:
- Format: ONNX (Open Neural Network Exchange)
- Input: Float tensor (1D for audio, 2D for spectrograms)
- Output: Float tensor matching input shape

### Extending Instruments

To add new instruments:

1. Add a new class in `InstrumentGenerator.h`
2. Implement synthesis in `InstrumentGenerator.cpp`
3. Add enable parameter in `PluginProcessor.cpp`
4. Add UI control in `MainComponent.cpp`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Credits

- **JUCE Framework**: https://juce.com/
- **ONNX Runtime**: https://onnxruntime.ai/
- **FXGenius LLC**: Plugin development

## Support

For issues, questions, or feature requests, please open an issue on GitHub:
https://github.com/fxgeniusllc-oss/Voice_Clone-VST/issues
