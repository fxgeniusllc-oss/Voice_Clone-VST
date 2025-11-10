# MAEVN Architecture

This document describes the internal architecture and design of the MAEVN VST3 plugin.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         DAW (Host)                               │
│                                                                   │
│  ┌────────────┐      ┌──────────────┐      ┌─────────────┐     │
│  │ MIDI Track │─────▶│ MAEVN VST3   │─────▶│ Audio Track │     │
│  └────────────┘      └──────────────┘      └─────────────┘     │
│                             │                                     │
│                      Transport Sync                              │
│                    (Tempo, Position)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MAEVN Plugin                               │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   PluginProcessor                         │  │
│  │  - Parameters (APVTS)                                     │  │
│  │  - State Management                                       │  │
│  │  - Process Block Callback                                 │  │
│  └────────────┬─────────────────────────────────┬────────────┘  │
│               │                                  │                │
│               ▼                                  ▼                │
│  ┌───────────────────────┐          ┌──────────────────────┐   │
│  │    AudioEngine        │          │   PluginEditor       │   │
│  │                       │          │   (GUI)              │   │
│  │  ┌─────────────────┐ │          │                      │   │
│  │  │ Instrument      │ │          │  ┌────────────────┐  │   │
│  │  │ Generator       │ │          │  │ MainComponent  │  │   │
│  │  │ - 808          │ │          │  │ - Controls     │  │   │
│  │  │ - Hi-Hat       │ │          │  │ - Sliders      │  │   │
│  │  │ - Snare        │ │          │  │ - Buttons      │  │   │
│  │  │ - Piano        │ │          │  └────────────────┘  │   │
│  │  │ - Synth        │ │          │  ┌────────────────┐  │   │
│  │  └─────────────────┘ │          │  │ Timeline       │  │   │
│  │                       │          │  │ Component      │  │   │
│  │  ┌─────────────────┐ │          │  └────────────────┘  │   │
│  │  │ VocalSynthesis  │ │          └──────────────────────┘   │
│  │  │ - TTS          │ │                                       │
│  │  │ - Vocoder      │ │                                       │
│  │  └─────────────────┘ │                                       │
│  │                       │                                       │
│  │  ┌─────────────────┐ │                                       │
│  │  │ FXChain         │ │                                       │
│  │  │ - Reverb       │ │                                       │
│  │  │ - Delay        │ │                                       │
│  │  │ - Distortion   │ │                                       │
│  │  │ - AI FX        │ │                                       │
│  │  └─────────────────┘ │                                       │
│  │                       │                                       │
│  │  ┌─────────────────┐ │                                       │
│  │  │ Arrangement     │ │                                       │
│  │  │ - ScriptParser │ │                                       │
│  │  │ - Timeline     │ │                                       │
│  │  └─────────────────┘ │                                       │
│  └───────────────────────┘                                       │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    State Management                        │  │
│  │  ┌─────────────────┐          ┌──────────────────┐       │  │
│  │  │ UndoManager     │          │ ParameterState   │       │  │
│  │  │ - History       │          │ - Automation     │       │  │
│  │  │ - Transactions  │          │ - Serialization  │       │  │
│  │  └─────────────────┘          └──────────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      AI Components                         │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │              ONNXInference                          │ │  │
│  │  │  - Model Loading                                    │ │  │
│  │  │  - Inference Engine                                 │ │  │
│  │  │  - Tensor Operations                                │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Audio Processing Path

```
MIDI Input
    │
    ▼
┌────────────────────┐
│ InstrumentGenerator│
│  - Note On/Off     │
│  - Synthesis       │
└────────┬───────────┘
         │
         ▼
   Audio Buffer
         │
         ▼
┌────────────────────┐
│  VocalSynthesis    │
│  - TTS Generation  │
│  - Vocoder         │
└────────┬───────────┘
         │
         ▼
   Audio Buffer
         │
         ▼
┌────────────────────┐
│     FXChain        │
│  - Distortion      │
│  - Delay           │
│  - Reverb          │
│  - AI FX           │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Master Section    │
│  - Gain            │
│  - Pan             │
└────────┬───────────┘
         │
         ▼
    Audio Output
```

### Arrangement Processing

```
DAW Transport
    │
    ▼
┌────────────────────┐
│  PlayHead Info     │
│  - PPQ Position    │
│  - BPM             │
│  - Time Signature  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Arrangement      │
│  - Update Position │
│  - Find Section    │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  ScriptParser      │
│  - Active Section  │
│  - Configuration   │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ AudioEngine Config │
│ - Enable/Disable   │
│ - Instruments      │
└────────────────────┘
```

## Component Details

### PluginProcessor

**Responsibility**: Main VST3 plugin interface
- Manages plugin lifecycle
- Handles parameter changes
- Coordinates audio processing
- Manages state serialization

**Key Methods**:
- `prepareToPlay()` - Initialize audio processing
- `processBlock()` - Process audio/MIDI
- `getStateInformation()` - Save state
- `setStateInformation()` - Load state

### AudioEngine

**Responsibility**: Audio processing coordinator
- Routes audio through components
- Manages sample rate and buffer size
- Synchronizes with DAW transport
- Applies arrangement configuration

**Subcomponents**:
- InstrumentGenerator
- VocalSynthesis
- FXChain
- Arrangement

### InstrumentGenerator

**Responsibility**: Generate trap instrument sounds
- Responds to MIDI input
- Synthesizes 5 instruments
- Manages polyphony

**Instruments**:
1. **Bass808**: Sub-bass with pitch envelope
2. **HiHat**: Noise-based percussion
3. **Snare**: Tone + noise hybrid
4. **Piano**: Harmonic synthesis
5. **Synth**: Modulated sawtooth

### VocalSynthesis

**Responsibility**: AI-powered vocal generation
- Text-to-speech conversion
- Neural vocoder processing
- Fallback formant synthesis

**Pipeline**:
```
Text Input
    ↓
TTS Model (ONNX)
    ↓
Mel-Spectrogram
    ↓
Vocoder Model (ONNX)
    ↓
Audio Waveform
```

### FXChain

**Responsibility**: Audio effects processing
- Serial effect chain
- Mix/bypass control
- DSP and AI effects

**Effects Order**:
1. Distortion (pre-effect)
2. Delay
3. Reverb
4. AI FX (post-effect)

### ScriptParser & Arrangement

**Responsibility**: Timeline and arrangement management
- Parse stage scripts
- Track playback position
- Configure sections

**Script Format**:
```
[SECTION:START:DURATION]
```

**Section Configuration**:
- Which instruments are active
- Vocal enable/disable
- Timing and duration

### State Management

**UndoManager**:
- Transaction-based undo/redo
- 100-level history
- Parameter change tracking

**ParameterState**:
- Automation data storage
- Timeline-based automation
- Linear interpolation

### UI Components

**MainComponent**:
- 4-section layout (Instruments, Vocals, FX, Master)
- Parameter controls
- Script editor
- Undo/Redo buttons

**TimelineComponent**:
- Visual arrangement display
- Color-coded sections
- Current position indicator

## Threading Model

### Audio Thread (Real-time)
- `processBlock()` callback
- All audio synthesis
- Effect processing
- **No allocations allowed**
- **No locks allowed**

### Message Thread (UI)
- Parameter changes
- UI updates
- Timeline rendering
- File I/O
- Model loading

### Thread Safety

- JUCE AudioProcessorValueTreeState handles parameter thread safety
- Audio thread reads parameters atomically
- UI thread updates parameters through APVTS
- No shared mutable state between threads

## Memory Management

### Audio Buffers
- Pre-allocated in `prepareToPlay()`
- Reused every process block
- No dynamic allocation during processing

### ONNX Models
- Loaded once on initialization
- Session persists for plugin lifetime
- Inference uses pre-allocated tensors

### State
- Serialized to XML
- Stored in plugin state
- Restored on load

## Extension Points

### Adding Instruments

1. Create instrument class in `InstrumentGenerator.h`
2. Add instance to `InstrumentGenerator`
3. Add parameter in `PluginProcessor`
4. Add UI control in `MainComponent`
5. Update documentation

### Adding Effects

1. Create effect class in `DSP/Effects.h`
2. Add to `FXChain`
3. Add parameter
4. Add UI control
5. Update documentation

### Adding Section Types

1. Update `ScriptParser::configureSectionByName()`
2. Define instrument configuration
3. Update examples
4. Update documentation

## Performance Considerations

### CPU Usage
- Instruments: ~5% per active instrument
- Vocals: ~10% (DSP) or variable (ONNX)
- Effects: ~5-10% per effect
- AI FX: Variable based on model

### Optimization Strategies
- Disable unused instruments
- Reduce effect mix when not needed
- Use DSP fallbacks instead of ONNX
- Increase DAW buffer size

## Dependencies

### JUCE Framework
- Audio processing (AudioProcessor)
- DSP modules (Reverb, WaveShaper)
- UI components (Component, Slider, etc.)
- Value tree state (parameters)

### ONNX Runtime (Optional)
- Model loading
- Inference execution
- Tensor operations

### C++ Standard Library
- Containers (vector, map)
- Smart pointers (unique_ptr, shared_ptr)
- Algorithms (sort, transform)

## File Structure

```
Voice_Clone-VST/
├── CMakeLists.txt              # Build configuration
├── Source/
│   ├── PluginProcessor.h/cpp   # Main processor
│   ├── PluginEditor.h/cpp      # Main editor
│   ├── Audio/                  # Audio generation
│   │   ├── AudioEngine.h/cpp
│   │   └── InstrumentGenerator.h/cpp
│   ├── DSP/                    # DSP effects
│   │   ├── FXChain.h/cpp
│   │   └── Effects.h/cpp
│   ├── AI/                     # AI components
│   │   ├── ONNXInference.h/cpp
│   │   ├── VocalSynthesis.h/cpp
│   │   └── AIEffects.h/cpp
│   ├── Parser/                 # Script parsing
│   │   ├── ScriptParser.h/cpp
│   │   └── Arrangement.h/cpp
│   ├── State/                  # State management
│   │   ├── UndoManager.h/cpp
│   │   └── ParameterState.h/cpp
│   └── UI/                     # User interface
│       ├── MainComponent.h/cpp
│       └── TimelineComponent.h/cpp
└── docs/                       # Documentation
```

## Future Architecture Improvements

### Potential Enhancements

1. **Multi-threading**: Parallel instrument processing
2. **SIMD**: Vectorized DSP operations
3. **Preset System**: Save/load arrangements and parameters
4. **MIDI CC**: Map MIDI controllers to parameters
5. **Sample Playback**: Import audio samples
6. **More Instruments**: Expand instrument library
7. **More Effects**: Additional FX options
8. **Modulation System**: LFOs and envelopes
9. **Step Sequencer**: Built-in pattern editor
10. **Recording**: Bounce audio internally

## Design Principles

1. **Modularity**: Clear separation of concerns
2. **Real-time Safety**: No allocations in audio thread
3. **Extensibility**: Easy to add features
4. **Performance**: Optimize critical paths
5. **Robustness**: Graceful degradation
6. **Documentation**: Well-documented code
7. **Testing**: Testable components
8. **User Experience**: Intuitive interface

## Questions?

For architectural questions or suggestions:
- Open a GitHub Discussion
- See CONTRIBUTING.md
- Review source code comments

---

*This architecture document is maintained alongside the codebase.*
