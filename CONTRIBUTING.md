# Contributing to MAEVN

Thank you for your interest in contributing to MAEVN! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Code Style](#code-style)
8. [Architecture](#architecture)

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling or insulting/derogatory comments
- Public or private harassment
- Publishing others' private information

## Getting Started

### Before You Begin

1. **Check existing issues**: Look for related issues or feature requests
2. **Open an issue first**: For major changes, discuss with maintainers first
3. **Fork the repository**: Create your own fork to work in
4. **Read the documentation**: Familiarize yourself with the codebase

### Types of Contributions

We welcome:
- **Bug fixes**: Fix issues or improve stability
- **Features**: Add new instruments, effects, or capabilities
- **Documentation**: Improve README, guides, or code comments
- **Examples**: Add new arrangement examples or tutorials
- **Testing**: Add tests or improve test coverage
- **Performance**: Optimize algorithms or reduce CPU usage

## Development Setup

### Prerequisites

1. Install build tools (see [BUILD.md](BUILD.md))
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Voice_Clone-VST.git
   cd Voice_Clone-VST
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/fxgeniusllc-oss/Voice_Clone-VST.git
   ```

### Building for Development

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
```

Debug builds include:
- Debug symbols
- Assertions enabled
- Logging enabled
- No optimization

### IDE Setup

#### Visual Studio Code
```json
{
  "cmake.buildDirectory": "${workspaceFolder}/build",
  "cmake.configureSettings": {
    "CMAKE_BUILD_TYPE": "Debug"
  }
}
```

#### CLion / Visual Studio
Open the root `CMakeLists.txt` as a project.

## Making Changes

### Workflow

1. **Create a branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make changes**: Edit code following our style guide

3. **Test locally**: Build and test your changes

4. **Commit**: Write clear commit messages
   ```bash
   git add .
   git commit -m "Add new synth waveform option"
   ```

5. **Push**: Push to your fork
   ```bash
   git push origin feature/my-new-feature
   ```

6. **Pull request**: Open a PR against the main repository

### Branch Naming

- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions

Examples:
- `feature/add-chorus-effect`
- `bugfix/fix-808-pitch-envelope`
- `docs/update-build-instructions`

### Commit Messages

Good commit messages:
```
Add reverb pre-delay parameter

- Add pre-delay parameter (0-100ms)
- Update ReverbEffect class
- Add UI control in FX section
- Update documentation
```

Bad commit messages:
```
Update stuff
Fixed it
WIP
```

## Testing

### Manual Testing

1. **Build standalone version**
2. **Test instruments**: Trigger each instrument via MIDI
3. **Test effects**: Adjust all effect parameters
4. **Test arrangement**: Parse different scripts
5. **Test undo/redo**: Verify state management
6. **Test in DAW**: Load in actual DAW environment

### Automated Testing (Future)

When unit tests are added:
```bash
cd build
ctest
```

### Performance Testing

Monitor CPU usage:
1. Open Activity Monitor / Task Manager
2. Play complex arrangement
3. Check CPU percentage
4. Compare with previous version

## Submitting Changes

### Pull Request Process

1. **Update documentation**: If you changed behavior
2. **Add examples**: If you added features
3. **Test thoroughly**: On your platform
4. **Fill PR template**: Describe your changes
5. **Wait for review**: Address feedback promptly

### PR Checklist

- [ ] Code builds without errors
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Manual testing completed
- [ ] Commit messages are clear
- [ ] No unrelated changes included

## Code Style

### General Guidelines

- **C++17 standard**: Use modern C++ features
- **JUCE conventions**: Follow JUCE framework patterns
- **Clear naming**: Use descriptive variable/function names
- **Comments**: Comment complex algorithms, not obvious code
- **Consistency**: Match existing code style

### Naming Conventions

```cpp
// Classes: PascalCase
class AudioEngine {};

// Functions/Methods: camelCase
void processAudio() {}

// Variables: camelCase
int sampleRate = 44100;

// Constants: UPPER_SNAKE_CASE
const int MAX_BUFFER_SIZE = 8192;

// Member variables: camelCase
double currentPhase;

// Private members: camelCase (no prefix)
bool isPlaying;
```

### File Organization

```cpp
// Header file structure
#pragma once

#include <JuceHeader.h>
// Other includes

class MyClass
{
public:
    // Public interface

private:
    // Private members
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MyClass)
};
```

```cpp
// Implementation file structure
#include "MyClass.h"

MyClass::MyClass()
{
    // Constructor
}

// Methods in same order as header
```

### Code Formatting

- **Indentation**: 4 spaces (no tabs)
- **Braces**: Opening brace on same line
- **Line length**: Prefer < 100 characters
- **Blank lines**: Separate logical sections

```cpp
// Good
if (condition)
{
    doSomething();
    doSomethingElse();
}

// Bad
if(condition){
doSomething();
doSomethingElse();
}
```

### Comments

```cpp
// Good: Explain why, not what
// Apply pitch sweep for classic 808 sound
double pitchEnv = std::exp(-time / decayTime);

// Bad: Obvious comment
// Multiply by 2
result = value * 2;

// Good: Document complex algorithms
/**
 * Implements formant synthesis using parallel resonators.
 * Each formant is a bandpass filter centered at specific frequencies.
 * 
 * @param input Input audio buffer
 * @param formantShift Multiplier for formant frequencies (0.5 - 2.0)
 */
void applyFormantFilter(AudioBuffer& input, float formantShift);
```

## Architecture

### Component Overview

```
MAEVN/
├── PluginProcessor     # Main audio processor, parameters
├── PluginEditor        # Main UI coordinator
├── Audio/
│   ├── AudioEngine     # Audio processing coordinator
│   └── InstrumentGenerator  # Synth engines
├── DSP/
│   ├── FXChain         # Effect chain manager
│   └── Effects         # DSP effect implementations
├── AI/
│   ├── ONNXInference   # ONNX Runtime wrapper
│   ├── VocalSynthesis  # TTS + vocoder
│   └── AIEffects       # Neural FX
├── Parser/
│   ├── ScriptParser    # Parse arrangement scripts
│   └── Arrangement     # Timeline manager
├── State/
│   ├── UndoManager     # Undo/redo system
│   └── ParameterState  # Automation
└── UI/
    ├── MainComponent   # Main interface
    └── TimelineComponent  # Visual timeline
```

### Adding New Features

#### Adding a New Instrument

1. Add to `InstrumentGenerator.h`:
```cpp
class NewInstrument
{
public:
    void prepare(double sampleRate);
    void process(AudioBuffer<float>& buffer, const MidiBuffer& midi);
    void noteOn(int note, float velocity);
    void noteOff();
private:
    // Implementation
};
```

2. Add instance to `InstrumentGenerator`
3. Add enable parameter in `PluginProcessor`
4. Add UI control in `MainComponent`

#### Adding a New Effect

1. Create in `DSP/Effects.h`:
```cpp
class NewEffect
{
public:
    void prepare(double sampleRate, int samplesPerBlock);
    void process(AudioBuffer<float>& buffer, float amount);
private:
    // DSP implementation
};
```

2. Add to `FXChain`
3. Add parameter
4. Add UI control

#### Adding a New Section Type

1. Update `ScriptParser::configureSectionByName()`
2. Add section configuration
3. Update documentation

### Best Practices

- **RAII**: Use constructors/destructors for resource management
- **const correctness**: Mark const methods and parameters
- **Move semantics**: Use `std::move` for large objects
- **Smart pointers**: Use `std::unique_ptr` / `std::shared_ptr`
- **JUCE patterns**: Use JUCE classes (AudioBuffer, String, etc.)

### Performance Considerations

- **Real-time safety**: No allocations in audio thread
- **SIMD**: Consider vectorization for DSP
- **Caching**: Cache calculated values
- **Profiling**: Profile before optimizing

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Security issues**: Email maintainers directly
- **Feature requests**: Open an Issue with [Feature Request] tag

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Appreciated by the community!

Thank you for contributing to MAEVN! 🎵
