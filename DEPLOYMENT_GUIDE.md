# MAEVN VST3 - Production Deployment Guide

**Version:** 1.0.0  
**Status:** ✅ PRODUCTION READY  
**Last Updated:** December 12, 2024

## 🚀 Quick Start for End Users

### Installation (Pre-built Binary)

1. **Download the VST3 plugin** from the releases page
2. **Copy to your system's VST3 directory:**

   - **Windows**: `C:\Program Files\Common Files\VST3\MAEVN.vst3`
   - **macOS**: `~/Library/Audio/Plug-Ins/VST3/MAEVN.vst3`
   - **Linux**: `~/.vst3/MAEVN.vst3`

3. **Rescan plugins** in your DAW
4. **Load MAEVN** as an instrument plugin on a new track
5. **Start creating!** See [QUICKSTART.md](QUICKSTART.md)

### System Requirements

**Minimum:**
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- OS: Windows 10+, macOS 10.13+, or Linux (Ubuntu 20.04+)
- DAW: Any VST3-compatible host

**Recommended:**
- CPU: Quad-core 2.5 GHz or better
- RAM: 8 GB or more
- SSD storage
- Audio interface with ASIO/CoreAudio drivers

## 🔧 Building from Source

### Prerequisites

Install the following before building:

**All Platforms:**
- CMake 3.15 or later
- Git
- C++17 compatible compiler

**Platform-Specific:**

**Windows:**
```cmd
# Install Visual Studio 2019+ with C++ development tools
# Download from: https://visualstudio.microsoft.com/
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install CMake
brew install cmake
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libasound2-dev \
    libx11-dev \
    libxcomposite-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxrender-dev \
    libfreetype6-dev \
    libglu1-mesa-dev \
    mesa-common-dev \
    libcurl4-openssl-dev
```

### Build Steps

1. **Clone the repository:**
```bash
git clone https://github.com/fxgeniusllc-oss/Voice_Clone-VST.git
cd Voice_Clone-VST
```

2. **Create build directory:**
```bash
mkdir build
cd build
```

3. **Configure with CMake:**

**Windows:**
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64
```

**macOS/Linux:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

4. **Build:**

**Windows:**
```cmd
cmake --build . --config Release
```

**macOS/Linux:**
```bash
cmake --build . --config Release -j$(nproc)
```

5. **Install (optional):**

The build automatically installs to your system VST3 directory. To install manually:

**Windows:**
```cmd
copy MAEVN_artefacts\Release\VST3\MAEVN.vst3 "C:\Program Files\Common Files\VST3\"
```

**macOS:**
```bash
cp -r MAEVN_artefacts/Release/VST3/MAEVN.vst3 ~/Library/Audio/Plug-Ins/VST3/
```

**Linux:**
```bash
cp -r MAEVN_artefacts/Release/VST3/MAEVN.vst3 ~/.vst3/
```

### Build Verification

Run the test suite to verify your build:

```bash
cd build
ctest --output-on-failure
```

Expected output:
```
Test project /path/to/build
    Start 1: BuildVerificationTests
1/4 Test #1: BuildVerificationTests ...........   Passed
    Start 2: ScriptParserTests
2/4 Test #2: ScriptParserTests ................   Passed
    Start 3: ArrangementTests
3/4 Test #3: ArrangementTests .................   Passed
    Start 4: AudioEngineTests
4/4 Test #4: AudioEngineTests .................   Passed

100% tests passed, 0 tests failed out of 4
```

## 🎛️ Features Overview

### Core Features ✅
- **5 Trap Instruments**: 808 Bass, Hi-Hat, Snare, Piano, Synth
- **Script-Based Arrangement**: Define song structure with simple syntax
- **FX Chain**: Reverb, Delay, Distortion, AI FX (with ONNX)
- **Vocal Synthesis**: TTS + Vocoder (requires ONNX models)
- **Undo/Redo**: Full history management (100 levels)
- **DAW Sync**: Automatic BPM and position synchronization
- **Parameter Automation**: All parameters DAW-automatable
- **Timeline Visualization**: See your arrangement in real-time

### Instruments

1. **808 Bass** - Deep sub-bass with pitch envelope
2. **Hi-Hat** - High-frequency percussion
3. **Snare** - Punchy snare drum
4. **Piano** - Melodic piano synthesis
5. **Synth** - Lead synthesizer

### Effects

1. **Reverb** - Spacial ambience
2. **Delay** - Echo with feedback
3. **Distortion** - Harmonic saturation
4. **AI FX** - Neural audio processing (optional)

## 📋 Usage Guide

### Basic Workflow

1. **Load MAEVN** in your DAW as an instrument
2. **Enable instruments** you want to use
3. **Create MIDI track** and send notes to MAEVN
4. **Edit stage script** to define arrangement:
   ```
   [INTRO:0:8] [VERSE:8:16] [HOOK:24:16] [OUTRO:40:8]
   ```
5. **Adjust effects** to taste
6. **Automate parameters** from your DAW

### MIDI Note Mapping

**Drums:**
- 808 Bass: Any note (responds to pitch)
- Hi-Hat: F#2-A#2 (MIDI 42-46)
- Snare: D1-E1 (MIDI 38-40)

**Melodic:**
- Piano: Full keyboard range
- Synth: Full keyboard range

### Stage Script Syntax

Format: `[SECTION:START:DURATION]`

**Section Types:**
- `HOOK` - Full instrumentation + vocals
- `VERSE` - Reduced instrumentation + vocals
- `808` - Bass only
- `INTRO` - Light instrumentation, no vocals
- `OUTRO` - Piano + vocals, no drums

**Example:**
```
[INTRO:0:4] [VERSE:4:12] [HOOK:16:16] [VERSE:32:12] [HOOK:44:16] [OUTRO:60:8]
```

See [examples/ARRANGEMENTS.md](examples/ARRANGEMENTS.md) for more examples.

## 🤖 AI Features (Optional)

### ONNX Runtime Setup

MAEVN supports AI-powered features when ONNX Runtime is available:

1. **Download ONNX Runtime:**
   - https://github.com/microsoft/onnxruntime/releases
   - Get the appropriate version for your platform

2. **Rebuild with ONNX Runtime:**
```bash
cmake .. \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXRUNTIME_LIB=/path/to/onnxruntime/lib/libonnxruntime.so \
  -DCMAKE_CXX_FLAGS="-DONNXRUNTIME_AVAILABLE"
```

3. **Provide ONNX Models:**
   - Place `.onnx` model files in appropriate directories
   - See [Models/README.md](Models/) for model specifications

**Without ONNX Runtime:**
- Plugin works normally with DSP fallbacks
- AI Vocals use formant synthesis
- AI FX are bypassed

## 📦 Distribution

### For Binary Distribution

1. **Build in Release mode** (see Build Steps above)
2. **Collect artifacts:**
   - Windows: `MAEVN_artefacts/Release/VST3/MAEVN.vst3/`
   - macOS: `MAEVN_artefacts/Release/VST3/MAEVN.vst3/`
   - Linux: `MAEVN_artefacts/Release/VST3/MAEVN.vst3/`

3. **Package with documentation:**
   - README.md
   - QUICKSTART.md
   - LICENSE
   - examples/ARRANGEMENTS.md

4. **Code signing** (recommended for macOS/Windows):
   - **macOS**: `codesign --force --deep --sign "Developer ID" MAEVN.vst3`
   - **Windows**: Use signtool.exe with your certificate

### Directory Structure for Distribution

```
MAEVN-v1.0.0/
├── MAEVN.vst3/          # VST3 plugin bundle
├── README.md            # Getting started
├── QUICKSTART.md        # Quick reference
├── LICENSE              # License file
├── examples/
│   └── ARRANGEMENTS.md  # Example arrangements
└── docs/
    ├── BUILD.md         # Build instructions
    └── TESTING.md       # Testing guide
```

## ✅ Verification Checklist

Before deploying to production:

- [ ] All tests pass (`ctest` shows 100% pass rate)
- [ ] Plugin loads in at least 3 different DAWs
- [ ] All instruments produce sound correctly
- [ ] Script parser handles valid and invalid input
- [ ] Undo/redo works for all parameter changes
- [ ] DAW sync works (tempo, position)
- [ ] State save/restore works correctly
- [ ] FX chain processes audio without clicks/pops
- [ ] CPU usage is acceptable (<20% active)
- [ ] No memory leaks during extended use
- [ ] Code signing completed (for distribution)

## 🐛 Known Issues & Limitations

### Current Version (1.0.0)

1. **ONNX Runtime**
   - Not included by default
   - Users must install separately for AI features
   - DSP fallbacks work without it

2. **Platform Testing**
   - Linux: Fully verified ✅
   - macOS: Build verified, runtime testing recommended ⚠️
   - Windows: Build verified, runtime testing recommended ⚠️

3. **GUI**
   - Basic layout functional
   - Advanced theming not yet implemented

## 📞 Support & Documentation

### Documentation Files

- **README.md** - Project overview and architecture
- **QUICKSTART.md** - Quick start guide for users
- **BUILD.md** - Detailed build instructions
- **ARCHITECTURE.md** - Technical architecture details
- **TESTING.md** - Testing procedures
- **CONTRIBUTING.md** - Contribution guidelines
- **examples/ARRANGEMENTS.md** - Example arrangements

### Getting Help

1. Check documentation in this repository
2. Review [examples/ARRANGEMENTS.md](examples/ARRANGEMENTS.md)
3. Search [GitHub Issues](https://github.com/fxgeniusllc-oss/Voice_Clone-VST/issues)
4. Open a new issue with:
   - Your platform and version
   - DAW name and version
   - Steps to reproduce
   - Error messages/logs

## 🔐 Security

### Best Practices

1. **Download from official sources only**
2. **Verify checksums** for downloaded binaries
3. **Keep dependencies updated** (JUCE, ONNX Runtime)
4. **Review code** before building from source

### Reporting Security Issues

Report security vulnerabilities privately via GitHub Security Advisories or email (see CONTRIBUTING.md).

## 📄 License

MAEVN is released under the license specified in the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **JUCE Framework** - Audio plugin framework
- **ONNX Runtime** - ML inference engine
- **Open Source Community** - Contributors and testers

---

**Ready to deploy?** Follow the steps above and you'll have MAEVN running in your DAW in minutes!

For questions or issues, visit: https://github.com/fxgeniusllc-oss/Voice_Clone-VST
