# MAEVN Build Guide

This guide provides detailed instructions for building the MAEVN VST3 plugin on different platforms.

**⚠️ Important Note:** MAEVN builds as a VST3 plugin, which is not compatible with Audacity. If you need to use MAEVN with Audacity, consider using the Standalone build or a VST3-compatible DAW. See [DAW Compatibility](#daw-compatibility) below.

## Table of Contents

1. [DAW Compatibility](#daw-compatibility)
2. [Prerequisites](#prerequisites)
3. [Windows Build](#windows-build)
4. [macOS Build](#macos-build)
5. [Linux Build](#linux-build)
6. [ONNX Runtime Integration](#onnx-runtime-integration)
7. [Exporting ONNX Models](#exporting-onnx-models)
8. [Troubleshooting](#troubleshooting)

## DAW Compatibility

**Plugin Format:** VST3 and Standalone

**✅ Compatible DAWs:**
- Ableton Live 10+
- FL Studio 20+
- Reaper 5.0+
- Bitwig Studio 3.0+
- Steinberg Cubase 10.5+
- Steinberg Nuendo
- PreSonus Studio One 4+
- Tracktion Waveform

**❌ NOT Compatible:**
- **Audacity** (does not support VST3 format - only VST2/LV2/AU)
- Pro Tools (requires AAX format)
- Older DAWs without VST3 support

**Workaround for Audacity Users:**
Use the Standalone build which can run independently without a DAW.

## Prerequisites

### Required Tools

- **CMake** 3.15 or later
- **Git** (for cloning and submodules)
- **C++17 compatible compiler**

### Platform-Specific Requirements

#### Windows
- Visual Studio 2019 or later (with C++ development tools)
- Windows 10 SDK

#### macOS
- Xcode 12 or later
- Xcode Command Line Tools
- macOS 10.13 or later

#### Linux
- GCC 9+ or Clang 10+
- Development packages (see Linux Build section)

## Windows Build

### Step 1: Install Visual Studio

Download and install Visual Studio 2022 Community Edition:
https://visualstudio.microsoft.com/downloads/

Make sure to select "Desktop development with C++" workload.

### Step 2: Clone Repository

```cmd
git clone https://github.com/fxgeniusllc-oss/Voice_Clone-VST.git
cd Voice_Clone-VST
```

### Step 3: Generate Visual Studio Project

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
```

### Step 4: Build

Option A - Using Visual Studio IDE:
1. Open `build/MAEVN.sln` in Visual Studio
2. Select "Release" configuration
3. Build → Build Solution (or press Ctrl+Shift+B)

Option B - Using command line:
```cmd
cmake --build . --config Release
```

### Step 5: Install Plugin

Copy the VST3 from `build/MAEVN_artefacts/Release/VST3/MAEVN.vst3` to:
```
C:\Program Files\Common Files\VST3\
```

## macOS Build

### Step 1: Install Xcode

1. Install Xcode from the App Store
2. Install command line tools:
```bash
xcode-select --install
```

### Step 2: Install CMake

Using Homebrew:
```bash
brew install cmake
```

Or download from: https://cmake.org/download/

### Step 3: Clone Repository

```bash
git clone https://github.com/fxgeniusllc-oss/Voice_Clone-VST.git
cd Voice_Clone-VST
```

### Step 4: Build

Option A - Xcode project:
```bash
mkdir build
cd build
cmake .. -G Xcode
open MAEVN.xcodeproj
```
Then build in Xcode (Product → Build or Cmd+B)

Option B - Unix Makefiles (faster):
```bash
mkdir build
cd build
cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Step 5: Install Plugin

Copy the VST3 bundle to your plugins folder:
```bash
cp -r build/MAEVN_artefacts/Release/VST3/MAEVN.vst3 ~/Library/Audio/Plug-Ins/VST3/
```

Or system-wide:
```bash
sudo cp -r build/MAEVN_artefacts/Release/VST3/MAEVN.vst3 /Library/Audio/Plug-Ins/VST3/
```

## Linux Build

### Step 1: Install Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libasound2-dev \
    libjack-jackd2-dev \
    libcurl4-openssl-dev \
    libfreetype6-dev \
    libx11-dev \
    libxcomposite-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxrender-dev \
    libwebkit2gtk-4.0-dev \
    libglu1-mesa-dev \
    mesa-common-dev
```

#### Fedora:
```bash
sudo dnf install -y \
    cmake \
    gcc-c++ \
    git \
    alsa-lib-devel \
    jack-audio-connection-kit-devel \
    libcurl-devel \
    freetype-devel \
    libX11-devel \
    libXcomposite-devel \
    libXcursor-devel \
    libXinerama-devel \
    libXrandr-devel \
    libXrender-devel \
    webkit2gtk3-devel \
    mesa-libGL-devel
```

#### Arch Linux:
```bash
sudo pacman -S --needed \
    base-devel \
    cmake \
    git \
    alsa-lib \
    jack \
    curl \
    freetype2 \
    libx11 \
    libxcomposite \
    libxcursor \
    libxinerama \
    libxrandr \
    libxrender \
    webkit2gtk \
    mesa
```

### Step 2: Clone Repository

```bash
git clone https://github.com/fxgeniusllc-oss/Voice_Clone-VST.git
cd Voice_Clone-VST
```

### Step 3: Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Step 4: Install Plugin

```bash
# User installation
mkdir -p ~/.vst3
cp -r build/MAEVN_artefacts/Release/VST3/MAEVN.vst3 ~/.vst3/

# Or system-wide installation
sudo mkdir -p /usr/lib/vst3
sudo cp -r build/MAEVN_artefacts/Release/VST3/MAEVN.vst3 /usr/lib/vst3/
```

## ONNX Runtime Integration

To enable AI features, you need to build with ONNX Runtime support.

### Option 1: Using Pre-built ONNX Runtime

1. Download ONNX Runtime from:
   https://github.com/microsoft/onnxruntime/releases

2. Extract to a known location

3. Build with ONNX Runtime paths:

**Windows:**
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DONNXRUNTIME_INCLUDE_DIR="C:/path/to/onnxruntime/include" ^
  -DONNXRUNTIME_LIB="C:/path/to/onnxruntime/lib/onnxruntime.lib" ^
  -DCMAKE_CXX_FLAGS="/DONNXRUNTIME_AVAILABLE"
```

**macOS/Linux:**
```bash
cmake .. \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXRUNTIME_LIB=/path/to/onnxruntime/lib/libonnxruntime.so \
  -DCMAKE_CXX_FLAGS="-DONNXRUNTIME_AVAILABLE"
```

### Option 2: System-wide ONNX Runtime

If ONNX Runtime is installed system-wide, CMake will find it automatically:

```bash
cmake .. -DCMAKE_CXX_FLAGS="-DONNXRUNTIME_AVAILABLE"
cmake --build . --config Release
```

### Note on ONNX Runtime

ONNX Runtime is **optional**. Without it:
- The plugin will build and run normally
- AI features will use DSP-based fallbacks
- TTS/vocoder will use simple formant synthesis
- AI effects will be bypassed

### Exporting ONNX Models

To export the AI models to ONNX format, use the provided export scripts:

**Prerequisites:**
- Python 3.10 or later
- PyTorch (`pip install torch`)
- ONNX (`pip install onnx`)

**Export all models:**
```bash
# Linux/macOS
./build_maevn_onnx.sh

# Windows
build_maevn_onnx.bat
```

**Export specific model types:**
```bash
# Drum synthesis models (808_ddsp, hihat_ddsp, snare_ddsp)
python3 scripts/export_drum_models.py

# Instrument synthesis models (piano_ddsp, synth_fm)
python3 scripts/export_instrument_models.py

# Vocal synthesis models (vocals_tts, vocals_hifigan)
python3 scripts/export_vocal_models.py
```

**Important Notes:**
- The export scripts use **ONNX opset version 18** to support all required operators (LayerNormalization, etc.)
- Ensure your ONNX Runtime version supports opset 18 (ONNX Runtime 1.12.0+)
- Models are exported with static batch sizes optimized for inference
- Exported models are saved to `Models/drums/`, `Models/instruments/`, and `Models/vocals/` directories

**Troubleshooting Model Export:**
- If you encounter LayerNormalization errors, ensure you're using the latest version of PyTorch
- For ONNX Runtime compatibility, download version 1.12.0 or later from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases)
- To verify an exported model: `python3 -c "import onnx; onnx.checker.check_model('path/to/model.onnx')"`

## Troubleshooting

### Common Issues

#### "JUCE modules not found"

**Solution**: CMake will automatically fetch JUCE. If it fails, check your internet connection and try:
```bash
rm -rf build
mkdir build
cd build
cmake ..
```

#### "Cannot find Visual Studio"

**Solution**: Specify the generator explicitly:
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64
```

#### Linux: "Cannot find ALSA/X11/etc."

**Solution**: Install missing development packages:
```bash
# On Ubuntu/Debian
sudo apt-get install libasound2-dev libx11-dev

# On Fedora
sudo dnf install alsa-lib-devel libX11-devel
```

#### macOS: "Code signing failed"

**Solution**: This is normal for development builds. The plugin will still work in DAWs that allow unsigned plugins.

To sign the plugin:
```bash
codesign --force --deep --sign - build/MAEVN_artefacts/Release/VST3/MAEVN.vst3
```

#### "Plugin crashes on load"

**Possible causes**:
1. Missing runtime dependencies (ONNX Runtime DLLs)
2. Incompatible DAW version
3. Plugin not built in Release mode

**Solution**:
1. Ensure Release build: `cmake --build . --config Release`
2. Check DAW logs for error messages
3. Test with standalone version first

### Build Optimization

#### Faster builds on multi-core systems:

**Windows:**
```cmd
cmake --build . --config Release -- /m
```

**macOS/Linux:**
```bash
make -j$(nproc)  # Linux
make -j$(sysctl -n hw.ncpu)  # macOS
```

#### Debug builds:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
```

### Clean rebuild:

```bash
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Additional Resources

- JUCE Documentation: https://docs.juce.com/
- JUCE Forums: https://forum.juce.com/
- CMake Documentation: https://cmake.org/documentation/
- ONNX Runtime: https://onnxruntime.ai/docs/

## Getting Help

If you encounter issues not covered here:

1. Check existing GitHub issues
2. Open a new issue with:
   - Your platform and version
   - CMake output
   - Build errors
   - Steps to reproduce

GitHub Issues: https://github.com/fxgeniusllc-oss/Voice_Clone-VST/issues
