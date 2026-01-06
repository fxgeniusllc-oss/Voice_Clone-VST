# MAEVN Google Colab Guide

This guide explains how to build, run, and use MAEVN (AI-Powered Vocal + Instrument Generator) using Google Colab.

## 🎯 Overview

The MAEVN Google Colab notebook allows you to:
- Build MAEVN from source in the cloud
- Export ONNX models for AI features
- Download the VST3 plugin for local use
- Save/load models from Google Drive across sessions

## 🚀 Quick Start

### Option 1: Direct Link (Recommended)

1. Open the notebook in Google Colab: 
   - Upload `colab_maevn.ipynb` to your Google Drive
   - Right-click → Open with → Google Colaboratory
   - Or use the direct link if available

2. Run all cells in order (Runtime → Run all)

3. Mount Google Drive when prompted (optional but recommended)

4. Wait for build to complete (~10-15 minutes)

5. Download the package when ready

### Option 2: Step-by-Step

1. **Open Google Colab**
   - Go to https://colab.research.google.com
   - Click File → Upload notebook
   - Select `colab_maevn.ipynb`

2. **Follow the notebook sections:**
   - Step 1: Install dependencies
   - Step 2: Mount Google Drive (optional)
   - Step 3: Clone repository
   - Step 4-6: Setup and export models
   - Step 7-8: Build MAEVN
   - Step 9-10: Verify and install
   - Step 11-12: Package and download

## 📋 Prerequisites

### Required
- Google Account (for Colab access)
- Web browser with JavaScript enabled
- ~30 minutes for first-time setup

### Optional
- Google Drive account (for model persistence)
- ~2GB free space in Google Drive

### Local Machine Requirements
For using the downloaded plugin:
- **Windows:** Windows 10+, VST3-compatible DAW
- **macOS:** macOS 10.13+, VST3-compatible DAW
- **Linux:** Recent distribution, VST3-compatible DAW

## 📖 Detailed Steps

### Step 1: Environment Setup

The notebook installs all required dependencies:
- CMake 3.15+
- Build tools (GCC, G++)
- JUCE dependencies (ALSA, X11, Freetype, etc.)
- Audio libraries

**Expected time:** 2-3 minutes

**Common issues:**
- If installation fails, restart runtime and retry
- Check that Colab session is active

### Step 2: Google Drive Integration

Mounting Google Drive enables:
- Persistent model storage across sessions
- Backup of build artifacts
- Faster subsequent builds (reuse models)

**To mount:**
1. Run the cell
2. Click the authentication link
3. Grant permission to access Drive
4. Copy the authorization code
5. Paste into the input field

**Optional:** You can skip this step, but you'll need to export models each session.

### Step 3: Clone Repository

Downloads the latest MAEVN source code from GitHub.

**What happens:**
- Clones https://github.com/fxgeniusllc-oss/Voice_Clone-VST
- Places code in `/content/Voice_Clone-VST`

**Expected time:** 30 seconds

### Step 4: Repository Setup

Creates directory structure for models and configuration.

**Creates:**
- `Models/drums/`
- `Models/instruments/`
- `Models/vocals/`

**Expected time:** 5 seconds

### Step 5: Restore Models (Optional)

If you've previously built MAEVN and backed up models to Google Drive, this step restores them.

**Skips model export if:**
- Models exist in `/content/drive/MyDrive/MAEVN/Models`
- All model types are present

**Expected time:** 10-30 seconds

### Step 6: Export ONNX Models

Generates default AI models for:
- Drum synthesis
- Instrument synthesis
- Vocal synthesis

**Installs:**
- PyTorch (CPU version)
- ONNX
- NumPy

**Expected time:** 3-5 minutes (first time), 1 minute (subsequent)

**Note:** These are placeholder models for demonstration. For production, use trained models.

### Step 7: Backup Models (Optional)

Saves exported models to Google Drive for reuse.

**Saves to:** `/content/drive/MyDrive/MAEVN/Models/`

**Includes:**
- All `.onnx` files
- `config.json`
- `metadata.json`

**Expected time:** 30 seconds

### Step 8: Build MAEVN

Compiles the plugin using CMake.

**Two sub-steps:**
1. **CMake Configure** - Sets up build system
2. **Build** - Compiles source code

**Expected time:** 10-15 minutes

**Progress indicators:**
- CMake fetches JUCE framework
- Compiles ~50+ source files
- Links VST3 and Standalone targets

**Note:** This is the longest step. Be patient!

### Step 9: Verify Build

Checks that compilation succeeded.

**Verifies:**
- Standalone executable exists
- VST3 plugin bundle exists
- Files have correct permissions

**Expected output:**
```
✓ Standalone executable found
  Location: /content/Voice_Clone-VST/build/MAEVN_artefacts/Release/Standalone/MAEVN
✓ VST3 plugin found
  Location: /content/Voice_Clone-VST/build/MAEVN_artefacts/Release/VST3/MAEVN.vst3
```

### Step 10: Install MAEVN

Installs to user directories in the Colab environment.

**Installs to:**
- `~/.local/bin/MAEVN` (standalone)
- `~/.vst3/MAEVN.vst3` (plugin)

**Expected time:** 10 seconds

**Note:** This is for testing in Colab. You'll download for local use.

### Step 11: Launch Test

Tests that the executable is valid (GUI won't display in Colab).

**Checks:**
- Executable exists
- Has correct permissions
- Is a valid ELF binary

**Expected time:** 5 seconds

### Step 12: Download Artifacts

Packages everything for download.

**Package includes:**
- `MAEVN.vst3` - VST3 plugin
- `MAEVN` - Standalone executable (Linux)
- `Models/` - ONNX models
- Documentation (README, QUICKSTART, BUILD)

**Creates:** `MAEVN_Package.tar.gz`

**Expected size:** 50-200 MB (depending on models)

**Expected time:** 30 seconds to create, varies for download

### Step 13: Backup Package (Optional)

Saves the complete package to Google Drive.

**Saves to:** `/content/drive/MyDrive/MAEVN/MAEVN_Package.tar.gz`

**Why backup:**
- Direct download from Drive anytime
- Faster than re-downloading from Colab
- Persistent across Colab sessions

**Expected time:** 1-2 minutes

## 💾 Using Downloaded Package

### Extraction

**Linux/macOS:**
```bash
tar -xzf MAEVN_Package.tar.gz
cd MAEVN_Package
```

**Windows:**
- Use 7-Zip, WinRAR, or built-in extraction
- Extract to a folder of your choice

### Installation

#### VST3 Plugin

**Windows:**
```cmd
copy MAEVN.vst3 "C:\Program Files\Common Files\VST3\"
```

**macOS:**
```bash
cp -r MAEVN.vst3 ~/Library/Audio/Plug-Ins/VST3/
```

**Linux:**
```bash
mkdir -p ~/.vst3
cp -r MAEVN.vst3 ~/.vst3/
```

#### Standalone (Linux only)

```bash
chmod +x MAEVN
./MAEVN
```

**Note:** Standalone builds for Windows/macOS require building on those platforms.

### DAW Setup

1. **Rescan Plugins**
   - Open your DAW
   - Go to plugin settings
   - Rescan VST3 plugins

2. **Load MAEVN**
   - Create new instrument track
   - Load MAEVN as VST3 instrument
   - Start making music!

### Compatible DAWs

✅ **Supported:**
- Ableton Live 10+
- FL Studio 20+
- Reaper 5.0+
- Bitwig Studio 3.0+
- Steinberg Cubase 10.5+
- Steinberg Nuendo
- PreSonus Studio One 4+
- Tracktion Waveform

❌ **Not Supported:**
- Audacity (no VST3 support - use standalone instead)
- Pro Tools (requires AAX format)
- Older DAWs without VST3

## 🔄 Workflow Tips

### First Time Setup (Full Build)

1. Run all cells sequentially
2. Mount Google Drive
3. Wait for build (~15-20 minutes total)
4. Backup models and package to Drive
5. Download package

**Total time:** ~20-25 minutes

### Subsequent Builds (With Drive Backup)

1. Mount Google Drive
2. Restore models from Drive (skip export)
3. Build plugin (~10-15 minutes)
4. Download package

**Total time:** ~12-18 minutes

**Time saved:** 5-7 minutes (skip model export)

### Quick Rebuild (Code Changes Only)

If you only changed code (not models):

1. Skip model export/restore steps
2. Delete build directory
3. Rebuild
4. Download

**Total time:** ~10-12 minutes

## 🐛 Troubleshooting

### Build Fails at CMake Configure

**Symptoms:**
- CMake can't find JUCE
- Missing dependencies error

**Solutions:**
1. Re-run "Environment Setup" cell
2. Restart Colab runtime
3. Check CMake version: `cmake --version` (need 3.15+)

### Build Fails at Compilation

**Symptoms:**
- Compiler errors
- Linker errors

**Solutions:**
1. Check available RAM: `free -h`
2. Close other notebooks
3. Restart runtime and retry
4. Use Colab Pro for more resources

### Model Export Fails

**Symptoms:**
- Python import errors
- PyTorch not found

**Solutions:**
1. Re-run "Install Python dependencies" cell
2. Verify PyTorch: `python3 -c "import torch; print(torch.__version__)"`
3. Clear pip cache: `pip cache purge`

### Google Drive Mount Fails

**Symptoms:**
- Permission denied
- Drive not accessible

**Solutions:**
1. Revoke permissions: Google Account → Security → Third-party apps
2. Re-run mount cell
3. Grant permissions again
4. Check Drive storage space

### Download Fails or Hangs

**Symptoms:**
- Browser times out
- File corrupt

**Solutions:**
1. Use Google Drive backup instead
2. Split package (download VST3 and models separately)
3. Try different browser
4. Check internet connection

### Package Too Large

**Symptoms:**
- Download extremely slow
- Package >500MB

**Solutions:**
1. Remove unnecessary model files
2. Download VST3 only (models separate)
3. Use Google Drive for large files
4. Compress models further

### Out of Memory During Build

**Symptoms:**
- "Killed" message during compilation
- RAM usage 100%

**Solutions:**
1. Reduce parallel jobs: `cmake --build . -j1`
2. Close all other notebooks
3. Restart runtime
4. Upgrade to Colab Pro (more RAM)

### Runtime Disconnects During Build

**Symptoms:**
- "Reconnecting..." message
- Lost progress

**Solutions:**
1. Keep browser tab active
2. Disable browser extensions that may interfere
3. Use stable internet connection
4. Run build in smaller steps (configure, then build)
5. Consider Colab Pro (longer runtime)

## 🔐 Security & Privacy

### What Data is Stored?

**In Colab Session:**
- Source code (public repository)
- Build artifacts (temporary)
- ONNX models (generated)

**In Google Drive (optional):**
- ONNX models
- Build package
- No personal data or credentials

### Safe Practices

✅ **Do:**
- Review notebook code before running
- Use Google Drive backup for models
- Download package to secure location
- Clear sensitive data after session

❌ **Don't:**
- Add API keys or credentials to notebook
- Share Drive folder with untrusted users
- Run modified notebooks from unknown sources

## 📊 Resource Usage

### Colab Free Tier

- **RAM:** ~12-13 GB available
- **Disk:** ~100 GB available
- **GPU:** Not used for this build
- **Runtime:** 12 hours max, may disconnect if idle

**MAEVN Requirements:**
- **RAM:** ~2-4 GB during build
- **Disk:** ~5-10 GB for build artifacts
- **Time:** 15-25 minutes

**Verdict:** Colab Free is sufficient for MAEVN builds.

### Colab Pro Benefits

- More RAM (25-50 GB)
- Longer runtime (24 hours)
- Faster CPUs
- Priority access

**Needed for MAEVN?** No, but helpful if:
- Building many times
- Running other notebooks simultaneously
- Need guaranteed uptime

## 🌐 Limitations

### What Works in Colab

✅ Building MAEVN from source
✅ Exporting ONNX models
✅ Packaging for download
✅ Testing build validity
✅ Google Drive backup/restore

### What Doesn't Work in Colab

❌ Running MAEVN GUI (no X11 display)
❌ Audio playback (no audio output)
❌ DAW integration (no DAW in Colab)
❌ Real-time audio processing

### Workarounds

For actual usage:
1. Download the package
2. Install on local machine
3. Use in your DAW

For testing builds:
- Verify executable exists
- Check file size is reasonable
- Test on local machine

## 🎓 Advanced Usage

### Custom Models

To use your own trained models:

1. Upload models to Google Drive
2. Modify restore cell to use your models
3. Update `Models/config.json` if needed
4. Build and package

### Building Specific Versions

To build a specific MAEVN version:

1. Modify clone cell:
   ```bash
   git clone --branch v1.2.3 https://github.com/fxgeniusllc-oss/Voice_Clone-VST.git
   ```

2. Continue with normal build process

### Parallel Builds

To speed up builds (if you have Colab Pro):

```bash
cmake --build . --config Release -j$(nproc)
```

### Custom CMake Options

To enable specific features:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_AVAILABLE=ON \
  -DCUSTOM_OPTION=value
```

## 📚 Additional Resources

- **Main README:** See `README.md` in package
- **Quick Start:** See `QUICKSTART.md` in package
- **Build Guide:** See `BUILD.md` in package
- **GitHub:** https://github.com/fxgeniusllc-oss/Voice_Clone-VST
- **Issues:** https://github.com/fxgeniusllc-oss/Voice_Clone-VST/issues

## 🤝 Contributing

Found an issue with the Colab notebook?

1. Check existing issues
2. Open new issue with:
   - Colab session details
   - Error messages
   - Steps to reproduce
3. Tag with "colab" label

## 📜 License

The Colab notebook is part of the MAEVN project and follows the same MIT License.

---

**Happy Building! 🎵**

If you find this useful, please star the repository on GitHub!
