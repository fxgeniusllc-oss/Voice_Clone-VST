# MAEVN Launcher and Installation Guide

This document explains how to use the MAEVN launcher scripts and installation utilities.

## Overview

MAEVN provides several scripts to help you build, install, and launch the application:

| Script | Platform | Purpose |
|--------|----------|---------|
| `setup_maevn_repo.sh/bat` | All | Creates Models directory structure |
| `build_maevn_onnx.sh/bat` | All | Exports ONNX models (requires Python) |
| `install_maevn.sh/bat` | All | Installs VST3 plugin and standalone app |
| `launch_maevn.sh/bat` | All | Launches MAEVN standalone application |
| `maevn.desktop` | Linux | Desktop entry for system integration |

## Quick Installation Workflow

### Full Installation Process

```bash
# 1. Setup repository
./setup_maevn_repo.sh  # Creates Models directories

# 2. Build MAEVN
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# 3. Install (VST3 + Standalone)
cd ..
./install_maevn.sh  # Installs to system

# 4. Launch
./launch_maevn.sh  # Run standalone
```

## Launcher Scripts

### launch_maevn.sh (Linux/macOS)

**Purpose:** Launches the MAEVN standalone application

**Usage:**
```bash
./launch_maevn.sh [options]
```

**Features:**
- Auto-detects platform (Linux/macOS)
- Searches multiple installation locations
- Validates executable exists before launching
- Checks for Models directory
- Passes command-line arguments to MAEVN

**Search Paths:**
- `build/MAEVN_artefacts/Release/Standalone/MAEVN` (build directory)
- `~/.local/bin/MAEVN` (user installation)
- `/usr/local/bin/MAEVN` (system installation)
- `/opt/MAEVN/MAEVN` (alternative system location)
- `/Applications/MAEVN.app/Contents/MacOS/MAEVN` (macOS)

**Example:**
```bash
# Launch with default settings
./launch_maevn.sh

# Launch with custom MIDI device (if supported)
./launch_maevn.sh --midi-device=hw:1,0
```

### launch_maevn.bat (Windows)

**Purpose:** Launches the MAEVN standalone application on Windows

**Usage:**
```cmd
launch_maevn.bat [options]
```

**Features:**
- Searches multiple installation locations
- Validates executable exists before launching
- Checks for Models directory
- Passes command-line arguments to MAEVN

**Search Paths:**
- `build\MAEVN_artefacts\Release\Standalone\MAEVN.exe` (build directory)
- `%LOCALAPPDATA%\Programs\MAEVN\MAEVN.exe` (user installation)
- `%ProgramFiles%\MAEVN\MAEVN.exe` (system installation)

**Example:**
```cmd
REM Launch MAEVN
launch_maevn.bat
```

## Installation Scripts

### install_maevn.sh (Linux/macOS)

**Purpose:** Installs MAEVN VST3 plugin and standalone application

**Usage:**
```bash
# User installation (recommended)
./install_maevn.sh

# System-wide installation (requires sudo)
sudo ./install_maevn.sh
```

**What it installs:**

**User Installation:**
- Standalone: `~/.local/bin/MAEVN`
- VST3 Plugin: `~/.vst3/MAEVN.vst3` (Linux) or `~/Library/Audio/Plug-Ins/VST3/MAEVN.vst3` (macOS)
- Desktop Entry: `~/.local/share/applications/maevn.desktop` (Linux only)

**System Installation (with sudo):**
- Standalone: `/usr/local/bin/MAEVN` (Linux) or `/Applications/MAEVN.app` (macOS)
- VST3 Plugin: `/usr/lib/vst3/MAEVN.vst3` (Linux) or `/Library/Audio/Plug-Ins/VST3/MAEVN.vst3` (macOS)
- Desktop Entry: `/usr/share/applications/maevn.desktop` (Linux only)

**Features:**
- Detects if running as root (system install) or user
- Creates necessary directories automatically
- Copies built artifacts to appropriate locations
- Sets executable permissions
- Installs desktop entry file (Linux)
- Updates desktop database (Linux, if available)

### install_maevn.bat (Windows)

**Purpose:** Installs MAEVN VST3 plugin and standalone application on Windows

**Usage:**
```cmd
REM User installation
install_maevn.bat

REM System-wide installation (run as Administrator)
REM Right-click install_maevn.bat -> Run as Administrator
install_maevn.bat
```

**What it installs:**

**User Installation:**
- Standalone: `%LOCALAPPDATA%\Programs\MAEVN\MAEVN.exe`
- VST3 Plugin: `%CommonProgramFiles%\VST3\MAEVN.vst3`
- Start Menu Shortcut: `%APPDATA%\Microsoft\Windows\Start Menu\Programs\MAEVN.lnk`

**System Installation (as Administrator):**
- Standalone: `%ProgramFiles%\MAEVN\MAEVN.exe`
- VST3 Plugin: `%CommonProgramFiles%\VST3\MAEVN.vst3`
- Start Menu Shortcut: `%ProgramData%\Microsoft\Windows\Start Menu\Programs\MAEVN.lnk`

**Features:**
- Detects administrator privileges
- Creates installation directories
- Copies built artifacts
- Creates Start Menu shortcuts using PowerShell

## Desktop Integration

### maevn.desktop (Linux)

**Purpose:** Desktop entry file for Linux application menus

**Location (after installation):**
- User: `~/.local/share/applications/maevn.desktop`
- System: `/usr/share/applications/maevn.desktop`

**Features:**
- Adds MAEVN to application menu
- Provides searchable entry in desktop environments (GNOME, KDE, etc.)
- Associates with AudioVideo and MIDI categories
- Includes keywords for better searchability

**Manual Installation:**
```bash
# Copy to applications directory
cp maevn.desktop ~/.local/share/applications/

# Update the Exec path to match your installation
sed -i 's|Exec=.*|Exec=/home/YOUR_USER/.local/bin/MAEVN|g' ~/.local/share/applications/maevn.desktop

# Make it executable
chmod +x ~/.local/share/applications/maevn.desktop

# Update desktop database (optional)
update-desktop-database ~/.local/share/applications/
```

## Troubleshooting

### Launcher Script Issues

**Problem:** "MAEVN executable not found"

**Solutions:**
1. Verify MAEVN was built successfully:
   ```bash
   ls -la build/MAEVN_artefacts/Release/Standalone/MAEVN
   ```

2. Build MAEVN if not already built:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ```

3. Run installation script:
   ```bash
   ./install_maevn.sh
   ```

**Problem:** "Permission denied"

**Solution:**
```bash
chmod +x launch_maevn.sh install_maevn.sh
```

### Installation Issues

**Problem:** Installation fails on Linux/macOS

**Solution:**
Check that you have write permissions:
```bash
# For user installation, ensure ~/.local/bin exists
mkdir -p ~/.local/bin

# For system installation, use sudo
sudo ./install_maevn.sh
```

**Problem:** Installation fails on Windows

**Solution:**
- For user installation: Run normally
- For system installation: Right-click and "Run as Administrator"

**Problem:** Desktop entry doesn't appear (Linux)

**Solutions:**
1. Manually update desktop database:
   ```bash
   update-desktop-database ~/.local/share/applications/
   ```

2. Log out and log back in

3. Restart your desktop environment

### Runtime Issues

**Problem:** ALSA/Audio device errors on Linux

**Solution:**
This is normal if no audio device is available. The standalone will still launch. To fix:
```bash
# Install ALSA/JACK
sudo apt-get install libasound2 jackd2
```

**Problem:** VST3 not detected by DAW

**Solutions:**
1. Ensure plugin is in correct location:
   - Windows: `C:\Program Files\Common Files\VST3\`
   - macOS: `~/Library/Audio/Plug-Ins/VST3/`
   - Linux: `~/.vst3/`

2. Rescan plugins in your DAW

3. Check DAW supports VST3 format

## Advanced Usage

### Custom Installation Paths

You can modify the installation scripts to use custom paths:

**Linux/macOS:**
Edit `install_maevn.sh` and modify:
```bash
INSTALL_DIR="/your/custom/path"
```

**Windows:**
Edit `install_maevn.bat` and modify:
```batch
set "INSTALL_DIR=C:\Your\Custom\Path"
```

### Environment Variables

The launcher scripts respect these environment variables:

**MAEVN_MODELS_DIR:**
Override the Models directory location:
```bash
export MAEVN_MODELS_DIR=/path/to/custom/models
./launch_maevn.sh
```

## Best Practices

1. **Always build in Release mode** for optimal performance
2. **Use the installation script** for automatic setup
3. **Keep Models directory** in the same directory as the launcher scripts
4. **Create symbolic links** for easy access if needed:
   ```bash
   ln -s ~/.local/bin/MAEVN ~/Desktop/MAEVN
   ```

## Distribution

When distributing MAEVN, include these files:

```
MAEVN-Distribution/
├── MAEVN.vst3/              # VST3 plugin
├── MAEVN or MAEVN.exe       # Standalone executable
├── launch_maevn.sh          # Linux/macOS launcher
├── launch_maevn.bat         # Windows launcher
├── install_maevn.sh         # Linux/macOS installer
├── install_maevn.bat        # Windows installer
├── maevn.desktop            # Linux desktop entry
├── README.md                # Documentation
├── QUICKSTART.md            # Quick start guide
├── LICENSE                  # License file
└── Models/                  # ONNX models directory
    ├── config.json
    └── ... (model files)
```

## Support

For additional help:
- See [BUILD.md](BUILD.md) for build instructions
- See [QUICKSTART.md](QUICKSTART.md) for usage guide
- Report issues: https://github.com/fxgeniusllc-oss/Voice_Clone-VST/issues
