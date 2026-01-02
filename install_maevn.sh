#!/bin/bash
#
# MAEVN Installation Script
# Installs MAEVN standalone application and creates desktop launcher
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "  MAEVN Installation Script"
echo "========================================="
echo

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "[WARN] Running as root. Installing system-wide."
    INSTALL_SYSTEM=1
    INSTALL_DIR="/usr/local/bin"
    DESKTOP_DIR="/usr/share/applications"
else
    echo "[INFO] Running as user. Installing to user directory."
    INSTALL_SYSTEM=0
    INSTALL_DIR="${HOME}/.local/bin"
    DESKTOP_DIR="${HOME}/.local/share/applications"
fi

echo

# Check if build exists
STANDALONE_EXEC="${SCRIPT_DIR}/build/MAEVN_artefacts/Release/Standalone/MAEVN"
if [ ! -f "${STANDALONE_EXEC}" ]; then
    echo "[ERROR] MAEVN standalone executable not found!"
    echo "        Expected at: ${STANDALONE_EXEC}"
    echo
    echo "Please build MAEVN first:"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build . --config Release"
    echo
    exit 1
fi

echo "[OK] Found MAEVN executable"

# Detect platform
PLATFORM="$(uname -s)"
case "${PLATFORM}" in
    Linux*)
        echo "[OK] Platform: Linux"
        ;;
    Darwin*)
        echo "[OK] Platform: macOS"
        echo "[INFO] On macOS, standalone apps are typically in Applications folder"
        INSTALL_DIR="/Applications"
        ;;
    *)
        echo "[ERROR] Unsupported platform: ${PLATFORM}"
        exit 1
        ;;
esac

echo

# Create installation directories
echo "Creating installation directories..."
mkdir -p "${INSTALL_DIR}"
if [ "${PLATFORM}" = "Linux" ]; then
    mkdir -p "${DESKTOP_DIR}"
fi
echo "[OK] Directories created"
echo

# Install standalone executable
echo "Installing MAEVN standalone..."
if [ "${PLATFORM}" = "Darwin" ]; then
    # macOS - copy the app bundle if it exists, otherwise copy the executable
    if [ -d "${SCRIPT_DIR}/build/MAEVN_artefacts/Release/Standalone/MAEVN.app" ]; then
        cp -r "${SCRIPT_DIR}/build/MAEVN_artefacts/Release/Standalone/MAEVN.app" "${INSTALL_DIR}/"
        echo "[OK] Installed to ${INSTALL_DIR}/MAEVN.app"
    else
        cp "${STANDALONE_EXEC}" "${INSTALL_DIR}/MAEVN"
        chmod +x "${INSTALL_DIR}/MAEVN"
        echo "[OK] Installed to ${INSTALL_DIR}/MAEVN"
    fi
else
    # Linux - copy executable
    cp "${STANDALONE_EXEC}" "${INSTALL_DIR}/MAEVN"
    chmod +x "${INSTALL_DIR}/MAEVN"
    echo "[OK] Installed to ${INSTALL_DIR}/MAEVN"
fi

echo

# Install VST3 plugin
VST3_SOURCE="${SCRIPT_DIR}/build/MAEVN_artefacts/Release/VST3/MAEVN.vst3"
if [ -d "${VST3_SOURCE}" ]; then
    echo "Installing VST3 plugin..."
    if [ "${PLATFORM}" = "Darwin" ]; then
        VST3_DIR="${HOME}/Library/Audio/Plug-Ins/VST3"
        if [ ${INSTALL_SYSTEM} -eq 1 ]; then
            VST3_DIR="/Library/Audio/Plug-Ins/VST3"
        fi
    else
        VST3_DIR="${HOME}/.vst3"
        if [ ${INSTALL_SYSTEM} -eq 1 ]; then
            VST3_DIR="/usr/lib/vst3"
        fi
    fi
    mkdir -p "${VST3_DIR}"
    cp -r "${VST3_SOURCE}" "${VST3_DIR}/"
    echo "[OK] VST3 plugin installed to ${VST3_DIR}/MAEVN.vst3"
    echo
fi

# Install desktop entry (Linux only)
if [ "${PLATFORM}" = "Linux" ]; then
    echo "Installing desktop entry..."
    DESKTOP_FILE="${SCRIPT_DIR}/maevn.desktop"
    if [ -f "${DESKTOP_FILE}" ]; then
        # Update Exec path in desktop file
        sed "s|Exec=.*|Exec=${INSTALL_DIR}/MAEVN|g" "${DESKTOP_FILE}" > "${DESKTOP_DIR}/maevn.desktop"
        chmod +x "${DESKTOP_DIR}/maevn.desktop"
        echo "[OK] Desktop entry installed to ${DESKTOP_DIR}/maevn.desktop"
        
        # Update desktop database if possible
        if command -v update-desktop-database &> /dev/null; then
            update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true
            echo "[OK] Desktop database updated"
        fi
    else
        echo "[WARN] Desktop entry file not found: ${DESKTOP_FILE}"
    fi
    echo
fi

# Installation summary
echo "========================================="
echo "  Installation Complete!"
echo "========================================="
echo
echo "MAEVN has been installed to your system."
echo
echo "To launch MAEVN standalone:"
if [ "${PLATFORM}" = "Darwin" ]; then
    echo "  - Open from Applications folder"
    echo "  - Or run: ${INSTALL_DIR}/MAEVN"
elif [ "${PLATFORM}" = "Linux" ]; then
    echo "  - Search for 'MAEVN' in your application menu"
    echo "  - Or run: ${INSTALL_DIR}/MAEVN"
    echo "  - Or use the launcher: ./launch_maevn.sh"
fi
echo
echo "VST3 Plugin Location:"
if [ "${PLATFORM}" = "Darwin" ]; then
    if [ ${INSTALL_SYSTEM} -eq 1 ]; then
        echo "  /Library/Audio/Plug-Ins/VST3/MAEVN.vst3"
    else
        echo "  ~/Library/Audio/Plug-Ins/VST3/MAEVN.vst3"
    fi
else
    if [ ${INSTALL_SYSTEM} -eq 1 ]; then
        echo "  /usr/lib/vst3/MAEVN.vst3"
    else
        echo "  ~/.vst3/MAEVN.vst3"
    fi
fi
echo
echo "Rescan plugins in your DAW to use MAEVN as a VST3 plugin."
echo
echo "For documentation, see:"
echo "  - README.md - Overview and features"
echo "  - QUICKSTART.md - Quick start guide"
echo "  - BUILD.md - Build instructions"
echo
