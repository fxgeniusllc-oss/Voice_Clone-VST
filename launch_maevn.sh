#!/bin/bash
#
# MAEVN Standalone Launcher
# Launches the MAEVN standalone application
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "  MAEVN Standalone Launcher"
echo "========================================="
echo

# Detect platform
PLATFORM="$(uname -s)"
case "${PLATFORM}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${PLATFORM}"
esac

echo "Platform: ${MACHINE}"
echo

# Define possible executable paths
STANDALONE_PATHS=(
    # Built executable in build directory
    "${SCRIPT_DIR}/build/MAEVN_artefacts/Release/Standalone/MAEVN"
    # Installed to user's local bin
    "${HOME}/.local/bin/MAEVN"
    # System-wide installation
    "/usr/local/bin/MAEVN"
    "/opt/MAEVN/MAEVN"
    # macOS application bundle
    "${SCRIPT_DIR}/build/MAEVN_artefacts/Release/Standalone/MAEVN.app/Contents/MacOS/MAEVN"
    "/Applications/MAEVN.app/Contents/MacOS/MAEVN"
    "${HOME}/Applications/MAEVN.app/Contents/MacOS/MAEVN"
)

# Function to find executable
find_executable() {
    for path in "${STANDALONE_PATHS[@]}"; do
        if [ -f "${path}" ] && [ -x "${path}" ]; then
            echo "${path}"
            return 0
        fi
    done
    return 1
}

# Find the executable
MAEVN_EXEC=$(find_executable)

if [ -z "${MAEVN_EXEC}" ]; then
    echo "[ERROR] MAEVN standalone executable not found!"
    echo
    echo "Please build MAEVN first or install it to one of these locations:"
    for path in "${STANDALONE_PATHS[@]}"; do
        echo "  - ${path}"
    done
    echo
    echo "To build MAEVN:"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build . --config Release"
    echo
    exit 1
fi

echo "[OK] Found MAEVN executable: ${MAEVN_EXEC}"
echo

# Check for Models directory
MODELS_DIR="${SCRIPT_DIR}/Models"
if [ ! -d "${MODELS_DIR}" ]; then
    echo "[WARN] Models directory not found at: ${MODELS_DIR}"
    echo "       AI features may not work. Run ./setup_maevn_repo.sh to create it."
    echo
fi

# Launch MAEVN
echo "Launching MAEVN..."
echo "========================================="
echo

# Change to script directory so relative paths work
cd "${SCRIPT_DIR}"

# Launch the application
exec "${MAEVN_EXEC}" "$@"
