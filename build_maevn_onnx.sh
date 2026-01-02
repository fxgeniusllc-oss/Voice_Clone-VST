#!/bin/bash
# MAEVN ONNX Model Build Script
# This script exports lightweight default ONNX models

echo "========================================"
echo "  MAEVN ONNX Model Export"
echo "========================================"
echo

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.10+ and try again."
    exit 1
fi

echo "[OK] Python found: $(python3 --version)"
echo

# Check for required packages
echo "Checking for required Python packages..."
if ! python3 -c "import torch" &> /dev/null; then
    echo "[WARN] PyTorch not found. Installing..."
    pip3 install torch
fi

if ! python3 -c "import torch.onnx" &> /dev/null; then
    echo "[ERROR] ONNX not available. Please install: pip3 install onnx"
    exit 1
fi

echo "[OK] Required packages found"
echo

# Export drum models
echo "Exporting drum models..."
python3 scripts/export_drum_models.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to export drum models"
    exit 1
fi
echo

# Export instrument models
echo "Exporting instrument models..."
python3 scripts/export_instrument_models.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to export instrument models"
    exit 1
fi
echo

# Export vocal models
echo "Exporting vocal models..."
python3 scripts/export_vocal_models.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to export vocal models"
    exit 1
fi
echo

echo "========================================"
echo "Model Export Complete!"
echo "========================================"
echo
echo "Models have been exported to the Models/ directory."
echo "These are simple placeholder models for demonstration."
echo
echo "For production use, replace with properly trained models."
echo "See scripts/README.md for more information."
echo
echo "Next step: Build the plugin using CMake"
echo "See BUILD.md for build instructions."
echo
