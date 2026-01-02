#!/bin/bash
# MAEVN Repository Setup Script
# This script creates necessary folders for the MAEVN plugin

echo "========================================"
echo "  MAEVN Repository Setup"
echo "========================================"
echo

# Create Models directory structure
echo "Creating Models directory structure..."
mkdir -p Models/drums
mkdir -p Models/instruments
mkdir -p Models/vocals

echo "[OK] Models directories created"
echo

# Create scripts directory
echo "Creating scripts directory..."
mkdir -p scripts
echo "[OK] Scripts directory created"
echo

# Check if config.json exists
if [ -f "Models/config.json" ]; then
    echo "[OK] Models/config.json already exists"
else
    echo "[WARN] Models/config.json not found - this should already exist in the repository"
fi

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Run ./build_maevn_onnx.sh to export default ONNX models"
echo "2. Build the plugin using CMake"
echo
echo "See BUILD.md for detailed build instructions."
echo
