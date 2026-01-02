@echo off
REM MAEVN ONNX Model Build Script
REM This script exports lightweight default ONNX models

echo ========================================
echo   MAEVN ONNX Model Export
echo ========================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and try again.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check for required packages
echo Checking for required Python packages...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [WARN] PyTorch not found. Installing...
    pip install torch
)

python -c "import torch.onnx" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ONNX not available. Please install: pip install onnx
    pause
    exit /b 1
)

echo [OK] Required packages found
echo.

REM Export drum models
echo Exporting drum models...
python scripts\export_drum_models.py
if errorlevel 1 (
    echo [ERROR] Failed to export drum models
    pause
    exit /b 1
)
echo.

REM Export instrument models
echo Exporting instrument models...
python scripts\export_instrument_models.py
if errorlevel 1 (
    echo [ERROR] Failed to export instrument models
    pause
    exit /b 1
)
echo.

REM Export vocal models
echo Exporting vocal models...
python scripts\export_vocal_models.py
if errorlevel 1 (
    echo [ERROR] Failed to export vocal models
    pause
    exit /b 1
)
echo.

echo ========================================
echo Model Export Complete!
echo ========================================
echo.
echo Models have been exported to the Models\ directory.
echo These are simple placeholder models for demonstration.
echo.
echo For production use, replace with properly trained models.
echo See scripts\README.md for more information.
echo.
echo Next step: Build the plugin using CMake
echo See BUILD.md for build instructions.
echo.
pause
