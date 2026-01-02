@echo off
REM MAEVN Repository Setup Script
REM This script creates necessary folders for the MAEVN plugin

echo ========================================
echo   MAEVN Repository Setup
echo ========================================
echo.

REM Create Models directory structure
echo Creating Models directory structure...
if not exist "Models" mkdir "Models"
if not exist "Models\drums" mkdir "Models\drums"
if not exist "Models\instruments" mkdir "Models\instruments"
if not exist "Models\vocals" mkdir "Models\vocals"

echo [OK] Models directories created
echo.

REM Create scripts directory
echo Creating scripts directory...
if not exist "scripts" mkdir "scripts"
echo [OK] Scripts directory created
echo.

REM Check if config.json exists
if exist "Models\config.json" (
    echo [OK] Models\config.json already exists
) else (
    echo [WARN] Models\config.json not found - this should already exist in the repository
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run build_maevn_onnx.bat to export default ONNX models
echo 2. Build the plugin using CMake
echo.
echo See BUILD.md for detailed build instructions.
echo.
pause
