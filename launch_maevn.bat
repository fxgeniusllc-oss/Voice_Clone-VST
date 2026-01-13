@echo off
REM MAEVN Standalone Launcher for Windows
REM Launches the MAEVN standalone application

setlocal enabledelayedexpansion

echo =========================================
echo   MAEVN Standalone Launcher
echo =========================================
echo.
echo MAEVN - AI-Powered Audio Synthesis
echo Production-quality sounds with DSP and optional AI enhancement
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo Platform: Windows
echo.

REM Define possible executable paths
set "PATHS[0]=%SCRIPT_DIR%\build\MAEVN_artefacts\Release\Standalone\MAEVN.exe"
set "PATHS[1]=%SCRIPT_DIR%\build\MAEVN_artefacts\Debug\Standalone\MAEVN.exe"
set "PATHS[2]=%LOCALAPPDATA%\Programs\MAEVN\MAEVN.exe"
set "PATHS[3]=%ProgramFiles%\MAEVN\MAEVN.exe"
set "PATHS[4]=%ProgramFiles(x86)%\MAEVN\MAEVN.exe"

REM Find the executable
set "MAEVN_EXEC="
for /L %%i in (0,1,4) do (
    if exist "!PATHS[%%i]!" (
        set "MAEVN_EXEC=!PATHS[%%i]!"
        goto :found
    )
)

:notfound
echo [ERROR] MAEVN standalone executable not found!
echo.
echo Please build MAEVN first or install it to one of these locations:
for /L %%i in (0,1,4) do (
    echo   - !PATHS[%%i]!
)
echo.
echo To build MAEVN:
echo   mkdir build ^&^& cd build
echo   cmake .. -G "Visual Studio 17 2022" -A x64
echo   cmake --build . --config Release
echo.
pause
exit /b 1

:found
echo [OK] Found MAEVN executable: %MAEVN_EXEC%
echo.

REM Check for Models directory
if not exist "%SCRIPT_DIR%\Models" (
    echo [INFO] Models directory not found at: %SCRIPT_DIR%\Models
    echo        Running setup_maevn_repo.bat to create it...
    echo.
    call "%SCRIPT_DIR%\setup_maevn_repo.bat"
)

REM Check if ONNX models exist
set ONNX_COUNT=0
for /r "%SCRIPT_DIR%\Models" %%f in (*.onnx) do (
    set /a ONNX_COUNT+=1
)

if %ONNX_COUNT% EQU 0 (
    echo [INFO] No ONNX AI models found - using production-quality DSP synthesis
    echo        This is normal and provides excellent sound quality.
    echo        ONNX models are optional enhancements.
    echo.
) else (
    echo [OK] Found %ONNX_COUNT% ONNX AI model(s) - AI-enhanced synthesis available
    echo.
)

REM Launch MAEVN
echo Launching MAEVN...
echo =========================================
echo.
echo First time using MAEVN? See FIRST_USE.md for a quick guide.
echo.

REM Change to script directory so relative paths work
cd /d "%SCRIPT_DIR%"

REM Launch the application
start "" "%MAEVN_EXEC%" %*
