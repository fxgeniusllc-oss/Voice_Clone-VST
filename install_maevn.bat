@echo off
REM MAEVN Installation Script for Windows
REM Installs MAEVN standalone application and VST3 plugin

setlocal enabledelayedexpansion

echo =========================================
echo   MAEVN Installation Script
echo =========================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] Running with administrator privileges.
    echo [INFO] Installing system-wide.
    set "INSTALL_SYSTEM=1"
    set "INSTALL_DIR=%ProgramFiles%\MAEVN"
) else (
    echo [INFO] Running as user. Installing to user directory.
    set "INSTALL_SYSTEM=0"
    set "INSTALL_DIR=%LOCALAPPDATA%\Programs\MAEVN"
)

echo.

REM Check if build exists
set "STANDALONE_EXEC=%SCRIPT_DIR%\build\MAEVN_artefacts\Release\Standalone\MAEVN.exe"
if not exist "%STANDALONE_EXEC%" (
    echo [ERROR] MAEVN standalone executable not found!
    echo         Expected at: %STANDALONE_EXEC%
    echo.
    echo Please build MAEVN first:
    echo   mkdir build ^&^& cd build
    echo   cmake .. -G "Visual Studio 17 2022" -A x64
    echo   cmake --build . --config Release
    echo.
    pause
    exit /b 1
)

echo [OK] Found MAEVN executable
echo.

REM Create installation directories
echo Creating installation directories...
mkdir "%INSTALL_DIR%" 2>nul
echo [OK] Directories created
echo.

REM Install standalone executable
echo Installing MAEVN standalone...
copy /Y "%STANDALONE_EXEC%" "%INSTALL_DIR%\MAEVN.exe" >nul
if %errorLevel% neq 0 (
    echo [ERROR] Failed to copy executable
    pause
    exit /b 1
)
echo [OK] Installed to %INSTALL_DIR%\MAEVN.exe
echo.

REM Install VST3 plugin
set "VST3_SOURCE=%SCRIPT_DIR%\build\MAEVN_artefacts\Release\VST3\MAEVN.vst3"
if exist "%VST3_SOURCE%" (
    echo Installing VST3 plugin...
    set "VST3_DIR=%CommonProgramFiles%\VST3"
    
    if not exist "!VST3_DIR!" mkdir "!VST3_DIR!"
    
    REM Copy VST3 bundle
    xcopy /E /I /Y "%VST3_SOURCE%" "!VST3_DIR!\MAEVN.vst3" >nul
    if !errorLevel! neq 0 (
        echo [WARN] Failed to copy VST3 plugin
    ) else (
        echo [OK] VST3 plugin installed to !VST3_DIR!\MAEVN.vst3
    )
    echo.
)

REM Create Start Menu shortcut (if user has permissions)
echo Creating Start Menu shortcut...
set "SHORTCUT_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
if %INSTALL_SYSTEM%==1 (
    set "SHORTCUT_DIR=%ProgramData%\Microsoft\Windows\Start Menu\Programs"
)

REM Use PowerShell to create shortcut
powershell -Command "$WS = New-Object -ComObject WScript.Shell; $SC = $WS.CreateShortcut('%SHORTCUT_DIR%\MAEVN.lnk'); $SC.TargetPath = '%INSTALL_DIR%\MAEVN.exe'; $SC.WorkingDirectory = '%INSTALL_DIR%'; $SC.Description = 'MAEVN AI-Powered Audio Synthesizer'; $SC.Save()" 2>nul

if %errorLevel% equ 0 (
    echo [OK] Start Menu shortcut created
) else (
    echo [WARN] Could not create Start Menu shortcut
)
echo.

REM Installation summary
echo =========================================
echo   Installation Complete!
echo =========================================
echo.
echo MAEVN has been installed to your system.
echo.
echo ✅ READY TO USE: MAEVN includes production-quality DSP synthesis
echo    All instruments (808, hi-hat, snare, piano, synth) and vocals
echo    work immediately with professional sound quality.
echo.
echo To launch MAEVN standalone:
echo   - Search for 'MAEVN' in Start Menu
echo   - Or run: %INSTALL_DIR%\MAEVN.exe
echo   - Or use the launcher: launch_maevn.bat
echo.
echo VST3 Plugin Location:
if %INSTALL_SYSTEM%==1 (
    echo   %CommonProgramFiles%\VST3\MAEVN.vst3
) else (
    echo   %CommonProgramFiles%\VST3\MAEVN.vst3
)
echo.
echo Rescan plugins in your DAW to use MAEVN as a VST3 plugin.
echo.
echo For documentation, see:
echo   - README.md - Overview and features
echo   - QUICKSTART.md - Quick start guide
echo   - BUILD.md - Build instructions
echo.
echo OPTIONAL: To add AI-enhanced synthesis (advanced users):
echo   Run: build_maevn_onnx.bat (requires Python 3.10+ and PyTorch)
echo   This is NOT required - DSP mode is production-ready
echo.
pause
