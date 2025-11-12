# MAEVN Testing Guide

This document describes the testing infrastructure for the MAEVN VST3 plugin project.

## Overview

The MAEVN project includes a comprehensive test suite to ensure production readiness and code quality. Tests cover:

- Build verification and environment configuration
- Core component functionality (parsers, audio engines, effects)
- Integration between components
- System compatibility

## Running Tests

### Prerequisites

- CMake 3.15 or later
- C++17 compiler
- JUCE dependencies (handled automatically by CMake)

### Build with Tests

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build . --config Release
```

### Run Tests

```bash
cd build
ctest
```

For verbose output:

```bash
ctest -V
```

To run specific tests:

```bash
ctest -R ScriptParserTests
ctest -R AudioEngineTests
```

## Test Suite

### 1. Build Verification Tests (`BuildVerificationTests`)

Verifies that the build environment is correctly configured.

**Tests:**
- JUCE version detection
- Basic JUCE functionality (String, Array, File)
- Audio buffer operations
- System information (OS, CPU, memory)

**Purpose:** Ensures the development environment meets minimum requirements.

### 2. Script Parser Tests (`ScriptParserTests`)

Tests the stage script parser that handles arrangement notation.

**Tests:**
- Basic parsing: `[INTRO:0:8] [VERSE:8:16] [HOOK:24:16]`
- Empty script handling
- Invalid script rejection
- Complex arrangements with 7+ sections

**Purpose:** Validates musical arrangement parsing logic.

### 3. Arrangement Tests (`ArrangementTests`)

Tests the arrangement system that tracks playback position and active sections.

**Tests:**
- Position tracking across sections
- Section boundary handling
- Empty arrangement behavior
- Default arrangement functionality

**Purpose:** Ensures correct timeline-based arrangement switching.

### 4. Audio Engine Tests (`AudioEngineTests`)

Tests the core audio processing engine.

**Tests:**
- Engine initialization and preparation
- Audio buffer processing (without crashing)
- Resource cleanup and release

**Purpose:** Validates audio engine stability and correctness.

## Test Results

All tests must pass before a production release. Current status:

```
100% tests passed, 0 tests failed out of 4
```

## Continuous Integration

Tests are automatically run on every commit through GitHub Actions (when configured). The CI pipeline:

1. Sets up the build environment
2. Installs dependencies
3. Compiles the project with tests
4. Runs all tests
5. Reports results

## Adding New Tests

To add a new test:

1. Create a test file in `Tests/` directory (e.g., `MyNewTests.cpp`)
2. Include necessary headers and write test functions
3. Add to `Tests/CMakeLists.txt`:

```cmake
add_maevn_test(MyNewTests MyNewTests.cpp)
target_sources(MyNewTests PRIVATE
    ${CMAKE_SOURCE_DIR}/Source/ComponentToTest.cpp
    # Add other dependencies
)
```

4. Rebuild and run tests

### Test Template

```cpp
#include <juce_core/juce_core.h>
#include <cassert>
#include <iostream>

void testFeatureX()
{
    std::cout << "Testing feature X..." << std::endl;
    
    // Test code here
    assert(condition && "Description of what should be true");
    
    std::cout << "✓ Feature X test passed" << std::endl;
}

int main()
{
    std::cout << "=== Running My Tests ===" << std::endl;
    
    try {
        testFeatureX();
        
        std::cout << "\n✅ All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
```

## Test Coverage

Current test coverage includes:

- ✅ Build environment verification
- ✅ Script parsing and validation
- ✅ Arrangement position tracking
- ✅ Audio engine initialization
- ⚠️ GUI components (manual testing required)
- ⚠️ VST3 host integration (DAW testing required)
- ⚠️ ONNX model loading (requires models)

## Manual Testing

Some aspects require manual testing in a DAW:

### Plugin Loading Test
1. Copy `MAEVN.vst3` to your VST3 directory
2. Open your DAW (Reaper, Ableton, FL Studio, etc.)
3. Scan for new plugins
4. Load MAEVN as an instrument

### Audio Output Test
1. Create a MIDI track
2. Load MAEVN
3. Play MIDI notes
4. Verify audio output

### Arrangement Test
1. Set playback position in DAW
2. Verify correct section activation
3. Test section transitions

### Parameter Automation Test
1. Enable automation for parameters
2. Record parameter changes
3. Verify automation playback

## Performance Testing

For production builds, verify:

- DSP processing < 1ms per buffer (64-512 samples)
- No clicks or pops during processing
- Stable operation over extended periods
- No memory leaks

Use tools like:
- Valgrind (Linux): `valgrind --leak-check=full ./MAEVN_Standalone`
- Activity Monitor (macOS)
- Task Manager (Windows)

## Troubleshooting

### Tests won't build
- Ensure JUCE dependencies are installed
- Check CMake configuration output
- Verify C++17 support

### Tests fail
- Check test output for specific failures
- Verify system meets minimum requirements
- Review recent code changes

### Tests pass but plugin doesn't work
- Manual testing required
- Check DAW compatibility
- Verify plugin format (VST3)

## Security Testing

Before release, run security checks:

```bash
# CodeQL analysis (if configured)
codeql database create --language=cpp

# Dependency vulnerability scan
# Check ONNX Runtime version for CVEs
```

## Performance Benchmarks

Target performance metrics:

- Plugin load time: < 1 second
- Parameter change latency: < 10ms
- DSP processing: < 1ms per buffer @ 44.1kHz
- Memory usage: < 200MB
- CPU usage: < 5% idle, < 30% peak

## Release Checklist

Before releasing a new version:

- [ ] All unit tests pass
- [ ] Manual DAW testing completed
- [ ] No memory leaks detected
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers incremented
- [ ] Security scan completed

## Support

For testing issues or questions:

- GitHub Issues: https://github.com/fxgeniusllc-oss/Voice_Clone-VST/issues
- Documentation: [README.md](../README.md)
- Build Guide: [BUILD.md](../BUILD.md)
