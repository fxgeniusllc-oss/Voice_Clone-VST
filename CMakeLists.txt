cmake_minimum_required(VERSION 3.15)

project(MAEVN VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE Framework
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 7.0.9
)
FetchContent_MakeAvailable(JUCE)

# ONNX Runtime
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    message(STATUS "ONNX Runtime not found via find_package, will use manual configuration")
    # Users can specify ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIB manually
endif()

# Plugin target
juce_add_plugin(MAEVN
    COMPANY_NAME "FXGenius"
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE FxGs
    PLUGIN_CODE Maev
    FORMATS VST3 Standalone
    PRODUCT_NAME "MAEVN"
    VST3_CATEGORIES "Instrument" "Synth" "Fx"
)

# Source files
target_sources(MAEVN
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/Audio/AudioEngine.cpp
        Source/Audio/InstrumentGenerator.cpp
        Source/DSP/FXChain.cpp
        Source/DSP/Effects.cpp
        Source/AI/ONNXInference.cpp
        Source/AI/VocalSynthesis.cpp
        Source/AI/AIEffects.cpp
        Source/Parser/ScriptParser.cpp
        Source/Parser/Arrangement.cpp
        Source/State/UndoManager.cpp
        Source/State/ParameterState.cpp
        Source/UI/MainComponent.cpp
        Source/UI/TimelineComponent.cpp
)

# Headers
target_include_directories(MAEVN
    PUBLIC
        Source
)

# ONNX Runtime linking (if found or manually specified)
if(onnxruntime_FOUND)
    target_link_libraries(MAEVN PRIVATE onnxruntime::onnxruntime)
elseif(DEFINED ONNXRUNTIME_INCLUDE_DIR AND DEFINED ONNXRUNTIME_LIB)
    target_include_directories(MAEVN PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(MAEVN PRIVATE ${ONNXRUNTIME_LIB})
endif()

# JUCE modules
target_link_libraries(MAEVN
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Compile definitions
target_compile_definitions(MAEVN
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
        JUCE_REPORT_APP_USAGE=0
        JUCE_STRICT_REFCOUNTEDPOINTER=1
)

# Testing
option(BUILD_TESTS "Build test suite" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
    message(STATUS "Tests enabled - use 'ctest' to run tests after building")
endif()
