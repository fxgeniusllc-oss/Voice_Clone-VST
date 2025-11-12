cmake_minimum_required(VERSION 3.15)

# Enable testing
enable_testing()

# Function to add a test executable
function(add_maevn_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    
    target_link_libraries(${TEST_NAME}
        PRIVATE
            juce::juce_core
            juce::juce_audio_basics
            juce::juce_data_structures
    )
    
    target_compile_definitions(${TEST_NAME}
        PRIVATE
            JUCE_STANDALONE_APPLICATION=1
            JUCE_USE_CURL=0
            JUCE_WEB_BROWSER=0
    )
    
    target_include_directories(${TEST_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/Source
    )
    
    # Add as a CTest test
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Build verification tests - basic JUCE and build environment
add_maevn_test(BuildVerificationTests BuildVerificationTests.cpp)

# Unit tests for individual components
add_maevn_test(ScriptParserTests ScriptParserTests.cpp)
target_sources(ScriptParserTests PRIVATE
    ${CMAKE_SOURCE_DIR}/Source/Parser/ScriptParser.cpp
    ${CMAKE_SOURCE_DIR}/Source/Parser/Arrangement.cpp
)

add_maevn_test(ArrangementTests ArrangementTests.cpp)
target_sources(ArrangementTests PRIVATE
    ${CMAKE_SOURCE_DIR}/Source/Parser/Arrangement.cpp
    ${CMAKE_SOURCE_DIR}/Source/Parser/ScriptParser.cpp
)

add_maevn_test(AudioEngineTests AudioEngineTests.cpp)
target_sources(AudioEngineTests PRIVATE
    ${CMAKE_SOURCE_DIR}/Source/Audio/AudioEngine.cpp
    ${CMAKE_SOURCE_DIR}/Source/Audio/InstrumentGenerator.cpp
    ${CMAKE_SOURCE_DIR}/Source/DSP/FXChain.cpp
    ${CMAKE_SOURCE_DIR}/Source/DSP/Effects.cpp
    ${CMAKE_SOURCE_DIR}/Source/AI/ONNXInference.cpp
    ${CMAKE_SOURCE_DIR}/Source/AI/VocalSynthesis.cpp
    ${CMAKE_SOURCE_DIR}/Source/AI/AIEffects.cpp
    ${CMAKE_SOURCE_DIR}/Source/Parser/Arrangement.cpp
    ${CMAKE_SOURCE_DIR}/Source/Parser/ScriptParser.cpp
)

target_link_libraries(AudioEngineTests
    PRIVATE
        juce::juce_audio_processors
        juce::juce_dsp
)

# Print test configuration
message(STATUS "Test executables configured:")
message(STATUS "  - BuildVerificationTests")
message(STATUS "  - ScriptParserTests")
message(STATUS "  - ArrangementTests")
message(STATUS "  - AudioEngineTests")
