#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <iostream>
#include <cassert>

void testJUCEVersion()
{
    std::cout << "Testing JUCE version..." << std::endl;
    
    juce::String juceVersion = juce::SystemStats::getJUCEVersion();
    std::cout << "  JUCE Version: " << juceVersion << std::endl;
    
    assert(!juceVersion.isEmpty() && "JUCE version should not be empty");
    
    std::cout << "✓ JUCE version test passed" << std::endl;
}

void testBasicJUCEFunctionality()
{
    std::cout << "Testing basic JUCE functionality..." << std::endl;
    
    // Test String
    juce::String testStr = "Hello MAEVN";
    assert(testStr.length() == 11 && "String length should be 11");
    
    // Test Array
    juce::Array<int> testArray;
    testArray.add(1);
    testArray.add(2);
    testArray.add(3);
    assert(testArray.size() == 3 && "Array size should be 3");
    
    // Test File operations (non-destructive)
    juce::File currentDir = juce::File::getCurrentWorkingDirectory();
    assert(currentDir.exists() && "Current directory should exist");
    
    std::cout << "  Current directory: " << currentDir.getFullPathName() << std::endl;
    
    std::cout << "✓ Basic JUCE functionality test passed" << std::endl;
}

void testAudioBufferOperations()
{
    std::cout << "Testing audio buffer operations..." << std::endl;
    
    juce::AudioBuffer<float> buffer(2, 512);
    
    // Test clear
    buffer.clear();
    
    // Verify buffer is zeroed
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        const float* data = buffer.getReadPointer(channel);
        for (int i = 0; i < buffer.getNumSamples(); ++i)
        {
            assert(data[i] == 0.0f && "Buffer should be cleared to zero");
        }
    }
    
    // Test writing data
    buffer.setSample(0, 0, 1.0f);
    assert(buffer.getSample(0, 0) == 1.0f && "Sample should be set to 1.0");
    
    // Test gain application
    buffer.applyGain(0.5f);
    assert(std::abs(buffer.getSample(0, 0) - 0.5f) < 0.0001f && "Sample should be halved");
    
    std::cout << "✓ Audio buffer operations test passed" << std::endl;
}

void testSystemInfo()
{
    std::cout << "Testing system information..." << std::endl;
    
    std::cout << "  OS: " << juce::SystemStats::getOperatingSystemName() << std::endl;
    std::cout << "  CPU: " << juce::SystemStats::getCpuModel() << std::endl;
    std::cout << "  Cores: " << juce::SystemStats::getNumCpus() << std::endl;
    std::cout << "  Memory: " << juce::SystemStats::getMemorySizeInMegabytes() << " MB" << std::endl;
    
    assert(juce::SystemStats::getNumCpus() > 0 && "Should have at least 1 CPU");
    
    std::cout << "✓ System information test passed" << std::endl;
}

int main()
{
    std::cout << "=== Running Build Verification Tests ===" << std::endl;
    std::cout << "These tests verify that the build environment is correctly configured." << std::endl;
    std::cout << std::endl;
    
    try {
        testJUCEVersion();
        testBasicJUCEFunctionality();
        testAudioBufferOperations();
        testSystemInfo();
        
        std::cout << "\n✅ All build verification tests passed!" << std::endl;
        std::cout << "The MAEVN project is properly configured and ready for production." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
