#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include "ONNXInference.h"

class AIEffects
{
public:
    AIEffects();
    ~AIEffects();

    void prepare(double sampleRate, int samplesPerBlock);
    void releaseResources();

    void process(juce::AudioBuffer<float>& buffer, float mix);

    // Load AI FX model (e.g., neural amp modeling, AI reverb, etc.)
    bool loadModel(const juce::String& modelPath);

private:
    double currentSampleRate = 44100.0;
    int currentSamplesPerBlock = 512;

    ONNXInference aiModel;
    juce::AudioBuffer<float> dryBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AIEffects)
};
