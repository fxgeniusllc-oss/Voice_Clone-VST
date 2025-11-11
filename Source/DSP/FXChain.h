#pragma once

#include <JuceHeader.h>
#include "Effects.h"
#include "../AI/AIEffects.h"

class FXChain
{
public:
    FXChain();
    ~FXChain();

    void prepare(double sampleRate, int samplesPerBlock);
    void releaseResources();

    void process(juce::AudioBuffer<float>& buffer, 
                 float reverbMix, float delayMix, 
                 float distortion, float aiFxMix);

private:
    ReverbEffect reverbEffect;
    DelayEffect delayEffect;
    DistortionEffect distortionEffect;
    AIEffects aiEffects;

    juce::AudioBuffer<float> tempBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(FXChain)
};
