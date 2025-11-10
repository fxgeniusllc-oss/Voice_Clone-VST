#include "FXChain.h"

FXChain::FXChain()
{
}

FXChain::~FXChain()
{
}

void FXChain::prepare(double sampleRate, int samplesPerBlock)
{
    reverbEffect.prepare(sampleRate, samplesPerBlock);
    delayEffect.prepare(sampleRate, samplesPerBlock);
    distortionEffect.prepare(sampleRate, samplesPerBlock);
    aiEffects.prepare(sampleRate, samplesPerBlock);

    tempBuffer.setSize(2, samplesPerBlock);
}

void FXChain::releaseResources()
{
    reverbEffect.releaseResources();
    delayEffect.releaseResources();
    distortionEffect.releaseResources();
    aiEffects.releaseResources();
}

void FXChain::process(juce::AudioBuffer<float>& buffer, 
                      float reverbMix, float delayMix, 
                      float distortion, float aiFxMix)
{
    // Apply distortion first (pre-effect)
    if (distortion > 0.01f)
    {
        distortionEffect.process(buffer, distortion);
    }

    // Apply delay
    if (delayMix > 0.01f)
    {
        delayEffect.process(buffer, delayMix);
    }

    // Apply reverb
    if (reverbMix > 0.01f)
    {
        reverbEffect.process(buffer, reverbMix);
    }

    // Apply AI effects (if enabled and ONNX model loaded)
    if (aiFxMix > 0.01f)
    {
        aiEffects.process(buffer, aiFxMix);
    }
}
