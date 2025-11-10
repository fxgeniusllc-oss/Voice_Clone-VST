#include "Effects.h"

// ReverbEffect implementation
ReverbEffect::ReverbEffect()
{
}

ReverbEffect::~ReverbEffect()
{
}

void ReverbEffect::prepare(double sampleRate, int samplesPerBlock)
{
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels = 2;

    reverb.prepare(spec);

    juce::dsp::Reverb::Parameters params;
    params.roomSize = 0.5f;
    params.damping = 0.5f;
    params.wetLevel = 0.33f;
    params.dryLevel = 0.4f;
    params.width = 1.0f;
    reverb.setParameters(params);

    dryBuffer.setSize(2, samplesPerBlock);
}

void ReverbEffect::releaseResources()
{
}

void ReverbEffect::process(juce::AudioBuffer<float>& buffer, float mix)
{
    // Store dry signal
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        dryBuffer.copyFrom(ch, 0, buffer, ch, 0, buffer.getNumSamples());

    // Process reverb
    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    reverb.process(context);

    // Mix dry and wet
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        auto* wet = buffer.getWritePointer(ch);
        auto* dry = dryBuffer.getReadPointer(ch);
        
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            wet[sample] = dry[sample] * (1.0f - mix) + wet[sample] * mix;
    }
}

// DelayEffect implementation
DelayEffect::DelayEffect()
{
}

DelayEffect::~DelayEffect()
{
}

void DelayEffect::prepare(double sr, int samplesPerBlock)
{
    sampleRate = sr;
    delayBuffer.setSize(2, maxDelayInSamples);
    delayBuffer.clear();
    delayPosition = 0;
    dryBuffer.setSize(2, samplesPerBlock);
}

void DelayEffect::releaseResources()
{
}

void DelayEffect::process(juce::AudioBuffer<float>& buffer, float mix)
{
    const int delayTimeInSamples = static_cast<int>(sampleRate * 0.375); // 375ms delay
    const float feedback = 0.4f;

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        dryBuffer.copyFrom(ch, 0, buffer, ch, 0, buffer.getNumSamples());

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        auto* channelData = buffer.getWritePointer(ch);
        auto* delayData = delayBuffer.getWritePointer(ch);
        
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        {
            int readPos = (delayPosition - delayTimeInSamples + maxDelayInSamples) % maxDelayInSamples;
            float delayedSample = delayData[readPos];
            
            float input = channelData[sample];
            float output = input + delayedSample * feedback;
            
            delayData[delayPosition] = output;
            channelData[sample] = input * (1.0f - mix) + delayedSample * mix;
            
            delayPosition = (delayPosition + 1) % maxDelayInSamples;
        }
    }
}

// DistortionEffect implementation
DistortionEffect::DistortionEffect()
{
}

DistortionEffect::~DistortionEffect()
{
}

void DistortionEffect::prepare(double sampleRate, int samplesPerBlock)
{
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels = 2;

    waveshaper.prepare(spec);
    
    // Soft clipping transfer function
    waveshaper.functionToUse = [](float x)
    {
        return std::tanh(x * 2.0f) * 0.5f;
    };
}

void DistortionEffect::releaseResources()
{
}

void DistortionEffect::process(juce::AudioBuffer<float>& buffer, float amount)
{
    if (amount < 0.01f)
        return;

    // Pre-gain based on distortion amount
    float preGain = 1.0f + amount * 5.0f;
    buffer.applyGain(preGain);

    // Apply waveshaping
    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    waveshaper.process(context);

    // Post-gain compensation
    buffer.applyGain(1.0f / (1.0f + amount));
}
