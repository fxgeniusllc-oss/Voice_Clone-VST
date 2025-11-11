#pragma once

#include <JuceHeader.h>

class ReverbEffect
{
public:
    ReverbEffect();
    ~ReverbEffect();

    void prepare(double sampleRate, int samplesPerBlock);
    void releaseResources();
    void process(juce::AudioBuffer<float>& buffer, float mix);

private:
    juce::dsp::Reverb reverb;
    juce::dsp::ProcessSpec spec;
    juce::AudioBuffer<float> dryBuffer;
};

class DelayEffect
{
public:
    DelayEffect();
    ~DelayEffect();

    void prepare(double sampleRate, int samplesPerBlock);
    void releaseResources();
    void process(juce::AudioBuffer<float>& buffer, float mix);

private:
    double sampleRate = 44100.0;
    static constexpr int maxDelayInSamples = 88200; // 2 seconds at 44.1kHz
    juce::AudioBuffer<float> delayBuffer;
    int delayPosition = 0;
    juce::AudioBuffer<float> dryBuffer;
};

class DistortionEffect
{
public:
    DistortionEffect();
    ~DistortionEffect();

    void prepare(double sampleRate, int samplesPerBlock);
    void releaseResources();
    void process(juce::AudioBuffer<float>& buffer, float amount);

private:
    juce::dsp::WaveShaper<float> waveshaper;
    juce::dsp::ProcessSpec spec;
};
