#include "AudioEngine.h"

AudioEngine::AudioEngine()
{
}

AudioEngine::~AudioEngine()
{
}

void AudioEngine::prepare(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentSamplesPerBlock = samplesPerBlock;

    instrumentGenerator.prepare(sampleRate, samplesPerBlock);
    vocalSynthesis.prepare(sampleRate, samplesPerBlock);
    fxChain.prepare(sampleRate, samplesPerBlock);
}

void AudioEngine::releaseResources()
{
    instrumentGenerator.releaseResources();
    vocalSynthesis.releaseResources();
    fxChain.releaseResources();
}

void AudioEngine::updatePlaybackPosition(const juce::AudioPlayHead::CurrentPositionInfo& positionInfo)
{
    lastPositionInfo = positionInfo;
    
    // Update arrangement based on playback position
    if (positionInfo.isPlaying)
    {
        arrangement.updatePosition(positionInfo.ppqPosition, 
                                   positionInfo.bpm,
                                   positionInfo.timeSigNumerator,
                                   positionInfo.timeSigDenominator);
    }
}

void AudioEngine::process(juce::AudioBuffer<float>& buffer, 
                          juce::MidiBuffer& midiMessages,
                          juce::AudioProcessorValueTreeState& parameters)
{
    buffer.clear();

    // Process instruments (808, hi-hat, snare, piano, synth)
    processInstruments(buffer, midiMessages, parameters);

    // Process AI vocals (TTS + vocoder)
    processVocals(buffer, parameters);

    // Apply FX chain (DSP + AI effects)
    processFX(buffer, parameters);

    // Apply master gain and pan
    auto masterGain = parameters.getRawParameterValue("masterGain")->load();
    auto masterPan = parameters.getRawParameterValue("masterPan")->load();

    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        float panGain = 1.0f;
        if (channel == 0) // Left
            panGain = std::min(1.0f, 1.0f - masterPan);
        else // Right
            panGain = std::min(1.0f, 1.0f + masterPan);

        buffer.applyGain(channel, 0, buffer.getNumSamples(), masterGain * panGain);
    }
}

void AudioEngine::processInstruments(juce::AudioBuffer<float>& buffer, 
                                     juce::MidiBuffer& midiMessages,
                                     juce::AudioProcessorValueTreeState& parameters)
{
    // Get current arrangement section
    auto currentSection = arrangement.getCurrentSection();

    // Enable/disable instruments based on parameters and arrangement
    bool enable808 = parameters.getRawParameterValue("enable808")->load() > 0.5f;
    bool enableHiHat = parameters.getRawParameterValue("enableHiHat")->load() > 0.5f;
    bool enableSnare = parameters.getRawParameterValue("enableSnare")->load() > 0.5f;
    bool enablePiano = parameters.getRawParameterValue("enablePiano")->load() > 0.5f;
    bool enableSynth = parameters.getRawParameterValue("enableSynth")->load() > 0.5f;

    // Process based on arrangement section
    if (currentSection == "808" || currentSection == "VERSE" || currentSection == "HOOK")
    {
        instrumentGenerator.process(buffer, midiMessages, 
                                   enable808, enableHiHat, enableSnare, 
                                   enablePiano, enableSynth);
    }
}

void AudioEngine::processVocals(juce::AudioBuffer<float>& buffer,
                                juce::AudioProcessorValueTreeState& parameters)
{
    bool enableVocals = parameters.getRawParameterValue("enableVocals")->load() > 0.5f;
    
    if (enableVocals)
    {
        auto vocalPitch = parameters.getRawParameterValue("vocalPitch")->load();
        auto vocalFormant = parameters.getRawParameterValue("vocalFormant")->load();
        
        vocalSynthesis.process(buffer, vocalPitch, vocalFormant);
    }
}

void AudioEngine::processFX(juce::AudioBuffer<float>& buffer,
                           juce::AudioProcessorValueTreeState& parameters)
{
    auto reverbMix = parameters.getRawParameterValue("reverbMix")->load();
    auto delayMix = parameters.getRawParameterValue("delayMix")->load();
    auto distortion = parameters.getRawParameterValue("distortion")->load();
    auto aiFxMix = parameters.getRawParameterValue("aiFxMix")->load();

    fxChain.process(buffer, reverbMix, delayMix, distortion, aiFxMix);
}
