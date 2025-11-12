#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "InstrumentGenerator.h"
#include "../DSP/FXChain.h"
#include "../AI/VocalSynthesis.h"
#include "../Parser/Arrangement.h"

class AudioEngine
{
public:
    AudioEngine();
    ~AudioEngine();

    void prepare(double sampleRate, int samplesPerBlock);
    void releaseResources();
    
    void process(juce::AudioBuffer<float>& buffer, 
                 juce::MidiBuffer& midiMessages,
                 juce::AudioProcessorValueTreeState& parameters);

    void updatePlaybackPosition(const juce::AudioPlayHead::CurrentPositionInfo& positionInfo);

    InstrumentGenerator& getInstrumentGenerator() { return instrumentGenerator; }
    VocalSynthesis& getVocalSynthesis() { return vocalSynthesis; }
    FXChain& getFXChain() { return fxChain; }
    Arrangement& getArrangement() { return arrangement; }

private:
    double currentSampleRate = 44100.0;
    int currentSamplesPerBlock = 512;
    
    InstrumentGenerator instrumentGenerator;
    VocalSynthesis vocalSynthesis;
    FXChain fxChain;
    Arrangement arrangement;

    juce::AudioPlayHead::CurrentPositionInfo lastPositionInfo;
    
    void processInstruments(juce::AudioBuffer<float>& buffer, 
                           juce::MidiBuffer& midiMessages,
                           juce::AudioProcessorValueTreeState& parameters);
    void processVocals(juce::AudioBuffer<float>& buffer,
                      juce::AudioProcessorValueTreeState& parameters);
    void processFX(juce::AudioBuffer<float>& buffer,
                  juce::AudioProcessorValueTreeState& parameters);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};
