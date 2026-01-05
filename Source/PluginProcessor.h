#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "Audio/AudioEngine.h"
#include "State/UndoManager.h"
#include "State/ParameterState.h"
#include "Parser/ScriptParser.h"

class MAEVNAudioProcessor : public juce::AudioProcessor
{
public:
    MAEVNAudioProcessor();
    ~MAEVNAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Custom methods
    AudioEngine& getAudioEngine() { return audioEngine; }
    MAEVNUndoManager& getUndoManager() { return undoManager; }
    ScriptParser& getScriptParser() { return scriptParser; }
    juce::AudioProcessorValueTreeState& getParameters() { return *parameters; }

private:
    AudioEngine audioEngine;
    MAEVNUndoManager undoManager;
    ParameterState parameterState;
    ScriptParser scriptParser;
    
    std::unique_ptr<juce::AudioProcessorValueTreeState> parameters;
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Standalone readiness tracking
    int prepareCallCount = 0;
    bool isStandaloneInitialized = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MAEVNAudioProcessor)
};
