#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "PluginProcessor.h"
#include "UI/MainComponent.h"

class MAEVNAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    MAEVNAudioProcessorEditor(MAEVNAudioProcessor&);
    ~MAEVNAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    MAEVNAudioProcessor& audioProcessor;
    MainComponent mainComponent;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MAEVNAudioProcessorEditor)
};
