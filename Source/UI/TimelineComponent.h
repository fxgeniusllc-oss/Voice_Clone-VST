#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "../PluginProcessor.h"

class TimelineComponent : public juce::Component, public juce::Timer
{
public:
    TimelineComponent(MAEVNAudioProcessor& processor);
    ~TimelineComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

private:
    MAEVNAudioProcessor& audioProcessor;
    
    double currentPosition = 0.0;
    juce::String currentSection;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TimelineComponent)
};
