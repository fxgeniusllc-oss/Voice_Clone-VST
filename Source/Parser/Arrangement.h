#pragma once

#include <juce_core/juce_core.h>
#include "ScriptParser.h"

class Arrangement
{
public:
    Arrangement();
    ~Arrangement();

    // Set arrangement script
    void setScript(const juce::String& script);
    
    // Update playback position
    void updatePosition(double ppqPosition, double bpm, int numerator, int denominator);
    
    // Get current section name
    juce::String getCurrentSection() const { return currentSectionName; }
    
    // Get current arrangement section data
    const ArrangementSection* getCurrentSectionData() const;

private:
    ScriptParser parser;
    double currentPPQ = 0.0;
    double currentBPM = 120.0;
    int currentNumerator = 4;
    int currentDenominator = 4;
    
    juce::String currentSectionName;
    ArrangementSection* currentSection = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Arrangement)
};
