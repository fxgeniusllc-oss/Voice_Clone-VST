#include "Arrangement.h"

Arrangement::Arrangement()
{
    // Set default arrangement
    setScript("[INTRO:0:4] [VERSE:4:12] [HOOK:12:20] [VERSE:20:28] [HOOK:28:36] [OUTRO:36:40]");
}

Arrangement::~Arrangement()
{
}

void Arrangement::setScript(const juce::String& script)
{
    if (parser.parseScript(script))
    {
        juce::Logger::writeToLog("Arrangement script parsed successfully");
        juce::Logger::writeToLog("Sections: " + juce::String(parser.getSections().size()));
    }
    else
    {
        juce::Logger::writeToLog("Failed to parse arrangement script");
    }
}

void Arrangement::updatePosition(double ppqPosition, double bpm, int numerator, int denominator)
{
    currentPPQ = ppqPosition;
    currentBPM = bpm;
    currentNumerator = numerator;
    currentDenominator = denominator;
    
    // Find current section
    currentSection = parser.getSectionAt(ppqPosition);
    
    if (currentSection != nullptr)
    {
        currentSectionName = currentSection->name;
    }
    else
    {
        currentSectionName = "";
    }
}

const ArrangementSection* Arrangement::getCurrentSectionData() const
{
    return currentSection;
}
