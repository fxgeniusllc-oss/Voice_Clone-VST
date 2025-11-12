#pragma once

#include <juce_core/juce_core.h>
#include <vector>
#include <map>

struct ArrangementSection
{
    juce::String name;        // e.g., "HOOK", "VERSE", "808"
    double startTime;         // In quarters/PPQ
    double duration;          // In quarters
    bool enable808 = false;
    bool enableHiHat = false;
    bool enableSnare = false;
    bool enablePiano = false;
    bool enableSynth = false;
    bool enableVocals = false;
};

class ScriptParser
{
public:
    ScriptParser();
    ~ScriptParser();

    // Parse a stage script string
    // Format example: "[HOOK:0:4] [VERSE:4:8] [808:8:12]"
    bool parseScript(const juce::String& script);
    
    // Get parsed sections
    const std::vector<ArrangementSection>& getSections() const { return sections; }
    
    // Clear all sections
    void clear();
    
    // Get section at specific time position (PPQ)
    ArrangementSection* getSectionAt(double ppqPosition);

private:
    std::vector<ArrangementSection> sections;
    
    // Parse individual section tag like "[HOOK:0:4]"
    bool parseSection(const juce::String& tag, ArrangementSection& section);
    
    // Configure section based on its name
    void configureSectionByName(ArrangementSection& section);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ScriptParser)
};
