#include "ScriptParser.h"

ScriptParser::ScriptParser()
{
}

ScriptParser::~ScriptParser()
{
}

bool ScriptParser::parseScript(const juce::String& script)
{
    clear();
    
    // Find all section tags in format [NAME:START:END]
    int pos = 0;
    while (pos < script.length())
    {
        int startBracket = script.indexOfChar(pos, '[');
        if (startBracket < 0)
            break;
            
        int endBracket = script.indexOfChar(startBracket, ']');
        if (endBracket < 0)
            break;
            
        juce::String tag = script.substring(startBracket, endBracket + 1);
        
        ArrangementSection section;
        if (parseSection(tag, section))
        {
            sections.push_back(section);
        }
        
        pos = endBracket + 1;
    }
    
    // Sort sections by start time
    std::sort(sections.begin(), sections.end(), 
              [](const ArrangementSection& a, const ArrangementSection& b) {
                  return a.startTime < b.startTime;
              });
    
    return !sections.empty();
}

bool ScriptParser::parseSection(const juce::String& tag, ArrangementSection& section)
{
    // Expected format: [NAME:START:END] or [NAME:START:DURATION]
    if (!tag.startsWith("[") || !tag.endsWith("]"))
        return false;
        
    juce::String content = tag.substring(1, tag.length() - 1);
    juce::StringArray parts = juce::StringArray::fromTokens(content, ":", "");
    
    if (parts.size() < 3)
        return false;
    
    section.name = parts[0].trim().toUpperCase();
    section.startTime = parts[1].getDoubleValue();
    
    // Third parameter is duration
    section.duration = parts[2].getDoubleValue();
    
    // Configure section based on name
    configureSectionByName(section);
    
    return true;
}

void ScriptParser::configureSectionByName(ArrangementSection& section)
{
    // Set defaults based on section name
    if (section.name == "HOOK")
    {
        section.enable808 = true;
        section.enableHiHat = true;
        section.enableSnare = true;
        section.enablePiano = true;
        section.enableSynth = true;
        section.enableVocals = true;
    }
    else if (section.name == "VERSE")
    {
        section.enable808 = true;
        section.enableHiHat = true;
        section.enableSnare = false;
        section.enablePiano = true;
        section.enableSynth = false;
        section.enableVocals = true;
    }
    else if (section.name == "808")
    {
        section.enable808 = true;
        section.enableHiHat = false;
        section.enableSnare = false;
        section.enablePiano = false;
        section.enableSynth = false;
        section.enableVocals = false;
    }
    else if (section.name == "INTRO")
    {
        section.enable808 = false;
        section.enableHiHat = true;
        section.enableSnare = false;
        section.enablePiano = true;
        section.enableSynth = true;
        section.enableVocals = false;
    }
    else if (section.name == "OUTRO")
    {
        section.enable808 = false;
        section.enableHiHat = false;
        section.enableSnare = false;
        section.enablePiano = true;
        section.enableSynth = false;
        section.enableVocals = true;
    }
    else
    {
        // Unknown section - enable everything
        section.enable808 = true;
        section.enableHiHat = true;
        section.enableSnare = true;
        section.enablePiano = true;
        section.enableSynth = true;
        section.enableVocals = true;
    }
}

void ScriptParser::clear()
{
    sections.clear();
}

ArrangementSection* ScriptParser::getSectionAt(double ppqPosition)
{
    for (auto& section : sections)
    {
        if (ppqPosition >= section.startTime && 
            ppqPosition < (section.startTime + section.duration))
        {
            return &section;
        }
    }
    return nullptr;
}
