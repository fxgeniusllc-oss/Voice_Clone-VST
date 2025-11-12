#pragma once

#include <juce_core/juce_core.h>
#include <juce_audio_processors/juce_audio_processors.h>

class ParameterState
{
public:
    ParameterState();
    ~ParameterState();

    // Save current state to XML
    std::unique_ptr<juce::XmlElement> saveToXml() const;
    
    // Load state from XML
    bool loadFromXml(const juce::XmlElement& xml);

    // Automation data
    struct AutomationPoint
    {
        double time;      // Time in PPQ
        float value;      // Parameter value
    };

    struct AutomationLane
    {
        juce::String parameterID;
        std::vector<AutomationPoint> points;
    };

    // Add automation point
    void addAutomationPoint(const juce::String& parameterID, double time, float value);
    
    // Get automation value at time
    float getAutomationValue(const juce::String& parameterID, double time) const;
    
    // Clear all automation
    void clearAutomation();

private:
    std::map<juce::String, AutomationLane> automationLanes;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ParameterState)
};
