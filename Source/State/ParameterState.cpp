#include "ParameterState.h"

ParameterState::ParameterState()
{
}

ParameterState::~ParameterState()
{
}

std::unique_ptr<juce::XmlElement> ParameterState::saveToXml() const
{
    auto xml = std::make_unique<juce::XmlElement>("ParameterState");
    
    auto automationXml = xml->createNewChildElement("Automation");
    
    for (const auto& [paramID, lane] : automationLanes)
    {
        auto laneXml = automationXml->createNewChildElement("Lane");
        laneXml->setAttribute("parameterID", paramID);
        
        for (const auto& point : lane.points)
        {
            auto pointXml = laneXml->createNewChildElement("Point");
            pointXml->setAttribute("time", point.time);
            pointXml->setAttribute("value", point.value);
        }
    }
    
    return xml;
}

bool ParameterState::loadFromXml(const juce::XmlElement& xml)
{
    if (!xml.hasTagName("ParameterState"))
        return false;
    
    clearAutomation();
    
    auto automationXml = xml.getChildByName("Automation");
    if (automationXml != nullptr)
    {
        for (auto* laneXml : automationXml->getChildIterator())
        {
            if (laneXml->hasTagName("Lane"))
            {
                juce::String paramID = laneXml->getStringAttribute("parameterID");
                
                for (auto* pointXml : laneXml->getChildIterator())
                {
                    if (pointXml->hasTagName("Point"))
                    {
                        double time = pointXml->getDoubleAttribute("time");
                        float value = static_cast<float>(pointXml->getDoubleAttribute("value"));
                        addAutomationPoint(paramID, time, value);
                    }
                }
            }
        }
    }
    
    return true;
}

void ParameterState::addAutomationPoint(const juce::String& parameterID, double time, float value)
{
    auto& lane = automationLanes[parameterID];
    lane.parameterID = parameterID;
    
    AutomationPoint point;
    point.time = time;
    point.value = value;
    
    lane.points.push_back(point);
    
    // Sort points by time
    std::sort(lane.points.begin(), lane.points.end(),
              [](const AutomationPoint& a, const AutomationPoint& b) {
                  return a.time < b.time;
              });
}

float ParameterState::getAutomationValue(const juce::String& parameterID, double time) const
{
    auto it = automationLanes.find(parameterID);
    if (it == automationLanes.end())
        return 0.0f;
    
    const auto& lane = it->second;
    if (lane.points.empty())
        return 0.0f;
    
    // Find surrounding points for interpolation
    if (time <= lane.points.front().time)
        return lane.points.front().value;
    
    if (time >= lane.points.back().time)
        return lane.points.back().value;
    
    // Linear interpolation between points
    for (size_t i = 0; i < lane.points.size() - 1; ++i)
    {
        if (time >= lane.points[i].time && time <= lane.points[i + 1].time)
        {
            double t = (time - lane.points[i].time) / 
                      (lane.points[i + 1].time - lane.points[i].time);
            return static_cast<float>(lane.points[i].value + 
                   t * (lane.points[i + 1].value - lane.points[i].value));
        }
    }
    
    return 0.0f;
}

void ParameterState::clearAutomation()
{
    automationLanes.clear();
}
