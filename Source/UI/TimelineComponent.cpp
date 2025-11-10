#include "TimelineComponent.h"

TimelineComponent::TimelineComponent(MAEVNAudioProcessor& processor)
    : audioProcessor(processor)
{
    startTimerHz(30); // Update 30 times per second
}

TimelineComponent::~TimelineComponent()
{
    stopTimer();
}

void TimelineComponent::timerCallback()
{
    currentSection = audioProcessor.getAudioEngine().getArrangement().getCurrentSection();
    repaint();
}

void TimelineComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    
    // Background
    g.setColour(juce::Colour(0xff1a1a1a));
    g.fillRoundedRectangle(bounds.toFloat(), 5.0f);
    
    // Border
    g.setColour(juce::Colour(0xff404040));
    g.drawRoundedRectangle(bounds.toFloat().reduced(1), 5.0f, 2.0f);
    
    // Timeline sections
    auto& sections = audioProcessor.getScriptParser().getSections();
    
    if (!sections.empty())
    {
        // Find total duration
        double totalDuration = 0.0;
        for (const auto& section : sections)
        {
            double endTime = section.startTime + section.duration;
            if (endTime > totalDuration)
                totalDuration = endTime;
        }
        
        if (totalDuration > 0.0)
        {
            auto timelineBounds = bounds.reduced(10);
            float pixelsPerQuarter = timelineBounds.getWidth() / static_cast<float>(totalDuration);
            
            // Draw sections
            for (const auto& section : sections)
            {
                float x = section.startTime * pixelsPerQuarter;
                float width = section.duration * pixelsPerQuarter;
                
                juce::Rectangle<float> sectionRect(
                    timelineBounds.getX() + x,
                    timelineBounds.getY() + 20,
                    width,
                    timelineBounds.getHeight() - 40
                );
                
                // Section color based on name
                juce::Colour sectionColour;
                if (section.name == "HOOK")
                    sectionColour = juce::Colour(0xff4a9eff);
                else if (section.name == "VERSE")
                    sectionColour = juce::Colour(0xff9d4aff);
                else if (section.name == "808")
                    sectionColour = juce::Colour(0xffff4a4a);
                else if (section.name == "INTRO")
                    sectionColour = juce::Colour(0xff4aff9d);
                else if (section.name == "OUTRO")
                    sectionColour = juce::Colour(0xffff9d4a);
                else
                    sectionColour = juce::Colour(0xff808080);
                
                // Highlight current section
                if (section.name == currentSection)
                {
                    g.setColour(sectionColour.brighter(0.3f));
                    g.fillRoundedRectangle(sectionRect.expanded(2), 3.0f);
                }
                
                g.setColour(sectionColour.withAlpha(0.6f));
                g.fillRoundedRectangle(sectionRect, 3.0f);
                
                g.setColour(sectionColour);
                g.drawRoundedRectangle(sectionRect, 3.0f, 1.5f);
                
                // Section name
                g.setColour(juce::Colours::white);
                g.setFont(12.0f);
                g.drawText(section.name, sectionRect, juce::Justification::centred);
            }
        }
    }
    
    // Title
    g.setColour(juce::Colours::white.withAlpha(0.8f));
    g.setFont(14.0f);
    g.drawText("Timeline Arrangement", bounds.removeFromTop(20).reduced(10), juce::Justification::centredLeft);
    
    // Current section display
    if (!currentSection.isEmpty())
    {
        g.setFont(juce::Font(16.0f, juce::Font::bold));
        g.drawText("Current: " + currentSection, 
                   bounds.removeFromBottom(20).reduced(10), 
                   juce::Justification::centredRight);
    }
}

void TimelineComponent::resized()
{
}
