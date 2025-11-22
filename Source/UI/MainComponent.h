#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "../PluginProcessor.h"
#include "TimelineComponent.h"

class MainComponent : public juce::Component
{
public:
    MainComponent(MAEVNAudioProcessor& processor);
    ~MainComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    MAEVNAudioProcessor& audioProcessor;
    
    // UI Components
    juce::Label titleLabel;
    
    // Control sections
    juce::GroupComponent instrumentGroup;
    juce::GroupComponent vocalGroup;
    juce::GroupComponent fxGroup;
    juce::GroupComponent masterGroup;
    
    // Instrument toggles
    juce::ToggleButton enable808Button;
    juce::ToggleButton enableHiHatButton;
    juce::ToggleButton enableSnareButton;
    juce::ToggleButton enablePianoButton;
    juce::ToggleButton enableSynthButton;
    
    // Vocal controls
    juce::ToggleButton enableVocalsButton;
    juce::Slider vocalPitchSlider;
    juce::Slider vocalFormantSlider;
    juce::Label vocalPitchLabel;
    juce::Label vocalFormantLabel;
    
    // FX controls
    juce::Slider reverbMixSlider;
    juce::Slider delayMixSlider;
    juce::Slider distortionSlider;
    juce::Slider aiFxMixSlider;
    juce::Label reverbLabel;
    juce::Label delayLabel;
    juce::Label distortionLabel;
    juce::Label aiFxLabel;
    
    // Master controls
    juce::Slider masterGainSlider;
    juce::Slider masterPanSlider;
    juce::Label masterGainLabel;
    juce::Label masterPanLabel;
    
    // Timeline
    TimelineComponent timeline;
    
    // Script editor
    juce::TextEditor scriptEditor;
    juce::TextButton parseScriptButton;
    juce::Label scriptLabel;
    
    // Undo/Redo buttons
    juce::TextButton undoButton;
    juce::TextButton redoButton;
    
    // Parameter attachments
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>> buttonAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    
    void setupButton(juce::ToggleButton& button, const juce::String& text, const juce::String& paramID);
    void setupSlider(juce::Slider& slider, juce::Label& label, const juce::String& text, const juce::String& paramID);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
