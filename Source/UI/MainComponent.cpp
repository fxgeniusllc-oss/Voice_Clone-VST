#include "MainComponent.h"

MainComponent::MainComponent(MAEVNAudioProcessor& processor)
    : audioProcessor(processor), timeline(processor)
{
    // Title
    titleLabel.setText("MAEVN - AI Vocal + Instrument Generator", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(24.0f, juce::Font::bold));
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);
    
    // Group components
    instrumentGroup.setText("Instruments");
    instrumentGroup.setTextLabelPosition(juce::Justification::centredTop);
    addAndMakeVisible(instrumentGroup);
    
    vocalGroup.setText("Vocals");
    vocalGroup.setTextLabelPosition(juce::Justification::centredTop);
    addAndMakeVisible(vocalGroup);
    
    fxGroup.setText("Effects");
    fxGroup.setTextLabelPosition(juce::Justification::centredTop);
    addAndMakeVisible(fxGroup);
    
    masterGroup.setText("Master");
    masterGroup.setTextLabelPosition(juce::Justification::centredTop);
    addAndMakeVisible(masterGroup);
    
    // Instrument controls
    setupButton(enable808Button, "808 Bass", "enable808");
    setupButton(enableHiHatButton, "Hi-Hat", "enableHiHat");
    setupButton(enableSnareButton, "Snare", "enableSnare");
    setupButton(enablePianoButton, "Piano", "enablePiano");
    setupButton(enableSynthButton, "Synth", "enableSynth");
    
    // Vocal controls
    setupButton(enableVocalsButton, "Enable Vocals", "enableVocals");
    setupSlider(vocalPitchSlider, vocalPitchLabel, "Pitch", "vocalPitch");
    setupSlider(vocalFormantSlider, vocalFormantLabel, "Formant", "vocalFormant");
    
    // FX controls
    setupSlider(reverbMixSlider, reverbLabel, "Reverb", "reverbMix");
    setupSlider(delayMixSlider, delayLabel, "Delay", "delayMix");
    setupSlider(distortionSlider, distortionLabel, "Distortion", "distortion");
    setupSlider(aiFxMixSlider, aiFxLabel, "AI FX", "aiFxMix");
    
    // Master controls
    setupSlider(masterGainSlider, masterGainLabel, "Gain", "masterGain");
    setupSlider(masterPanSlider, masterPanLabel, "Pan", "masterPan");
    
    // Timeline
    addAndMakeVisible(timeline);
    
    // Script editor
    scriptLabel.setText("Stage Script:", juce::dontSendNotification);
    scriptLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(scriptLabel);
    
    scriptEditor.setMultiLine(false);
    scriptEditor.setReturnKeyStartsNewLine(false);
    scriptEditor.setText("[INTRO:0:4] [VERSE:4:12] [HOOK:12:20] [VERSE:20:28] [HOOK:28:36] [OUTRO:36:40]");
    addAndMakeVisible(scriptEditor);
    
    parseScriptButton.setButtonText("Parse Script");
    parseScriptButton.onClick = [this]()
    {
        audioProcessor.getScriptParser().parseScript(scriptEditor.getText());
        audioProcessor.getAudioEngine().getArrangement().setScript(scriptEditor.getText());
    };
    addAndMakeVisible(parseScriptButton);
    
    // Undo/Redo buttons
    undoButton.setButtonText("Undo");
    undoButton.onClick = [this]()
    {
        audioProcessor.getUndoManager().undo();
    };
    addAndMakeVisible(undoButton);
    
    redoButton.setButtonText("Redo");
    redoButton.onClick = [this]()
    {
        audioProcessor.getUndoManager().redo();
    };
    addAndMakeVisible(redoButton);
}

MainComponent::~MainComponent()
{
}

void MainComponent::setupButton(juce::ToggleButton& button, const juce::String& text, const juce::String& paramID)
{
    button.setButtonText(text);
    addAndMakeVisible(button);
    
    auto attachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), paramID, button);
    buttonAttachments.push_back(std::move(attachment));
}

void MainComponent::setupSlider(juce::Slider& slider, juce::Label& label, 
                                const juce::String& text, const juce::String& paramID)
{
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
    addAndMakeVisible(slider);
    
    label.setText(text, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centred);
    label.attachToComponent(&slider, false);
    addAndMakeVisible(label);
    
    auto attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), paramID, slider);
    sliderAttachments.push_back(std::move(attachment));
}

void MainComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff2a2a2a));
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    // Title
    titleLabel.setBounds(bounds.removeFromTop(40));
    bounds.removeFromTop(10);
    
    // Undo/Redo buttons (top right)
    auto undoRedoBounds = bounds.removeFromTop(30);
    redoButton.setBounds(undoRedoBounds.removeFromRight(80));
    undoRedoBounds.removeFromRight(5);
    undoButton.setBounds(undoRedoBounds.removeFromRight(80));
    bounds.removeFromTop(10);
    
    // Script editor
    auto scriptBounds = bounds.removeFromTop(30);
    scriptLabel.setBounds(scriptBounds.removeFromLeft(100));
    parseScriptButton.setBounds(scriptBounds.removeFromRight(100));
    scriptBounds.removeFromLeft(5);
    scriptBounds.removeFromRight(5);
    scriptEditor.setBounds(scriptBounds);
    bounds.removeFromTop(10);
    
    // Timeline
    timeline.setBounds(bounds.removeFromTop(80));
    bounds.removeFromTop(10);
    
    // Control sections
    auto controlBounds = bounds;
    int sectionWidth = controlBounds.getWidth() / 4;
    
    // Instruments
    auto instBounds = controlBounds.removeFromLeft(sectionWidth).reduced(5);
    instrumentGroup.setBounds(instBounds);
    instBounds.reduce(10, 25);
    
    int buttonHeight = 30;
    enable808Button.setBounds(instBounds.removeFromTop(buttonHeight));
    instBounds.removeFromTop(5);
    enableHiHatButton.setBounds(instBounds.removeFromTop(buttonHeight));
    instBounds.removeFromTop(5);
    enableSnareButton.setBounds(instBounds.removeFromTop(buttonHeight));
    instBounds.removeFromTop(5);
    enablePianoButton.setBounds(instBounds.removeFromTop(buttonHeight));
    instBounds.removeFromTop(5);
    enableSynthButton.setBounds(instBounds.removeFromTop(buttonHeight));
    
    // Vocals
    auto vocalBounds = controlBounds.removeFromLeft(sectionWidth).reduced(5);
    vocalGroup.setBounds(vocalBounds);
    vocalBounds.reduce(10, 25);
    
    enableVocalsButton.setBounds(vocalBounds.removeFromTop(buttonHeight));
    vocalBounds.removeFromTop(15);
    
    int knobSize = 80;
    auto vocalKnobBounds = vocalBounds.removeFromTop(knobSize + 40);
    vocalPitchSlider.setBounds(vocalKnobBounds.removeFromLeft(vocalKnobBounds.getWidth() / 2).withSizeKeepingCentre(knobSize, knobSize + 40));
    vocalFormantSlider.setBounds(vocalKnobBounds.withSizeKeepingCentre(knobSize, knobSize + 40));
    
    // FX
    auto fxBounds = controlBounds.removeFromLeft(sectionWidth).reduced(5);
    fxGroup.setBounds(fxBounds);
    fxBounds.reduce(10, 25);
    
    auto fxRow1 = fxBounds.removeFromTop(knobSize + 40);
    reverbMixSlider.setBounds(fxRow1.removeFromLeft(fxRow1.getWidth() / 2).withSizeKeepingCentre(knobSize, knobSize + 40));
    delayMixSlider.setBounds(fxRow1.withSizeKeepingCentre(knobSize, knobSize + 40));
    
    auto fxRow2 = fxBounds.removeFromTop(knobSize + 40);
    distortionSlider.setBounds(fxRow2.removeFromLeft(fxRow2.getWidth() / 2).withSizeKeepingCentre(knobSize, knobSize + 40));
    aiFxMixSlider.setBounds(fxRow2.withSizeKeepingCentre(knobSize, knobSize + 40));
    
    // Master
    auto masterBounds = controlBounds.removeFromLeft(sectionWidth).reduced(5);
    masterGroup.setBounds(masterBounds);
    masterBounds.reduce(10, 25);
    
    auto masterKnobBounds = masterBounds.removeFromTop(knobSize + 40);
    masterGainSlider.setBounds(masterKnobBounds.removeFromLeft(masterKnobBounds.getWidth() / 2).withSizeKeepingCentre(knobSize, knobSize + 40));
    masterPanSlider.setBounds(masterKnobBounds.withSizeKeepingCentre(knobSize, knobSize + 40));
}
