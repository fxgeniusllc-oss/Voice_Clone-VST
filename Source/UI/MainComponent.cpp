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
    
    // Voice model management
    setupVoiceModelControls();
    
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
    vocalBounds.removeFromTop(10);
    
    // Voice model controls
    voiceModelLabel.setBounds(vocalBounds.removeFromTop(20));
    vocalBounds.removeFromTop(2);
    voiceModelSelector.setBounds(vocalBounds.removeFromTop(24));
    vocalBounds.removeFromTop(5);
    
    auto buttonRow = vocalBounds.removeFromTop(28);
    loadModelButton.setBounds(buttonRow.removeFromLeft(buttonRow.getWidth() / 2 - 2));
    buttonRow.removeFromLeft(4);
    cloneVoiceButton.setBounds(buttonRow);
    vocalBounds.removeFromTop(5);
    
    modelStatusLabel.setBounds(vocalBounds.removeFromTop(20));
    vocalBounds.removeFromTop(10);
    
    // Pitch and formant knobs
    int knobSize = 70;
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

// Voice model management implementation
void MainComponent::setupVoiceModelControls()
{
    // Voice model label
    voiceModelLabel.setText("Voice Model:", juce::dontSendNotification);
    voiceModelLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(voiceModelLabel);
    
    // Voice model selector
    voiceModelSelector.setTextWhenNothingSelected("Select a voice model...");
    voiceModelSelector.setTextWhenNoChoicesAvailable("No models available");
    voiceModelSelector.onChange = [this]() { onVoiceModelSelected(); };
    addAndMakeVisible(voiceModelSelector);
    
    // Load model button
    loadModelButton.setButtonText("Load Model");
    loadModelButton.setTooltip("Browse and load a custom voice model file");
    loadModelButton.onClick = [this]() { onLoadModelButtonClicked(); };
    addAndMakeVisible(loadModelButton);
    
    // Clone voice button
    cloneVoiceButton.setButtonText("Clone Voice");
    cloneVoiceButton.setTooltip("Record audio to create a new voice clone");
    cloneVoiceButton.onClick = [this]() { onCloneVoiceButtonClicked(); };
    addAndMakeVisible(cloneVoiceButton);
    
    // Model status label
    modelStatusLabel.setText("Status: No model loaded", juce::dontSendNotification);
    modelStatusLabel.setJustificationType(juce::Justification::centredLeft);
    modelStatusLabel.setFont(juce::Font(12.0f));
    modelStatusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(modelStatusLabel);
    
    // Load available models
    loadAvailableVoiceModels();
}

void MainComponent::loadAvailableVoiceModels()
{
    voiceModelSelector.clear(juce::dontSendNotification);
    
    // Add default/built-in models
    voiceModelSelector.addItem("Default Voice (DSP Formant)", 1);
    voiceModelSelector.addItem("Neural TTS Model", 2);
    
    // Scan Models/vocals directory for available models
    juce::File modelsDir = juce::File::getCurrentWorkingDirectory().getChildFile("Models").getChildFile("vocals");
    
    if (modelsDir.exists() && modelsDir.isDirectory())
    {
        int itemId = 3;
        auto files = modelsDir.findChildFiles(juce::File::findFiles, false, "*.onnx");
        
        for (const auto& file : files)
        {
            voiceModelSelector.addItem(file.getFileNameWithoutExtension(), itemId++);
        }
    }
    
    // Set default selection
    voiceModelSelector.setSelectedId(1, juce::dontSendNotification);
    updateModelStatus("Using default DSP formant synthesis");
}

void MainComponent::onLoadModelButtonClicked()
{
    // Create file chooser for loading ONNX models
    auto chooser = std::make_unique<juce::FileChooser>(
        "Select a voice model file",
        juce::File::getCurrentWorkingDirectory().getChildFile("Models").getChildFile("vocals"),
        "*.onnx"
    );
    
    auto chooserFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    
    chooser->launchAsync(chooserFlags, [this](const juce::FileChooser& fc)
    {
        auto file = fc.getResult();
        
        if (file.existsAsFile())
        {
            // Load TTS model
            bool ttsLoaded = audioProcessor.getAudioEngine().getVocalSynthesis()
                .loadTTSModel(file.getFullPathName());
            
            if (ttsLoaded)
            {
                // Try to find corresponding vocoder model
                juce::File vocoderFile = file.getSiblingFile("vocals_hifigan.onnx");
                if (vocoderFile.existsAsFile())
                {
                    audioProcessor.getAudioEngine().getVocalSynthesis()
                        .loadVocoderModel(vocoderFile.getFullPathName());
                }
                
                updateModelStatus("Loaded: " + file.getFileName());
                
                // Add to selector if not already present
                bool alreadyPresent = false;
                for (int i = 0; i < voiceModelSelector.getNumItems(); ++i)
                {
                    if (voiceModelSelector.getItemText(i) == file.getFileNameWithoutExtension())
                    {
                        alreadyPresent = true;
                        voiceModelSelector.setSelectedItemIndex(i);
                        break;
                    }
                }
                
                if (!alreadyPresent)
                {
                    int newId = voiceModelSelector.getNumItems() + 1;
                    voiceModelSelector.addItem(file.getFileNameWithoutExtension(), newId);
                    voiceModelSelector.setSelectedId(newId);
                }
            }
            else
            {
                updateModelStatus("Failed to load model");
                juce::AlertWindow::showMessageBoxAsync(
                    juce::AlertWindow::WarningIcon,
                    "Model Load Error",
                    "Failed to load the voice model. Please check the file format.",
                    "OK"
                );
            }
        }
    });
}

void MainComponent::onCloneVoiceButtonClicked()
{
    // Show dialog for voice cloning
    juce::AlertWindow::showMessageBoxAsync(
        juce::AlertWindow::InfoIcon,
        "Voice Cloning",
        "Voice cloning feature:\n\n"
        "To clone a voice, you need to:\n"
        "1. Record 5-10 minutes of clean audio samples\n"
        "2. Use external voice cloning tools (RVC, So-VITS-SVC, etc.)\n"
        "3. Export the trained model to ONNX format\n"
        "4. Load the model using the 'Load Model' button\n\n"
        "Quick clone from recording is a planned feature for future releases.\n\n"
        "For now, you can:\n"
        "• Use pre-trained models from Hugging Face\n"
        "• Train models using TensorFlow/PyTorch\n"
        "• Export to ONNX and load here",
        "OK"
    );
}

void MainComponent::onVoiceModelSelected()
{
    int selectedId = voiceModelSelector.getSelectedId();
    juce::String selectedText = voiceModelSelector.getText();
    
    if (selectedId == 1)
    {
        // Default DSP formant synthesis
        updateModelStatus("Using default DSP formant synthesis");
    }
    else if (selectedId == 2)
    {
        // Neural TTS model
        juce::File ttsFile = juce::File::getCurrentWorkingDirectory()
            .getChildFile("Models").getChildFile("vocals").getChildFile("vocals_tts.onnx");
        juce::File vocoderFile = juce::File::getCurrentWorkingDirectory()
            .getChildFile("Models").getChildFile("vocals").getChildFile("vocals_hifigan.onnx");
        
        if (ttsFile.existsAsFile() && vocoderFile.existsAsFile())
        {
            bool loaded = audioProcessor.getAudioEngine().getVocalSynthesis()
                .loadTTSModel(ttsFile.getFullPathName());
            loaded &= audioProcessor.getAudioEngine().getVocalSynthesis()
                .loadVocoderModel(vocoderFile.getFullPathName());
            
            if (loaded)
            {
                updateModelStatus("Loaded: Neural TTS + HiFi-GAN");
            }
            else
            {
                updateModelStatus("Failed to load neural models");
            }
        }
        else
        {
            updateModelStatus("Neural models not found - using DSP fallback");
        }
    }
    else
    {
        // Custom model selected
        updateModelStatus("Selected: " + selectedText);
    }
}

void MainComponent::updateModelStatus(const juce::String& status)
{
    modelStatusLabel.setText("Status: " + status, juce::dontSendNotification);
}

