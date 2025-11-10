#include "PluginEditor.h"

MAEVNAudioProcessorEditor::MAEVNAudioProcessorEditor(MAEVNAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p), mainComponent(p)
{
    addAndMakeVisible(mainComponent);
    setSize(900, 600);
    setResizable(true, true);
    setResizeLimits(800, 500, 1600, 1200);
}

MAEVNAudioProcessorEditor::~MAEVNAudioProcessorEditor()
{
}

void MAEVNAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff1a1a1a));
}

void MAEVNAudioProcessorEditor::resized()
{
    mainComponent.setBounds(getLocalBounds());
}
