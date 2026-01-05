#include "PluginProcessor.h"
#include "PluginEditor.h"

MAEVNAudioProcessor::MAEVNAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    parameters = std::make_unique<juce::AudioProcessorValueTreeState>(
        *this, &undoManager.getJuceUndoManager(), "Parameters", createParameterLayout());
}

MAEVNAudioProcessor::~MAEVNAudioProcessor()
{
}

juce::AudioProcessorValueTreeState::ParameterLayout MAEVNAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    // Master controls
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "masterGain", "Master Gain", 
        juce::NormalisableRange<float>(0.0f, 1.0f), 0.8f));
    
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "masterPan", "Master Pan", 
        juce::NormalisableRange<float>(-1.0f, 1.0f), 0.0f));

    // Instrument enables
    layout.add(std::make_unique<juce::AudioParameterBool>(
        "enable808", "Enable 808", true));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        "enableHiHat", "Enable Hi-Hat", true));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        "enableSnare", "Enable Snare", true));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        "enablePiano", "Enable Piano", true));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        "enableSynth", "Enable Synth", true));

    // Vocal parameters
    layout.add(std::make_unique<juce::AudioParameterBool>(
        "enableVocals", "Enable Vocals", true));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "vocalPitch", "Vocal Pitch", 
        juce::NormalisableRange<float>(-12.0f, 12.0f), 0.0f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "vocalFormant", "Vocal Formant", 
        juce::NormalisableRange<float>(0.5f, 2.0f), 1.0f));

    // FX parameters
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "reverbMix", "Reverb Mix", 
        juce::NormalisableRange<float>(0.0f, 1.0f), 0.2f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "delayMix", "Delay Mix", 
        juce::NormalisableRange<float>(0.0f, 1.0f), 0.15f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "distortion", "Distortion", 
        juce::NormalisableRange<float>(0.0f, 1.0f), 0.0f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "aiFxMix", "AI FX Mix", 
        juce::NormalisableRange<float>(0.0f, 1.0f), 0.0f));

    return layout;
}

const juce::String MAEVNAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool MAEVNAudioProcessor::acceptsMidi() const
{
    return true;
}

bool MAEVNAudioProcessor::producesMidi() const
{
    return false;
}

bool MAEVNAudioProcessor::isMidiEffect() const
{
    return false;
}

double MAEVNAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int MAEVNAudioProcessor::getNumPrograms()
{
    return 1;
}

int MAEVNAudioProcessor::getCurrentProgram()
{
    return 0;
}

void MAEVNAudioProcessor::setCurrentProgram(int index)
{
    juce::ignoreUnused(index);
}

const juce::String MAEVNAudioProcessor::getProgramName(int index)
{
    juce::ignoreUnused(index);
    return {};
}

void MAEVNAudioProcessor::changeProgramName(int index, const juce::String& newName)
{
    juce::ignoreUnused(index, newName);
}

void MAEVNAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    // Check if we're running in standalone mode
    #if JucePlugin_Build_Standalone
    // Perform 3 sequential prepare calls for plugin readiness in standalone mode
    if (!isStandaloneInitialized)
    {
        // Call prepare 3 times to ensure all resources are fully initialized
        for (int i = 0; i < 3; ++i)
        {
            audioEngine.prepare(sampleRate, samplesPerBlock);
            prepareCallCount++;
        }
        
        isStandaloneInitialized = true;
        
        juce::Logger::writeToLog("MAEVN Standalone: Plugin readiness initialized with " 
                                 + juce::String(prepareCallCount) + " prepare calls");
    }
    else
    {
        // Normal prepare call after initialization
        audioEngine.prepare(sampleRate, samplesPerBlock);
    }
    #else
    // VST3 mode - normal single prepare call
    audioEngine.prepare(sampleRate, samplesPerBlock);
    #endif
}

void MAEVNAudioProcessor::releaseResources()
{
    audioEngine.releaseResources();
    
    // Reset standalone initialization flag so next prepareToPlay will perform 3 prepare calls again
    #if JucePlugin_Build_Standalone
    isStandaloneInitialized = false;
    prepareCallCount = 0;
    #endif
}

bool MAEVNAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    return true;
}

void MAEVNAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                       juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    
    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // Get host time info for DAW sync
    auto playHead = getPlayHead();
    if (playHead != nullptr)
    {
        juce::AudioPlayHead::CurrentPositionInfo positionInfo;
        if (playHead->getCurrentPosition(positionInfo))
        {
            audioEngine.updatePlaybackPosition(positionInfo);
        }
    }

    // Process audio through the engine
    audioEngine.process(buffer, midiMessages, *parameters);
}

bool MAEVNAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* MAEVNAudioProcessor::createEditor()
{
    return new MAEVNAudioProcessorEditor(*this);
}

void MAEVNAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = parameters->copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void MAEVNAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName(parameters->state.getType()))
            parameters->replaceState(juce::ValueTree::fromXml(*xmlState));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MAEVNAudioProcessor();
}
