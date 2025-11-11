#include "AIEffects.h"

AIEffects::AIEffects()
{
}

AIEffects::~AIEffects()
{
}

void AIEffects::prepare(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentSamplesPerBlock = samplesPerBlock;
    
    dryBuffer.setSize(2, samplesPerBlock);
}

void AIEffects::releaseResources()
{
    aiModel.releaseModel();
}

bool AIEffects::loadModel(const juce::String& modelPath)
{
    return aiModel.loadModel(modelPath);
}

void AIEffects::process(juce::AudioBuffer<float>& buffer, float mix)
{
    if (!aiModel.isModelLoaded() || mix < 0.01f)
        return;

    // Store dry signal
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        dryBuffer.copyFrom(ch, 0, buffer, ch, 0, buffer.getNumSamples());

    // Prepare input for AI model
    std::vector<float> input;
    input.reserve(buffer.getNumSamples() * buffer.getNumChannels());
    
    // Interleave channels for model input
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
    {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            input.push_back(buffer.getSample(ch, sample));
        }
    }

    // Run AI inference
    std::vector<float> output;
    if (aiModel.inferAudio(input, output))
    {
        // De-interleave output back to buffer
        int outputIndex = 0;
        for (int sample = 0; sample < buffer.getNumSamples() && outputIndex < output.size(); ++sample)
        {
            for (int ch = 0; ch < buffer.getNumChannels() && outputIndex < output.size(); ++ch)
            {
                buffer.setSample(ch, sample, output[outputIndex++]);
            }
        }

        // Mix wet and dry signals
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            auto* wet = buffer.getWritePointer(ch);
            auto* dry = dryBuffer.getReadPointer(ch);
            
            for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
                wet[sample] = dry[sample] * (1.0f - mix) + wet[sample] * mix;
        }
    }
    else
    {
        // If inference fails, pass through dry signal
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            buffer.copyFrom(ch, 0, dryBuffer, ch, 0, buffer.getNumSamples());
    }
}
