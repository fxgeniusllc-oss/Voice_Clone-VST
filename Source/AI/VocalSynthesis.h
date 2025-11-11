#pragma once

#include <JuceHeader.h>
#include "ONNXInference.h"

class VocalSynthesis
{
public:
    VocalSynthesis();
    ~VocalSynthesis();

    void prepare(double sampleRate, int samplesPerBlock);
    void releaseResources();

    void process(juce::AudioBuffer<float>& buffer, float pitch, float formant);

    // Set text for TTS (text-to-speech)
    void setText(const juce::String& text);
    
    // Load TTS and vocoder models
    bool loadTTSModel(const juce::String& modelPath);
    bool loadVocoderModel(const juce::String& modelPath);

private:
    double currentSampleRate = 44100.0;
    int currentSamplesPerBlock = 512;

    ONNXInference ttsModel;
    ONNXInference vocoderModel;
    
    juce::String currentText;
    std::vector<float> vocalbuffer;
    int vocalPlaybackPosition = 0;
    
    // Simple vocoder simulation (when ONNX models not available)
    void generateSimpleVocal(juce::AudioBuffer<float>& buffer, float pitch, float formant);
    
    double phase = 0.0;
    juce::Random random;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VocalSynthesis)
};
