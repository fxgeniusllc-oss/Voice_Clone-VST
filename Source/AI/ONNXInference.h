#pragma once

#include <JuceHeader.h>
#include <memory>
#include <vector>

// Forward declare ONNX Runtime types to avoid requiring headers at compile time
namespace Ort {
    class Env;
    class Session;
    class SessionOptions;
    class Value;
}

class ONNXInference
{
public:
    ONNXInference();
    ~ONNXInference();

    bool loadModel(const juce::String& modelPath);
    void releaseModel();
    
    bool isModelLoaded() const { return modelLoaded; }
    
    // Run inference with audio buffer
    bool inferAudio(const std::vector<float>& input, std::vector<float>& output);
    
    // Run inference with arbitrary tensor shapes
    bool infer(const std::vector<float>& input, 
               const std::vector<int64_t>& inputShape,
               std::vector<float>& output,
               std::vector<int64_t>& outputShape);

private:
    bool modelLoaded = false;
    
    // ONNX Runtime objects (using forward declaration)
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ONNXInference)
};
