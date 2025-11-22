#pragma once

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>
#include <vector>

// Forward declare ONNX Runtime types when available
#ifdef ONNXRUNTIME_AVAILABLE
namespace Ort {
    class Env;
    class Session;
    class SessionOptions;
    class Value;
}
#endif

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
    
#ifdef ONNXRUNTIME_AVAILABLE
    // ONNX Runtime objects (only when available)
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
#endif

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ONNXInference)
};
