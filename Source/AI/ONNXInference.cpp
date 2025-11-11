#include "ONNXInference.h"

// Only include ONNX headers if available
#ifdef ONNXRUNTIME_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

ONNXInference::ONNXInference()
{
#ifdef ONNXRUNTIME_AVAILABLE
    try {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MAEVNPlugin");
        sessionOptions = std::make_unique<Ort::SessionOptions>();
        sessionOptions->SetIntraOpNumThreads(1);
        sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    } catch (...) {
        juce::Logger::writeToLog("Failed to initialize ONNX Runtime");
    }
#else
    juce::Logger::writeToLog("ONNX Runtime not available - AI features disabled");
#endif
}

ONNXInference::~ONNXInference()
{
    releaseModel();
}

bool ONNXInference::loadModel(const juce::String& modelPath)
{
#ifdef ONNXRUNTIME_AVAILABLE
    try {
        if (!juce::File(modelPath).existsAsFile())
        {
            juce::Logger::writeToLog("Model file not found: " + modelPath);
            return false;
        }

        session = std::make_unique<Ort::Session>(*env, modelPath.toWideCharPointer(), *sessionOptions);
        modelLoaded = true;
        juce::Logger::writeToLog("ONNX model loaded successfully: " + modelPath);
        return true;
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Failed to load ONNX model: " + juce::String(e.what()));
        return false;
    }
#else
    juce::ignoreUnused(modelPath);
    juce::Logger::writeToLog("ONNX Runtime not available - cannot load model");
    return false;
#endif
}

void ONNXInference::releaseModel()
{
#ifdef ONNXRUNTIME_AVAILABLE
    session.reset();
#endif
    modelLoaded = false;
}

bool ONNXInference::inferAudio(const std::vector<float>& input, std::vector<float>& output)
{
#ifdef ONNXRUNTIME_AVAILABLE
    if (!modelLoaded || !session)
        return false;

    try {
        // Simple 1D audio processing: [batch, samples]
        std::vector<int64_t> inputShape = {1, static_cast<int64_t>(input.size())};
        std::vector<int64_t> outputShape;
        
        return infer(input, inputShape, output, outputShape);
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("ONNX inference failed: " + juce::String(e.what()));
        return false;
    }
#else
    juce::ignoreUnused(input, output);
    return false;
#endif
}

bool ONNXInference::infer(const std::vector<float>& input, 
                         const std::vector<int64_t>& inputShape,
                         std::vector<float>& output,
                         std::vector<int64_t>& outputShape)
{
#ifdef ONNXRUNTIME_AVAILABLE
    if (!modelLoaded || !session)
        return false;

    try {
        // Create input tensor
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(input.data()), input.size(),
            inputShape.data(), inputShape.size());

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        const char* inputName = session->GetInputName(0, allocator);
        const char* outputName = session->GetOutputName(0, allocator);

        // Run inference
        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr},
            &inputName, &inputTensor, 1,
            &outputName, 1);

        // Extract output
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto tensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        outputShape = tensorInfo.GetShape();
        
        size_t outputSize = 1;
        for (auto dim : outputShape)
            outputSize *= dim;
        
        output.assign(outputData, outputData + outputSize);
        
        return true;
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("ONNX inference failed: " + juce::String(e.what()));
        return false;
    }
#else
    juce::ignoreUnused(input, inputShape, output, outputShape);
    return false;
#endif
}
