#pragma once

#include <JuceHeader.h>

class InstrumentGenerator
{
public:
    InstrumentGenerator();
    ~InstrumentGenerator();

    void prepare(double sampleRate, int samplesPerBlock);
    void releaseResources();

    void process(juce::AudioBuffer<float>& buffer, 
                 juce::MidiBuffer& midiMessages,
                 bool enable808, bool enableHiHat, bool enableSnare,
                 bool enablePiano, bool enableSynth);

private:
    double currentSampleRate = 44100.0;
    
    // Trap instrument generators
    class Bass808
    {
    public:
        void prepare(double sampleRate);
        void process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi);
        void noteOn(int midiNote, float velocity);
        void noteOff();
    private:
        double sampleRate = 44100.0;
        double phase = 0.0;
        double frequency = 0.0;
        float velocity = 0.0f;
        float envelope = 0.0f;
        bool isPlaying = false;
    };

    class HiHat
    {
    public:
        void prepare(double sampleRate);
        void process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi);
        void trigger(float velocity);
    private:
        double sampleRate = 44100.0;
        juce::Random random;
        float envelope = 0.0f;
        float velocity = 0.0f;
    };

    class Snare
    {
    public:
        void prepare(double sampleRate);
        void process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi);
        void trigger(float velocity);
    private:
        double sampleRate = 44100.0;
        juce::Random random;
        float envelope = 0.0f;
        float velocity = 0.0f;
        double phase = 0.0;
    };

    class Piano
    {
    public:
        void prepare(double sampleRate);
        void process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi);
        void noteOn(int midiNote, float velocity);
        void noteOff();
    private:
        double sampleRate = 44100.0;
        double phase = 0.0;
        double frequency = 0.0;
        float velocity = 0.0f;
        float envelope = 0.0f;
        bool isPlaying = false;
    };

    class Synth
    {
    public:
        void prepare(double sampleRate);
        void process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi);
        void noteOn(int midiNote, float velocity);
        void noteOff();
    private:
        double sampleRate = 44100.0;
        double phase = 0.0;
        double lfoPhase = 0.0;
        double frequency = 0.0;
        float velocity = 0.0f;
        float envelope = 0.0f;
        bool isPlaying = false;
    };

    Bass808 bass808;
    HiHat hiHat;
    Snare snare;
    Piano piano;
    Synth synth;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InstrumentGenerator)
};
