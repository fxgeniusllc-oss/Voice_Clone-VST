#include "InstrumentGenerator.h"

InstrumentGenerator::InstrumentGenerator()
{
}

InstrumentGenerator::~InstrumentGenerator()
{
}

void InstrumentGenerator::prepare(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(samplesPerBlock);
    currentSampleRate = sampleRate;
    
    bass808.prepare(sampleRate);
    hiHat.prepare(sampleRate);
    snare.prepare(sampleRate);
    piano.prepare(sampleRate);
    synth.prepare(sampleRate);
}

void InstrumentGenerator::releaseResources()
{
}

void InstrumentGenerator::process(juce::AudioBuffer<float>& buffer, 
                                  juce::MidiBuffer& midiMessages,
                                  bool enable808, bool enableHiHat, bool enableSnare,
                                  bool enablePiano, bool enableSynth)
{
    if (enable808)
        bass808.process(buffer, midiMessages);
    
    if (enableHiHat)
        hiHat.process(buffer, midiMessages);
    
    if (enableSnare)
        snare.process(buffer, midiMessages);
    
    if (enablePiano)
        piano.process(buffer, midiMessages);
    
    if (enableSynth)
        synth.process(buffer, midiMessages);
}

// Bass808 implementation
void InstrumentGenerator::Bass808::prepare(double sr)
{
    sampleRate = sr;
}

void InstrumentGenerator::Bass808::process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    for (const auto metadata : midi)
    {
        auto message = metadata.getMessage();
        if (message.isNoteOn())
            noteOn(message.getNoteNumber(), message.getFloatVelocity());
        else if (message.isNoteOff())
            noteOff();
    }

    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        auto* data = buffer.getWritePointer(channel);
        
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        {
            if (isPlaying && envelope > 0.0001f)
            {
                // Generate 808 bass with pitch sweep
                double pitchEnv = std::exp(-sample / (sampleRate * 0.1));
                double currentFreq = frequency * (1.0 + 2.0 * pitchEnv);
                
                double sine = std::sin(phase);
                double output = sine * envelope * velocity;
                
                data[sample] += static_cast<float>(output * 0.5);
                
                phase += 2.0 * juce::MathConstants<double>::pi * currentFreq / sampleRate;
                if (phase > 2.0 * juce::MathConstants<double>::pi)
                    phase -= 2.0 * juce::MathConstants<double>::pi;
                
                envelope *= 0.9998f; // Decay
            }
        }
    }
}

void InstrumentGenerator::Bass808::noteOn(int midiNote, float vel)
{
    frequency = juce::MidiMessage::getMidiNoteInHertz(midiNote);
    velocity = vel;
    envelope = 1.0f;
    phase = 0.0;
    isPlaying = true;
}

void InstrumentGenerator::Bass808::noteOff()
{
    // 808 continues to ring out
}

// HiHat implementation
void InstrumentGenerator::HiHat::prepare(double sr)
{
    sampleRate = sr;
}

void InstrumentGenerator::HiHat::process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    for (const auto metadata : midi)
    {
        auto message = metadata.getMessage();
        if (message.isNoteOn() && message.getNoteNumber() >= 42 && message.getNoteNumber() <= 46)
            trigger(message.getFloatVelocity());
    }

    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        auto* data = buffer.getWritePointer(channel);
        
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        {
            if (envelope > 0.0001f)
            {
                // Generate noise for hi-hat
                float noise = random.nextFloat() * 2.0f - 1.0f;
                float output = noise * envelope * velocity;
                
                data[sample] += output * 0.3f;
                
                envelope *= 0.995f; // Fast decay
            }
        }
    }
}

void InstrumentGenerator::HiHat::trigger(float vel)
{
    velocity = vel;
    envelope = 1.0f;
}

// Snare implementation
void InstrumentGenerator::Snare::prepare(double sr)
{
    sampleRate = sr;
}

void InstrumentGenerator::Snare::process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    for (const auto metadata : midi)
    {
        auto message = metadata.getMessage();
        if (message.isNoteOn() && message.getNoteNumber() >= 38 && message.getNoteNumber() <= 40)
            trigger(message.getFloatVelocity());
    }

    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        auto* data = buffer.getWritePointer(channel);
        
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        {
            if (envelope > 0.0001f)
            {
                // Mix noise and tone for snare
                float noise = random.nextFloat() * 2.0f - 1.0f;
                double tone = std::sin(phase);
                float output = (noise * 0.6f + tone * 0.4f) * envelope * velocity;
                
                data[sample] += output * 0.4f;
                
                phase += 2.0 * juce::MathConstants<double>::pi * 200.0 / sampleRate;
                if (phase > 2.0 * juce::MathConstants<double>::pi)
                    phase -= 2.0 * juce::MathConstants<double>::pi;
                
                envelope *= 0.996f; // Medium decay
            }
        }
    }
}

void InstrumentGenerator::Snare::trigger(float vel)
{
    velocity = vel;
    envelope = 1.0f;
    phase = 0.0;
}

// Piano implementation
void InstrumentGenerator::Piano::prepare(double sr)
{
    sampleRate = sr;
}

void InstrumentGenerator::Piano::process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    for (const auto metadata : midi)
    {
        auto message = metadata.getMessage();
        if (message.isNoteOn())
            noteOn(message.getNoteNumber(), message.getFloatVelocity());
        else if (message.isNoteOff())
            noteOff();
    }

    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        auto* data = buffer.getWritePointer(channel);
        
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        {
            if (isPlaying && envelope > 0.0001f)
            {
                // Simple piano-like tone with harmonics
                double fundamental = std::sin(phase);
                double harmonic2 = std::sin(phase * 2.0) * 0.5;
                double harmonic3 = std::sin(phase * 3.0) * 0.25;
                
                double output = (fundamental + harmonic2 + harmonic3) * envelope * velocity;
                
                data[sample] += static_cast<float>(output * 0.3);
                
                phase += 2.0 * juce::MathConstants<double>::pi * frequency / sampleRate;
                if (phase > 2.0 * juce::MathConstants<double>::pi)
                    phase -= 2.0 * juce::MathConstants<double>::pi;
                
                envelope *= 0.9995f; // Slow decay
            }
        }
    }
}

void InstrumentGenerator::Piano::noteOn(int midiNote, float vel)
{
    frequency = juce::MidiMessage::getMidiNoteInHertz(midiNote);
    velocity = vel;
    envelope = 1.0f;
    phase = 0.0;
    isPlaying = true;
}

void InstrumentGenerator::Piano::noteOff()
{
    isPlaying = false;
}

// Synth implementation
void InstrumentGenerator::Synth::prepare(double sr)
{
    sampleRate = sr;
}

void InstrumentGenerator::Synth::process(juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    for (const auto metadata : midi)
    {
        auto message = metadata.getMessage();
        if (message.isNoteOn())
            noteOn(message.getNoteNumber(), message.getFloatVelocity());
        else if (message.isNoteOff())
            noteOff();
    }

    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        auto* data = buffer.getWritePointer(channel);
        
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        {
            if (isPlaying && envelope > 0.0001f)
            {
                // Synth with LFO modulation
                double lfo = std::sin(lfoPhase) * 0.1 + 1.0;
                double sawWave = (phase / juce::MathConstants<double>::pi) - 1.0;
                double output = sawWave * lfo * envelope * velocity;
                
                data[sample] += static_cast<float>(output * 0.25);
                
                phase += 2.0 * juce::MathConstants<double>::pi * frequency / sampleRate;
                if (phase > 2.0 * juce::MathConstants<double>::pi)
                    phase -= 2.0 * juce::MathConstants<double>::pi;
                
                lfoPhase += 2.0 * juce::MathConstants<double>::pi * 5.0 / sampleRate;
                if (lfoPhase > 2.0 * juce::MathConstants<double>::pi)
                    lfoPhase -= 2.0 * juce::MathConstants<double>::pi;
                
                envelope *= 0.9997f; // Sustained decay
            }
        }
    }
}

void InstrumentGenerator::Synth::noteOn(int midiNote, float vel)
{
    frequency = juce::MidiMessage::getMidiNoteInHertz(midiNote);
    velocity = vel;
    envelope = 1.0f;
    phase = 0.0;
    lfoPhase = 0.0;
    isPlaying = true;
}

void InstrumentGenerator::Synth::noteOff()
{
    isPlaying = false;
}
