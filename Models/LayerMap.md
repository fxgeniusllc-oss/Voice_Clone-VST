# 🧠 ONNX Layer Map - Explainability Documentation

## Overview

This document provides detailed explanations of the neural network architectures used in MAEVN's ONNX models. Understanding each layer's purpose is crucial for debugging, optimization, and maintaining deterministic inference.

---

## 🎤 Vocal Models

### 1. Vocal TTS (vocals_tts.onnx)

**Architecture**: Tacotron2-based Text-to-Speech

**Purpose**: Convert text/phoneme input into mel-spectrogram representations of speech

#### Layer-by-Layer Breakdown

##### Input Layer
- **Name**: `text_input`
- **Shape**: `[batch_size, max_seq_length]`
- **Type**: Integer (phoneme indices)
- **Purpose**: Encoded phoneme sequence representing input text
- **Range**: 0-255 (phoneme vocabulary size)

##### Embedding Layer
- **Name**: `phoneme_embedding`
- **Shape**: `[batch_size, max_seq_length, 512]`
- **Type**: Learned embeddings
- **Purpose**: Convert discrete phoneme indices to continuous vector representations
- **Notes**: 512-dimensional embeddings allow rich phonetic representation

##### Encoder (3 Convolutional Layers)
- **Layers**: `encoder_conv1`, `encoder_conv2`, `encoder_conv3`
- **Kernel Size**: 5x1
- **Channels**: 512 → 512 → 512
- **Activation**: ReLU + Batch Normalization + Dropout(0.5)
- **Purpose**: Extract high-level phonetic features and temporal context
- **Notes**: Convolutional structure captures local phonetic patterns

##### Bidirectional LSTM Encoder
- **Name**: `encoder_lstm`
- **Hidden Size**: 512
- **Layers**: 1 bidirectional layer (forward + backward)
- **Purpose**: Capture long-range dependencies in speech sequence
- **Output Shape**: `[batch_size, seq_length, 512]`
- **Notes**: Bidirectional allows context from both past and future phonemes

##### Attention Mechanism
- **Type**: Location-Sensitive Attention
- **Components**:
  - **Location Features**: `attention_location_conv` (32 filters, kernel 31)
  - **Query Projection**: `attention_query` (128 dims)
  - **Memory Projection**: `attention_memory` (128 dims)
  - **Energy Computation**: `attention_v` (scalar output per position)
- **Purpose**: Align encoder outputs with decoder time steps
- **Notes**: Location-sensitivity prevents attention from skipping or repeating

##### Decoder (2-layer LSTM + Prenet)
- **Prenet**:
  - **Layers**: `prenet_fc1` (256), `prenet_fc2` (256)
  - **Activation**: ReLU + Dropout(0.5)
  - **Purpose**: Regularize decoder inputs
  
- **LSTM Decoder**:
  - **Name**: `decoder_lstm1`, `decoder_lstm2`
  - **Hidden Size**: 1024 per layer
  - **Purpose**: Generate mel-spectrogram frame by frame
  
##### Output Projection
- **Name**: `mel_projection`
- **Output Shape**: `[batch_size, time_steps, 80]`
- **Purpose**: Project decoder hidden state to mel-spectrogram
- **Notes**: 80 mel bins standard for speech synthesis

##### Postnet (5 Convolutional Layers)
- **Layers**: `postnet_conv1` through `postnet_conv5`
- **Kernel Size**: 5x1
- **Channels**: 512 → 512 → 512 → 512 → 80
- **Activation**: Tanh (intermediate), Linear (final)
- **Purpose**: Refine mel-spectrogram prediction, add residual corrections
- **Notes**: Residual connection improves fine details

##### Stop Token Prediction
- **Name**: `stop_token_projection`
- **Output Shape**: `[batch_size, time_steps, 1]`
- **Purpose**: Predict when to stop generating (sigmoid output)
- **Threshold**: 0.5

**Total Parameters**: ~13M  
**Inference Time**: ~150ms per sentence (CPU)  
**Determinism**: Yes (no stochastic layers at inference)

---

### 2. Vocal HiFi-GAN (vocals_hifigan.onnx)

**Architecture**: HiFi-GAN V1 Vocoder

**Purpose**: Convert mel-spectrogram to high-fidelity audio waveform

#### Layer-by-Layer Breakdown

##### Input Layer
- **Name**: `mel_input`
- **Shape**: `[batch_size, 80, time_frames]`
- **Type**: Float32 (mel-spectrogram)
- **Range**: Typically [-11.5, 2.5] (log scale)

##### Initial Convolution
- **Name**: `input_conv`
- **Kernel Size**: 7
- **Padding**: Reflection padding (3)
- **Channels**: 80 → 512
- **Purpose**: Initial feature extraction from mel-spectrogram

##### Upsampling Blocks (4 blocks)
Each block contains:

**Block 1** (Upsample 8x):
- **Transpose Conv**: `upsample_conv1`
  - Kernel: 16, Stride: 8, Padding: 4
  - Channels: 512 → 256
- **Multi-Receptive Field Fusion (MRF)**:
  - 3 parallel branches with dilated convolutions
  - Kernels: 3, 7, 11 with varying dilations
  - Purpose: Capture patterns at multiple time scales
  
**Block 2** (Upsample 8x):
- Channels: 256 → 128
- Similar MRF structure

**Block 3** (Upsample 2x):
- Channels: 128 → 64
- Similar MRF structure

**Block 4** (Upsample 2x):
- Channels: 64 → 32
- Similar MRF structure

**Total Upsampling**: 8 × 8 × 2 × 2 = 256x

##### Output Convolution
- **Name**: `output_conv`
- **Kernel Size**: 7
- **Activation**: Tanh
- **Channels**: 32 → 1
- **Output Range**: [-1, 1] (audio waveform)

##### Activation Functions
- **LeakyReLU**: Slope 0.1 (prevents dead neurons)
- **Tanh**: Final output activation

**Total Parameters**: ~1.5M  
**Inference Time**: ~25ms per second of audio (CPU)  
**Determinism**: Yes (fully convolutional, no dropout)

---

## 🥁 Drum Models (DDSP Architecture)

### DDSP Shared Architecture

All drum models (808, hi-hat, snare) use Differentiable Digital Signal Processing (DDSP) architecture.

#### Common Structure

##### Input Layer
- **Name**: `control_params`
- **Shape**: `[batch_size, num_frames, num_params]`
- **Type**: Float32 control signals
- **Parameters vary by instrument**

##### Encoder Network
- **Layers**: `encoder_fc1` (512), `encoder_fc2` (512), `encoder_fc3` (256)
- **Activation**: ReLU + Layer Normalization
- **Purpose**: Process control parameters into internal representation

##### Harmonic Synthesizer
- **Component**: `harmonic_synth`
- **Parameters**:
  - **Amplitudes**: Per-harmonic amplitude control (100 harmonics)
  - **Harmonic Distribution**: Learned harmonic structure
  - **F0 Contour**: Fundamental frequency over time
- **Purpose**: Generate harmonic content via additive synthesis
- **Output**: Sum of sinusoids

##### Filtered Noise Synthesizer
- **Component**: `noise_synth`
- **Filter Type**: Time-varying FIR filter
- **Filter Length**: 65 samples
- **Purpose**: Generate noise component (essential for drums)
- **Output**: Filtered white noise

##### Reverb Module
- **Type**: Differentiable reverb
- **Parameters**: Room size, damping, mix
- **Purpose**: Add spatial characteristics
- **Implementation**: Learned FIR filter (2048 taps)

##### Mixer
- **Combines**:
  - Harmonic synthesis output
  - Filtered noise output
  - Reverb
- **Learnable weights** for each component

##### Output
- **Sample Rate**: 44.1 kHz
- **Bit Depth**: 32-bit float
- **Range**: [-1, 1]

---

### 3. 808 Bass (808_ddsp.onnx)

**Specific Parameters**:
- `pitch`: MIDI note (20-60, sub-bass range)
- `amplitude`: Volume (0-1)
- `decay`: Decay time in seconds (0.1-3.0)
- `distortion`: Saturation amount (0-1)

**Unique Characteristics**:
- **Heavy harmonic content**: First 5 harmonics emphasized
- **Long decay**: Sustain component up to 3 seconds
- **Sub-bass focus**: Fundamental typically 40-80 Hz
- **Distortion module**: Soft-clipping saturation

**Inference Time**: ~5ms  
**Parameters**: ~500K

---

### 4. Hi-Hat (hihat_ddsp.onnx)

**Specific Parameters**:
- `brightness`: Spectral centroid control (0-1)
- `decay`: Release time (0.05-0.5s)
- `pitch`: Tonal center (0-1, typically high)

**Unique Characteristics**:
- **Noise-dominant**: 90% noise, 10% harmonics
- **Fast decay**: Typically 50-200ms
- **High-frequency emphasis**: Energy centered 6-12 kHz
- **Brightness control**: Adjusts filter cutoff

**Inference Time**: ~3ms  
**Parameters**: ~300K

---

### 5. Snare (snare_ddsp.onnx)

**Specific Parameters**:
- `pitch`: Fundamental frequency (150-300 Hz)
- `snap`: Transient intensity (0-1)
- `body`: Tonal component mix (0-1)
- `decay`: Release time (0.1-0.8s)

**Unique Characteristics**:
- **Dual component**: Tonal body + noise snare
- **Transient emphasis**: Attack time < 1ms
- **Body resonance**: Modal synthesis for shell
- **Snare buzz**: Filtered noise for snare wires

**Inference Time**: ~4ms  
**Parameters**: ~400K

---

## 🎹 Instrument Models

### 6. Piano (piano_ddsp.onnx)

**Architecture**: DDSP with Modal Synthesis

**Parameters**:
- `pitch`: MIDI note (21-108, full piano range)
- `velocity`: Strike velocity (0-127)
- `sustain`: Pedal state (0-1)

#### Layer Structure

##### Input Processing
- **Velocity Curves**: Non-linear mapping of velocity to amplitude
- **Pitch-Dependent Timbre**: Different synthesis per register

##### Modal Synthesis Engine
- **Modes per Note**: 12 modal frequencies
- **Mode Parameters**:
  - Frequency (harmonic + inharmonic)
  - Decay time (varies by register)
  - Amplitude
- **Purpose**: Simulate string vibrations

##### Hammer Impact Model
- **Attack Transient**: 5ms impact simulation
- **Spectral Tilt**: Velocity-dependent brightness

##### Soundboard Resonance
- **Impulse Response**: 4096-sample learned IR
- **Purpose**: Add body resonance and room interaction

##### Sympathetic Resonance
- **Cross-coupling**: Adjacent strings vibrate
- **Sustain Pedal**: Sustain allows all strings to resonate

**Inference Time**: ~8ms  
**Parameters**: ~1.2M

---

### 7. FM Synth (synth_fm.onnx)

**Architecture**: Neural FM Synthesis

**Parameters**:
- `carrier_freq`: Base frequency (20-20000 Hz)
- `modulator_freq`: Modulation frequency (ratio of carrier)
- `mod_index`: Modulation depth (0-20)
- `envelope`: ADSR parameters (4D vector)

#### Layer Structure

##### Operator Network
- **Carrier Oscillator**: Sine wave generation
- **Modulator Oscillator**: FM modulation source
- **FM Algorithm**: Phase modulation

##### Envelope Generator
- **ADSR**: Attack, Decay, Sustain, Release
- **Neural Network**: 3-layer MLP for envelope shaping
- **Per-Parameter Control**: Independent envelopes

##### Filter Section
- **Type**: Learned time-varying filter
- **Cutoff Modulation**: Envelope and velocity sensitive
- **Resonance**: Adjustable Q factor

##### Effects Chain
- **Chorus**: Slight detuning effect
- **Reverb**: Spatial ambience

**Inference Time**: ~6ms  
**Parameters**: ~800K

---

## 🔍 Model Optimization Notes

### Real-Time Requirements

All models must meet:
- **Latency**: < 10ms per inference call
- **Buffer Stability**: Consistent timing, no dropouts
- **CPU Usage**: < 20% per model instance
- **Memory**: < 100MB per model

### Optimization Techniques Used

1. **Quantization**: INT8 where possible (not for drums due to quality loss)
2. **Operator Fusion**: Combine consecutive operations
3. **Graph Optimization**: Remove unnecessary nodes
4. **SIMD Instructions**: Leverage CPU vector operations
5. **Batching**: Process multiple frames together when possible

### Debugging Tips

- **Check Input Ranges**: Models expect normalized inputs
- **Validate Shapes**: ONNX is strict about tensor dimensions
- **Monitor Inference Time**: Profile each model separately
- **Test Edge Cases**: Silent input, extreme parameters
- **Verify Determinism**: Same input should always produce same output

---

## 📊 Performance Comparison

| Model | Parameters | Inference (ms) | Memory (MB) | CPU Usage |
|-------|-----------|----------------|-------------|-----------|
| vocals_tts.onnx | 13M | 150 | 45 | Medium |
| vocals_hifigan.onnx | 1.5M | 25 | 30 | Low |
| 808_ddsp.onnx | 500K | 5 | 10 | Very Low |
| hihat_ddsp.onnx | 300K | 3 | 8 | Very Low |
| snare_ddsp.onnx | 400K | 4 | 9 | Very Low |
| piano_ddsp.onnx | 1.2M | 8 | 15 | Low |
| synth_fm.onnx | 800K | 6 | 12 | Low |

**Total Memory**: ~129 MB (all models loaded)  
**Total Inference**: ~201 ms (sequential processing)

---

## 🛡️ Safety & Validation

### Pre-Deployment Checklist

- [ ] Model produces stable output for all valid inputs
- [ ] No NaN or Inf values in output
- [ ] Inference time meets real-time requirements
- [ ] Memory usage within bounds
- [ ] Output audio clips to [-1, 1] range
- [ ] No audible artifacts (clicks, pops)
- [ ] Deterministic inference verified
- [ ] Model metadata updated
- [ ] LayerMap documentation current

### Known Limitations

1. **TTS Model**: Limited to English phonemes, ~30 character alphabet
2. **Vocoder**: Quality degrades below 22.05 kHz sample rate
3. **DDSP Models**: Parameter ranges must be respected for natural sound
4. **Piano Model**: No pedal noise or key click simulation
5. **FM Synth**: Limited to single operator FM (not 6-operator like DX7)

---

## 📚 References

### Academic Papers
- Tacotron 2: Natural TTS Synthesis (Shen et al., 2018)
- HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (Kong et al., 2020)
- DDSP: Differentiable Digital Signal Processing (Engel et al., 2020)

### Implementation Resources
- ONNX Runtime Documentation
- JUCE DSP Module Documentation
- Real-Time Audio Programming Best Practices

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-15  
**Maintained By**: AI Agent + Documentation Agent
