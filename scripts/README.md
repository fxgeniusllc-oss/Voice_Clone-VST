# MAEVN ONNX Export Scripts

This directory contains Python scripts for exporting ONNX models used by the MAEVN plugin.

## Scripts

### export_drum_models.py
Exports drum synthesis models (808, hihat, snare) to ONNX format.

**Usage:**
```bash
python3 scripts/export_drum_models.py
```

**Requirements:**
- Python 3.10+
- PyTorch
- ONNX

### export_instrument_models.py
Exports instrument synthesis models (piano, synth) to ONNX format.

**Usage:**
```bash
python3 scripts/export_instrument_models.py
```

**Requirements:**
- Python 3.10+
- PyTorch
- ONNX

### export_vocal_models.py
Exports vocal synthesis models (TTS, vocoder) to ONNX format.

**Usage:**
```bash
python3 scripts/export_vocal_models.py
```

**Requirements:**
- Python 3.10+
- PyTorch
- ONNX

## Installation

Install required dependencies:

```bash
pip install torch onnx
```

Or using conda:

```bash
conda install pytorch onnx -c pytorch
```

## Note on Model Quality

The scripts in this directory export **simple placeholder models** for demonstration purposes. These models:

- Are lightweight and fast to export
- Demonstrate the expected model structure and interface
- Allow the plugin to build and run without requiring large pre-trained models
- Should be replaced with proper trained models for production use

## Creating Production Models

For production-quality synthesis, you should:

1. **Drums**: Train DDSP (Differentiable Digital Signal Processing) models on real drum samples
2. **Instruments**: Train neural synthesis models on instrument recordings
3. **Vocals**: Use pre-trained TTS (e.g., Tacotron 2) and vocoder models (e.g., HiFi-GAN)

## Resources

- **DDSP**: https://github.com/magenta/ddsp
- **Tacotron 2**: https://github.com/NVIDIA/tacotron2
- **HiFi-GAN**: https://github.com/jik876/hifi-gan
- **ONNX**: https://onnx.ai/
- **PyTorch ONNX**: https://pytorch.org/docs/stable/onnx.html

## Fallback Behavior

If ONNX models are not available, MAEVN will use DSP-based fallback synthesis as configured in `Models/config.json`.
