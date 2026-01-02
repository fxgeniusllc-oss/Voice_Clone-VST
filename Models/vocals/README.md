# Vocal Models

This directory contains ONNX models for vocal synthesis.

## Expected Models

- `vocals_tts.onnx` - Text-to-Speech model (converts text to mel-spectrogram)
- `vocals_hifigan.onnx` - Vocoder model (converts mel-spectrogram to audio)

## Model Format

All models should be in ONNX format (.onnx extension).

## Creating Models

To create your own vocal synthesis models:

1. Train or use pre-trained TTS and vocoder models
2. Export both models to ONNX format:
   ```python
   import torch.onnx
   
   # Export TTS model
   torch.onnx.export(tts_model, tts_input, "vocals_tts.onnx")
   
   # Export vocoder model
   torch.onnx.export(vocoder_model, mel_input, "vocals_hifigan.onnx")
   ```
3. Place both .onnx files in this directory

## Recommended Models

- **TTS**: Tacotron 2, FastSpeech 2, or similar
- **Vocoder**: HiFi-GAN, WaveGlow, or similar

## Fallback

If models are not present, MAEVN will use formant-based synthesis as a fallback.
