# Drum Models

This directory contains ONNX models for drum synthesis.

## Expected Models

- `808_ddsp.onnx` - Sub-bass 808 synthesis model
- `hihat_ddsp.onnx` - Hi-hat synthesis model  
- `snare_ddsp.onnx` - Snare drum synthesis model

## Model Format

All models should be in ONNX format (.onnx extension).

## Creating Models

To create your own drum synthesis models:

1. Train a DDSP (Differentiable Digital Signal Processing) model for the desired drum sound
2. Export the model to ONNX format using Python:
   ```python
   import torch.onnx
   
   # Export your trained model
   torch.onnx.export(model, dummy_input, "drum_model.onnx")
   ```
3. Place the .onnx file in this directory
4. Update `Models/config.json` if using a different filename

## Fallback

If models are not present, MAEVN will use DSP-based fallback synthesis as configured in `Models/config.json`.
