# Instrument Models

This directory contains ONNX models for instrument synthesis.

## Expected Models

- `piano_ddsp.onnx` - Piano synthesis model
- `synth_fm.onnx` - FM synthesis model

## Model Format

All models should be in ONNX format (.onnx extension).

## Creating Models

To create your own instrument synthesis models:

1. Train a DDSP or other neural synthesis model for the desired instrument
2. Export the model to ONNX format using Python:
   ```python
   import torch.onnx
   
   # Export your trained model
   torch.onnx.export(model, dummy_input, "instrument_model.onnx")
   ```
3. Place the .onnx file in this directory
4. Update `Models/config.json` if using a different filename

## Fallback

If models are not present, MAEVN will use DSP-based fallback synthesis (sine waves) as configured in `Models/config.json`.
