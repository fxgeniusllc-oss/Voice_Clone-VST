# MAEVN Quick Start Guide

This guide will help you get started with MAEVN quickly.

## Installation

### 1. Binary Installation (Recommended for Users)

If you have a pre-built binary:

1. Copy the VST3 file to your system's plugin folder:
   - **Windows**: `C:\Program Files\Common Files\VST3\`
   - **macOS**: `~/Library/Audio/Plug-Ins/VST3/`
   - **Linux**: `~/.vst3/`

2. Open your DAW and rescan plugins

3. Load MAEVN as an instrument plugin

### 2. Building from Source (For Developers)

See [BUILD.md](BUILD.md) for detailed build instructions.

**Quick build:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## First Use

### Loading the Plugin

1. Open your DAW (see compatibility list below)
2. Create a new MIDI/Instrument track
3. Load MAEVN as a VST3 instrument
4. Send MIDI notes to hear the instruments

**Compatible DAWs (VST3 Required):**
- Ableton Live 10+, FL Studio 20+, Reaper 5.0+, Bitwig Studio 3.0+
- Steinberg Cubase/Nuendo 10.5+, PreSonus Studio One 4+, Tracktion Waveform

**Not Compatible:**
- ❌ Audacity (does not support VST3 format)

**Alternative:** Use MAEVN in Standalone mode (no DAW required)

### Basic Configuration

The MAEVN interface is divided into 4 sections:

#### 1. Instruments (Left Panel)
Toggle on/off trap instruments:
- **808 Bass** - Sub bass (responds to any MIDI note)
- **Hi-Hat** - High frequency percussion (MIDI notes 42-46)
- **Snare** - Snare drum (MIDI notes 38-40)
- **Piano** - Melodic piano (any MIDI note)
- **Synth** - Lead synthesizer (any MIDI note)

#### 2. Vocals (Center-Left Panel)
- **Enable Vocals** - Turn on/off vocal synthesis
- **Pitch** - Adjust vocal pitch (-12 to +12 semitones)
- **Formant** - Change vocal character (0.5 to 2.0)

#### 3. Effects (Center-Right Panel)
- **Reverb** - Space and ambience
- **Delay** - Echo effect with feedback
- **Distortion** - Saturation and drive
- **AI FX** - Neural effects (requires ONNX model)

#### 4. Master (Right Panel)
- **Gain** - Overall volume
- **Pan** - Stereo positioning

### Stage Script Editor

At the top of the plugin, you'll see a script editor with text like:
```
[INTRO:0:4] [VERSE:4:12] [HOOK:12:20]
```

This defines your song arrangement:
- **Format**: `[SECTION:START:DURATION]`
- **START**: Position in quarter notes (beats)
- **DURATION**: Length in quarter notes

**Section Types:**
- `HOOK` - Full instrumentation + vocals
- `VERSE` - Selected instruments + vocals
- `808` - Only 808 bass
- `INTRO` - Intro section with reduced instruments
- `OUTRO` - Outro section

**Example arrangement:**
```
[INTRO:0:4] [VERSE:4:16] [HOOK:20:16] [808:36:8] [HOOK:44:16] [OUTRO:60:8]
```

Click "Parse Script" to apply changes.

### Timeline View

The visual timeline shows:
- Color-coded sections
- Current playback position
- Active section name

Colors:
- Blue = HOOK
- Purple = VERSE
- Red = 808
- Green = INTRO
- Orange = OUTRO

## Common Workflows

### Workflow 1: Quick Beat

1. Enable 808, Hi-Hat, and Snare
2. Draw MIDI notes:
   - 808: C1 on the quarter notes
   - Hi-Hat: F#2 on eighth notes
   - Snare: D1 on beats 2 and 4
3. Add Reverb and Delay to taste
4. Adjust Master Gain

### Workflow 2: Melodic Track

1. Enable Piano and Synth
2. Disable drums (808, Hi-Hat, Snare)
3. Play chord progressions on Piano
4. Play lead melody on Synth (different MIDI channel or track)
5. Enable Vocals with text
6. Apply Reverb generously

### Workflow 3: Arranged Song

1. Write your stage script:
   ```
   [INTRO:0:8] [VERSE:8:16] [HOOK:24:16] [VERSE:40:16] [HOOK:56:16] [OUTRO:72:8]
   ```
2. Parse the script
3. Play along with the arrangement
4. Different instruments activate in each section automatically
5. Use undo/redo for parameter tweaks

## MIDI Control

### Drum Mapping
- **808 Bass**: C0-B8 (any note, follows pitch)
- **Hi-Hat**: F#2-A#2 (notes 42-46)
- **Snare**: D1-E1 (notes 38-40)

### Melodic Instruments
- **Piano**: Full keyboard range
- **Synth**: Full keyboard range

### Tips
- Layer multiple MIDI tracks for complex arrangements
- Use velocity for dynamics (all instruments respond to velocity)
- Automate parameters from your DAW

## AI Features (Advanced)

To use AI-powered vocals and effects:

### 1. Obtain ONNX Models

You need three types of models:
- **TTS Model**: Text-to-speech (text → mel-spectrogram)
- **Vocoder Model**: Neural vocoder (mel-spectrogram → audio)
- **AI FX Model**: Neural audio processing

Popular sources:
- Hugging Face (https://huggingface.co/models)
- ONNX Model Zoo (https://github.com/onnx/models)
- Train your own models

### 2. Convert Models to ONNX

If your models are in PyTorch/TensorFlow:
```python
# PyTorch example
import torch.onnx

model = YourModel()
dummy_input = torch.randn(1, 80, 100)  # Example shape
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 3. Load Models

Use the plugin API to load models:
```cpp
// In your DAW's plugin host or via JUCE API
processor.getAudioEngine()
    .getVocalSynthesis()
    .loadTTSModel("/path/to/tts.onnx");

processor.getAudioEngine()
    .getVocalSynthesis()
    .loadVocoderModel("/path/to/vocoder.onnx");

processor.getAudioEngine()
    .getFXChain()
    .getAIEffects()
    .loadModel("/path/to/ai_fx.onnx");
```

### 4. Enable AI Features

- Toggle "Enable Vocals" for AI vocal synthesis
- Increase "AI FX" knob for neural effects

**Note**: Without ONNX models, MAEVN uses high-quality DSP fallbacks.

## Automation

### DAW Automation

All parameters can be automated from your DAW:
1. Enable automation recording
2. Adjust parameters while recording
3. Edit automation curves in your DAW

### Built-in Automation

MAEVN includes timeline-based automation:
- Parameters can be automated to timeline position
- Automation follows PPQ (Pulse Per Quarter) timing
- Syncs perfectly with DAW transport

## Undo/Redo

- Click "Undo" to reverse last parameter change
- Click "Redo" to reapply undone change
- Up to 100 levels of undo history
- Keyboard shortcuts (if DAW supports):
  - Ctrl+Z / Cmd+Z: Undo
  - Ctrl+Y / Cmd+Shift+Z: Redo

## Performance Tips

### Reducing CPU Usage

1. **Disable unused instruments**: Turn off instruments you're not using
2. **Reduce effects**: Lower reverb/delay mix when not needed
3. **AI FX**: Keep AI FX at 0 unless specifically needed
4. **Buffer size**: Increase audio buffer size in DAW preferences

### Best Practices

- Use Release builds (not Debug) for performance
- Test with standalone version first
- Monitor CPU usage in your DAW
- Bounce tracks to audio when arrangement is finalized

## Troubleshooting

### No Sound

1. Check master gain is not at 0
2. Ensure at least one instrument is enabled
3. Verify MIDI is being received (check DAW MIDI monitor)
4. Check audio routing in DAW

### Crackling/Glitches

1. Increase audio buffer size
2. Disable AI effects if enabled
3. Reduce number of active instruments
4. Check CPU usage

### Plugin Won't Load

1. Ensure VST3 is in correct folder
2. Rescan plugins in DAW
3. Check DAW logs for errors
4. Verify plugin matches your system architecture (x64)

### Stage Script Not Working

1. Click "Parse Script" after editing
2. Check syntax: `[NAME:START:DURATION]`
3. Ensure no overlapping sections
4. Start time must be before end time

## Next Steps

- **Experiment**: Try different instrument combinations
- **Learn**: Study the example arrangements
- **Create**: Build your own stage scripts
- **Extend**: Load custom ONNX models for unique sounds
- **Share**: Export your arrangements and share with others

## Resources

- **Documentation**: [README.md](README.md)
- **Build Guide**: [BUILD.md](BUILD.md)
- **GitHub**: https://github.com/fxgeniusllc-oss/Voice_Clone-VST
- **Issues**: https://github.com/fxgeniusllc-oss/Voice_Clone-VST/issues

## Support

Need help?
1. Check this guide first
2. Read the full README
3. Search GitHub issues
4. Open a new issue with details

Happy music making! 🎵
