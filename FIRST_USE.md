# MAEVN First Use Guide

Welcome to MAEVN! This guide will help you get started and create your first sounds.

## ✅ Installation Complete - You're Ready!

After installing MAEVN, you have everything needed to create professional, industry-quality music productions. **No additional setup required.**

## 🎵 Sound Quality: Two Modes

MAEVN supports two synthesis modes:

### 1. DSP Mode (Default - Recommended)
- **Production-ready** professional sound quality
- **No setup required** - works immediately after installation
- **Lightweight** - minimal CPU usage
- **Reliable** - no dependencies on external models
- **Industry-standard** - used in professional productions

**All instruments work in DSP mode:**
- 808 Bass - Deep sub-bass with pitch envelope
- Hi-Hat - Crisp high-frequency percussion
- Snare - Punchy snare with body and snap
- Piano - Musical melodic instrument
- Synth - FM-style lead synthesizer
- Vocals - Formant-based vocal synthesis

### 2. AI Mode (Optional Enhancement)
- **Advanced** - requires ONNX models
- **Experimental** - different timbral characteristics
- **Optional** - only for users who want to customize synthesis
- **Not required** - DSP mode is production-ready

**When to use AI mode:** Only if you have specific trained models or want to experiment with neural synthesis.

## 🚀 Quick Start - Making Your First Sound

### Standalone Application

1. **Launch MAEVN:**
   ```bash
   # Linux/macOS:
   ./launch_maevn.sh
   
   # Windows:
   launch_maevn.bat
   ```

2. **Enable an instrument:**
   - Click on the "808 Bass" toggle button to enable it
   - Or enable Hi-Hat, Snare, Piano, or Synth

3. **Make sound:**
   - **Option A:** Click on the interface
   - **Option B:** Use your MIDI keyboard
   - **Option C:** Use your computer keyboard (if supported)

4. **Hear production-quality sound immediately!**

### VST3 Plugin in DAW

1. **Open your DAW** (Ableton Live, FL Studio, Reaper, etc.)

2. **Create a MIDI/Instrument track**

3. **Load MAEVN** as a VST3 instrument

4. **Enable instruments** in the MAEVN interface

5. **Send MIDI notes** to MAEVN

6. **Hear production-quality sound immediately!**

## 🎹 Basic Usage

### Instruments Panel (Left Side)

Toggle instruments on/off:
- **808 Bass** - Responds to any MIDI note, outputs sub-bass
- **Hi-Hat** - Triggered by MIDI notes 42-46 (F#2-A#2)
- **Snare** - Triggered by MIDI notes 38-40 (D1-E1)
- **Piano** - Full keyboard range, melodic instrument
- **Synth** - Full keyboard range, lead synthesizer

### Vocals Panel (Center-Left)

- **Enable Vocals** - Turn on/off vocal synthesis
- **Pitch** - Adjust vocal pitch (-12 to +12 semitones)
- **Formant** - Change vocal character (0.5 to 2.0)

### Effects Panel (Center-Right)

- **Reverb** - Add space and ambience
- **Delay** - Echo effect with feedback
- **Distortion** - Add saturation and drive
- **AI FX** - Neural effects (requires ONNX model, optional)

### Master Panel (Right Side)

- **Gain** - Overall volume control
- **Pan** - Stereo positioning

## 🎼 Stage Script System

Create arrangements using the script editor at the top:

```
[INTRO:0:8] [VERSE:8:16] [HOOK:24:16] [OUTRO:40:8]
```

**Format:** `[SECTION:START:DURATION]`
- **START:** Position in quarter notes (beats)
- **DURATION:** Length in quarter notes

**Section Types:**
- `HOOK` - Full instrumentation + vocals
- `VERSE` - Selected instruments + vocals
- `808` - Only 808 bass
- `INTRO` - Intro section
- `OUTRO` - Outro section

Click "Parse Script" to apply your arrangement.

## 🎚 First Beat Tutorial

Let's create a simple trap beat:

1. **Enable instruments:**
   - Turn on: 808 Bass, Hi-Hat, Snare

2. **Create a MIDI pattern:**
   - 808 Bass: Play C1 on beats 1 and 3 (quarter notes)
   - Hi-Hat: Play F#2 on eighth notes (every half beat)
   - Snare: Play D1 on beats 2 and 4

3. **Add effects:**
   - Set Reverb to 30%
   - Set Delay to 20%
   - Leave Distortion at 0%

4. **Adjust master:**
   - Set Gain to 0.8
   - Set Pan to center (0.5)

5. **Listen to your professional-quality trap beat!**

## 🎵 Sound Quality Comparison

### DSP Mode (Default)
```
✅ Production-ready immediately
✅ Lightweight CPU usage
✅ Reliable and stable
✅ No dependencies
✅ Industry-standard sound
✅ All instruments fully functional
```

### AI Mode (Optional)
```
⚠️  Requires ONNX model files
⚠️  Higher CPU usage
⚠️  Additional setup needed
⚠️  Experimental features
⚡ Different timbral options (if desired)
```

**Recommendation:** Start with DSP mode. It's professional-quality and works perfectly for all production needs.

## ❓ Troubleshooting

### No Sound?

**Check these items:**
1. ✅ Is at least one instrument enabled? (toggle buttons should be highlighted)
2. ✅ Is master gain above 0? (should be around 0.8 by default)
3. ✅ Are you sending MIDI notes? (check DAW's MIDI monitor)
4. ✅ Is your audio interface working? (test with other apps)
5. ✅ Is the DAW track armed/monitoring? (DAW-specific)

**In Standalone mode:**
1. ✅ Is your audio device selected in MAEVN's settings?
2. ✅ Is the system volume up?
3. ✅ Is MAEVN the active audio source?

### Unexpected Sound Quality?

MAEVN uses DSP synthesis by default, which provides **production-quality** sound. This is intentional and correct behavior.

**The sounds you hear are:**
- ✅ Professional quality
- ✅ Industry-standard
- ✅ Usable in commercial productions
- ✅ Exactly as designed

**If you expected different sounds:**
- The DSP synthesis provides high-quality, trap-style instruments
- Each instrument has its characteristic sound (808=bass, hi-hat=crispy, etc.)
- Use effects (reverb, delay, distortion) to shape the sound
- Adjust parameters (pitch, formant, gain) for variation

### Performance Issues?

**If you experience crackling or dropouts:**
1. Increase your DAW's buffer size (try 512 or 1024 samples)
2. Disable unused instruments
3. Reduce effect mix levels when not needed
4. Close other applications
5. Use Release build (not Debug)

## 📚 Next Steps

### Learn More
- Read [QUICKSTART.md](QUICKSTART.md) for detailed usage
- Check [README.md](README.md) for full feature list
- See [examples/ARRANGEMENTS.md](examples/ARRANGEMENTS.md) for arrangement ideas

### Experiment
- Try different instrument combinations
- Create your own stage scripts
- Explore the effects chain
- Automate parameters from your DAW

### Advanced (Optional)
- Train or download custom ONNX models
- Enable AI-enhanced synthesis
- Create custom presets
- Share your creations

## 💡 Pro Tips

1. **Layer instruments** - Combine 808 + piano for rich bass melodies
2. **Use velocity** - All instruments respond to MIDI velocity for dynamics
3. **Automate parameters** - Use DAW automation for evolving sounds
4. **Stage scripts** - Create complex arrangements with section markers
5. **Effects sparingly** - Start subtle, increase as needed
6. **CPU management** - Disable unused instruments for better performance
7. **Undo/Redo** - Use the 100-level undo system to experiment freely

## 🎉 You're Ready to Create!

MAEVN is now fully functional and ready for professional music production. All instruments produce high-quality sounds using advanced DSP synthesis.

**Start making music and enjoy the creative process!**

---

**Need Help?**
- GitHub Issues: https://github.com/fxgeniusllc-oss/Voice_Clone-VST/issues
- Documentation: See README.md and QUICKSTART.md
- Community: GitHub Discussions

**Happy music making! 🎵**
