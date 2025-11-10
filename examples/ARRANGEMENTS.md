# MAEVN Example Arrangements

This directory contains example stage scripts for different musical styles and arrangements.

## How to Use

1. Copy the arrangement text from any example below
2. Paste it into MAEVN's stage script editor
3. Click "Parse Script"
4. Play your DAW and watch the arrangement come to life

## Example 1: Simple Trap Beat

**Style**: Basic trap arrangement with 808 and drums
**Duration**: 32 bars (128 quarter notes)

```
[INTRO:0:8] [808:8:16] [VERSE:24:32] [HOOK:56:32] [OUTRO:88:40]
```

**Description**:
- Intro: 2 bars of minimal instrumentation
- 808 Section: 4 bars of just 808 bass
- Verse: 8 bars with reduced instrumentation
- Hook: 8 bars with full instrumentation
- Outro: 10 bars fade out

## Example 2: Full Song Structure

**Style**: Complete song with intro, verses, hooks, bridge
**Duration**: 80 bars (320 quarter notes)

```
[INTRO:0:16] [VERSE:16:32] [HOOK:48:32] [VERSE:80:32] [HOOK:112:32] [VERSE:144:16] [HOOK:160:64] [OUTRO:224:96]
```

**Description**:
- Intro: 4 bars
- Verse 1: 8 bars
- Hook 1: 8 bars
- Verse 2: 8 bars
- Hook 2: 8 bars
- Bridge: 4 bars (as verse)
- Final Hook: 16 bars (extended)
- Outro: 24 bars

## Example 3: Minimal Electronic

**Style**: Minimal electronic with gradual build
**Duration**: 64 bars (256 quarter notes)

```
[808:0:32] [808:32:32] [VERSE:64:64] [HOOK:128:64] [808:192:32] [OUTRO:224:32]
```

**Description**:
- 808 Only: 8 bars
- 808 Only: 8 bars (add variation)
- Verse: 16 bars (bring in melodic elements)
- Hook: 16 bars (full arrangement)
- 808 Breakdown: 8 bars
- Outro: 8 bars

## Example 4: Quick Demo

**Style**: Short demonstration of all sections
**Duration**: 24 bars (96 quarter notes)

```
[INTRO:0:8] [VERSE:8:16] [HOOK:24:24] [808:48:8] [OUTRO:56:40]
```

**Description**:
- Intro: 2 bars
- Verse: 4 bars
- Hook: 6 bars
- 808: 2 bars
- Outro: 10 bars

## Example 5: Build and Drop

**Style**: EDM-style build and drop
**Duration**: 32 bars (128 quarter notes)

```
[INTRO:0:16] [VERSE:16:16] [VERSE:32:16] [HOOK:48:32] [HOOK:80:32] [OUTRO:112:16]
```

**Description**:
- Intro: 4 bars (buildup)
- Verse 1: 4 bars (tension)
- Verse 2: 4 bars (more tension)
- Drop/Hook 1: 8 bars (release)
- Drop/Hook 2: 8 bars (full energy)
- Outro: 4 bars (wind down)

## Example 6: Hip-Hop Structure

**Style**: Traditional hip-hop song structure
**Duration**: 96 bars (384 quarter notes)

```
[INTRO:0:8] [VERSE:8:48] [HOOK:56:32] [VERSE:88:48] [HOOK:136:32] [VERSE:168:48] [HOOK:216:32] [OUTRO:248:136]
```

**Description**:
- Intro: 2 bars
- Verse 1: 12 bars (16 bars - 2 intro)
- Hook 1: 8 bars
- Verse 2: 12 bars
- Hook 2: 8 bars
- Verse 3: 12 bars
- Hook 3: 8 bars
- Outro: 34 bars

## Example 7: Ambient Progression

**Style**: Slow ambient build with vocals
**Duration**: 48 bars (192 quarter notes)

```
[INTRO:0:32] [VERSE:32:32] [VERSE:64:32] [HOOK:96:32] [HOOK:128:32] [OUTRO:160:32]
```

**Description**:
- Long Intro: 8 bars
- Verse 1: 8 bars (minimal)
- Verse 2: 8 bars (add layers)
- Hook 1: 8 bars (full)
- Hook 2: 8 bars (sustained)
- Outro: 8 bars (fade)

## Example 8: Experimental

**Style**: Experimental with alternating sections
**Duration**: 40 bars (160 quarter notes)

```
[808:0:8] [VERSE:8:8] [808:16:8] [HOOK:24:16] [808:40:8] [VERSE:48:16] [HOOK:64:32] [OUTRO:96:64]
```

**Description**:
- 808: 2 bars
- Verse: 2 bars
- 808: 2 bars
- Hook: 4 bars
- 808: 2 bars
- Verse: 4 bars
- Hook: 8 bars
- Extended Outro: 16 bars

## Custom Section Tips

### Section Characteristics

**HOOK** (Full Arrangement):
- All instruments enabled
- Vocals enabled
- Maximum energy
- Use for choruses and drops

**VERSE** (Reduced Arrangement):
- 808, Hi-Hat, Piano enabled
- Vocals enabled
- Medium energy
- Use for verses and bridges

**808** (Bass Only):
- Only 808 bass enabled
- No vocals
- Minimal energy
- Use for breaks and transitions

**INTRO** (Light Arrangement):
- Hi-Hat, Piano, Synth enabled
- No 808, no vocals
- Building energy
- Use for song intros

**OUTRO** (Fade Out):
- Piano and vocals
- No drums
- Decreasing energy
- Use for song outros

## Creating Your Own

### Formula: `[NAME:START:DURATION]`

- **NAME**: Section type (HOOK, VERSE, 808, INTRO, OUTRO)
- **START**: When to start (in quarter notes from beginning)
- **DURATION**: How long to last (in quarter notes)

### Tips:

1. **Timing**: 4 quarter notes = 1 bar (in 4/4 time)
2. **Planning**: Sketch on paper first
3. **Variety**: Mix different section types
4. **Flow**: Consider energy levels between sections
5. **Space**: Leave gaps between sections if needed

### Common Durations:

- Short section: 8-16 quarters (2-4 bars)
- Medium section: 16-32 quarters (4-8 bars)
- Long section: 32-64 quarters (8-16 bars)
- Extended: 64+ quarters (16+ bars)

## DAW Integration

### Aligning with DAW

1. Set your DAW to 4/4 time signature
2. Set tempo (e.g., 120 BPM)
3. MAEVN syncs automatically with DAW position
4. Timeline in MAEVN shows current section

### MIDI Programming

- Draw MIDI notes in your DAW
- Different sections will use different instruments automatically
- Layer multiple MIDI tracks for complexity

## Performance Ideas

1. **Live Jamming**: Play MIDI keyboard live with arrangement
2. **Pattern Recording**: Record MIDI patterns per section
3. **Parameter Automation**: Automate FX per section
4. **Section Variations**: Create multiple versions, test which works best

## Need More Help?

- See [QUICKSTART.md](../QUICKSTART.md) for basic usage
- See [README.md](../README.md) for detailed documentation
- See [BUILD.md](../BUILD.md) for building from source

Happy composing! 🎹
