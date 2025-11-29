# Music Short Videos Auto‑Generator (GPU-ready)

A fully automated YouTube Shorts generator that creates psychedelic, beat‑reactive visual clips using:
- Random video sources
- Random effect chains (multi‑effect pipelines)
- Intelligent audio title extraction
- Reverse reverb intro & reverb tail outro
- Automatic mastering (normalization + limiter)
- GPU-ready architecture (optional)
- Customizable render options

The generator:
1. Picks a random video from `input/videos/`
2. Picks a random audio track from `input/music/`
3. Extracts the title from metadata (fallback: cleaned filename)
4. Generates a random chain of 2–4 visual effects
5. Applies beat‑reactive video warping
6. Applies reverse‑reverb audio effects
7. Normalizes and limits audio to avoid clipping
8. Exports a vertical 9:16 Short with synced audio

---

# 📁 Directory Structure

```
project/
│
├── input/
│   ├── videos/      # Source videos (mp4/mov)
│   └── music/       # Source audio (mp3/wav)
│
├── output/
│   ├── *.mp4        # Generated Shorts
│   └── effects_preview/
│
├── temp/            # Temporary audio files
└── src/
    ├── generator.py
    ├── config/
    │   └── settings.py
    ├── utils/
    │   ├── file_utils.py
    │   ├── logging_utils.py
    │   └── title_utils.py
    ├── video/
    │   ├── effects.py
    │   ├── effect_chains.py
    │   ├── renderer.py
    │   └── transforms.py
    └── audio/
        ├── loader.py
        ├── analysis.py
        ├── effects.py
        └── exporter.py
```

---

# ⚙️ Installation (Ubuntu / Linux)

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To check MoviePy 2.x installation:

```
python3 -c "import moviepy; print(moviepy.__version__)"
```

---

# ▶️ Running the Generator

## Generate 1 Short:
```
python src/generator.py
```

## Generate multiple Shorts:
```
python src/generator.py --count 5
```

## Generate previews of all effects:
```
python src/generator.py --preview-effects
```

---

# 🎨 Visual Effect Chains

Every generated Short uses **random chains of 2–4 effects**, such as:

```
kaleidoscope → distort → rgb_vibration
ripple → block_glitch
mandala → feedback → chroma_shift → distort
```

Effects respond to:
- bass energy
- hihat energy
- beat timestamps
- time parameter `t`

You can add your own effects in `src/video/effects.py`.

---

# 🎧 Audio Pipeline

The audio is processed through:

1. Reverse reverb intro  
2. Reverb tail outro  
3. Normalization  
4. Soft limiter (prevents clipping)  
5. Final gain normalization  

This ensures Shorts never get distorted.

---

# 🧠 Title Auto‑Generation

Using `mutagen`, the title is extracted from audio metadata:

- MP3 ID3  
- FLAC tags  
- M4A atoms  

If unavailable, filename is cleaned:

```
"03. My Song (final master).mp3" → "My Song"
```

This title becomes the output filename and can be used for YouTube uploads.

---

# 🎬 Video Formatting

Final Shorts are exported as:

- 1080×1920 (vertical)
- 60 FPS
- H.264, AAC audio
- Perfect YouTube Shorts compatibility

---

# 📦 Command Line Options

```
python src/generator.py --help
```

Outputs:

```
--count N             Generate N shorts
--preview-effects     Export one preview clip per effect
```

---

# Author
Jakub Krysakowski with great support from the artificial helper.
