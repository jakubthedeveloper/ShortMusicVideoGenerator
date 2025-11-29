
# 🎥 Psychedelic Shorts Auto-Generator
Automated generator of YouTube Shorts from your own videos and music.  
Creates short vertical (9:16) clips with dynamic psychedelic, glitch, kaleidoscope and beat-reactive effects, synchronized to your audio (bass, hihat, beats).  
Audio is enhanced with reverse-reverb intros and reverb tails.

---

## ✨ Features

### 🎬 Video
- Over 20 video effects:
  - glitch, RGB split, block glitch, scanlines
  - kaleidoscope, mandala, swirl, ripple, wave
  - pixel sorting (horizontal & vertical)
  - hue shift, neon pulse, solarize
  - beat-reactive zoom, ripple, glow, RGB shake
  - bass-reactive distortions & zooming
  - hihat-reactive glitches & flashes
- Automatic 9:16 vertical conversion
- Smooth frame processing with OpenCV
- FPS consistent video rendering (default: 30 FPS)

### 🎧 Audio
- Beat detection via `librosa`
- Bass and hihat band detection  
- Reverse-reverb intro (cinematic swell)
- Reverb tail outro
- Full stereo audio processing using Pedalboard

### 🎛 Automation
- Automatically picks random video + random audio
- Generates multiple shorts using `--count N`
- Generates preview clips for every effect using `--preview-effects`
- Clean temporary audio file handling

---

## 📁 Project Structure

```
project_root/
│
├── src/
│   ├── generator.py
│   ├── video/
│   │   ├── effects.py
│   │   ├── renderer.py
│   │   └── transforms.py
│   ├── audio/
│   │   ├── loader.py
│   │   ├── analysis.py
│   │   ├── effects.py
│   │   └── exporter.py
│   ├── utils/
│   │   ├── file_utils.py
│   │   └── logging_utils.py
│   └── config/
│       ├── settings.py
│       └── presets.py (optional)
│
├── input/
│   ├── videos/   # put your .mp4/.mov source videos here
│   └── music/    # put your .mp3/.wav music files here
│
├── output/
│   ├── short_XXXX.mp4  # final generated shorts
│   └── effects_preview/ # previews of all effects (optional)
│
├── temp/         # temporary audio exports
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

### 1. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your assets
Place your input files into:

```
input/videos/
input/music/
```

Supported formats:
- Video: `.mp4`, `.mov`
- Audio: `.mp3`, `.wav`

---

## ▶️ Usage

### Generate one short

```bash
python src/generator.py
```

### Generate multiple shorts

```bash
python src/generator.py --count 10
```

### Generate preview clips for all effects

```bash
python src/generator.py --preview-effects
```

Previews appear in:

```
output/effects_preview/
```

---

## 🔧 Configuration

Edit:

```
src/config/settings.py
```

Available options:
- `CLIP_MIN_DURATION`
- `CLIP_MAX_DURATION`
- `FPS`

---

## 🧩 Adding New Effects

Add your effect to:

```
src/video/effects.py
```

And include it in:

```python
VIDEO_EFFECTS = [ ... ]
```

---

## 🗂 Output

All generated clips appear in:

```
output/
```

Each clip is named:

```
short_XXXX.mp4
```

---

## 🧪 Requirements

- Python 3.9–3.12
- FFmpeg installed (MoviePy uses it)
- For Linux/macOS: libsndfile for audio I/O

macOS:
```bash
brew install libsndfile
```

Ubuntu/Debian:
```bash
sudo apt install libsndfile1
```

---

## 🤝 License

You own 100% of the videos and audio you generate.  
YouTube upload automation is supported externally (coming soon).

---

## 📬 Author

Your personal AI assistant for video automation 😉  
Powered by MoviePy, Librosa, OpenCV, Pedalboard and a lot of creative math.
