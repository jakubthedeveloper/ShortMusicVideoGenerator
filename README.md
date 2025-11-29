# 🎥 Music Short Videos Auto-Generator (with GPU S)

Automated generator of YouTube Shorts from your own videos and music.
Creates short vertical (9:16) clips with dynamic psychedelic, glitch, kaleidoscope,
and beat‑reactive GPU‑accelerated effects, synchronized to your audio (bass, hihat, beats).
Audio is enhanced with reverse‑reverb intros and long cinematic tails.

---

## ✨ Features

### 🎬 Video (GPU-Accelerated)
Over **20+ GPU-powered effects** using **CuPy** for massive speedups:

- glitch, RGB split, block glitch
- kaleidoscope, mandala twist
- swirl, ripple, wave
- pixel sorting (H/V)
- hue shift, neon pulse, solarize
- beat reactive: zoom, ripple, glow, RGB shake
- bass reactive: distortions, zoom pulses

All heavy math (sinus warps, distortions, sorting, matrices)
is now executed on **GPU**, giving **20×–300× faster rendering** than CPU.

### 🎧 Audio (High-Quality DSP)
- Beat detection via librosa  
- Bass + hihat band detection  
- Reverse-reverb intro  
- Reverb tail outro  
- Stereo DSP via Pedalboard  

### 🎛 Automation
- Random video + music selection
- Multi‑short rendering: `--count N`
- Effect preview mode: `--preview-effects`
- All temp files handled automatically

---

## 📁 Project Structure

project_root/
```
├── src/
│   ├── generator.py
│   ├── video/
│   │   ├── effects.py        # GPU/CuPy effects
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
│   ├── videos/
│   └── music/
├── output/
│   ├── short_XXXX.mp4
│   └── effects_preview/
├── temp/
├── README.md
└── requirements.txt
```

---

## 🔧 Installation (Ubuntu)

### 1. System packages
```
sudo apt update
sudo apt install ffmpeg python3.12-venv python3-dev build-essential libsndfile1
```

### 2. Virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies
```
pip install -r requirements.txt
```

### 4. Install CuPy (GPU support)

Check CUDA:
```
nvidia-smi
```

Install:
```
# CUDA 12.x
pip install cupy-cuda12x
# CUDA 11.x
pip install cupy-cuda11x
```

Verify:
```
python -c "import cupy as cp; print(cp.ones(5)*2)"
```

---

## ▶️ Usage

### Generate one short
```
python src/generator.py
```

### Generate multiple shorts
```
python src/generator.py --count 10
```

### Generate all effect previews
```
python src/generator.py --preview-effects
```

Previews go to:
```
output/effects_preview/
```

---

## 🧩 Add Your Own Effects

Create a function in:

```
src/video/effects.py
```

Example:
```
def my_effect(frame, t, beats, bass, hihat):
    ...
```

Register it in:
```
VIDEO_EFFECTS = [..., my_effect]
```

You can use:
- CuPy for GPU operations  
- NumPy + OpenCV for CPU operations  

---

## 📝 License

Free for personal and commercial use.
