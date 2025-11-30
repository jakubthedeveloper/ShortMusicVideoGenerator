import os
import random
import sys
import argparse
from utils.title_utils import generate_title_from_audio
from video.effect_chains import random_chain

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    ImageSequenceClip
)

from utils.file_utils import pick_random_file
from utils.logging_utils import log

from video.effects import VIDEO_EFFECTS
from video.transforms import to_vertical_9_16
from video.renderer import apply_video_effect

from audio.loader import load_audio
from audio.analysis import detect_beats, get_audio_bands
from audio.effects import apply_audio_effects
from audio.exporter import save_temp_audio

from config.settings import CLIP_MIN_DURATION, CLIP_MAX_DURATION, FPS

SMOOTH_BASS = 0.1   # lower = more smooth (0.05–0.3)
SMOOTH_HIHAT = 0.15

def generate_short():
    # -----------------------------
    # SELECT INPUTS
    # -----------------------------
    video_path = pick_random_file("input/videos", ("mp4", "mov"))
    music_path = pick_random_file("input/music", ("mp3", "wav"))

    log(f"Selected video: {video_path}")
    log(f"Selected audio: {music_path}")

    title = generate_title_from_audio(music_path)
    print(f"[★] Generated title: {title}")

    # -----------------------------
    # LOAD VIDEO
    # -----------------------------
    clip = VideoFileClip(video_path)

    # -----------------------------
    # SAFE RANDOM SUBCLIP SELECTION
    # -----------------------------
    video_duration = clip.duration

    # pick random duration inside allowed range
    duration = random.uniform(CLIP_MIN_DURATION, CLIP_MAX_DURATION)

    # clamp duration if video is shorter
    duration = min(duration, video_duration - 0.1)

    # if video is extremely short: use entire clip
    if duration <= 0:
        duration = video_duration

    # valid start time
    max_start = max(0, video_duration - duration)
    start = random.uniform(0, max_start)

    # MOVIEPY 2.x: subclipped()
    subclip = clip.subclipped(start, start + duration)

    # -----------------------------
    # AUDIO ANALYSIS
    # -----------------------------
    beats = detect_beats(music_path)
    bass_energy, hihat_energy = get_audio_bands(music_path, fps=FPS)

    # -----------------------------
    # SELECT EFFECT
    # -----------------------------
    #effect_fn = random.choice(VIDEO_EFFECTS)
    chain = random_chain()
    chain_names = " → ".join(fn.__name__ for fn in chain)
    log(f"Selected effect chain: {chain_names}")

    # -----------------------------
    # PROCESS VIDEO FRAMES
    # -----------------------------
    frames = []
    total_frames = int(subclip.duration * FPS)

    for i, frame in enumerate(subclip.iter_frames(fps=FPS)):
        t = i / FPS

        if i == 0:
            bass_s = bass_energy[0]
            hihat_s = hihat_energy[0]
        else:
            bass_s = bass_s * (1 - SMOOTH_BASS) + bass_energy[i] * SMOOTH_BASS
            hihat_s = hihat_s * (1 - SMOOTH_HIHAT) + hihat_energy[i] * SMOOTH_HIHAT
    
        processed = apply_video_effect(frame, t, beats, bass_s, hihat_s, chain)

        frames.append(processed)

    video = ImageSequenceClip(frames, fps=FPS)
    video = to_vertical_9_16(video)

    # -----------------------------
    # PROCESS AUDIO
    # -----------------------------
    audio = load_audio(music_path)
    audio = apply_audio_effects(audio)
    temp_audio_path = save_temp_audio(audio)

    audio_clip = AudioFileClip(temp_audio_path)
    audio_clip = audio_clip.subclipped(0, video.duration)

    # MOVIEPY 2.x: with_audio()
    video = video.with_audio(audio_clip)

    # -----------------------------
    # EXPORT
    # -----------------------------
    os.makedirs("output", exist_ok=True)

    safe_title = title.replace("/", "-").replace("\\", "-").replace(":", " ").strip()
    out_path = f"output/{safe_title} - short.mp4"

    log(f"Exporting: {out_path}")

    video.write_videofile(
        out_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
    )

    log("Generation complete.")


def generate_effect_previews():
    os.makedirs("output/effects_preview", exist_ok=True)

    video_path = pick_random_file("input/videos", ("mp4", "mov"))
    log(f"Preview source video: {video_path}")

    clip = VideoFileClip(video_path)
    subclip = clip.subclipped(0, min(3, clip.duration))

    fps_list = int(subclip.duration * FPS)
    beats = []
    bass = [0.5] * fps_list
    hihat = [0.5] * fps_list

    for effect_fn in VIDEO_EFFECTS:
        log(f"Generating preview: {effect_fn.__name__}")

        frames = []
        for i, frame in enumerate(subclip.iter_frames(fps=FPS)):
            t = i / FPS
            processed = apply_video_effect(
                frame, t, beats, bass[i], hihat[i], effect_fn
            )
            frames.append(processed)

        video = ImageSequenceClip(frames, fps=FPS)
        video = to_vertical_9_16(video)

        out_path = f"output/effects_preview/{effect_fn.__name__}.mp4"
        video.write_videofile(out_path, fps=FPS, codec="libx264", audio=False)


def generate_multiple(count):
    for i in range(count):
        log(f"--- Generating clip {i+1}/{count} ---")
        generate_short()

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Psychedelic Shorts Auto-Generator (GPU edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="How many shorts to generate."
    )

    parser.add_argument(
        "--preview-effects",
        action="store_true",
        help="Generate a preview sample for every video effect."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.preview_effects:
        generate_effect_previews()
    else:
        for _ in range(args.count):
            generate_short()

