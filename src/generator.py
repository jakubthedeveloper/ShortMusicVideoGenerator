import os
import random
import sys

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


def generate_short():
    # -----------------------------
    # SELECT INPUTS
    # -----------------------------
    video_path = pick_random_file("input/videos", ("mp4", "mov"))
    music_path = pick_random_file("input/music", ("mp3", "wav"))

    log(f"Selected video: {video_path}")
    log(f"Selected audio: {music_path}")

    # -----------------------------
    # LOAD VIDEO
    # -----------------------------
    clip = VideoFileClip(video_path)

    duration = random.uniform(CLIP_MIN_DURATION, CLIP_MAX_DURATION)
    start = random.uniform(0, max(0, clip.duration - duration))

    # MOVIEPY 2.x
    subclip = clip.subclipped(start, start + duration)

    # -----------------------------
    # AUDIO ANALYSIS
    # -----------------------------
    beats = detect_beats(music_path)
    bass_energy, hihat_energy = get_audio_bands(music_path, fps=FPS)

    # -----------------------------
    # SELECT EFFECT
    # -----------------------------
    effect_fn = random.choice(VIDEO_EFFECTS)
    log(f"Selected effect: {effect_fn.__name__}")

    # -----------------------------
    # PROCESS VIDEO FRAMES
    # -----------------------------
    frames = []
    total_frames = int(subclip.duration * FPS)

    for i, frame in enumerate(subclip.iter_frames(fps=FPS)):
        t = i / FPS

        bass = bass_energy[min(i, len(bass_energy)-1)]
        hihat = hihat_energy[min(i, len(hihat_energy)-1)]

        processed = apply_video_effect(
            frame, t, beats, bass, hihat, effect_fn
        )
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

    out_path = f"output/short_{random.randint(1000,9999)}.mp4"
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


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--preview-effects" in args:
        generate_effect_previews()
        sys.exit(0)

    if "--count" in args:
        idx = args.index("--count") + 1
        if idx < len(args):
            count = int(args[idx])
            generate_multiple(count)
            sys.exit(0)
        else:
            log("ERROR: missing number after --count")
            sys.exit(1)

    generate_short()

