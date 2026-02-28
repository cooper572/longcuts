#!/usr/bin/env python3
"""High-performance YouTube Shorts generation pipeline for long videos.

Key optimization principles:
1. Transcribe once (GPU Whisper) on the full source video.
2. Rank transcript segments and pick only top-N candidate windows.
3. Cut selected windows first with FFmpeg stream-copy (no re-encode).
4. Run face detection + crop only on extracted short clips.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import torch
import whisper
import yt_dlp


TRIGGER_WORDS = {
    "secret",
    "shocking",
    "mistake",
    "why",
    "truth",
    "warning",
    "exposed",
    "never",
    "crazy",
    "surprising",
    "you",
    "nobody",
}


@dataclass
class ScoredSegment:
    start: float
    end: float
    text: str
    score: float


def run_cmd(cmd: Sequence[str]) -> None:
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(cmd)}\n{process.stdout}")


def download_youtube_mp4(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tpl = str(output_dir / "source.%(ext)s")
    ydl_opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_tpl,
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = Path(ydl.prepare_filename(info))

    # If format merge changed extension, normalize to .mp4 path.
    if downloaded.suffix.lower() != ".mp4":
        alt = downloaded.with_suffix(".mp4")
        if alt.exists():
            downloaded = alt

    if not downloaded.exists():
        raise FileNotFoundError(f"Download output not found: {downloaded}")

    normalized = output_dir / "source.mp4"
    if downloaded != normalized:
        shutil.move(str(downloaded), str(normalized))
        downloaded = normalized

    return downloaded


def transcribe_with_whisper(video_path: Path, model_name: str = "medium") -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(video_path),
        verbose=False,
        fp16=(device == "cuda"),
        word_timestamps=False,
    )
    return result


def sentence_score(text: str) -> float:
    clean = text.strip().lower()
    words = re.findall(r"[a-zA-Z']+", clean)
    word_count = len(words)

    trigger_hits = sum(1 for w in words if w in TRIGGER_WORDS)
    trigger_score = trigger_hits * 2.0

    question_score = 1.5 if "?" in clean else 0.0

    # Slight boost for substance without over-rewarding very long lines.
    if 8 <= word_count <= 35:
        length_score = 1.0
    elif word_count > 35:
        length_score = 0.5
    else:
        length_score = 0.0

    return trigger_score + question_score + length_score


def rank_segments(segments: Iterable[dict], top_n: int) -> List[ScoredSegment]:
    scored: List[ScoredSegment] = []
    for seg in segments:
        text = seg.get("text", "")
        score = sentence_score(text)
        scored.append(
            ScoredSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=text,
                score=score,
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)

    # Deduplicate near-identical timestamps to avoid overlapping picks.
    selected: List[ScoredSegment] = []
    for cand in scored:
        if len(selected) >= top_n:
            break
        if any(abs(cand.start - s.start) < 20 for s in selected):
            continue
        selected.append(cand)

    return selected


def cut_clip_stream_copy(source_video: Path, output_clip: Path, start_sec: float, duration_sec: float) -> None:
    output_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration_sec:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_clip),
    ]
    run_cmd(cmd)


def detect_face_centers(video_path: Path, detect_every: int = 10) -> List[tuple[int, int] | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip for detection: {video_path}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    centers: List[tuple[int, int] | None] = []
    last_center: tuple[int, int] | None = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        center = last_center
        if frame_idx % detect_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                center = (x + w // 2, y + h // 2)
                last_center = center

        centers.append(center)
        frame_idx += 1

    cap.release()
    return centers


def smooth_centers(centers: List[tuple[int, int] | None], alpha: float = 0.25) -> List[tuple[int, int]]:
    if not centers:
        return []

    valid = [c for c in centers if c is not None]
    if not valid:
        return [(0, 0)] * len(centers)

    # Fill missing centers by nearest-neighbor propagation.
    filled: List[tuple[int, int]] = [valid[0]] * len(centers)
    last = valid[0]
    for i, c in enumerate(centers):
        if c is not None:
            last = c
        filled[i] = last
    last = filled[-1]
    for i in range(len(filled) - 1, -1, -1):
        if centers[i] is None:
            filled[i] = last
        else:
            last = centers[i]  # type: ignore[assignment]

    # EMA smoothing for stable crop.
    smoothed: List[tuple[int, int]] = []
    sx, sy = filled[0]
    for cx, cy in filled:
        sx = int(alpha * cx + (1 - alpha) * sx)
        sy = int(alpha * cy + (1 - alpha) * sy)
        smoothed.append((sx, sy))
    return smoothed


def render_vertical_clip(
    input_clip: Path,
    output_clip: Path,
    centers: List[tuple[int, int]],
    out_width: int = 1080,
    out_height: int = 1920,
) -> None:
    cap = cv2.VideoCapture(str(input_clip))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip for rendering: {input_clip}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_w = min(in_w, int(in_h * (9 / 16)))
    crop_h = in_h

    output_clip.parent.mkdir(parents=True, exist_ok=True)
    tmp_raw = output_clip.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(tmp_raw),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )

    idx = 0
    default_center = (in_w // 2, in_h // 2)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cx, _ = centers[idx] if idx < len(centers) else default_center
        left = max(0, min(in_w - crop_w, cx - crop_w // 2))
        crop = frame[:, left : left + crop_w]
        out = cv2.resize(crop, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
        writer.write(out)
        idx += 1

    cap.release()
    writer.release()

    # Fast final encode, preferring GPU NVENC when available.
    nvenc_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_raw),
        "-i",
        str(input_clip),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "24",
        "-c:a",
        "aac",
        "-shortest",
        str(output_clip),
    ]

    cpu_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_raw),
        "-i",
        str(input_clip),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-shortest",
        str(output_clip),
    ]

    try:
        run_cmd(nvenc_cmd)
    except RuntimeError:
        run_cmd(cpu_cmd)
    finally:
        if tmp_raw.exists():
            tmp_raw.unlink()


def build_pipeline(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    clips_dir = work_dir / "clips"
    final_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Downloading source video...")
    source_video = download_youtube_mp4(args.url, work_dir)

    print("[2/5] Transcribing once with Whisper...")
    transcript = transcribe_with_whisper(source_video, model_name=args.whisper_model)
    (work_dir / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")

    print("[3/5] Ranking high-potential segments...")
    top_segments = rank_segments(transcript["segments"], args.top_n)

    print("[4/5] Cutting selected clips with stream copy...")
    for i, seg in enumerate(top_segments, start=1):
        clip_path = clips_dir / f"clip_{i:02d}.mp4"
        cut_clip_stream_copy(source_video, clip_path, seg.start, args.clip_len)

    print("[5/5] Face-aware vertical render only on selected clips...")
    for i, seg in enumerate(top_segments, start=1):
        src_clip = clips_dir / f"clip_{i:02d}.mp4"
        final_clip = final_dir / f"short_{i:02d}.mp4"
        centers = detect_face_centers(src_clip, detect_every=args.detect_every)
        smoothed = smooth_centers(centers, alpha=args.smooth_alpha)
        render_vertical_clip(src_clip, final_clip, smoothed)
        print(f"Saved: {final_clip} | score={seg.score:.2f} | start={seg.start:.2f}s")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized YouTube Shorts generation pipeline")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--work-dir", default="work", help="Temporary workspace directory")
    parser.add_argument("--output-dir", default="output_shorts", help="Directory for final rendered shorts")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model: small/medium/large-v3")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top clips to produce")
    parser.add_argument("--clip-len", type=float, default=45.0, help="Fixed short length in seconds")
    parser.add_argument("--detect-every", type=int, default=10, help="Run face detection every N frames")
    parser.add_argument("--smooth-alpha", type=float, default=0.25, help="EMA smoothing factor for face center")
    return parser.parse_args()


if __name__ == "__main__":
    build_pipeline(parse_args())
