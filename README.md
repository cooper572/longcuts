# Optimized YouTube Shorts Pipeline (Colab T4)

This repository contains a performance-first Shorts generation pipeline designed for long-form videos (e.g., 2 hours).

## Why this is fast

The pipeline avoids processing the full video frame-by-frame:

1. **Download once** in MP4.
2. **Transcribe once** with Whisper on GPU.
3. **Score + rank transcript segments** by viral potential.
4. **Cut only top-N windows** with FFmpeg stream copy (`-c copy`, no re-encode).
5. **Run face detection + 9:16 rendering only on selected clips**.

This architecture dramatically reduces total frame processing and can approach ~90% time savings compared with full-video render-then-select workflows.

## Viral potential heuristic

Each transcript segment is scored using:

- Emotional trigger words (`secret`, `shocking`, `mistake`, `why`, ...)
- Presence of a question mark (`?`) as engagement signal
- Slight sentence-length boost for substantive lines

Top segments are deduplicated by timestamp proximity to reduce near-duplicate clips.

## Colab setup (T4)

```bash
!apt-get update -qq && apt-get install -y ffmpeg
!pip install -q --upgrade yt-dlp openai-whisper opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> If your Colab runtime already includes CUDA-enabled PyTorch, keep that version and install only missing packages.

## Usage

```bash
python shorts_pipeline.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --work-dir work \
  --output-dir output_shorts \
  --whisper-model medium \
  --top-n 5 \
  --clip-len 45 \
  --detect-every 10 \
  --smooth-alpha 0.25
```

## Outputs

- `work/source.mp4` – downloaded source
- `work/transcript.json` – full transcript result
- `work/clips/clip_XX.mp4` – raw extracted windows (stream-copy)
- `output_shorts/short_XX.mp4` – final vertical shorts

## Performance tips for T4

- Use `--whisper-model medium` for speed/quality balance.
- Increase `--detect-every` (e.g., 12 or 15) for faster face tracking.
- Keep `--top-n` low (3–7) when iterating.
- Keep clip length fixed (`--clip-len 45`) to stabilize runtime.
- Pipeline auto-tries `h264_nvenc` for encode and falls back to CPU `libx264`.

## Notes

- FFmpeg stream-copy clip extraction is near-instant versus re-encoding.
- Final rendering still decodes/encodes frames, but only for selected clips.
