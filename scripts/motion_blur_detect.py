#!/usr/bin/env python3
"""
motion_blur_detect.py
---------------------
Layer B visual quality check for egocentric manipulation video datasets.

Detects motion blur in video clips using Laplacian variance analysis.
Designed as a semi-automated filter within a multi-layer QA pipeline —
identifies frames where hand motion blur may make pose keypoints
undetectable for model training.

This is v1 of the implementation. See KNOWN LIMITATIONS below.

Usage:
    python motion_blur_detect.py <video_file>
    python motion_blur_detect.py <video_folder>   # batch mode

Output:
    - Console report: blurry frame timestamps + severity
    - Sharpness curve chart saved as PNG
    - CSV report saved alongside the input video

Requirements:
    pip install opencv-python numpy matplotlib

Author: Louise Wang

---------------------------------------------------------------------
KNOWN LIMITATIONS (v1)

This implementation applies Laplacian variance to the full frame.
This causes a systematic false-negative issue in high-texture scenes:
background patterns (metal grids, wood grain, patterned fabric) inflate
the per-frame variance score, masking blur in the hand region.

Planned improvements:
  v2 — ROI-based detection
       Use MediaPipe Hand Landmark to localize the hand region first,
       then compute Laplacian variance only within that bounding box.
       Eliminates background texture interference entirely.

  v3 — Optical flow analysis
       Compute per-frame motion vectors (Farneback method) to measure
       movement speed rather than static sharpness. This enables
       distinction between:
         • Hand-motion blur  → operator moved too fast (actionable)
         • Tool-vibration blur → tool spinning at high RPM (not operator fault)
       The v1 approach cannot distinguish these two cases.

v2 and v3 implementations are in progress.
---------------------------------------------------------------------
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import sys
from pathlib import Path

# ── Configurable thresholds ───────────────────────────────────────────────────
# Tune these based on your camera setup and scene characteristics.
# Lower values = more sensitive (flags more frames as blurry).

SAMPLE_INTERVAL_SEC = 1    # seconds between sampled frames
BLUR_THRESHOLD      = 80   # Laplacian variance below this → blur (Major)
SEVERE_THRESHOLD    = 40   # Laplacian variance below this → severe blur (Blocker)

# ─────────────────────────────────────────────────────────────────────────────


def fmt_time(seconds):
    """Convert seconds to mm:ss string."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def analyze_video(video_path):
    """
    Analyze a single video file for motion blur.

    Samples one frame per SAMPLE_INTERVAL_SEC, computes Laplacian variance
    for each frame, and flags frames below threshold as blurry.

    Returns list of blur records (dicts with timestamp, score, severity).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path.name}")
        return []

    fps           = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration      = total_frames / fps
    step          = int(fps * SAMPLE_INTERVAL_SEC)

    print(f"\n[{video_path.name}]")
    print(f"  Duration: {fmt_time(duration)}  |  FPS: {fps:.0f}  |  Sample interval: {SAMPLE_INTERVAL_SEC}s")
    print(f"  Thresholds — blur: <{BLUR_THRESHOLD}  |  severe: <{SEVERE_THRESHOLD}\n")

    timestamps, scores = [], []
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        timestamps.append(frame_idx / fps)
        scores.append(score)
        frame_idx += step

    cap.release()

    times_arr  = np.array(timestamps)
    scores_arr = np.array(scores)

    # ── Print blur frame list ─────────────────────────────────────────────────
    print(f"  {'Time':>8}  {'Score':>8}  Severity")
    print("  " + "-" * 35)

    blur_records = []
    for t, s in zip(times_arr, scores_arr):
        if s < BLUR_THRESHOLD:
            if s < SEVERE_THRESHOLD:
                severity = "SEVERE"
                tag      = "🔴 SEVERE BLUR"
                action   = "Blocker — reshoot required"
            else:
                severity = "BLUR"
                tag      = "🟡 BLUR"
                action   = "Major — human review recommended"

            print(f"  {fmt_time(t):>8}  {s:>8.1f}  {tag}")
            blur_records.append({
                "timestamp":    fmt_time(t),
                "score":        round(s, 1),
                "severity":     severity,
                "action":       action,
            })

    if not blur_records:
        print("  ✅ No significant blur detected")

    # ── Stats ─────────────────────────────────────────────────────────────────
    blur_mask   = scores_arr < BLUR_THRESHOLD
    severe_mask = scores_arr < SEVERE_THRESHOLD
    print(f"\n  Sampled frames: {len(scores)}")
    print(f"  Blurry:  {blur_mask.sum()} ({blur_mask.mean()*100:.1f}%)")
    print(f"  Severe:  {severe_mask.sum()}")
    print(f"  Mean score: {scores_arr.mean():.1f}  |  Min: {scores_arr.min():.1f} @ {fmt_time(times_arr[scores_arr.argmin()])}")

    # ── CSV export ────────────────────────────────────────────────────────────
    out_dir  = video_path.parent
    csv_path = out_dir / f"{video_path.stem}_blur_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "score", "severity", "action"])
        writer.writeheader()
        writer.writerows(blur_records)
    print(f"\n  CSV saved: {csv_path.name}")

    # ── Sharpness curve chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    for t, s in zip(times_arr, scores_arr):
        if s < SEVERE_THRESHOLD:
            ax.axvspan(t - 0.5, t + 0.5, alpha=0.35, color='#ff4444', zorder=1)
        elif s < BLUR_THRESHOLD:
            ax.axvspan(t - 0.5, t + 0.5, alpha=0.2,  color='#ffaa00', zorder=1)

    ax.axhline(BLUR_THRESHOLD,   color='#ffaa00', linestyle='--', lw=1.2,
               label=f'Blur threshold ({BLUR_THRESHOLD})')
    ax.axhline(SEVERE_THRESHOLD, color='#ff4444', linestyle='--', lw=1.2,
               label=f'Severe threshold ({SEVERE_THRESHOLD})')
    ax.plot(times_arr, scores_arr, color='#00d4ff', lw=1.8, zorder=3,
            label='Sharpness score (Laplacian variance)')
    ax.fill_between(times_arr, scores_arr, alpha=0.15, color='#00d4ff', zorder=2)

    xticks = np.arange(0, duration + 1, 30)
    ax.set_xticks(xticks)
    ax.set_xticklabels([fmt_time(t) for t in xticks], color='#aaaacc', fontsize=9)
    ax.set_xlabel("Time", color='#aaaacc')
    ax.set_ylabel("Sharpness Score (Laplacian Variance)", color='#aaaacc')
    ax.set_title(f"Motion Blur Detection (v1) — {video_path.name}",
                 color='white', fontsize=12, pad=10)
    ax.tick_params(colors='#aaaacc')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#333355',
              labelcolor='white', fontsize=9)

    # Annotate known limitation
    ax.text(0.01, 0.04,
            "⚠ v1 limitation: full-frame analysis — high-texture backgrounds may cause false negatives. "
            "See v2/v3 roadmap in docstring.",
            transform=ax.transAxes, fontsize=7.5, color='#aaaacc', style='italic')

    plt.tight_layout()
    chart_path = out_dir / f"{video_path.stem}_blur_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved: {chart_path.name}\n")

    return blur_records


def run_folder(folder):
    """Batch mode: scan all video files in a folder."""
    folder = Path(folder)
    VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v",
                  ".MOV", ".MP4", ".AVI", ".MKV"}
    videos = sorted([f for f in folder.iterdir() if f.suffix in VIDEO_EXTS])

    if not videos:
        print(f"No video files found in: {folder}")
        return

    print(f"Batch mode: {len(videos)} videos found in {folder}")
    for v in videos:
        analyze_video(v)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python motion_blur_detect.py video.mp4")
        print("  python motion_blur_detect.py /path/to/folder/")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        run_folder(target)
    else:
        analyze_video(target)
