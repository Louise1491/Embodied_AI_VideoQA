#!/usr/bin/env python3
"""
technical_check.py
------------------
Layer A automated QA script for egocentric manipulation video datasets.

Checks technical parameters against configurable thresholds for robot
training data collection standards. Designed as the first-pass filter
in a multi-layer QA pipeline — catches format/spec issues at near-zero
cost before human review begins.

Usage:
    python technical_check.py <video_folder>
    python technical_check.py .             # scan current directory

Output:
    - Console report per video (pass/review/reject + reason)
    - Batch summary with reject rate
    - CSV report saved to the scanned folder

Requirements:
    - ffprobe (part of ffmpeg): https://ffmpeg.org/download.html
    - Python 3.8+

Author: Louise Wang
"""

import subprocess
import json
import os
import sys
import csv
from pathlib import Path

# ── Configurable thresholds ───────────────────────────────────────────────────
# Adjust these to match your collection standard's requirements

REQUIRED_WIDTH = 1920          # pixels
REQUIRED_HEIGHT = 1080         # pixels
REQUIRED_FPS = 30              # frames per second (±0.5 tolerance)
MIN_DURATION_SEC = 30          # hard floor: clips shorter than this are rejected
TARGET_DURATION_SEC = 60       # target duration (EgoVerse reference: ~61s/episode)
MIN_BITRATE_KBPS = 8000        # below this: reject (heavy compression)
TARGET_BITRATE_KBPS = 12000    # below this: warn (quality at risk)
ALLOWED_CODECS = ["h264", "hevc"]

# ─────────────────────────────────────────────────────────────────────────────


def get_video_info(filepath):
    """
    Extract video metadata using ffprobe.
    Returns parsed JSON dict, or None if file cannot be read.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(filepath)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def parse_fps(fps_raw):
    """Parse ffprobe frame rate string (e.g. '30/1' or '29.97')."""
    try:
        if "/" in fps_raw:
            num, den = fps_raw.split("/")
            return round(int(num) / int(den), 2)
        return round(float(fps_raw), 2)
    except Exception:
        return 0


def check_video(filepath):
    """
    Run all Layer A technical checks on a single video file.

    Returns a dict with:
        file     : filename
        verdict  : PASS | REVIEW | REJECT | ERROR
        blockers : list of Blocker-level issues (must fix before resubmit)
        warnings : list of Major-level issues (should fix)
        notes    : list of items requiring human follow-up
        params   : dict of raw extracted parameters
    """
    info = get_video_info(filepath)
    blockers = []
    warnings = []
    notes = []

    if not info:
        return {
            "file": Path(filepath).name,
            "verdict": "ERROR",
            "blockers": ["Could not read file — may be corrupted or unsupported format"],
            "warnings": [], "notes": [], "params": {}
        }

    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        None
    )
    fmt = info.get("format", {})

    if not video_stream:
        return {
            "file": Path(filepath).name,
            "verdict": "REJECT",
            "blockers": ["No video stream found in file"],
            "warnings": [], "notes": [], "params": {}
        }

    # ── Extract raw parameters ────────────────────────────────────────────────
    width    = video_stream.get("width", 0)
    height   = video_stream.get("height", 0)
    codec    = video_stream.get("codec_name", "unknown")
    duration = float(fmt.get("duration", 0))
    bitrate  = int(fmt.get("bit_rate", 0)) // 1000  # convert to kbps
    fps      = parse_fps(video_stream.get("avg_frame_rate", "0/1"))
    filename = Path(filepath).name

    # Detect if a de-fisheye / rectified version exists alongside this file
    is_undistorted = "_undistorted" in filename.lower()

    params = {
        "resolution": f"{width}x{height}",
        "fps":        f"{fps}",
        "codec":      codec,
        "duration_s": round(duration, 1),
        "bitrate_kbps": bitrate,
    }

    # ── Check 1: Resolution ───────────────────────────────────────────────────
    if width != REQUIRED_WIDTH or height != REQUIRED_HEIGHT:
        blockers.append(
            f"Resolution mismatch: got {width}x{height}, "
            f"required {REQUIRED_WIDTH}x{REQUIRED_HEIGHT}"
        )

    # ── Check 2: Frame rate ───────────────────────────────────────────────────
    if abs(fps - REQUIRED_FPS) > 0.5:
        blockers.append(
            f"Frame rate mismatch: got {fps} fps, required {REQUIRED_FPS} fps"
        )

    # ── Check 3: Duration ─────────────────────────────────────────────────────
    if duration < MIN_DURATION_SEC:
        blockers.append(
            f"Clip too short: {duration:.1f}s — hard minimum is {MIN_DURATION_SEC}s"
        )
    elif duration < TARGET_DURATION_SEC:
        warnings.append(
            f"Duration {duration:.1f}s is below target ({TARGET_DURATION_SEC}s). "
            f"Acceptable but re-shoot recommended for richer training signal."
        )

    # ── Check 4: Codec ────────────────────────────────────────────────────────
    if codec.lower() not in ALLOWED_CODECS:
        warnings.append(
            f"Unexpected codec: {codec}. Preferred: H.264 (h264)"
        )

    # ── Check 5: Bitrate ─────────────────────────────────────────────────────
    if bitrate < MIN_BITRATE_KBPS:
        blockers.append(
            f"Bitrate too low: {bitrate} kbps — "
            f"minimum {MIN_BITRATE_KBPS} kbps required (heavy compression detected)"
        )
    elif bitrate < TARGET_BITRATE_KBPS:
        warnings.append(
            f"Bitrate {bitrate} kbps is below recommended target "
            f"({TARGET_BITRATE_KBPS} kbps). Fine detail may be lost in dark regions."
        )

    # ── Check 6: Fisheye / rectification flag ────────────────────────────────
    if not is_undistorted:
        notes.append(
            "Filename does not contain '_undistorted'. "
            "If filmed with a fisheye lens, submit a rectified version "
            "OR provide camera intrinsics (fx, fy, cx, cy, distortion coefficients)."
        )

    # ── Verdict ───────────────────────────────────────────────────────────────
    if blockers:
        verdict = "REJECT"
    elif warnings:
        verdict = "REVIEW"   # passes Layer A but flags for human attention
    else:
        verdict = "PASS"     # clear for Layer B / human content review

    return {
        "file": filename, "verdict": verdict,
        "blockers": blockers, "warnings": warnings,
        "notes": notes, "params": params
    }


def run_batch(folder):
    """Scan a folder and run technical checks on all video files found."""
    folder = Path(folder)
    VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v",
                  ".MP4", ".MOV", ".AVI", ".MKV"}
    video_files = sorted([f for f in folder.iterdir() if f.suffix in VIDEO_EXTS])

    if not video_files:
        print(f"No video files found in: {folder}")
        return

    print(f"\nScanning: {folder}")
    print(f"Videos found: {len(video_files)}\n")
    print("=" * 70)

    results = []
    summary = {"PASS": 0, "REVIEW": 0, "REJECT": 0, "ERROR": 0}
    icons = {"PASS": "✅", "REVIEW": "🟡", "REJECT": "❌", "ERROR": "💥"}

    for vf in video_files:
        print(f"Checking: {vf.name} ...")
        r = check_video(vf)
        results.append(r)
        summary[r["verdict"]] = summary.get(r["verdict"], 0) + 1

        print(f"\n{icons.get(r['verdict'], '?')} [{r['verdict']}] {r['file']}")
        for k, v in r["params"].items():
            print(f"    {k}: {v}")
        for b in r["blockers"]:
            print(f"    ⛔ BLOCKER: {b}")
        for w in r["warnings"]:
            print(f"    ⚠️  WARNING: {w}")
        for n in r["notes"]:
            print(f"    📌 NOTE: {n}")
        print()

    # ── Batch summary ─────────────────────────────────────────────────────────
    total = len(video_files)
    print("=" * 70)
    print("BATCH SUMMARY")
    print(f"  ✅ PASS   (proceed to human review): {summary['PASS']}")
    print(f"  🟡 REVIEW (flag for attention):       {summary['REVIEW']}")
    print(f"  ❌ REJECT (technical failure):        {summary['REJECT']}")
    if summary["ERROR"]:
        print(f"  💥 ERROR  (unreadable files):         {summary['ERROR']}")
    reject_rate = (summary["REJECT"] + summary["ERROR"]) / total * 100
    print(f"\n  Layer A reject rate: {reject_rate:.1f}%")
    print("=" * 70)

    # ── CSV export ────────────────────────────────────────────────────────────
    csv_path = folder / "qa_technical_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "verdict",
            "resolution", "fps", "codec", "duration_s", "bitrate_kbps",
            "blockers", "warnings", "notes"
        ])
        for r in results:
            writer.writerow([
                r["file"], r["verdict"],
                r["params"].get("resolution", ""),
                r["params"].get("fps", ""),
                r["params"].get("codec", ""),
                r["params"].get("duration_s", ""),
                r["params"].get("bitrate_kbps", ""),
                " | ".join(r["blockers"]),
                " | ".join(r["warnings"]),
                " | ".join(r["notes"]),
            ])
    print(f"\nCSV report saved to: {csv_path}\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    run_batch(target)
