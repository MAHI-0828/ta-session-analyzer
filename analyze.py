#!/usr/bin/env python3
"""
CLI entry point for Lecture Screenshot Analyzer.
Usage: python analyze.py --folder ./screenshots --batch "DS-Batch-12" --module "Module-3-SQL"
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

from core import (
    PARAMETERS, PARAMETER_LABELS,
    load_image_file, analyze_image,
    build_csv_rows, aggregate_scores,
)


def collect_images(folder: Path) -> list:
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in extensions])


def extract_frames_from_video(video_path: Path, output_dir: Path, count: int = 10) -> list:
    import subprocess
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr}")

    duration = float(probe.stdout.strip())
    interval = duration / count

    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path),
         "-vf", f"fps=1/{interval:.2f}",
         "-vframes", str(count),
         str(output_dir / "frame_%02d.png"), "-y"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    frames = sorted(output_dir.glob("frame_*.png"))
    print(f"  Extracted {len(frames)} frames into {output_dir}")
    return frames


def print_report_card(batch: str, module: str, results: list):
    print(f"\n{'='*65}")
    print(f"  SESSION REPORT")
    print(f"  Batch  : {batch}")
    print(f"  Module : {module}")
    print(f"  Frames : {len(results)} screenshots analyzed")
    print(f"{'='*65}")

    averages = aggregate_scores(results)
    flagged = []

    for param in PARAMETERS:
        avg = averages.get(param)
        if avg is None:
            continue
        bar = "█" * round(avg) + "░" * (5 - round(avg))
        flag = "  ← FLAG" if avg < 3.5 else ""
        if avg < 3.5:
            flagged.append((PARAMETER_LABELS[param], avg))
        print(f"  {PARAMETER_LABELS[param]:<38} {bar}  {avg:.1f}/5{flag}")

    overall_scores = [r["overall_score"] for r in results if isinstance(r.get("overall_score"), (int, float))]
    if overall_scores:
        overall = sum(overall_scores) / len(overall_scores)
        bar = "█" * round(overall) + "░" * (5 - round(overall))
        print(f"\n  {'OVERALL':<38} {bar}  {overall:.1f}/5")

    if flagged:
        flagged.sort(key=lambda x: x[1])
        print(f"\n  Top issues to fix:")
        for label, avg in flagged[:3]:
            print(f"    • {label} ({avg:.1f}/5)")

    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze lecture screenshots across 16 visual quality parameters."
    )
    parser.add_argument("--folder",  "-f", required=True,  help="Folder containing screenshots")
    parser.add_argument("--batch",   "-b", required=True,  help="Batch name, e.g. 'DS-Batch-12'")
    parser.add_argument("--module",  "-m", required=True,  help="Lecture module, e.g. 'Module-3-SQL'")
    parser.add_argument("--video",   "-v",                 help="Video file — auto-extracts frames via ffmpeg")
    parser.add_argument("--frames",  type=int, default=10, help="Frames to extract from video (default: 10)")
    parser.add_argument("--output",  "-o",                 help="Output directory (default: same as --folder)")
    parser.add_argument("--api-key", "-k",                 help="Google AI Studio API key (or set GOOGLE_API_KEY env var)")
    args = parser.parse_args()

    folder     = Path(args.folder)
    output_dir = Path(args.output) if args.output else folder
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: video file not found: {video_path}")
            sys.exit(1)
        print(f"Extracting {args.frames} frames from '{video_path.name}'...")
        extract_frames_from_video(video_path, folder, args.frames)

    images = collect_images(folder)
    if not images:
        print(f"No images found in {folder}")
        sys.exit(1)
    print(f"Found {len(images)} screenshots. Starting analysis...\n")

    import os
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("Error: provide --api-key or set GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    results = []

    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name} ...", end=" ", flush=True)
        try:
            pil_image = load_image_file(img_path)
            analysis  = analyze_image(api_key, pil_image)
            analysis["screenshot"]  = img_path.name
            analysis["analyzed_at"] = datetime.now().isoformat()
            results.append(analysis)
            print(f"done  (overall {analysis.get('overall_score', '?')}/5)")
        except json.JSONDecodeError as e:
            print(f"FAILED (bad JSON: {e})")
        except Exception as e:
            print(f"FAILED ({e})")

    if not results:
        print("No results to save.")
        sys.exit(1)

    slug      = f"{args.batch}_{args.module}".replace(" ", "-").replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{slug}_{timestamp}"

    json_path = output_dir / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"batch_name": args.batch, "lecture_module": args.module,
                   "analyzed_at": timestamp, "results": results},
                  f, indent=2, ensure_ascii=False)

    csv_path = output_dir / f"{base_name}.csv"
    rows = build_csv_rows(args.batch, args.module, results)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nJSON saved : {json_path}")
    print(f"CSV saved  : {csv_path}")
    print_report_card(args.batch, args.module, results)


if __name__ == "__main__":
    main()
