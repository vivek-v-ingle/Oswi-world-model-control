#!/usr/bin/env python3
"""
Utility script to trim demonstration videos to the active pick-and-place segment.
"""

import sys
import argparse
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Trim Demonstration Video")
    parser.add_argument("--input", "-i", required=True, help="Input video file (.mp4)")
    parser.add_argument("--output", "-o", default="data/demos/demo_trimmed.mp4", help="Output trimmed video (.mp4)")
    parser.add_argument("--start", "-s", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", "-e", type=float, default=0.0, help="End time in seconds (0 for end of video)")
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        in_path = PROJECT_ROOT / args.input
    if not in_path.exists():
        print(f"[Error] File not found: {args.input}")
        return

    out_path = Path(args.output) if Path(args.output).is_absolute() else PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    start_sec = max(0.0, args.start)
    end_sec = args.end if args.end > 0 else duration

    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames)

    print("=" * 70)
    print(" Video Demonstration Trimmer ")
    print("=" * 70)
    print(f"Input: {in_path} ({total_frames} frames, {fps:.1f} fps, {duration:.2f}s)")
    print(f"Trimming range: {start_sec:.2f}s (frame {start_frame}) -> {end_sec:.2f}s (frame {end_frame})")
    print(f"Output: {out_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0

    curr_frame = start_frame
    while curr_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1
        curr_frame += 1

    cap.release()
    writer.release()

    print(f"✓ Successfully wrote {written} frames ({written / fps:.2f}s) to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
