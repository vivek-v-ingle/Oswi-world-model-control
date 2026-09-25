#!/usr/bin/env python3
"""
Step 0: Interactive Video Demonstration Recorder.
Records a live demonstration from the top-view camera (e.g. ZED / USB webcam)
and saves it as an MP4 video formatted for OSVI-WM input.
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Record Top-View Demo Video for OSVI-WM")
    parser.add_argument(
        "--output",
        "-o",
        default="data/demos/live_demo.mp4",
        help="Path where output MP4 will be saved",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (/dev/videoX)")
    parser.add_argument("--fps", type=float, default=15.0, help="Video recording frame rate")
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=0.0,
        help="Auto-stop after D seconds (0 for manual stop via Ctrl+C / 'q')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" OSVI-WM Live Demonstration Recorder ")
    print("=" * 70)
    print(f"Connecting to Camera Index: {args.camera_id} ...")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {args.camera_id}.")
        return

    # Read test frame to get dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("[Error] Failed to capture test frame from camera.")
        cap.release()
        return

    h, w = test_frame.shape[:2]
    is_stereo = w > h * 1.7
    out_w = w // 2 if is_stereo else w
    out_h = h

    print(f"Camera stream active: Input={w}x{h} -> Output Crop={out_w}x{out_h} (Stereo Left Crop: {is_stereo})")
    print(f"Target Output: {out_path} @ {args.fps} FPS")
    print("\n[Controls] Starting recording now...")
    if args.duration > 0:
        print(f"Recording will automatically stop after {args.duration:.1f} seconds.")
    else:
        print("Press Ctrl+C in terminal when your demonstration is finished.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (out_w, out_h))

    start_time = time.time()
    frames_recorded = 0
    frame_interval = 1.0 / args.fps

    try:
        while True:
            t_loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # If stereo (e.g. ZED), take left half
            if is_stereo:
                crop = frame[:, :out_w]
            else:
                crop = frame

            writer.write(crop)
            frames_recorded += 1
            elapsed = time.time() - start_time

            print(f"\rRecording: {elapsed:5.1f}s | Frames: {frames_recorded:4d}", end="", flush=True)

            if args.duration > 0 and elapsed >= args.duration:
                print(f"\n[Info] Reached duration limit ({args.duration}s).")
                break

            sleep_time = frame_interval - (time.time() - t_loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Info] Recording stopped by user.")
    finally:
        cap.release()
        writer.release()

    print("\n" + "=" * 70)
    print(f"✓ Video successfully saved to: {out_path}")
    print(f"✓ Total Frames: {frames_recorded} | Duration: {frames_recorded / args.fps:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
