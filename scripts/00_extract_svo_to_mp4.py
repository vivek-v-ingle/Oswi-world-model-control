#!/usr/bin/env python3
"""
ZED SVO/SVO2 to MP4 Video & Frame Extractor.
Reads ZED .svo or .svo2 files, extracts the left monocular camera view,
and exports directly to a standard .mp4 video ready for OSVI-WM.
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    import pyzed.sl as sl
    PYZED_AVAILABLE = True
except ImportError:
    PYZED_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ZED SVO/SVO2 to MP4")
    parser.add_argument("--input", "-i", required=True, help="Path to .svo or .svo2 file")
    parser.add_argument("--output", "-o", default="data/demos/svo_demo.mp4", help="Output .mp4 video path")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end-frame", type=int, default=0, help="End frame index (0 for end of SVO)")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end-sec", type=float, default=0.0, help="End time in seconds")
    parser.add_argument("--fps", type=float, default=15.0, help="Output MP4 frame rate")
    return parser.parse_args()


def main():
    if not PYZED_AVAILABLE:
        print("[Error] pyzed.sl is not available in current python environment.")
        print("Run with python3.10: python3.10 scripts/00_extract_svo_to_mp4.py ...")
        sys.exit(1)

    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[Error] Input SVO file not found: {in_path}")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" ZED SVO -> MP4 Demonstration Converter ")
    print("=" * 70)
    print(f"Loading SVO file: {in_path}")

    # Initialize ZED camera with SVO playback
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(in_path))
    init_params.svo_real_time_mode = False  # Fast offline decoding
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Failed to open SVO file: {status}")
        sys.exit(1)

    total_frames = zed.get_svo_number_of_frames()
    svo_fps = zed.get_camera_information().camera_configuration.fps or 15.0
    res = zed.get_camera_information().camera_configuration.resolution
    w, h = res.width, res.height

    print(f"SVO Loaded: {total_frames} frames @ {svo_fps:.1f} FPS | Resolution: {w}x{h}")

    # Determine start and end frames
    if args.start_sec > 0:
        start_f = int(args.start_sec * svo_fps)
    else:
        start_f = max(0, args.start_frame)

    if args.end_sec > 0:
        end_f = min(total_frames, int(args.end_sec * svo_fps))
    elif args.end_frame > 0:
        end_f = min(total_frames, args.end_frame)
    else:
        end_f = total_frames

    print(f"Extracting frame range: {start_f} -> {end_f} (Total frames: {end_f - start_f})")
    print(f"Output MP4: {out_path} @ {args.fps} FPS")

    # Set SVO position to start frame
    zed.set_svo_position(start_f)

    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))

    image_mat = sl.Mat()
    frames_written = 0

    while zed.get_svo_position() < end_f:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            bgra = image_mat.get_data()
            bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            writer.write(bgr)
            frames_written += 1
            curr_pos = zed.get_svo_position()
            print(f"\rProgress: Frame {curr_pos}/{end_f} ({frames_written} written)", end="", flush=True)
        else:
            break

    writer.release()
    zed.close()

    print("\n" + "=" * 70)
    print(f"✓ Video successfully exported to: {out_path}")
    print(f"✓ Total Frames: {frames_written} | Duration: {frames_written / args.fps:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
