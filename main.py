#!/usr/bin/env python3
"""
OSVI-WM (One-Shot Visual Imitation with World Models) on Fairino FR10.
Unified Entry Point CLI for Inference, Visualization, Calibration, and Robot Execution.
"""

import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="OSVI-WM: One-Shot Visual Imitation on Fairino Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  inference   Run offline trajectory prediction on sample demo (scripts/01_offline_inference.py)
  visualize   Generate side-by-side demo vs execution video & plots (scripts/05_visualize_demo_vs_execution.py)
  dry-run     Run IK preflight checks and waypoint reachability verification (scripts/03_robot_dry_run.py)
  execute     Deploy live trajectory on Fairino FR10 arm with camera (scripts/04_execute_on_fairino.py)
  calibrate   Interactive 4-point Eye-to-Base calibration (calibration/calibrate_4point.py)
  record      Record a top-view demonstration video (.mp4) (scripts/00_record_demo.py)
        """,
    )
    parser.add_argument(
        "mode",
        choices=["inference", "visualize", "dry-run", "execute", "calibrate", "record"],
        nargs="?",
        default="inference",
        help="Execution mode (default: inference)",
    )
    parser.add_argument("--demo-video", default="data/demos/bottle_top_view.mp4", help="Path to demo video")
    parser.add_argument("--checkpoint", default="checkpoints/pp_model.pt", help="Path to checkpoint")
    parser.add_argument("--ip", default="192.168.57.2", help="Fairino Robot Controller IP")
    parser.add_argument("--speed", type=float, default=8.0, help="Movement velocity percentage")
    parser.add_argument("--mock-robot", action="store_true", help="Run without physical robot connection")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")

    args, extra_args = parser.parse_known_args()

    script_map = {
        "inference": ["scripts/01_offline_inference.py", "--checkpoint", args.checkpoint],
        "visualize": ["scripts/05_visualize_demo_vs_execution.py", "--demo-video", args.demo_video, "--checkpoint", args.checkpoint],
        "dry-run": ["scripts/03_robot_dry_run.py", "--checkpoint", args.checkpoint],
        "execute": ["scripts/04_execute_on_fairino.py", "--ip", args.ip, "--speed", str(args.speed), "--checkpoint", args.checkpoint],
        "calibrate": ["calibration/calibrate_4point.py", "--ip", args.ip, "--camera-id", str(args.camera_id)],
        "record": ["scripts/00_record_demo.py", "--camera-id", str(args.camera_id)],
    }

    if args.mock_robot and args.mode in ("dry-run", "execute"):
        script_map[args.mode].append("--mock-robot")

    target_cmd = [sys.executable, str(PROJECT_ROOT / script_map[args.mode][0])] + script_map[args.mode][1:] + extra_args

    print(f"[OSVI-WM] Launching mode '{args.mode}': {' '.join(target_cmd)}\n")
    sys.exit(subprocess.call(target_cmd))


if __name__ == "__main__":
    main()
