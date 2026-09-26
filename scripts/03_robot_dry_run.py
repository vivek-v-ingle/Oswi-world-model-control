#!/usr/bin/env python3
"""
Step 3: Robot Dry-Run and Inverse Kinematics Preflight Check.
Validates that all predicted 3D waypoints are collision-free and kinematics-feasible
without commanding physical arm movements.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference_engine import InferenceEngine
from perception.video_loader import DemonstrationLoader
from perception.camera_stream import CameraStream
from robot.fairino_driver import FairinoDriver
from robot.trajectory_follower import TrajectoryFollower
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Robot Dry Run & IK Validation")
    parser.add_argument("--demo-video", required=False, help="Path to 10-frame teacher demo video (.mp4)")
    parser.add_argument("--demo-dir", required=False, help="Path to folder containing demo images")
    parser.add_argument("--camera", default="opencv", choices=["opencv", "zed"], help="Camera type")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index for OpenCV")
    parser.add_argument("--use-camera", action="store_true", help="Capture real live frame from camera")
    parser.add_argument("--ip", default="192.168.57.2", help="Fairino Robot IP")
    parser.add_argument("--mock", action="store_true", default=False, help="Force Mock mode for offline testing")
    parser.add_argument("--speed", type=float, default=10.0, help="Movement velocity percentage")
    parser.add_argument("--checkpoint", default="checkpoints/pp_model.pt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Demo video start timestamp in seconds")
    parser.add_argument("--end-sec", type=float, default=0.0, help="Demo video end timestamp in seconds")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Safety lift offset in mm")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not checkpoint.exists():
        checkpoint = PROJECT_ROOT / "checkpoints" / "metaworld_model.pt"
    calibration = PROJECT_ROOT / "calibration" / "cam2base_calibration.json"

    print("=" * 70)
    print(" Fairino FR10 Dry-Run & Kinematics Preflight Check ")
    print("=" * 70)

    # 1. Load Demo
    if args.demo_video:
        print(f"Loading teacher demo from video: {args.demo_video} (range: {args.start_sec}s - {args.end_sec or 'end'}s)")
        teacher_frames = DemonstrationLoader.load_video(
            args.demo_video,
            max_frames=10,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
        )
    elif args.demo_dir:
        print(f"Loading teacher demo from image directory: {args.demo_dir}")
        teacher_frames = DemonstrationLoader.load_image_folder(args.demo_dir, max_frames=10)
    else:
        print("[Info] No demo path provided. Using synthetic demonstration frames for testing.")
        teacher_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]

    # 2. Capture or simulate observation
    if args.use_camera:
        camera = CameraStream(camera_type=args.camera, camera_id=args.camera_id)
        camera.start()
        print("Capturing workspace observation image from camera...")
        obs_frame = camera.get_frame()
        camera.stop()
    else:
        obs_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # 3. Run inference to get 3D waypoints
    print("\nRunning World Model Foreseeing Inference...")
    engine = InferenceEngine(checkpoint_path=checkpoint, calibration_path=calibration)
    result = engine.predict(teacher_frames=teacher_frames, current_obs=obs_frame, rollout_horizon=16)
    robot_wps = result["robot_waypoints"]

    # 4. Connect driver
    driver = FairinoDriver(robot_ip=args.ip, mock=args.mock)
    driver.connect()

    # 5. Preflight IK check
    follower = TrajectoryFollower(
        driver=driver,
        speed=args.speed,
        min_z_mm=-300.0,
        max_z_mm=4000.0,
        z_offset_mm=args.z_offset,
    )
    ok, validated = follower.preflight_check(robot_wps)

    if ok:
        print("\n[Dry Run Summary]")
        print(f"✓ All {len(validated)} waypoints passed Inverse Kinematics & workspace limits.")
        print("✓ System is READY for safe physical execution on the Fairino robot.")
    else:
        print("\n[Dry Run Summary] Preflight check found kinematic violations.")

    print("=" * 70)


if __name__ == "__main__":
    main()
