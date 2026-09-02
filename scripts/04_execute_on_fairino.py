#!/usr/bin/env python3
"""
Step 4: End-to-End Live Deployment on Fairino Robot with Camera Stream.
Captures live workspace image (ZED/USB camera) + teacher demo video,
runs OSVI-WM world model inference, and executes the trajectory safely on the Fairino arm.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference_engine import InferenceEngine
from perception.video_loader import DemonstrationLoader
from perception.camera_stream import CameraStream
from robot.fairino_driver import FairinoDriver
from robot.gripper import GripperController
from robot.trajectory_follower import TrajectoryFollower


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy OSVI-WM on Fairino Robot")
    parser.add_argument("--demo-video", required=False, help="Path to 10-frame teacher demo video (.mp4)")
    parser.add_argument("--demo-dir", required=False, help="Path to folder containing demo images")
    parser.add_argument("--camera", default="opencv", choices=["opencv", "zed"], help="Camera type")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index for OpenCV")
    parser.add_argument("--ip", default="192.168.57.2", help="Fairino Robot IP")
    parser.add_argument("--speed", type=float, default=10.0, help="Movement velocity percentage (1-100)")
    parser.add_argument("--use-gripper", action="store_true", default=True, help="Enable gripper actuation")
    parser.add_argument("--mock-robot", action="store_true", help="Run without physical robot connection")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Safety lift offset in mm")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = PROJECT_ROOT / "checkpoints" / "metaworld_model.pt"
    calibration = PROJECT_ROOT / "calibration" / "cam2base_calibration.json"

    print("=" * 70)
    print(" OSVI-WM Live Deployment on Fairino FR10 ")
    print("=" * 70)

    # 1. Load Teacher Demonstration (10 frames)
    if args.demo_video:
        print(f"Loading teacher demo from video: {args.demo_video}")
        teacher_frames = DemonstrationLoader.load_video(args.demo_video, max_frames=10)
    elif args.demo_dir:
        print(f"Loading teacher demo from image directory: {args.demo_dir}")
        teacher_frames = DemonstrationLoader.load_image_folder(args.demo_dir, max_frames=10)
    else:
        print("[Info] No demo path provided. Using synthetic demonstration frames for testing.")
        teacher_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]

    # 2. Capture Initial Workspace Image
    camera = CameraStream(camera_type=args.camera, camera_id=args.camera_id)
    camera.start()
    print("Capturing workspace observation image...")
    obs_frame = camera.get_frame()
    camera.stop()

    # 3. Predict Trajectory using OSVI-WM World Model
    print("\nRunning World Model Foreseeing Inference...")
    engine = InferenceEngine(checkpoint_path=checkpoint, calibration_path=calibration)
    result = engine.predict(teacher_frames=teacher_frames, current_obs=obs_frame, rollout_horizon=16)
    robot_wps = result["robot_waypoints"]

    # 4. Connect to Fairino Robot
    driver = FairinoDriver(robot_ip=args.ip, mock=args.mock_robot)
    if not driver.connect():
        print("[Warning] Could not connect to physical robot. Running in simulation/mock mode.")

    gripper = GripperController(driver=driver) if args.use_gripper else None
    follower = TrajectoryFollower(
        driver=driver,
        gripper=gripper,
        speed=args.speed,
        min_z_mm=-300.0,
        max_z_mm=4000.0,
        z_offset_mm=args.z_offset,
    )

    # 5. Execute Safe Trajectory
    success = follower.execute(
        waypoints=robot_wps,
        start_with_movej=True,
        return_to_start=False,
    )

    if success:
        print("\n✓ OSVI-WM Robot Task Execution Finished Successfully!")
    else:
        print("\n✗ OSVI-WM Execution encountered errors.")


if __name__ == "__main__":
    main()
