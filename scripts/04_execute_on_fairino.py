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
    parser.add_argument("--checkpoint", default="checkpoints/pp_model.pt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Demo video start timestamp in seconds")
    parser.add_argument("--end-sec", type=float, default=0.0, help="Demo video end timestamp in seconds")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Safety lift offset in mm")
    parser.add_argument("--min-z", type=float, default=-600.0, help="Minimum allowable Z height in mm")
    parser.add_argument("--max-z", type=float, default=4000.0, help="Maximum allowable Z height in mm")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not checkpoint.exists():
        checkpoint = PROJECT_ROOT / "checkpoints" / "metaworld_model.pt"
    calibration = PROJECT_ROOT / "calibration" / "cam2base_calibration.json"

    print("=" * 70)
    print(" OSVI-WM Live Deployment on Fairino FR10 ")
    print("=" * 70)

    # 1. Load Teacher Demonstration (10 frames)
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
    raw_wps = result["raw_waypoints"][0].detach().cpu().numpy()

    # Save visual debug plot
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_path = output_dir / "live_execution_debug.png"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(15, 8))
        # Top 10 frames
        for idx in range(min(10, len(teacher_frames))):
            ax_t = fig.add_subplot(2, 5, idx + 1)
            ax_t.imshow(teacher_frames[idx])
            ax_t.set_title(f"Demo Frame {idx}", fontsize=8)
            ax_t.axis("off")

        # Bottom Left: Live frame with projected trajectory
        ax_live = fig.add_subplot(2, 2, 3)
        ax_live.imshow(obs_frame)
        h_obs, w_obs = obs_frame.shape[:2]
        for i, wp in enumerate(raw_wps):
            u, v, d, g = wp
            px = int((u + 1.0) / 2.0 * w_obs)
            py = int((v + 1.0) / 2.0 * h_obs)
            col = "lime" if i == 0 else ("red" if i == len(raw_wps) - 1 else "yellow")
            ax_live.scatter(px, py, color=col, s=35, zorder=5)
            ax_live.text(px + 2, py - 2, f"{i}", color="white", fontsize=7, weight="bold")
        ax_live.set_title("Live Camera + Predicted Waypoint Overlays", fontsize=9)
        ax_live.axis("off")

        # Bottom Right: 3D Robot Trajectory
        ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
        xs = [wp["x_mm"] for wp in robot_wps]
        ys = [wp["y_mm"] for wp in robot_wps]
        zs = [wp["z_mm"] + args.z_offset for wp in robot_wps]
        ax_3d.plot(xs, ys, zs, "b-o", linewidth=1.5)
        ax_3d.scatter(xs[0], ys[0], zs[0], color="green", s=50, label="Start")
        ax_3d.scatter(xs[-1], ys[-1], zs[-1], color="red", s=50, label="End")
        ax_3d.set_xlabel("X (mm)", fontsize=8)
        ax_3d.set_ylabel("Y (mm)", fontsize=8)
        ax_3d.set_zlabel("Z (mm)", fontsize=8)
        ax_3d.set_title("3D Fairino Execution Path (mm)", fontsize=9)
        ax_3d.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(str(vis_path), dpi=150)
        plt.close()
        print(f"✓ Visual debug image saved to: {vis_path}")
    except Exception as e:
        print(f"[Warning] Could not generate visual debug plot: {e}")

    # 4. Connect to Fairino Robot
    driver = FairinoDriver(robot_ip=args.ip, mock=args.mock_robot)
    if not driver.connect():
        print("[Warning] Could not connect to physical robot. Running in simulation/mock mode.")

    gripper = GripperController(driver=driver) if args.use_gripper else None
    follower = TrajectoryFollower(
        driver=driver,
        gripper=gripper,
        speed=args.speed,
        min_z_mm=args.min_z,
        max_z_mm=args.max_z,
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
