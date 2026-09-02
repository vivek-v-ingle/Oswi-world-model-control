#!/usr/bin/env python3
"""
Step 2: 3D Trajectory & Camera Projection Visualizer.
Plots the predicted 3D Cartesian waypoints and saves a visualization image
(ideal for headless SSH viewing).
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference_engine import InferenceEngine


def main():
    checkpoint = PROJECT_ROOT / "checkpoints" / "metaworld_model.pt"
    calibration = PROJECT_ROOT / "calibration" / "cam2base_calibration.json"
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running OSVI-WM Trajectory Foreseeing & Visualization...")
    engine = InferenceEngine(checkpoint_path=checkpoint, calibration_path=calibration)

    dummy_demo = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
    dummy_obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    result = engine.predict(teacher_frames=dummy_demo, current_obs=dummy_obs, rollout_horizon=16)
    robot_wps = result["robot_waypoints"]

    xs = [wp["x_mm"] for wp in robot_wps]
    ys = [wp["y_mm"] for wp in robot_wps]
    zs = [wp["z_mm"] for wp in robot_wps]
    grips = [wp["gripper_val"] for wp in robot_wps]

    fig = plt.figure(figsize=(12, 6))

    # 3D Trajectory Plot
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(xs, ys, zs, "b-o", linewidth=2, markersize=6, label="Predicted Waypoints")
    ax1.scatter(xs[0], ys[0], zs[0], color="green", s=100, label="Start Waypoint")
    ax1.scatter(xs[-1], ys[-1], zs[-1], color="red", s=100, label="End Waypoint")

    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        ax1.text(x, y, z, f" {i}", fontsize=9)

    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_title("OSVI-WM Foreseen 3D Robot Trajectory")
    ax1.legend()
    ax1.grid(True)

    # Gripper profile plot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(len(grips)), grips, "r-s", linewidth=2, label="Gripper State (1=close, -1=open)")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Waypoint Index")
    ax2.set_ylabel("Grasp Activation")
    ax2.set_title("Predicted Gripper Action Profile")
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    save_path = output_dir / "trajectory_3d.png"
    plt.savefig(str(save_path), dpi=150)
    plt.close()

    print(f"✓ 3D Trajectory visualization saved to: {save_path}")


if __name__ == "__main__":
    main()
