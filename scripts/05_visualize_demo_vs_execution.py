#!/usr/bin/env python3
"""
Side-by-Side Visualizer: Source Demonstration Video vs OSVI-WM Trajectory Execution.
Generates an animated comparison MP4 and high-resolution figure.
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference_engine import InferenceEngine
from perception.video_loader import DemonstrationLoader
from perception.camera_stream import CameraStream


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Demo vs OSVI-WM Execution")
    parser.add_argument("--demo-video", default="data/demos/bottle_top_view.mp4", help="Path to teacher demo video")
    parser.add_argument("--checkpoint", default="checkpoints/pp_model.pt", help="Path to checkpoint")
    parser.add_argument("--output-video", default="outputs/demo_vs_prediction.mp4", help="Path to save comparison video")
    parser.add_argument("--output-plot", default="outputs/demo_vs_prediction.png", help="Path to save static figure")
    return parser.parse_args()


def main():
    args = parse_args()
    demo_path = PROJECT_ROOT / args.demo_video if not Path(args.demo_video).is_absolute() else Path(args.demo_video)
    ckpt_path = PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    calib_path = PROJECT_ROOT / "calibration" / "cam2base_calibration.json"

    print("=" * 70)
    print(" OSVI-WM: Teacher Demo vs Execution Visualizer ")
    print("=" * 70)

    if not demo_path.exists():
        print(f"[Error] Demo video not found at: {demo_path}")
        return

    # 1. Load teacher demo frames (10 keyframes)
    teacher_frames = DemonstrationLoader.load_video(str(demo_path), max_frames=10)
    print(f"Loaded {len(teacher_frames)} keyframes from {demo_path.name}")

    # 2. Get observation frame (first frame of demo or live camera)
    obs_frame = teacher_frames[0].copy()

    # 3. Predict Trajectory
    print("Running OSVI-WM Model Inference...")
    engine = InferenceEngine(checkpoint_path=ckpt_path, calibration_path=calib_path)
    result = engine.predict(teacher_frames=teacher_frames, current_obs=obs_frame, rollout_horizon=16)

    raw_wps = result["raw_waypoints"][0].detach().cpu().numpy()
    robot_wps = result["robot_waypoints"]

    # 4. Create High-Resolution Static Comparison Plot
    out_plot = PROJECT_ROOT / args.output_plot
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 9), facecolor="#1a1a24")
    gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.2)

    # Top Row: Teacher Demonstration Keyframes (10 frames)
    for i in range(10):
        row = 0 if i < 5 else 1
        col = i % 5
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(teacher_frames[i])
        ax.set_title(f"Demo Frame {i+1}/10", color="white", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Bottom Left: Live Workspace Observation with Predicted 2D Waypoints
    ax_traj = fig.add_subplot(gs[2, :3])
    ax_traj.imshow(obs_frame)
    h_obs, w_obs = obs_frame.shape[:2]

    # Draw trajectory line
    pixel_pts = []
    for i, wp in enumerate(raw_wps):
        u, v, d, g = wp
        px = int((u + 1.0) / 2.0 * w_obs)
        py = int((v + 1.0) / 2.0 * h_obs)
        pixel_pts.append((px, py, g))

    for i in range(len(pixel_pts) - 1):
        p1 = pixel_pts[i]
        p2 = pixel_pts[i + 1]
        c = "cyan" if p1[2] < 0 else "orange"
        ax_traj.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, linewidth=2.5, zorder=4)

    for i, (px, py, g) in enumerate(pixel_pts):
        if i == 0:
            col, label = "#00ff00", "Start"
        elif i == len(pixel_pts) - 1:
            col, label = "#ff3333", "Release"
        elif g > 0 and (i == 1 or pixel_pts[i-1][2] <= 0):
            col, label = "#ffcc00", "Grasp"
        else:
            col, label = "#ffffff", f"{i}"
        ax_traj.scatter(px, py, color=col, s=70, edgecolors="black", linewidth=1.5, zorder=5)
        ax_traj.text(px + 4, py - 4, label, color="white", fontsize=8, weight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    ax_traj.set_title("Predicted 2D Pixel Trajectory (Orange: Gripper Closed | Cyan: Open)", color="white", fontsize=11, fontweight="bold")
    ax_traj.axis("off")

    # Bottom Right: 3D Metric Arm Trajectory
    ax_3d = fig.add_subplot(gs[2, 3:], projection="3d", facecolor="#1a1a24")
    ax_3d.set_facecolor("#1a1a24")
    xs = [wp["x_mm"] for wp in robot_wps]
    ys = [wp["y_mm"] for wp in robot_wps]
    zs = [wp["z_mm"] for wp in robot_wps]
    ax_3d.plot(xs, ys, zs, color="#00d4ff", linewidth=2.5, marker="o", markersize=5)
    ax_3d.scatter(xs[0], ys[0], zs[0], color="#00ff00", s=90, label="Start (W0)")
    ax_3d.scatter(xs[-1], ys[-1], zs[-1], color="#ff3333", s=90, label="Place (W14)")

    ax_3d.set_xlabel("X (mm)", color="white", fontsize=8)
    ax_3d.set_ylabel("Y (mm)", color="white", fontsize=8)
    ax_3d.set_zlabel("Z (mm)", color="white", fontsize=8)
    ax_3d.tick_params(colors="white", labelsize=7)
    ax_3d.set_title("3D Fairino Robot Trajectory (mm)", color="white", fontsize=11, fontweight="bold")
    ax_3d.legend(fontsize=8, loc="upper right")

    plt.savefig(str(out_plot), dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✓ Generated high-res visual comparison plot: {out_plot}")

    # 5. Create Animated Comparison Video (.mp4)
    out_vid = PROJECT_ROOT / args.output_video
    out_vid.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(demo_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

    target_h, target_w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_vid), fourcc, fps, (target_w * 2, target_h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop left eye if stereo
        h, w = frame.shape[:2]
        if w > h * 1.7:
            frame = frame[:, : w // 2]
        left_view = cv2.resize(frame, (target_w, target_h))

        # Right side: observation + dynamic trajectory overlay
        right_view = cv2.resize(cv2.cvtColor(obs_frame, cv2.COLOR_RGB2BGR), (target_w, target_h))

        # Current waypoint index based on video progress
        progress = min(1.0, frame_idx / max(1, total_frames - 1))
        curr_wp_idx = int(progress * (len(pixel_pts) - 1))

        # Draw full trajectory faintly
        for j in range(len(pixel_pts) - 1):
            p1 = (int(pixel_pts[j][0] * target_w / w_obs), int(pixel_pts[j][1] * target_h / h_obs))
            p2 = (int(pixel_pts[j+1][0] * target_w / w_obs), int(pixel_pts[j+1][1] * target_h / h_obs))
            cv2.line(right_view, p1, p2, (80, 80, 80), 2)

        # Draw active trajectory up to current progress
        for j in range(curr_wp_idx):
            p1 = (int(pixel_pts[j][0] * target_w / w_obs), int(pixel_pts[j][1] * target_h / h_obs))
            p2 = (int(pixel_pts[j+1][0] * target_w / w_obs), int(pixel_pts[j+1][1] * target_h / h_obs))
            c = (255, 200, 0) if pixel_pts[j][2] < 0 else (0, 140, 255)
            cv2.line(right_view, p1, p2, c, 3)

        # Draw active cursor
        curr_p = (int(pixel_pts[curr_wp_idx][0] * target_w / w_obs), int(pixel_pts[curr_wp_idx][1] * target_h / h_obs))
        g_state = pixel_pts[curr_wp_idx][2]
        cur_col = (0, 255, 0) if g_state < 0 else (0, 0, 255)
        cv2.circle(right_view, curr_p, 8, cur_col, -1)
        cv2.circle(right_view, curr_p, 14, (255, 255, 255), 2)

        # Header Titles
        cv2.putText(left_view, "SOURCE TEACHER DEMO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(right_view, "OSVI-WM PREDICTED EXECUTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        status_text = "GRIPPER: CLOSED (GRASPING)" if g_state > 0 else "GRIPPER: OPEN"
        cv2.putText(right_view, f"Waypoint {curr_wp_idx+1}/{len(pixel_pts)} | {status_text}",
                    (20, target_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        combined = np.hstack([left_view, right_view])
        writer.write(combined)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"✓ Generated side-by-side comparison video: {out_vid}")
    print("\nVisualizer execution complete!")


if __name__ == "__main__":
    main()
