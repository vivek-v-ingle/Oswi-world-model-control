#!/usr/bin/env python3
"""
Interactive 4-Point Touch-Off Eye-to-Base Calibration Tool for Fairino FR10 + Overhead Camera.

Workflow:
  1. Captures a clear live frame from the top-view camera (ZED/USB).
  2. Opens a GUI window where you click 4 distinct reference points on the table workspace (e.g. 4 marked points, corners of tape, or objects).
  3. For each of the 4 points, you move the Fairino arm (via teach pendant or drag-teach) to touch that exact point with the gripper tip, and press ENTER.
  4. The script queries actual robot TCP [X, Y, Z] directly from the Fairino controller.
  5. Computes the optimal rigid/affine transformation matrix T_cam2base using SVD/Least-Squares.
  6. Evaluates calibration residual error (RMSE in mm) and saves to calibration/cam2base_calibration.json!
"""

import sys
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perception.camera_stream import CameraStream
from robot.fairino_driver import FairinoDriver


def compute_rigid_transform(points_cam, points_robot):
    """
    Computes optimal 4x4 transformation matrix T such that:
    points_robot_hom = points_cam_hom @ T.T
    using SVD / Umeyama algorithm.
    """
    assert len(points_cam) == len(points_robot) and len(points_cam) >= 3

    p_cam = np.array(points_cam, dtype=np.float64)  # (N, 3)
    p_rob = np.array(points_robot, dtype=np.float64)  # (N, 3)

    # Centroids
    centroid_cam = np.mean(p_cam, axis=0)
    centroid_rob = np.mean(p_rob, axis=0)

    # Center the points
    cam_centered = p_cam - centroid_cam
    rob_centered = p_rob - centroid_rob

    # Covariance matrix H
    H = cam_centered.T @ rob_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection check
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_rob - R @ centroid_cam

    # Form 4x4 homogeneous matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def main():
    parser = argparse.ArgumentParser(description="4-Point Touch-Off Eye-to-Base Calibration")
    parser.add_argument("--ip", default="192.168.57.2", help="Fairino Robot Controller IP")
    parser.add_argument("--camera", default="opencv", choices=["opencv", "zed"], help="Camera type")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth scale factor (mm)")
    parser.add_argument(
        "--output",
        default="calibration/cam2base_calibration.json",
        help="Path to save updated calibration json",
    )
    args = parser.parse_args()

    out_file = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Fairino FR10 + ZED Camera 4-Point Touch-Off Calibration ")
    print("=" * 70)

    # 1. Connect to Fairino Robot
    driver = FairinoDriver(robot_ip=args.ip, mock=False)
    if not driver.connect():
        print(f"[Error] Could not connect to Fairino robot at {args.ip}. Check Ethernet cable.")
        return

    # 2. Open Live Camera Stream
    print("\nOpening live camera stream...")
    camera = CameraStream(camera_type=args.camera, camera_id=args.camera_id)
    camera.start()

    # Warmup
    for _ in range(10):
        camera.get_frame()
        time.sleep(0.05)

    clicked_pixels = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pixels) < 4:
            clicked_pixels.append((x, y))
            idx = len(clicked_pixels)
            print(f"  [Selected Point {idx}] Image Pixel: (u={x}, v={y})")

    window_name = "Live Calibration Stream: LEFT-CLICK 4 Points (P1->P4)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n[Step 1] Look at the live camera window and LEFT-CLICK 4 points across your table:")
    print("         (P1: Top-Left, P2: Top-Right, P3: Bottom-Right, P4: Bottom-Left).")
    print("         The window will proceed once 4 points are clicked.")

    h_img, w_img = 376, 672
    while len(clicked_pixels) < 4:
        frame_rgb = camera.get_frame()
        if frame_rgb is None or frame_rgb.size == 0 or np.all(frame_rgb == 0):
            time.sleep(0.03)
            continue

        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = frame.shape[:2]

        display = frame.copy()
        # Draw all currently clicked points
        for i, (px, py) in enumerate(clicked_pixels, start=1):
            cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
            cv2.circle(display, (px, py), 12, (0, 255, 0), 2)
            cv2.putText(
                display,
                f"P{i} ({px},{py})",
                (px + 10, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            display,
            f"Click Point P{len(clicked_pixels)+1}/4 on table (ESC to exit)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            print("Calibration cancelled by user.")
            camera.stop()
            cv2.destroyAllWindows()
            return

    camera.stop()
    cv2.destroyAllWindows()

    print(f"\n✓ Successfully selected 4 image points:")
    for i, (u, v) in enumerate(clicked_pixels, start=1):
        print(f"  P{i}: Pixel (u={u}, v={v})")

    # 4. Robot Touch-Off Collection
    robot_points = []
    camera_3d_points = []
    default_rpy = None

    print("\n" + "=" * 70)
    print(" [Step 2] Touch-Off: Move Fairino Arm to Touch Each Point ")
    print("=" * 70)

    for i, (u_px, v_px) in enumerate(clicked_pixels, start=1):
        print(f"\n--- Point {i}/4: Pixel (u={u_px}, v={v_px}) ---")
        print("  -> Use the Fairino Teach Pendant (or drag-teach mode) to move the gripper tip")
        print(f"     to physically touch Point P{i} on the table.")
        input(f"  -> When gripper is in contact with Point P{i}, PRESS ENTER to read robot TCP coordinates...")

        tcp = driver.get_actual_tcp_pose()
        x, y, z = tcp[0], tcp[1], tcp[2]
        rpy = tcp[3:6]
        if default_rpy is None:
            default_rpy = [float(r) for r in rpy]

        print(f"  ✓ Recorded Robot TCP P{i}: X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm | RPY={rpy}")
        robot_points.append([x, y, z])

        # Normalized coordinates u, v in [-1, 1]
        u_norm = (u_px / w_img) * 2.0 - 1.0
        v_norm = (v_px / h_img) * 2.0 - 1.0
        # In OSVI-WM: hom_im_coords = [u * d, v * d, d] * depth_scale (default 1000.0 mm)
        d_scale = args.depth_scale
        camera_3d_points.append([u_norm * d_scale, v_norm * d_scale, d_scale])

    # 5. Compute Transformation Matrix
    print("\nComputing optimal T_cam2base transformation matrix via least-squares...")
    
    # Linear affine mapping: [X_rob, Y_rob, Z_rob, 1]^T = T @ [u_norm*Z, v_norm*Z, Z, 1]^T
    # Or 3D-to-3D rigid transform:
    T = compute_rigid_transform(camera_3d_points, robot_points)

    # Compute Residual Errors
    cam_hom = np.hstack([camera_3d_points, np.ones((4, 1))])
    pred_robot = (cam_hom @ T.T)[:, :3]
    errors = np.linalg.norm(pred_robot - np.array(robot_points), axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    print("\n" + "=" * 70)
    print(" Calibration Results ")
    print("=" * 70)
    print("Computed T_cam2base Matrix:")
    for row in T:
        print(f"  [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}, {row[3]:12.6f}]")

    print(f"\nPoint-by-Point Residual Errors (mm):")
    for i, err in enumerate(errors, start=1):
        print(f"  P{i}: Error = {err:6.2f} mm")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} mm")

    # 6. Save to JSON
    calib_data = {
        "source": "Fairino FR10 + ZED Camera 4-Point Touch-Off Calibration",
        "units": "millimeters",
        "date_calibrated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rmse_mm": float(rmse),
        "default_rpy": default_rpy or [180.0, 0.0, 0.0],
        "T_cam2base": T.tolist(),
        "affine_camera_to_robot": T[:3, :].tolist(),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    print(f"\n✓ Calibration successfully saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
