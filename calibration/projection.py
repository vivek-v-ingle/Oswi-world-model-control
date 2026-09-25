"""
Coordinate projection utilities for mapping OSVI-WM waypoints (u, v, depth, gripper)
to Fairino robot base frame coordinates (X, Y, Z in mm, gripper).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import torch


class CoordinateProjector:
    def __init__(self, calibration_path: Union[str, Path]):
        self.calibration_path = Path(calibration_path)
        if not self.calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found at {self.calibration_path}")

        with open(self.calibration_path, "r", encoding="utf-8") as f:
            self.calib_data = json.load(f)

        if "T_cam2base" in self.calib_data:
            self.T_cam2base = np.array(self.calib_data["T_cam2base"], dtype=np.float64)
        elif "affine_camera_to_robot" in self.calib_data:
            affine = np.array(self.calib_data["affine_camera_to_robot"], dtype=np.float64)
            if affine.shape == (3, 4):
                self.T_cam2base = np.vstack([affine, [0.0, 0.0, 0.0, 1.0]])
            else:
                self.T_cam2base = affine
        else:
            raise ValueError("Calibration file missing 'T_cam2base' or 'affine_camera_to_robot'.")

        self.default_rpy = self.calib_data.get(
            "default_rpy", [178.51, -1.88, 12.12]
        )

    def image_waypoints_to_camera_3d(
        self,
        waypoints_2d: np.ndarray,
        depth_scale: float = 1000.0,
    ) -> np.ndarray:
        """
        Converts normalized/pixel waypoints [u, v, depth, gripper] to camera 3D [X_c, Y_c, Z_c, gripper].
        
        Args:
            waypoints_2d: Array of shape (N, 4) with (u, v, depth, gripper).
            depth_scale: Scale factor to convert normalized depth to mm.
        Returns:
            Array of shape (N, 4) with (X_c, Y_c, Z_c, gripper).
        """
        N = waypoints_2d.shape[0]
        u = waypoints_2d[:, 0]
        v = waypoints_2d[:, 1]
        # Camera optical depth is positive forward into the workspace (Z_cam > 0)
        d = np.abs(waypoints_2d[:, 2])
        gripper = waypoints_2d[:, 3]

        # In OSVI-WM: hom_im_coords = [u * d, v * d, d]
        # In meters or scaled depth:
        x_c = u * d * depth_scale
        y_c = v * d * depth_scale
        z_c = d * depth_scale

        return np.column_stack([x_c, y_c, z_c, gripper])

    def camera_3d_to_robot_base(self, points_cam: np.ndarray) -> np.ndarray:
        """
        Transforms camera 3D points [X_c, Y_c, Z_c, gripper] into Fairino robot base frame [X_r, Y_r, Z_r, gripper].
        
        Args:
            points_cam: Array of shape (N, 4) where first 3 columns are camera XYZ in mm.
        Returns:
            Array of shape (N, 4) in robot base coordinates in mm.
        """
        xyz_cam = points_cam[:, :3]
        gripper = points_cam[:, 3:]

        # Homogeneous coordinates (N, 4)
        ones = np.ones((xyz_cam.shape[0], 1), dtype=np.float64)
        xyz_hom = np.hstack([xyz_cam, ones])

        # Multiply by T_cam2base: (N, 4) @ (4, 4)^T -> (N, 4)
        xyz_robot_hom = xyz_hom @ self.T_cam2base.T
        xyz_robot = xyz_robot_hom[:, :3]

        return np.hstack([xyz_robot, gripper])

    def project(self, raw_waypoints: Union[np.ndarray, torch.Tensor]) -> List[Dict[str, float]]:
        """
        Full projection pipeline from raw OSVI-WM output tensor (N, 4) to list of robot waypoint dictionaries.
        
        Returns:
            List of dicts: [{'x': ..., 'y': ..., 'z': ..., 'gripper': ..., 'event': 'open'/'close'}]
        """
        if isinstance(raw_waypoints, torch.Tensor):
            raw_waypoints = raw_waypoints.detach().cpu().numpy()

        if raw_waypoints.ndim == 3:
            raw_waypoints = raw_waypoints[0]  # Take first batch item

        cam_3d = self.image_waypoints_to_camera_3d(raw_waypoints)
        robot_3d = self.camera_3d_to_robot_base(cam_3d)

        waypoint_list = []
        for i, row in enumerate(robot_3d):
            x, y, z, g = row
            event = "close" if g > 0.0 else "open"
            waypoint_list.append({
                "index": i,
                "x_mm": float(x),
                "y_mm": float(y),
                "z_mm": float(z),
                "gripper_val": float(g),
                "event": event,
                "rpy": self.default_rpy,
            })

        return waypoint_list
