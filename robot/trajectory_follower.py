"""
Safe Cartesian Trajectory Follower for Fairino Robot.
"""

from typing import Dict, List, Optional
import time
from robot.fairino_driver import FairinoDriver
from robot.gripper import GripperController


class TrajectoryFollower:
    def __init__(
        self,
        driver: FairinoDriver,
        gripper: Optional[GripperController] = None,
        speed: float = 10.0,
        min_z_mm: float = -300.0,
        max_z_mm: float = 1200.0,
        z_offset_mm: float = 0.0,
    ):
        self.driver = driver
        self.gripper = gripper
        self.speed = speed
        self.min_z_mm = min_z_mm
        self.max_z_mm = max_z_mm
        self.z_offset_mm = z_offset_mm

    def preflight_check(self, waypoints: List[Dict]) -> Tuple_IK:
        """
        Validates safety bounds and IK reachability for all trajectory waypoints.
        """
        print(f"\n--- Running Preflight Check for {len(waypoints)} Waypoints ---")
        failures = []
        validated_waypoints = []

        for wp in waypoints:
            x = wp["x_mm"]
            y = wp["y_mm"]
            z = wp["z_mm"] + self.z_offset_mm
            rpy = wp.get("rpy", [180.0, 0.0, 0.0])

            # Safety workspace limits check
            if z < self.min_z_mm or z > self.max_z_mm:
                print(f"[Safety Alarm] Waypoint {wp['index']} Z={z:.1f}mm outside limits [{self.min_z_mm}, {self.max_z_mm}]")
                failures.append((wp["index"], f"Z limit violation: {z:.1f}mm"))
                continue

            desc_pos = [x, y, z, rpy[0], rpy[1], rpy[2]]
            ok, joint_pos = self.driver.check_inverse_kinematics(desc_pos)

            if not ok:
                failures.append((wp["index"], f"IK failed for pose {desc_pos}"))
            else:
                wp_valid = dict(wp)
                wp_valid["target_pose"] = desc_pos
                wp_valid["joint_pos"] = joint_pos
                validated_waypoints.append(wp_valid)

        if failures:
            print(f"[Error] Preflight check FAILED with {len(failures)} issue(s):")
            for idx, reason in failures[:5]:
                print(f"  - Waypoint {idx}: {reason}")
            return False, []

        print(f"✓ Preflight check PASSED! All {len(waypoints)} waypoints are safely reachable.")
        return True, validated_waypoints

    def execute(
        self,
        waypoints: List[Dict],
        step_delay: float = 0.05,
        start_with_movej: bool = True,
        return_to_start: bool = False,
    ) -> bool:
        """
        Executes preflight check and then commands the robot through all waypoints.
        """
        ok, validated = self.preflight_check(waypoints)
        if not ok:
            print("[Execution Aborted] Preflight check failed.")
            return False

        self.driver.prepare_auto()

        if self.gripper is not None:
            self.gripper.setup()

        first_wp = validated[0]
        start_pose = first_wp["target_pose"]

        # Move to initial waypoint safely
        print(f"\nMoving to start pose: {start_pose}")
        if start_with_movej:
            self.driver.move_j(start_pose, vel=self.speed, joint_pos=first_wp.get("joint_pos"))
        else:
            self.driver.move_l(start_pose, vel=self.speed)

        # Execute trajectory waypoints
        print(f"\nExecuting {len(validated)} trajectory waypoints via MoveL...")
        for i, wp in enumerate(validated):
            pose = wp["target_pose"]
            j_pos = wp.get("joint_pos")
            event = wp.get("event", "")

            success = self.driver.move_l(pose, vel=self.speed, joint_pos=j_pos)
            if not success:
                print(f"[Error] MoveL failed at waypoint {i}!")
                return False

            if self.gripper is not None and event:
                self.gripper.apply_event(event)

            time.sleep(step_delay)

        if return_to_start:
            print(f"Returning to start pose: {start_pose}")
            self.driver.move_l(start_pose, vel=self.speed)

        self.driver.soft_refresh()
        print("✓ Trajectory execution completed successfully!")
        return True


from typing import Tuple as Tuple_IK
