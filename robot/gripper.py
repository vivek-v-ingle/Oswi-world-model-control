"""
Fairino End-Effector Gripper Controller.
"""

from typing import Optional
import time


class GripperController:
    def __init__(
        self,
        driver,
        gripper_index: int = 1,
        open_pos: int = 50,
        close_pos: int = 90,
        vel: int = 30,
        force: int = 40,
    ):
        self.driver = driver
        self.gripper_index = gripper_index
        self.open_pos = open_pos
        self.close_pos = close_pos
        self.vel = vel
        self.force = force

    def setup(self):
        if self.driver.mock or self.driver.robot is None:
            return
        robot = self.driver.robot
        # Set gripper configuration: company 6, device 0, soft version 0, bus 0
        robot.SetGripperConfig(self.gripper_index, 6, 0, 0, 0)
        time.sleep(0.5)
        robot.ActGripper(self.gripper_index, 1)
        time.sleep(1.0)
        print("Gripper initialized and activated.")

    def open(self):
        if self.driver.mock or self.driver.robot is None:
            print(f"[Mock Gripper] OPEN (pos: {self.open_pos})")
            return
        self.driver.robot.MoveGripper(
            self.gripper_index, self.open_pos, self.vel, self.force, 30000, 0
        )
        time.sleep(0.3)

    def close(self):
        if self.driver.mock or self.driver.robot is None:
            print(f"[Mock Gripper] CLOSE (pos: {self.close_pos})")
            return
        self.driver.robot.MoveGripper(
            self.gripper_index, self.close_pos, self.vel, self.force, 30000, 0
        )
        time.sleep(0.3)

    def apply_event(self, event: str):
        event_clean = (event or "").strip().lower()
        if event_clean in {"close", "closed", "grasp", "grab", "pick"}:
            self.close()
        elif event_clean in {"open", "release", "drop", "place"}:
            self.open()
