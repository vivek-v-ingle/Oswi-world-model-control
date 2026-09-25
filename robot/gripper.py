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
        try:
            # Set gripper configuration: company 6, device 0, soft version 0, bus 0
            robot.SetGripperConfig(6, 0, 0, 0)
            time.sleep(0.3)
        except Exception as e:
            print(f"[Warning] SetGripperConfig notice: {e}")

        try:
            robot.ActGripper(self.gripper_index, 1)
            time.sleep(0.5)
            print("Gripper initialized and activated.")
        except Exception as e:
            print(f"[Warning] ActGripper notice: {e}")

    def _send_gripper_cmd(self, target_pos: int):
        if self.driver.mock or self.driver.robot is None:
            return
        robot = self.driver.robot
        try:
            # 10-arg signature for current Fairino SDK: (index, pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque)
            robot.MoveGripper(self.gripper_index, target_pos, self.vel, self.force, 30000, 0, 0, 0, 0, 0)
        except TypeError:
            try:
                # 6-arg fallback for older SDK builds
                robot.MoveGripper(self.gripper_index, target_pos, self.vel, self.force, 30000, 0)
            except Exception as e:
                print(f"[Warning] Gripper motion notice: {e}")
        except Exception as e:
            print(f"[Warning] Gripper motion notice: {e}")
        time.sleep(0.3)

    def open(self):
        if self.driver.mock or self.driver.robot is None:
            print(f"[Mock Gripper] OPEN (pos: {self.open_pos})")
            return
        print(f"[Gripper] Opening (pos: {self.open_pos}%)")
        self._send_gripper_cmd(self.open_pos)

    def close(self):
        if self.driver.mock or self.driver.robot is None:
            print(f"[Mock Gripper] CLOSE (pos: {self.close_pos})")
            return
        print(f"[Gripper] Closing (pos: {self.close_pos}%)")
        self._send_gripper_cmd(self.close_pos)

    def apply_event(self, event: str):
        event_clean = (event or "").strip().lower()
        if event_clean in {"close", "closed", "grasp", "grab", "pick"}:
            self.close()
        elif event_clean in {"open", "release", "drop", "place"}:
            self.open()
