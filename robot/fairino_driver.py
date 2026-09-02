"""
Fairino Robot XML-RPC Interface Driver.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add Fairino Python SDK path
SDK_LINUX = Path(__file__).resolve().parent / "fairino_sdk" / "linux"
SDK_WINDOWS = Path(__file__).resolve().parent / "fairino_sdk" / "windows"

if SDK_LINUX.exists() and sys.platform.startswith("linux"):
    sys.path.insert(0, str(SDK_LINUX))
elif SDK_WINDOWS.exists():
    sys.path.insert(0, str(SDK_WINDOWS))

try:
    from fairino import Robot
    FAIRINO_SDK_AVAILABLE = True
except ImportError:
    Robot = None
    FAIRINO_SDK_AVAILABLE = False


class FairinoDriver:
    def __init__(
        self,
        robot_ip: str = "192.168.57.2",
        tool: int = 2,
        user: int = 0,
        mock: bool = False,
    ):
        self.robot_ip = robot_ip
        self.tool = tool
        self.user = user
        self.mock = mock or (not FAIRINO_SDK_AVAILABLE)
        self.robot = None

        if self.mock:
            print(f"[Info] Running FairinoDriver in MOCK mode (IP: {robot_ip}).")

    def connect(self) -> bool:
        if self.mock:
            return True

        print(f"Connecting to Fairino robot at {self.robot_ip}...")
        try:
            self.robot = Robot.RPC(self.robot_ip)
            if not getattr(self.robot, "is_connect", getattr(self.robot, "is_conect", False)):
                rpc = getattr(self.robot, "robot", None)
                if rpc is not None:
                    raw_ip = rpc.GetControllerIP()
                    print(f"XML-RPC connected to {raw_ip}")
                    Robot.RPC.is_connect = True
            print("Successfully connected to Fairino Robot.")
            return True
        except Exception as e:
            print(f"Connection failed: {e}. Switching to Mock mode.")
            self.mock = True
            return False

    def soft_refresh(self):
        if self.mock or self.robot is None:
            return
        print("Performing soft refresh on robot controller...")
        for name in ("StopMove", "ProgramStop", "ResetAllError", "DragTeachSwitch"):
            method = getattr(self.robot, name, None)
            if method is not None:
                try:
                    if name == "DragTeachSwitch":
                        method(0)
                    else:
                        method()
                except Exception as e:
                    print(f"{name} unavailable ({e})")
            time.sleep(0.1)

    def prepare_auto(self):
        if self.mock or self.robot is None:
            return
        self.soft_refresh()
        self.robot.RobotEnable(1)
        mode = self.robot.Mode(0)  # 0 is Auto mode
        print(f"Robot Mode set to Auto (code: {mode})")

    def get_actual_tcp_pose(self) -> List[float]:
        """Returns [x, y, z, rx, ry, rz] in mm and degrees."""
        if self.mock or self.robot is None:
            return [0.0, 0.0, 0.0, 180.0, 0.0, 0.0]
        ret = self.robot.GetActualTCPPose()
        if isinstance(ret, tuple) and len(ret) > 1:
            return list(ret[1])
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def check_inverse_kinematics(
        self,
        desc_pos: List[float],
        ref_joints: Optional[List[float]] = None,
    ) -> Tuple[bool, List[float]]:
        """
        Validates whether target pose [x, y, z, rx, ry, rz] has a valid IK solution.
        """
        if self.mock or self.robot is None:
            return True, [0.0] * 6

        ref = ref_joints if ref_joints is not None else [0.0] * 6
        ret = self.robot.GetInverseKin(0, desc_pos, -1)  # 0 for type
        if isinstance(ret, tuple) and ret[0] == 0:
            return True, list(ret[1])
        return False, []

    def move_l(
        self,
        desc_pos: List[float],
        vel: float = 10.0,
        joint_pos: Optional[List[float]] = None,
    ) -> bool:
        """
        Executes linear Cartesian motion to desc_pos [x, y, z, rx, ry, rz].
        """
        if self.mock or self.robot is None:
            print(f"[Mock MoveL] Target TCP: {desc_pos} @ {vel}% speed")
            return True

        j_pos = joint_pos if joint_pos is not None else [0.0] * 6
        result = self.robot.MoveL(
            desc_pos=desc_pos,
            tool=self.tool,
            user=self.user,
            joint_pos=j_pos,
            vel=vel,
        )
        code = result[0] if isinstance(result, tuple) else result
        return code == 0

    def move_j(
        self,
        desc_pos: List[float],
        vel: float = 10.0,
        joint_pos: Optional[List[float]] = None,
    ) -> bool:
        """
        Executes joint motion to target pose.
        """
        if self.mock or self.robot is None:
            print(f"[Mock MoveJ] Target TCP: {desc_pos} @ {vel}% speed")
            return True

        j_pos = joint_pos if joint_pos is not None else [0.0] * 6
        result = self.robot.MoveJ(
            joint_pos=j_pos,
            desc_pos=desc_pos,
            tool=self.tool,
            user=self.user,
            vel=vel,
        )
        code = result[0] if isinstance(result, tuple) else result
        return code == 0
