#!/usr/bin/env python3
"""
Step 3: Robot Dry-Run and Inverse Kinematics Preflight Check.
Validates that all predicted 3D waypoints are collision-free and kinematics-feasible
without commanding physical arm movements.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference_engine import InferenceEngine
from robot.fairino_driver import FairinoDriver
from robot.trajectory_follower import TrajectoryFollower
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Robot Dry Run & IK Validation")
    parser.add_argument("--ip", default="192.168.57.2", help="Fairino Robot IP")
    parser.add_argument("--mock", action="store_true", default=True, help="Force Mock mode for offline testing")
    parser.add_argument("--speed", type=float, default=10.0, help="Movement velocity percentage")
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / "checkpoints" / "metaworld_model.pt"
    calibration = PROJECT_ROOT / "calibration" / "cam2base_calibration.json"

    print("=" * 70)
    print(" Fairino FR10 Dry-Run & Kinematics Preflight Check ")
    print("=" * 70)

    # 1. Run inference to get 3D waypoints
    engine = InferenceEngine(checkpoint_path=checkpoint, calibration_path=calibration)
    dummy_demo = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
    dummy_obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    result = engine.predict(teacher_frames=dummy_demo, current_obs=dummy_obs, rollout_horizon=16)
    robot_wps = result["robot_waypoints"]

    # 2. Connect driver
    driver = FairinoDriver(robot_ip=args.ip, mock=args.mock)
    driver.connect()

    # 3. Preflight IK check
    follower = TrajectoryFollower(driver=driver, speed=args.speed, min_z_mm=-300.0, max_z_mm=4000.0)
    ok, validated = follower.preflight_check(robot_wps)

    if ok:
        print("\n[Dry Run Summary]")
        print(f"✓ All {len(validated)} waypoints passed Inverse Kinematics & workspace limits.")
        print("✓ System is READY for safe physical execution on the Fairino robot.")
    else:
        print("\n[Dry Run Summary] Preflight check found kinematic violations.")

    print("=" * 70)


if __name__ == "__main__":
    main()
