#!/usr/bin/env python3
"""
Step 1: Offline Inference Verification.
Runs OSVI-WM World Model on a 10-frame demonstration and observation image,
decodes the future latent rollout, and maps predicted waypoints into Fairino base coordinates.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference_engine import InferenceEngine


def main():
    checkpoint = PROJECT_ROOT / "checkpoints" / "metaworld_model.pt"
    calibration = PROJECT_ROOT / "calibration" / "cam2base_calibration.json"

    print("=" * 70)
    print(" OSVI-WM Offline Inference & Trajectory Foreseeing ")
    print("=" * 70)

    engine = InferenceEngine(
        checkpoint_path=checkpoint,
        calibration_path=calibration,
        latent_dim=256,
        waypoints=5,
        sub_waypoints=True,
    )

    # Generate synthetic 10-frame demo and initial observation for offline validation
    print("\n[Input] Simulating 10-frame Teacher Demo Video and Workspace Observation...")
    dummy_demo = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
    dummy_obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    print("[Inference] Rolling out Forward Transformer World Model across latent horizon...")
    result = engine.predict(
        teacher_frames=dummy_demo,
        current_obs=dummy_obs,
        rollout_horizon=16,
    )

    raw_wps = result["raw_waypoints"]
    print(f"\n✓ Raw Output Tensor Shape: {raw_wps.shape} (Batch, Num Waypoints, [u, v, depth, gripper])")

    robot_wps = result.get("robot_waypoints", [])
    print(f"✓ Transformed into {len(robot_wps)} Fairino Base Frame Waypoints (mm):\n")

    header = f"{'Index':<6} | {'X (mm)':<10} | {'Y (mm)':<10} | {'Z (mm)':<10} | {'Gripper':<8} | {'Event':<8}"
    print(header)
    print("-" * len(header))
    for wp in robot_wps:
        print(
            f"{wp['index']:<6} | {wp['x_mm']:<10.2f} | {wp['y_mm']:<10.2f} | {wp['z_mm']:<10.2f} | {wp['gripper_val']:<8.2f} | {wp['event']:<8}"
        )

    print("\n" + "=" * 70)
    print("✓ Offline inference test PASSED successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
