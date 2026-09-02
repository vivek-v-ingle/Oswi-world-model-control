# OSVI-WM on Fairino FR10: Complete Research & Engineering Guide

**Project Name:** `Oswi-world-model-control`  
**Target Hardware:** Fairino FR10 Collaborative Robot + Fixed RGB/ZED Camera  
**Theoretical Foundation:** *OSVI-WM: One-Shot Visual Imitation for Unseen Tasks using World-Model-Guided Trajectory Generation* (NeurIPS 2025)

---

## 1. Executive Summary & What OSVI-WM Is

**One-Shot Visual Imitation (OSVI)** addresses a fundamental challenge in robot learning: given a **single video demonstration** ($C$) of a task performed by a teacher (which may be a human, a different robot arm, or a simulation), how can a robot agent execute that **unseen task** in its own workspace?

Traditional imitation learning fails on unseen tasks because it requires hundreds of demonstrations per task. OSVI-WM solves this by decoupling high-level intent from low-level execution via a **Latent World Model**:

1. **Visual Encoder ($E$):** Encodes the 10-frame teacher demonstration and the agent's current workspace view into dense spatio-temporal feature maps.
2. **Intent/Action Model ($A$):** Uses causal non-local attention to infer the latent task direction between the demonstration context and current state.
3. **Transformer Forward World Model ($\mathcal{F}$):** Autoregressively rolls out and **foresees** future latent states $\hat{s}_{t+1}, \dots, \hat{s}_{t+H}$ step-by-step.
4. **Attentive Query Waypoint Decoder ($W$):** Decodes the foreseen latent rollout into **3D/4D Cartesian waypoints** $(u, v, d, \text{gripper})$ in camera space.
5. **Coordinate Projection ($T_{\text{cam2base}}$):** Transforms image waypoints into physical robot base coordinates $[X, Y, Z, \text{gripper}]^T$ (in millimeters).
6. **Robot Controller:** Fairino's internal Cartesian controller (`MoveL` / `MoveJ`) tracks the waypoints smoothly.

---

## 2. What We Are Doing in This Project

Rather than embedding world model code into monolithic frameworks, we built a **clean, dedicated repository (`Oswi-world-model-control`)** that connects:

$$\text{Teacher Video Demonstration } (10\text{ frames}) + \text{Live Camera Observation}$$
$$\Downarrow$$
$$\textbf{OSVI-WM Transformer World Model Foreseeing Loop}$$
$$\Downarrow$$
$$\text{Projected Cartesian Waypoints } [X, Y, Z, \text{Grasp}] \text{ in mm}$$
$$\Downarrow$$
$$\textbf{Fairino FR10 Python XML-RPC Driver } (\text{MoveL} + \text{Gripper})$$

---

## 3. Current Status & What Has Been Verified

| Component | Status | Verification Details |
| :--- | :--- | :--- |
| **Model Checkpoints** | **Verified** | Pretrained Meta-World (`768MB`) & Pick-and-Place (`1.1GB`) checkpoints loaded successfully. |
| **Offline Inference** | **Verified** | `scripts/01_offline_inference.py` decodes 10-frame inputs into 15 valid 3D Cartesian waypoints. |
| **Trajectory Visualizer** | **Verified** | `scripts/02_visualize_trajectory.py` generates headless 3D waypoint plots & gripper profiles (`outputs/trajectory_3d.png`). |
| **Kinematics Preflight Check** | **Verified** | `scripts/03_robot_dry_run.py` validates workspace bounds and Inverse Kinematics (`GetInverseKin`) in mock/dry-run mode. |
| **Fairino XML-RPC Integration** | **Verified** | Pure Python SDK interface (`robot/fairino_driver.py`) with `soft_refresh`, `prepare_auto`, and `MoveL`. |
| **Camera Extrinsics** | **Configured** | $T_{\text{cam2base}}$ calibration matrix integrated in `calibration/cam2base_calibration.json`. |

---

## 4. Prerequisites for the Fairino Lab Workstation

When you clone this repository onto the workstation in the Fairino lab, verify the following:

### Hardware Setup
- **Robot Arm:** Fairino FR10 powered on, in **Auto Mode** on the teach pendant, with emergency stop released.
- **Network:** Workstation connected to Fairino controller network (default IP: `192.168.57.2`). Test with `ping 192.168.57.2`.
- **Camera:** Fixed overhead or side-mounted camera (ZED 2 / ZED Mini or standard USB webcam) observing the workspace.
- **Gripper:** Electric gripper configured on tool bus (Index 1).

### Software Setup
- Linux Ubuntu 20.04 / 22.04 LTS with CUDA drivers.
- Python 3.10+ managed via `uv`:
  ```bash
  # Install UV if not present
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

---

## 5. Step-by-Step Instructions: Deploying on Fairino

### Step 5.1: Clone and Set Up on Lab Workstation
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/Oswi-world-model-control.git
cd Oswi-world-model-control

# Create virtual environment and install dependencies
uv sync
source .venv/bin/activate
```

> **Note on Checkpoints:** Checkpoint weights (`checkpoints/*.pt`) are large files. Copy them directly from the server or download them into the `checkpoints/` directory:
> - `checkpoints/metaworld_model.pt`
> - `checkpoints/pp_model.pt`

### Step 5.2: Test Dry Run (No Robot Motion)
Verify that the workstation sees the robot IP and passes kinematics preflight:
```bash
python scripts/03_robot_dry_run.py --ip 192.168.57.2 --mock
```

### Step 5.3: Run Live Execution with ZED Camera
Provide a 10-frame video of the task (e.g. human/robot demonstration) and execute:
```bash
python scripts/04_execute_on_fairino.py \
  --demo-video data/sample_demos/pick_place_demo.mp4 \
  --camera zed \
  --ip 192.168.57.2 \
  --speed 10.0 \
  --use-gripper
```

---

## 6. Datasets & Fine-Tuning: Do We Need to Fine-Tune?

### Option A: Zero-Shot Direct Inference (No Fine-Tuning Needed)
- **How it works:** The pretrained OSVI-WM world model was trained on diverse Pick-and-Place and Meta-World manipulation trajectories. Because the model outputs normalized spatial waypoints, it generalizes **zero-shot** to novel objects and arrangements.
- **When to use:** For standard pick-and-place, pushing, reaching, and button pressing in clear workspace environments.

### Option B: Fine-Tuning on Real Fairino Demonstrations (Optional for Complex Tasks)
- **When to fine-tune:** If your physical setup has drastic visual differences (harsh lighting, unusual table textures, occlusions) or precise multi-stage contact requirements (e.g. insertion, turning a valve).
- **How much data is needed?** Only **15 to 30 demonstration trajectories** on the Fairino setup.
- **Fine-Tuning Procedure:**
  1. Record 20 trajectories on the Fairino: save $(I_0, \dots, I_T)$ video frames and $(X, Y, Z, \text{gripper})$ robot poses.
  2. Store them in `data/fairino_dataset/`.
  3. Run the fine-tuning script with learning rate $\eta = 5 \times 10^{-5}$ for 5–10 epochs using the pretrained checkpoint as initialization.

---

## 7. Troubleshooting & Safety Checklist

1. **Emergency Stop:** Always keep the hardware E-Stop button within reach during initial runs.
2. **Speed Limiting:** Start with `--speed 5.0` or `--speed 10.0` (percentage of maximum joint speed) to observe motion trajectories safely.
3. **Z-Height Offset:** If testing above a table, pass `--z-offset 50.0` to lift the entire trajectory 50mm above the surface as a safe clearance margin.
4. **IK Failure Warning:** If `preflight_check` fails on a waypoint, adjust the `default_rpy` in `calibration/cam2base_calibration.json` to better match the tool approach angle.
