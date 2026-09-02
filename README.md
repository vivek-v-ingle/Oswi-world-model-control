# Oswi-World-Model-Control: One-Shot Visual Imitation on Fairino Robot

A clean, modular, standalone deployment repository for **OSVI-WM (One-Shot Visual Imitation using World-Model-Guided Trajectory Generation)** integrated with the **Fairino FR10** collaborative robot arm.

---

## 🚀 Key Highlights

1. **Self-Contained World Model Core**: Directly implements the NeurIPS 2025 OSVI-WM visual encoder, spatio-temporal intent attention, and autoregressive Transformer forward dynamics model.
2. **Offline-First & Headless-Ready**: Run inference, generate 3D trajectory visualizations, and perform Inverse Kinematics (IK) preflight checks on headless servers via SSH before touching physical hardware.
3. **Direct Python XML-RPC Control**: Communicates directly with the Fairino robot controller via lightweight Python XML-RPC—no heavy ROS 2 middleware required for core trajectory tracking.
4. **Cross-Platform & Open-Source**: Seamlessly clones between remote GPU compute servers (Linux/CUDA) and lab workstations with ZED cameras.

---

## 🏗️ Repository Architecture

```text
Oswi-world-model-control/
├── checkpoints/
│   ├── metaworld_model.pt         # Pretrained OSVI-WM weights
│   └── pp_model.pt                # Pick-and-Place cross-embodiment weights
├── calibration/
│   ├── cam2base_calibration.json  # Camera extrinsics matrix (T_cam2base)
│   └── projection.py              # Projects (u, v, depth) -> Fairino base frame (mm)
├── core/
│   ├── model.py                   # Top-level OSVIWorldModel architecture
│   ├── system_model.py            # Autoregressive Transformer World Model
│   ├── resnet_encoder.py          # ResNet-18 visual feature extractor
│   ├── traj_embed.py              # Spatio-temporal non-local attention action model
│   ├── attentive_pooler.py        # Query pooler for trajectory decoding
│   └── inference_engine.py        # Clean high-level prediction API
├── perception/
│   ├── transforms.py              # Image preprocessing & normalization (224x224)
│   ├── video_loader.py            # 10-frame demonstration video loader
│   └── camera_stream.py           # ZED & USB camera stream adapter (with mock fallback)
├── robot/
│   ├── fairino_driver.py          # Direct Fairino XML-RPC controller (MoveL, MoveJ, State)
│   ├── gripper.py                 # Gripper activation and grasp commands
│   ├── trajectory_follower.py     # Safe Cartesian follower with IK preflight check
│   └── fairino_sdk/               # Official Fairino Python SDK (Linux & Windows)
├── configs/
│   ├── model_config.yaml          # World model hyperparameters
│   └── fairino_robot.yaml         # Robot IP, speed, safety limits
├── scripts/
│   ├── 01_offline_inference.py    # Step 1: Run inference & print 3D waypoints table
│   ├── 02_visualize_trajectory.py # Step 2: Plot 3D trajectory (saves to outputs/trajectory_3d.png)
│   ├── 03_robot_dry_run.py        # Step 3: Run IK & reachability preflight check
│   └── 04_execute_on_fairino.py   # Step 4: Live deployment with Camera + Fairino FR10
├── pyproject.toml
└── README.md
```

---

## 📦 Quickstart & Installation

Managed with **`uv`**:

```bash
cd Oswi-world-model-control
uv sync
source .venv/bin/activate
```

---

## 🧪 Phased Execution Workflow

### Step 1: Offline Inference & Trajectory Foreseeing
Run the forward world model on a 10-frame demonstration and observe the decoded 3D waypoints:
```bash
python scripts/01_offline_inference.py
```

### Step 2: 3D Trajectory Visualization
Generate a 3D trajectory plot and gripper profile (saves image to `outputs/trajectory_3d.png`):
```bash
python scripts/02_visualize_trajectory.py
```

### Step 3: Robot IK Preflight Verification
Verify that every predicted waypoint is within physical safety bounds and has a valid Inverse Kinematics solution:
```bash
python scripts/03_robot_dry_run.py --mock
```

### Step 4: Live Deployment on Fairino FR10
Deploy on the lab workstation connected to the Fairino robot:
```bash
python scripts/04_execute_on_fairino.py \
  --demo-video data/sample_demos/screwdriver_demo.mp4 \
  --camera zed \
  --ip 192.168.57.2 \
  --speed 10.0
```

---

## 📄 License & Open-Source Notice
This repository is 100% open-source compatible (combines public NeurIPS 2025 OSVI-WM architecture with standard Fairino OEM client SDK and modular Python robotics control).
