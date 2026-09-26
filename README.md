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
│   ├── metaworld_model.pt         # Pretrained MetaWorld weights (224x224)
│   └── pp_model.pt                # Pick-and-Place cross-embodiment weights (256x320)
├── calibration/
│   ├── cam2base_calibration.json  # Camera extrinsics matrix (T_cam2base)
│   ├── calibrate_4point.py        # Interactive 4-point touch-off Eye-to-Base calibration
│   └── projection.py              # Projects (u, v, depth) -> Fairino base frame (mm)
├── core/
│   ├── model.py                   # Top-level OSVIWorldModel architecture
│   ├── system_model.py            # Autoregressive Transformer World Model
│   ├── resnet_encoder.py          # ResNet-18 visual feature extractor
│   ├── traj_embed.py              # Spatio-temporal non-local attention action model
│   ├── attentive_pooler.py        # Query pooler for trajectory decoding
│   └── inference_engine.py        # Clean high-level prediction API (multi-resolution)
├── perception/
│   ├── transforms.py              # Image preprocessing & normalization
│   ├── video_loader.py            # 10-frame demonstration video loader
│   └── camera_stream.py           # ZED & USB camera stream adapter (with mock fallback)
├── robot/
│   ├── fairino_driver.py          # Direct Fairino XML-RPC controller (MoveL, MoveJ, IK search)
│   ├── gripper.py                 # Gripper activation and grasp commands
│   ├── trajectory_follower.py     # Safe Cartesian follower with IK preflight check
│   └── fairino_sdk/               # Official Fairino Python SDK (Linux & Windows)
├── configs/
│   ├── model_config.yaml          # World model hyperparameters
│   └── fairino_robot.yaml         # Robot IP, speed, safety limits
├── scripts/
│   ├── 00_extract_svo_to_mp4.py   # ZED .svo/.svo2 to .mp4 video extractor
│   ├── 00_record_demo.py          # Live top-view demonstration recorder
│   ├── 00_trim_video.py           # Video trimming & cropping utility
│   ├── 01_offline_inference.py    # Step 1: Run inference & print 3D waypoints table
│   ├── 02_visualize_trajectory.py # Step 2: Plot 3D trajectory (saves to outputs/trajectory_3d.png)
│   ├── 03_robot_dry_run.py        # Step 3: Run IK & reachability preflight check
│   ├── 04_execute_on_fairino.py   # Step 4: Live deployment with Camera + Fairino FR10
│   └── 05_visualize_demo_vs_execution.py # Step 5: Side-by-side demo vs prediction MP4 generator
├── main.py                        # Unified Master CLI runner
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

## 🧪 Unified Execution CLI (`main.py`)

You can execute all pipeline stages directly through `python main.py <mode>`:

### 1. Offline Trajectory Foreseeing
```bash
python main.py inference --checkpoint checkpoints/pp_model.pt
```

### 2. Side-by-Side Visualizer & Video Generator
```bash
python main.py visualize --demo-video data/demos/bottle_top_view.mp4
```

### 3. Preflight Inverse Kinematics Verification
```bash
python main.py dry-run --mock-robot
```

### 4. Physical Robot Execution (Fairino FR10)
```bash
python main.py execute --ip 192.168.57.2 --speed 8.0
```

### 5. Interactive 4-Point Eye-to-Base Calibration
```bash
python main.py calibrate --ip 192.168.57.2 --camera-id 0
```

---

## 🔬 Research Findings: 2D World Model vs Physical Manipulator

* **Visual Imitation**: OSVI-WM reliably infers macro pick-and-place task sequences directly from unsegmented video in normalized 2D image space.
* **Extrinsics Dependency**: 2D image-space world models **strictly require precise extrinsic Eye-to-Base calibration ($T_{\text{cam2base}}$)** for physical grasping. Physical deployments without millimeter-accurate calibration suffer from projective spatial drift, establishing the necessity for cross-view joint representation models (e.g. JEPA) or direct 3D visual grounding.

---

## 📄 License & Open-Source Notice
This repository is 100% open-source compatible (Apache 2.0 / MIT).

