"""
High-Level Inference Engine for OSVI-WM with Fairino Projection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from core.model import OSVIWorldModel
from calibration.projection import CoordinateProjector
from perception.transforms import preprocess_frame, preprocess_sequence


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        calibration_path: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        latent_dim: int = 256,
        waypoints: int = 5,
        sub_waypoints: bool = True,
        image_resolution: Optional[Tuple[int, int]] = None,
    ):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Auto-detect resolution from checkpoint if not explicitly provided
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cleaned_state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }

        if image_resolution is None:
            # Check output head shape (40960 for 256x320 / 80 patches, 25088 for 224x224 / 49 patches)
            out_dim = cleaned_state_dict.get("forward_model.output_head.weight", None)
            if out_dim is not None and out_dim.shape[0] == 40960:
                image_resolution = (256, 320)
            else:
                image_resolution = (224, 224)

        self.image_resolution = image_resolution
        self.model = OSVIWorldModel(
            latent_dim=latent_dim,
            waypoints=waypoints,
            sub_waypoints=sub_waypoints,
            image_resolution=self.image_resolution,
        ).to(self.device)

        self.model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Loaded OSVI-WM weights from {self.checkpoint_path.name} (Resolution: {self.image_resolution})")
        self.model.eval()

        self.projector = None
        if calibration_path is not None:
            self.projector = CoordinateProjector(calibration_path)

    def predict(
        self,
        teacher_frames: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        current_obs: Union[np.ndarray, torch.Tensor],
        rollout_horizon: int = 16,
    ) -> Dict:
        """
        Runs world-model inference to predict trajectory waypoints.

        Args:
            teacher_frames: 10 RGB frames (Numpy array or List of (H, W, 3) arrays).
            current_obs: Current RGB observation image (H, W, 3) or (2, H, W, 3).
            rollout_horizon: World model latent foreseeing horizon.
        Returns:
            Dict containing:
                - 'raw_waypoints': (1, N, 4) tensor (u, v, depth, gripper)
                - 'robot_waypoints': List of 3D dicts in Fairino base frame (if calibration provided)
        """
        if not isinstance(teacher_frames, torch.Tensor):
            context_tensor = preprocess_sequence(teacher_frames, target_len=10, target_size=self.image_resolution)
        else:
            context_tensor = teacher_frames

        if not isinstance(current_obs, torch.Tensor):
            obs_tensor = preprocess_frame(current_obs, repeat_frames=2, target_size=self.image_resolution)
        else:
            obs_tensor = current_obs

        context_tensor = context_tensor.to(self.device)
        obs_tensor = obs_tensor.to(self.device)

        with torch.no_grad():
            out = self.model(obs_tensor, context_tensor, T_tot=rollout_horizon)

        raw_wps = out["waypoints"]  # Shape: (1, 15, 4)
        result = {
            "raw_waypoints": raw_wps,
            "forward_states": out["forward_states"],
        }

        if self.projector is not None:
            result["robot_waypoints"] = self.projector.project(raw_wps)

        return result
