"""
Video and Demonstration File Loader for OSVI-WM.
"""

from pathlib import Path
from typing import List, Union
import cv2
import numpy as np


class DemonstrationLoader:
    @staticmethod
    def load_video(video_path: Union[str, Path], max_frames: int = 10) -> List[np.ndarray]:
        """
        Reads video file and extracts uniformly sampled RGB frames.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        raw_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_frames.append(frame_rgb)
        cap.release()

        if not raw_frames:
            raise RuntimeError(f"Could not read frames from {path}")

        if len(raw_frames) < max_frames:
            indices = np.linspace(0, len(raw_frames) - 1, max_frames, dtype=int)
            return [raw_frames[i] for i in indices]

        indices = np.linspace(0, len(raw_frames) - 1, max_frames, dtype=int)
        return [raw_frames[i] for i in indices]

    @staticmethod
    def load_image_folder(folder_path: Union[str, Path], max_frames: int = 10) -> List[np.ndarray]:
        """
        Reads ordered image files (.png, .jpg) from a directory.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {folder}")

        files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")))
        if not files:
            raise FileNotFoundError(f"No image files found in {folder}")

        indices = np.linspace(0, len(files) - 1, max_frames, dtype=int)
        selected = [files[i] for i in indices]

        frames = []
        for p in selected:
            img = cv2.imread(str(p))
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames
