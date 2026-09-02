"""
Camera Stream Adapter for ZED and Standard USB Cameras.
Gracefully handles headless environments by providing a Mock camera mode.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class CameraStream:
    def __init__(self, camera_type: str = "opencv", camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720)):
        self.camera_type = camera_type.lower()
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.zed = None

    def start(self):
        if self.camera_type == "opencv":
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            if not self.cap.isOpened():
                print(f"[Warning] Could not open OpenCV camera {self.camera_id}. Operating in Mock mode.")
                self.cap = None
        elif self.camera_type == "zed":
            try:
                import pyzed.sl as sl
                self.zed = sl.Camera()
                init_params = sl.InitParameters()
                init_params.camera_resolution = sl.RESOLUTION.HD720
                init_params.camera_fps = 30
                status = self.zed.open(init_params)
                if status != sl.ERROR_CODE.SUCCESS:
                    print(f"[Warning] Failed to open ZED camera: {status}. Operating in Mock mode.")
                    self.zed = None
            except ImportError:
                print("[Warning] pyzed is not installed. Operating in Mock mode.")
                self.zed = None

    def get_frame(self) -> np.ndarray:
        """
        Captures one RGB frame (H, W, 3).
        """
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.zed is not None:
            import pyzed.sl as sl
            image = sl.Mat()
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                frame_bgra = image.get_data()
                return cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2RGB)

        # Fallback Mock Frame (useful for headless testing)
        return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

    def stop(self):
        if self.cap is not None:
            self.cap.release()
        if self.zed is not None:
            self.zed.close()
