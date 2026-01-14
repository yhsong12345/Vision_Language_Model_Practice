"""
Sim2Real Utilities

Utilities for bridging simulation to real-world deployment:
- Domain randomization
- Noise augmentation
- Camera calibration
- Action space mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization."""
    # Visual randomization
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: Tuple[float, float] = (-0.1, 0.1)

    # Noise
    gaussian_noise_std: float = 0.02
    salt_pepper_prob: float = 0.01
    blur_kernel_range: Tuple[int, int] = (1, 5)

    # Geometric
    random_crop_scale: Tuple[float, float] = (0.9, 1.0)
    random_rotation_range: Tuple[float, float] = (-5, 5)  # degrees

    # Camera
    camera_fov_range: Tuple[float, float] = (55, 65)  # degrees
    camera_position_noise: float = 0.01  # meters
    camera_rotation_noise: float = 1.0  # degrees

    # Physics
    friction_range: Tuple[float, float] = (0.5, 1.5)
    mass_range: Tuple[float, float] = (0.8, 1.2)
    damping_range: Tuple[float, float] = (0.8, 1.2)

    # Lighting
    light_intensity_range: Tuple[float, float] = (0.5, 1.5)
    light_color_variation: float = 0.1

    # Background
    use_random_backgrounds: bool = False
    background_paths: List[str] = field(default_factory=list)


class DomainRandomization:
    """
    Domain randomization for sim2real transfer.

    Applies random perturbations to observations to improve
    policy generalization from simulation to real world.
    """

    def __init__(self, config: DomainRandomizationConfig = None):
        self.config = config or DomainRandomizationConfig()

    def randomize_image(
        self,
        image: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Apply visual domain randomization to image.

        Args:
            image: (B, C, H, W) or (C, H, W) tensor in [0, 1]
            training: Whether in training mode

        Returns:
            Randomized image
        """
        if not training:
            return image

        # Handle single image
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Color jittering
        image = self._color_jitter(image)

        # Add noise
        image = self._add_noise(image)

        # Blur (occasionally)
        if np.random.random() < 0.3:
            image = self._random_blur(image)

        # Clamp to valid range
        image = torch.clamp(image, 0, 1)

        if squeeze:
            image = image.squeeze(0)

        return image

    def _color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply color jittering."""
        B, C, H, W = image.shape

        # Brightness
        brightness = torch.empty(B, 1, 1, 1, device=image.device).uniform_(
            self.config.brightness_range[0],
            self.config.brightness_range[1],
        )
        image = image * brightness

        # Contrast
        contrast = torch.empty(B, 1, 1, 1, device=image.device).uniform_(
            self.config.contrast_range[0],
            self.config.contrast_range[1],
        )
        mean = image.mean(dim=(2, 3), keepdim=True)
        image = (image - mean) * contrast + mean

        return image

    def _add_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(image) * self.config.gaussian_noise_std
        return image + noise

    def _random_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random blur."""
        k_min, k_max = self.config.blur_kernel_range
        kernel_size = np.random.randint(k_min, k_max + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        sigma = kernel_size / 3.0
        x = torch.arange(kernel_size, device=image.device) - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.outer(kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.repeat(image.shape[1], 1, 1, 1)

        # Apply blur
        padding = kernel_size // 2
        blurred = F.conv2d(image, kernel_2d, padding=padding, groups=image.shape[1])

        return blurred

    def randomize_physics(self) -> Dict[str, float]:
        """
        Generate randomized physics parameters.

        Returns:
            Dictionary of physics parameters
        """
        return {
            "friction": np.random.uniform(*self.config.friction_range),
            "mass_scale": np.random.uniform(*self.config.mass_range),
            "damping": np.random.uniform(*self.config.damping_range),
        }

    def randomize_camera(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Generate randomized camera parameters.

        Returns:
            Dictionary of camera parameters
        """
        return {
            "fov": np.random.uniform(*self.config.camera_fov_range),
            "position_offset": np.random.randn(3) * self.config.camera_position_noise,
            "rotation_offset": np.random.randn(3) * self.config.camera_rotation_noise,
        }

    def randomize_lighting(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Generate randomized lighting parameters.

        Returns:
            Dictionary of lighting parameters
        """
        return {
            "intensity": np.random.uniform(*self.config.light_intensity_range),
            "color": 1.0 + np.random.randn(3) * self.config.light_color_variation,
        }


@dataclass
class SensorNoiseConfig:
    """Configuration for sensor noise simulation."""
    # Camera
    camera_noise_std: float = 0.01
    camera_motion_blur_prob: float = 0.1
    camera_dropout_prob: float = 0.01

    # LiDAR
    lidar_noise_std: float = 0.02  # meters
    lidar_dropout_prob: float = 0.05
    lidar_beam_divergence: float = 0.001  # radians

    # IMU
    imu_accel_noise_std: float = 0.01  # m/s^2
    imu_gyro_noise_std: float = 0.001  # rad/s
    imu_bias_instability: float = 0.0001

    # Encoder
    encoder_noise_std: float = 0.001  # radians
    encoder_quantization: float = 0.001


class SensorNoiseAugmentation:
    """
    Add realistic sensor noise for sim2real transfer.

    Simulates various sensor imperfections commonly
    found in real-world robotic systems.
    """

    def __init__(self, config: SensorNoiseConfig = None):
        self.config = config or SensorNoiseConfig()

    def add_camera_noise(
        self,
        image: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Add realistic camera noise."""
        if not training:
            return image

        # Gaussian noise
        noise = torch.randn_like(image) * self.config.camera_noise_std
        image = image + noise

        # Random pixel dropout
        if np.random.random() < self.config.camera_dropout_prob:
            mask = torch.rand_like(image) > 0.01
            image = image * mask

        return torch.clamp(image, 0, 1)

    def add_lidar_noise(
        self,
        points: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Add realistic LiDAR noise.

        Args:
            points: (N, 3+) point cloud [x, y, z, ...]

        Returns:
            Noisy point cloud
        """
        if not training:
            return points

        # Position noise
        xyz_noise = torch.randn_like(points[:, :3]) * self.config.lidar_noise_std
        points = points.clone()
        points[:, :3] = points[:, :3] + xyz_noise

        # Random point dropout
        mask = torch.rand(len(points)) > self.config.lidar_dropout_prob
        points = points[mask]

        return points

    def add_imu_noise(
        self,
        imu_data: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Add realistic IMU noise.

        Args:
            imu_data: (..., 6) tensor [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

        Returns:
            Noisy IMU data
        """
        if not training:
            return imu_data

        imu_data = imu_data.clone()

        # Accelerometer noise
        accel_noise = torch.randn_like(imu_data[..., :3]) * self.config.imu_accel_noise_std
        imu_data[..., :3] = imu_data[..., :3] + accel_noise

        # Gyroscope noise
        gyro_noise = torch.randn_like(imu_data[..., 3:6]) * self.config.imu_gyro_noise_std
        imu_data[..., 3:6] = imu_data[..., 3:6] + gyro_noise

        return imu_data

    def add_encoder_noise(
        self,
        positions: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Add realistic encoder noise."""
        if not training:
            return positions

        # Gaussian noise
        noise = torch.randn_like(positions) * self.config.encoder_noise_std
        positions = positions + noise

        # Quantization
        if self.config.encoder_quantization > 0:
            positions = torch.round(positions / self.config.encoder_quantization) * self.config.encoder_quantization

        return positions


class ActionSpaceMapper:
    """
    Maps actions between simulation and real robot.

    Handles differences in:
    - Action ranges and scaling
    - Control modes (position, velocity, torque)
    - Safety limits
    """

    def __init__(
        self,
        sim_action_bounds: Tuple[np.ndarray, np.ndarray],
        real_action_bounds: Tuple[np.ndarray, np.ndarray],
        action_type: str = "position",
    ):
        """
        Args:
            sim_action_bounds: (low, high) bounds in simulation
            real_action_bounds: (low, high) bounds on real robot
            action_type: "position", "velocity", or "torque"
        """
        self.sim_low, self.sim_high = sim_action_bounds
        self.real_low, self.real_high = real_action_bounds
        self.action_type = action_type

        # Compute scaling
        self.sim_range = self.sim_high - self.sim_low
        self.real_range = self.real_high - self.real_low
        self.scale = self.real_range / (self.sim_range + 1e-8)
        self.offset = self.real_low - self.sim_low * self.scale

    def sim_to_real(self, action: np.ndarray) -> np.ndarray:
        """Convert simulation action to real robot action."""
        # Linear mapping
        real_action = action * self.scale + self.offset

        # Clamp to real bounds
        real_action = np.clip(real_action, self.real_low, self.real_high)

        return real_action

    def real_to_sim(self, action: np.ndarray) -> np.ndarray:
        """Convert real robot action to simulation action."""
        # Inverse mapping
        sim_action = (action - self.offset) / (self.scale + 1e-8)

        # Clamp to sim bounds
        sim_action = np.clip(sim_action, self.sim_low, self.sim_high)

        return sim_action


class CameraCalibration:
    """
    Camera calibration utilities for sim2real transfer.

    Handles differences between simulated and real cameras.
    """

    def __init__(
        self,
        intrinsics: np.ndarray = None,
        distortion: np.ndarray = None,
        extrinsics: np.ndarray = None,
    ):
        """
        Args:
            intrinsics: 3x3 camera intrinsic matrix
            distortion: Distortion coefficients [k1, k2, p1, p2, k3]
            extrinsics: 4x4 camera extrinsic matrix (camera to world)
        """
        self.intrinsics = intrinsics
        self.distortion = distortion
        self.extrinsics = extrinsics

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from image."""
        try:
            import cv2
        except ImportError:
            return image

        if self.intrinsics is None or self.distortion is None:
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.intrinsics, self.distortion, (w, h), 1, (w, h)
        )

        undistorted = cv2.undistort(
            image, self.intrinsics, self.distortion, None, new_camera_matrix
        )

        # Crop to ROI
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to image coordinates."""
        if self.intrinsics is None:
            raise ValueError("Intrinsics not set")

        # Transform to camera frame if extrinsics provided
        if self.extrinsics is not None:
            points_3d = self._transform_points(points_3d, np.linalg.inv(self.extrinsics))

        # Project
        points_2d = points_3d[:, :2] / points_3d[:, 2:3]
        points_2d = points_2d @ self.intrinsics[:2, :2].T + self.intrinsics[:2, 2]

        return points_2d

    def _transform_points(
        self,
        points: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """Apply 4x4 transform to 3D points."""
        ones = np.ones((len(points), 1))
        points_h = np.hstack([points, ones])
        transformed = points_h @ transform.T
        return transformed[:, :3]


def create_sim2real_augmentation(
    domain_randomization: bool = True,
    sensor_noise: bool = True,
    domain_config: DomainRandomizationConfig = None,
    noise_config: SensorNoiseConfig = None,
) -> nn.Module:
    """
    Create a combined sim2real augmentation module.

    Args:
        domain_randomization: Enable domain randomization
        sensor_noise: Enable sensor noise
        domain_config: Domain randomization config
        noise_config: Sensor noise config

    Returns:
        Augmentation module
    """

    class Sim2RealAugmentation(nn.Module):
        def __init__(self):
            super().__init__()
            self.domain_rand = DomainRandomization(domain_config) if domain_randomization else None
            self.sensor_noise = SensorNoiseAugmentation(noise_config) if sensor_noise else None

        def forward(
            self,
            image: torch.Tensor,
            training: bool = True,
        ) -> torch.Tensor:
            if self.domain_rand is not None:
                image = self.domain_rand.randomize_image(image, training)

            if self.sensor_noise is not None:
                image = self.sensor_noise.add_camera_noise(image, training)

            return image

    return Sim2RealAugmentation()


if __name__ == "__main__":
    print("Sim2Real Utilities")
    print("=" * 50)

    # Test domain randomization
    config = DomainRandomizationConfig()
    dr = DomainRandomization(config)

    image = torch.rand(1, 3, 224, 224)
    randomized = dr.randomize_image(image)
    print(f"Domain randomization: {image.shape} -> {randomized.shape}")

    # Test sensor noise
    noise_config = SensorNoiseConfig()
    sn = SensorNoiseAugmentation(noise_config)

    noisy_image = sn.add_camera_noise(image)
    print(f"Sensor noise: {image.shape} -> {noisy_image.shape}")

    # Test LiDAR noise
    points = torch.randn(1000, 4)  # x, y, z, intensity
    noisy_points = sn.add_lidar_noise(points)
    print(f"LiDAR noise: {points.shape} -> {noisy_points.shape}")

    # Test action space mapper
    sim_bounds = (np.array([-1, -1, -1]), np.array([1, 1, 1]))
    real_bounds = (np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
    mapper = ActionSpaceMapper(sim_bounds, real_bounds)

    sim_action = np.array([0.5, -0.5, 0.0])
    real_action = mapper.sim_to_real(sim_action)
    print(f"Action mapping: {sim_action} -> {real_action}")

    print("\nAll tests passed!")
