# Autonomous Driving Sensor Processing

This document covers sensor processing and fusion for autonomous driving VLA models, including camera, LiDAR, radar, and multi-modal fusion techniques.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Sensor Processing Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Camera    │  │   LiDAR     │  │   Radar     │  │    IMU      │        │
│  │  (RGB/Depth)│  │ (3D Points) │  │ (Velocity)  │  │ (Pose/Accel)│        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         ▼                ▼                ▼                ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Vision    │  │  PointNet   │  │   Radar     │  │    IMU      │        │
│  │   Encoder   │  │   Encoder   │  │   Encoder   │  │   Encoder   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                         │
│                                   ▼                                         │
│                          ┌─────────────────┐                               │
│                          │  Sensor Fusion  │                               │
│                          │  (BEV / Early / │                               │
│                          │   Late Fusion)  │                               │
│                          └────────┬────────┘                               │
│                                   │                                         │
│                                   ▼                                         │
│                          Fused Features for VLA                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Sensor Modalities

### Sensor Summary

| Sensor | Data Type | Resolution | Range | Frequency | Use Case |
|--------|-----------|------------|-------|-----------|----------|
| Camera (RGB) | Images | 1920×1080 | ~200m | 30-60 Hz | Scene understanding, lane detection |
| Camera (Stereo) | Depth | 1280×720 | ~100m | 30 Hz | 3D reconstruction |
| LiDAR | Point cloud | 64-128 beams | 200m | 10-20 Hz | 3D object detection, mapping |
| Radar | Velocity/Range | Low | 250m | 20 Hz | Velocity estimation, bad weather |
| IMU | Pose/Acceleration | N/A | N/A | 100-1000 Hz | Ego-motion, localization |
| GPS/GNSS | Position | ~1m accuracy | Global | 10 Hz | Global localization |
| CAN Bus | Vehicle state | N/A | N/A | 100 Hz | Steering, throttle, speed |

---

## Camera Processing

### Multi-Camera Setup

```python
class MultiCameraEncoder(nn.Module):
    """
    Encode images from multiple cameras for 360° perception.

    Typical setup:
    - Front: Main perception camera
    - Front-Left/Right: Side view for lane changes
    - Rear: Reverse and rear monitoring
    - Side-Left/Right: Blind spot monitoring
    """

    def __init__(
        self,
        num_cameras: int = 6,
        image_size: Tuple[int, int] = (224, 224),
        backbone: str = "resnet50",
        output_dim: int = 512,
    ):
        super().__init__()
        self.num_cameras = num_cameras

        # Shared backbone (weight sharing across cameras)
        self.backbone = self._create_backbone(backbone)

        # Camera-specific projection
        self.camera_projections = nn.ModuleList([
            nn.Linear(2048, output_dim) for _ in range(num_cameras)
        ])

        # Cross-camera attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(output_dim * num_cameras, output_dim)

    def forward(
        self,
        images: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images: Dict mapping camera names to tensors (B, 3, H, W)
                   e.g., {"front": tensor, "front_left": tensor, ...}

        Returns:
            fused_features: (B, output_dim)
        """
        camera_features = []

        for i, (name, img) in enumerate(images.items()):
            # Extract features with shared backbone
            feat = self.backbone(img)
            feat = feat.mean(dim=[2, 3])  # Global average pooling

            # Camera-specific projection
            feat = self.camera_projections[i](feat)
            camera_features.append(feat)

        # Stack camera features: (B, num_cameras, output_dim)
        stacked = torch.stack(camera_features, dim=1)

        # Cross-camera attention
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Flatten and project
        fused = attended.flatten(start_dim=1)
        output = self.output_proj(fused)

        return output

    def _create_backbone(self, name: str) -> nn.Module:
        import torchvision.models as models

        if name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            return nn.Sequential(*list(backbone.children())[:-2])
        elif name == "efficientnet":
            backbone = models.efficientnet_b0(pretrained=True)
            return backbone.features
```

### Camera Intrinsics and Extrinsics

```python
@dataclass
class CameraConfig:
    """Camera calibration parameters."""

    # Intrinsic matrix (3x3)
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y

    # Distortion coefficients
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    # Extrinsic (camera to vehicle transform)
    position: Tuple[float, float, float] = (0, 0, 0)  # x, y, z in meters
    rotation: Tuple[float, float, float] = (0, 0, 0)  # roll, pitch, yaw in radians

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def project_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to image coordinates."""
        # Transform to camera frame
        points_cam = self._world_to_camera(points_3d)

        # Project to image
        points_2d = points_cam[:, :2] / points_cam[:, 2:3]
        points_2d[:, 0] = points_2d[:, 0] * self.fx + self.cx
        points_2d[:, 1] = points_2d[:, 1] * self.fy + self.cy

        return points_2d


# Typical multi-camera setup for autonomous vehicle
CAMERA_CONFIGS = {
    "front": CameraConfig(
        fx=1000, fy=1000, cx=960, cy=540,
        position=(2.0, 0.0, 1.5),
        rotation=(0, 0, 0),
    ),
    "front_left": CameraConfig(
        fx=1000, fy=1000, cx=960, cy=540,
        position=(1.5, 0.8, 1.5),
        rotation=(0, 0, 0.52),  # 30 degrees
    ),
    "front_right": CameraConfig(
        fx=1000, fy=1000, cx=960, cy=540,
        position=(1.5, -0.8, 1.5),
        rotation=(0, 0, -0.52),
    ),
    "rear": CameraConfig(
        fx=1000, fy=1000, cx=960, cy=540,
        position=(-1.0, 0.0, 1.5),
        rotation=(0, 0, 3.14),  # 180 degrees
    ),
    "side_left": CameraConfig(
        fx=1000, fy=1000, cx=960, cy=540,
        position=(0.0, 1.0, 1.5),
        rotation=(0, 0, 1.57),  # 90 degrees
    ),
    "side_right": CameraConfig(
        fx=1000, fy=1000, cx=960, cy=540,
        position=(0.0, -1.0, 1.5),
        rotation=(0, 0, -1.57),
    ),
}
```

---

## LiDAR Processing

### Point Cloud Encoder

```python
class LiDAREncoder(nn.Module):
    """
    Encode LiDAR point cloud for autonomous driving.

    Supports multiple architectures:
    - PointNet/PointNet++
    - VoxelNet
    - PointPillars
    """

    def __init__(
        self,
        architecture: str = "pointpillars",
        num_points: int = 100000,
        point_dim: int = 4,  # x, y, z, intensity
        output_dim: int = 512,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 4.0),
        point_cloud_range: List[float] = [-50, -50, -3, 50, 50, 1],
    ):
        super().__init__()
        self.architecture = architecture
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        if architecture == "pointnet":
            self.encoder = PointNetEncoder(point_dim, output_dim)
        elif architecture == "pointpillars":
            self.encoder = PointPillarsEncoder(
                point_dim, output_dim, voxel_size, point_cloud_range
            )
        elif architecture == "voxelnet":
            self.encoder = VoxelNetEncoder(
                point_dim, output_dim, voxel_size, point_cloud_range
            )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 4) point cloud [x, y, z, intensity]

        Returns:
            features: (B, output_dim) or (B, H, W, output_dim) for BEV
        """
        return self.encoder(points)


class PointPillarsEncoder(nn.Module):
    """
    PointPillars: Fast encoder for LiDAR point clouds.

    Converts point cloud to pseudo-image (pillars) for efficient 2D CNN processing.
    """

    def __init__(
        self,
        point_dim: int = 4,
        output_dim: int = 512,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 4.0),
        point_cloud_range: List[float] = [-50, -50, -3, 50, 50, 1],
        max_points_per_pillar: int = 32,
        max_pillars: int = 12000,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        # Pillar feature extraction
        self.pillar_encoder = nn.Sequential(
            nn.Linear(point_dim + 5, 64),  # +5 for relative position and cluster center
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # 2D backbone for pseudo-image
        self.backbone = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

        # Compute grid size
        self.nx = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.ny = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 4) point cloud

        Returns:
            bev_features: (B, output_dim, H, W)
        """
        batch_size = points.shape[0]

        # Voxelize points into pillars
        pillars, coords = self._create_pillars(points)

        # Encode pillars
        pillar_features = self.pillar_encoder(pillars)
        pillar_features = pillar_features.max(dim=2)[0]  # Max pooling over points

        # Scatter to pseudo-image
        pseudo_image = self._scatter_to_image(pillar_features, coords, batch_size)

        # 2D CNN backbone
        bev_features = self.backbone(pseudo_image)

        return bev_features

    def _create_pillars(self, points):
        """Convert point cloud to pillars (voxels in z-dimension)."""
        # Implementation depends on specific voxelization library
        # (e.g., spconv, torch_scatter)
        pass

    def _scatter_to_image(self, features, coords, batch_size):
        """Scatter pillar features to 2D pseudo-image."""
        pseudo_image = torch.zeros(
            batch_size, features.shape[-1], self.ny, self.nx,
            device=features.device
        )
        # Scatter features to corresponding positions
        return pseudo_image
```

### LiDAR Configuration

```python
@dataclass
class LiDARConfig:
    """LiDAR sensor configuration."""

    # Sensor specs
    num_beams: int = 64           # Velodyne VLP-64, Ouster OS1-64
    rotation_frequency: float = 10.0  # Hz
    points_per_second: int = 1_300_000

    # Range
    min_range: float = 0.5        # meters
    max_range: float = 120.0      # meters

    # Field of view
    horizontal_fov: float = 360.0  # degrees
    vertical_fov: float = 40.0     # degrees (typically -25 to +15)

    # Position (in vehicle frame)
    position: Tuple[float, float, float] = (0.0, 0.0, 1.8)  # x, y, z
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll, pitch, yaw

    # Processing
    point_cloud_range: List[float] = field(
        default_factory=lambda: [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0]
    )
    voxel_size: Tuple[float, float, float] = (0.2, 0.2, 4.0)


# Common LiDAR configurations
LIDAR_CONFIGS = {
    "velodyne_vlp64": LiDARConfig(
        num_beams=64,
        max_range=120.0,
        points_per_second=1_300_000,
    ),
    "ouster_os1_128": LiDARConfig(
        num_beams=128,
        max_range=200.0,
        points_per_second=2_600_000,
    ),
    "hesai_pandar128": LiDARConfig(
        num_beams=128,
        max_range=200.0,
        points_per_second=3_456_000,
    ),
}
```

---

## Radar Processing

### Radar Encoder

```python
class RadarEncoder(nn.Module):
    """
    Encode radar detections for autonomous driving.

    Radar advantages:
    - Works in bad weather (rain, fog, snow)
    - Provides velocity information directly
    - Long range detection

    Radar disadvantages:
    - Low angular resolution
    - Sparse detections
    - Multi-path reflections
    """

    def __init__(
        self,
        max_detections: int = 100,
        detection_dim: int = 7,  # x, y, z, vx, vy, rcs, snr
        output_dim: int = 256,
    ):
        super().__init__()
        self.max_detections = max_detections

        # Point-wise encoding
        self.point_encoder = nn.Sequential(
            nn.Linear(detection_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Set abstraction (like PointNet)
        self.set_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        detections: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            detections: (B, N, 7) radar detections
                       [x, y, z, vx, vy, rcs, snr]
            mask: (B, N) valid detection mask

        Returns:
            features: (B, output_dim)
        """
        # Encode each detection
        point_features = self.point_encoder(detections)

        # Apply mask
        if mask is not None:
            point_features = point_features * mask.unsqueeze(-1)

        # Global max pooling
        global_features = point_features.max(dim=1)[0]

        # Final encoding
        output = self.set_encoder(global_features)

        return output


@dataclass
class RadarConfig:
    """Radar sensor configuration."""

    # Sensor specs
    max_range: float = 250.0      # meters
    range_resolution: float = 0.5  # meters
    velocity_range: Tuple[float, float] = (-50, 50)  # m/s
    velocity_resolution: float = 0.1  # m/s

    # Field of view
    azimuth_fov: float = 120.0    # degrees (horizontal)
    elevation_fov: float = 20.0   # degrees (vertical)
    angular_resolution: float = 1.0  # degrees

    # Position
    position: Tuple[float, float, float] = (2.5, 0.0, 0.5)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Detection
    max_detections: int = 100
    detection_threshold: float = 10.0  # dB SNR


# Common radar configurations
RADAR_CONFIGS = {
    "front_long_range": RadarConfig(
        max_range=250.0,
        azimuth_fov=20.0,
        position=(2.5, 0.0, 0.5),
    ),
    "front_corner_left": RadarConfig(
        max_range=100.0,
        azimuth_fov=120.0,
        position=(2.0, 0.8, 0.5),
        rotation=(0, 0, 0.52),
    ),
    "front_corner_right": RadarConfig(
        max_range=100.0,
        azimuth_fov=120.0,
        position=(2.0, -0.8, 0.5),
        rotation=(0, 0, -0.52),
    ),
    "rear_corner_left": RadarConfig(
        max_range=100.0,
        azimuth_fov=120.0,
        position=(-1.0, 0.8, 0.5),
        rotation=(0, 0, 2.62),
    ),
    "rear_corner_right": RadarConfig(
        max_range=100.0,
        azimuth_fov=120.0,
        position=(-1.0, -0.8, 0.5),
        rotation=(0, 0, -2.62),
    ),
}
```

---

## IMU and Vehicle State

### IMU Encoder

```python
class IMUEncoder(nn.Module):
    """
    Encode IMU data for ego-motion estimation.

    IMU provides:
    - Linear acceleration (ax, ay, az)
    - Angular velocity (wx, wy, wz)
    - Orientation (roll, pitch, yaw) - from sensor fusion
    """

    def __init__(
        self,
        input_dim: int = 9,       # 3 accel + 3 gyro + 3 orientation
        hidden_dim: int = 64,
        output_dim: int = 64,
        sequence_length: int = 10,
    ):
        super().__init__()
        self.sequence_length = sequence_length

        # Temporal encoding with LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        imu_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imu_sequence: (B, T, 9) IMU readings over time

        Returns:
            features: (B, output_dim)
        """
        lstm_out, _ = self.lstm(imu_sequence)
        last_hidden = lstm_out[:, -1, :]
        output = self.output_proj(last_hidden)
        return output


class VehicleStateEncoder(nn.Module):
    """
    Encode vehicle state from CAN bus.

    Vehicle state includes:
    - Speed, steering angle
    - Throttle, brake
    - Gear, turn signals
    """

    def __init__(
        self,
        input_dim: int = 8,       # speed, steering, throttle, brake, gear, etc.
        output_dim: int = 32,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, vehicle_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vehicle_state: (B, 8) current vehicle state

        Returns:
            features: (B, output_dim)
        """
        return self.encoder(vehicle_state)
```

---

## Sensor Fusion

### Bird's Eye View (BEV) Fusion

```python
class BEVFusion(nn.Module):
    """
    Fuse multi-modal sensors in Bird's Eye View space.

    BEV representation advantages:
    - Unified coordinate system for all sensors
    - Natural for driving (top-down view)
    - Easy to fuse camera, LiDAR, radar
    """

    def __init__(
        self,
        camera_encoder: MultiCameraEncoder,
        lidar_encoder: LiDAREncoder,
        radar_encoder: RadarEncoder,
        bev_size: Tuple[int, int] = (200, 200),
        bev_resolution: float = 0.5,  # meters per pixel
        output_dim: int = 512,
    ):
        super().__init__()
        self.camera_encoder = camera_encoder
        self.lidar_encoder = lidar_encoder
        self.radar_encoder = radar_encoder

        self.bev_size = bev_size
        self.bev_resolution = bev_resolution

        # Camera to BEV projection (LSS-style)
        self.camera_to_bev = CameraToBEV(
            camera_dim=512,
            bev_size=bev_size,
            bev_resolution=bev_resolution,
        )

        # Fusion backbone
        self.fusion_backbone = nn.Sequential(
            nn.Conv2d(512 + 512 + 256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 3, padding=1),
        )

    def forward(
        self,
        cameras: Dict[str, torch.Tensor],
        lidar: torch.Tensor,
        radar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse all sensors in BEV space.

        Returns:
            bev_features: (B, output_dim, H, W)
        """
        # Camera features to BEV
        camera_features = self.camera_encoder(cameras)
        camera_bev = self.camera_to_bev(camera_features, cameras)

        # LiDAR already in BEV
        lidar_bev = self.lidar_encoder(lidar)

        # Radar to BEV
        radar_features = self.radar_encoder(radar)
        radar_bev = self._radar_to_bev(radar_features)

        # Concatenate and fuse
        fused = torch.cat([camera_bev, lidar_bev, radar_bev], dim=1)
        output = self.fusion_backbone(fused)

        return output


class CameraToBEV(nn.Module):
    """
    Lift camera features to BEV using depth estimation (LSS).

    LSS: Lift, Splat, Shoot
    """

    def __init__(
        self,
        camera_dim: int,
        bev_size: Tuple[int, int],
        bev_resolution: float,
        depth_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 60.0),
    ):
        super().__init__()
        self.depth_bins = depth_bins
        self.depth_range = depth_range

        # Depth distribution prediction
        self.depth_net = nn.Sequential(
            nn.Conv2d(camera_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, depth_bins, 1),
        )

        # BEV encoder
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(camera_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )

    def forward(
        self,
        camera_features: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lift camera features to BEV.

        1. Predict depth distribution for each pixel
        2. Lift features to 3D frustum
        3. Splat 3D features to BEV grid
        """
        # Predict depth distribution
        depth_probs = F.softmax(self.depth_net(camera_features), dim=1)

        # Create 3D frustum points
        frustum = self._create_frustum(camera_features.shape[2:])

        # Lift features with depth weighting
        lifted = self._lift(camera_features, depth_probs, frustum)

        # Transform to vehicle frame
        transformed = self._transform_to_vehicle(lifted, camera_extrinsics)

        # Splat to BEV
        bev = self._splat_to_bev(transformed)

        return bev
```

### Early vs Late Fusion

```python
class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate raw sensor data before encoding.

    Pros: Rich multi-modal interactions
    Cons: Computationally expensive, sensor-specific preprocessing
    """

    def __init__(self, output_dim: int = 512):
        super().__init__()
        # Project all sensors to common dimension first
        self.camera_proj = nn.Linear(512, 256)
        self.lidar_proj = nn.Linear(512, 256)
        self.radar_proj = nn.Linear(256, 256)

        # Joint encoder
        self.joint_encoder = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(
        self,
        camera_feat: torch.Tensor,
        lidar_feat: torch.Tensor,
        radar_feat: torch.Tensor,
    ) -> torch.Tensor:
        cam = self.camera_proj(camera_feat)
        lid = self.lidar_proj(lidar_feat)
        rad = self.radar_proj(radar_feat)

        concat = torch.cat([cam, lid, rad], dim=-1)
        return self.joint_encoder(concat)


class LateFusion(nn.Module):
    """
    Late fusion: Process sensors independently, fuse at decision level.

    Pros: Modular, sensor dropout during training
    Cons: Limited cross-modal interaction
    """

    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.camera_head = nn.Linear(512, output_dim)
        self.lidar_head = nn.Linear(512, output_dim)
        self.radar_head = nn.Linear(256, output_dim)

        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(
        self,
        camera_feat: torch.Tensor,
        lidar_feat: torch.Tensor,
        radar_feat: torch.Tensor,
    ) -> torch.Tensor:
        cam = self.camera_head(camera_feat)
        lid = self.lidar_head(lidar_feat)
        rad = self.radar_head(radar_feat)

        # Weighted sum
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * cam + weights[1] * lid + weights[2] * rad

        return fused
```

---

## Sensor Data Augmentation

### Camera Augmentation

```python
import torchvision.transforms as T

camera_augmentation = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
    ),
    T.RandomGrayscale(p=0.1),
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### LiDAR Augmentation

```python
class LiDARAugmentation:
    """Augmentation for LiDAR point clouds."""

    def __init__(
        self,
        rotation_range: float = 0.1,      # radians
        translation_range: float = 0.5,   # meters
        scale_range: Tuple[float, float] = (0.95, 1.05),
        dropout_ratio: float = 0.1,
        noise_std: float = 0.02,
    ):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.dropout_ratio = dropout_ratio
        self.noise_std = noise_std

    def __call__(self, points: np.ndarray) -> np.ndarray:
        # Random rotation around z-axis
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        points[:, :3] = points[:, :3] @ rot_matrix.T

        # Random translation
        translation = np.random.uniform(
            -self.translation_range,
            self.translation_range,
            size=3
        )
        points[:, :3] += translation

        # Random scaling
        scale = np.random.uniform(*self.scale_range)
        points[:, :3] *= scale

        # Random dropout
        keep_mask = np.random.rand(len(points)) > self.dropout_ratio
        points = points[keep_mask]

        # Add noise
        noise = np.random.normal(0, self.noise_std, points[:, :3].shape)
        points[:, :3] += noise

        return points
```

---

## Training with Multiple Sensors

### Multi-Sensor Dataset

```python
class MultiSensorDrivingDataset(torch.utils.data.Dataset):
    """Dataset with multiple sensor modalities."""

    def __init__(
        self,
        data_path: str,
        sensors: List[str] = ["camera", "lidar", "radar"],
        augment: bool = True,
    ):
        self.data_path = data_path
        self.sensors = sensors
        self.augment = augment

        self.samples = self._load_samples()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        output = {}

        # Load camera images
        if "camera" in self.sensors:
            cameras = {}
            for cam_name in ["front", "front_left", "front_right", "rear"]:
                img = self._load_image(sample[f"camera_{cam_name}"])
                if self.augment:
                    img = camera_augmentation(img)
                cameras[cam_name] = img
            output["cameras"] = cameras

        # Load LiDAR
        if "lidar" in self.sensors:
            points = self._load_pointcloud(sample["lidar"])
            if self.augment:
                points = LiDARAugmentation()(points)
            output["lidar"] = torch.tensor(points, dtype=torch.float32)

        # Load radar
        if "radar" in self.sensors:
            radar = self._load_radar(sample["radar"])
            output["radar"] = torch.tensor(radar, dtype=torch.float32)

        # Load IMU
        if "imu" in self.sensors:
            imu = self._load_imu(sample["imu"])
            output["imu"] = torch.tensor(imu, dtype=torch.float32)

        # Load labels
        output["action"] = torch.tensor(sample["action"], dtype=torch.float32)

        return output
```

### Training Command

```bash
python train/finetune/vla_finetuner.py \
    --pretrained-vlm ./output/stage1b/best \
    --dataset nuscenes \
    --sensors camera,lidar,radar \
    --fusion-type bev \
    --output-dir ./output/multi_sensor_vla
```

---

## Datasets with Sensor Data

| Dataset | Cameras | LiDAR | Radar | IMU | Source |
|---------|---------|-------|-------|-----|--------|
| nuScenes | 6 cameras | 1 LiDAR | 5 radar | Yes | [nuscenes.org](https://www.nuscenes.org/) |
| Waymo Open | 5 cameras | 5 LiDAR | - | Yes | [waymo.com/open](https://waymo.com/open) |
| KITTI | 4 cameras | 1 LiDAR | - | Yes | [cvlibs.net](http://www.cvlibs.net/datasets/kitti/) |
| Argoverse 2 | 7 cameras | 2 LiDAR | - | Yes | [argoverse.org](https://www.argoverse.org/) |
| CARLA | Configurable | Configurable | Configurable | Yes | [carla.org](https://carla.org/) |

---

## Best Practices

### Sensor Synchronization

1. **Timestamp alignment**: Synchronize all sensors to common timestamp
2. **Interpolation**: Interpolate higher-frequency sensors to match lower-frequency
3. **Latency compensation**: Account for sensor processing delays

### Sensor Dropout Training

```python
class SensorDropout(nn.Module):
    """
    Random sensor dropout during training for robustness.
    """

    def __init__(self, dropout_probs: Dict[str, float]):
        super().__init__()
        self.dropout_probs = dropout_probs

    def forward(
        self,
        sensors: Dict[str, torch.Tensor],
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not training:
            return sensors

        output = {}
        for name, tensor in sensors.items():
            if torch.rand(1).item() > self.dropout_probs.get(name, 0.0):
                output[name] = tensor
            else:
                output[name] = torch.zeros_like(tensor)

        return output
```

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Complete training pipeline
- [Training Datasets](training_datasets.md) - Dataset documentation
- [Deployment](deployment.md) - Deploy sensor-fused models
