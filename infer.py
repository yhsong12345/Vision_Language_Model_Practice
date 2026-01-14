#!/usr/bin/env python3
"""
VLA Inference Pipeline

Standalone inference script for running trained VLA models.
Supports various input modes and deployment scenarios.

Usage:
    # Single image
    python infer.py --image robot_view.jpg --instruction "Pick up the red cube"

    # Video processing
    python infer.py --video task.mp4 --instruction "Complete the assembly"

    # Benchmark
    python infer.py --benchmark --model ./checkpoints/vla_model.pt

    # Interactive mode
    python infer.py --interactive --model ./checkpoints/vla_model.pt
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

# Framework imports
from model.utils import get_device, count_parameters


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Model
    model_path: str = None
    model_type: str = "vla"  # vla, driving, humanoid

    # Device
    device: str = "auto"
    precision: str = "fp32"  # fp32, fp16, bf16

    # Image
    image_size: int = 224
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)

    # Action
    action_dim: int = 7
    action_names: List[str] = field(default_factory=lambda: [
        "x", "y", "z", "roll", "pitch", "yaw", "gripper"
    ])

    # Performance
    warmup_runs: int = 5
    batch_size: int = 1
    use_torch_compile: bool = False

    # Output
    output_dir: str = "./inference_output"
    save_actions: bool = True
    verbose: bool = True


class VLAInferenceEngine:
    """
    High-performance VLA inference engine.

    Features:
    - Automatic model loading and optimization
    - Multiple input formats (image, video, stream)
    - Action smoothing and filtering
    - Performance monitoring
    - Logging and recording
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = get_device(config.device)

        # Load model
        self.model = self._load_model()

        # Setup precision
        self._setup_precision()

        # Compile model if requested
        if config.use_torch_compile and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Warmup
        self._warmup()

        # Metrics tracking
        self.inference_times = []
        self.action_history = []

        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)

        if config.verbose:
            self._print_config()

    def _load_model(self) -> nn.Module:
        """Load model from checkpoint or create default."""
        if self.config.model_path and os.path.exists(self.config.model_path):
            print(f"Loading model from {self.config.model_path}")
            checkpoint = torch.load(self.config.model_path, map_location=self.device)

            # Create appropriate model based on type
            model = self._create_model()

            # Load weights
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

            return model.to(self.device)
        else:
            print("Creating default demo model...")
            return self._create_model().to(self.device)

    def _create_model(self) -> nn.Module:
        """Create model architecture."""

        class SimpleVLA(nn.Module):
            """Simple VLA for demo purposes."""

            def __init__(self, action_dim: int = 7):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                )
                self.head = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, action_dim),
                    nn.Tanh(),
                )

            def forward(self, image: torch.Tensor) -> torch.Tensor:
                features = self.encoder(image)
                return self.head(features)

        return SimpleVLA(self.config.action_dim)

    def _setup_precision(self):
        """Setup model precision."""
        if self.config.precision == "fp16":
            self.model = self.model.half()
            self.dtype = torch.float16
        elif self.config.precision == "bf16":
            self.model = self.model.to(torch.bfloat16)
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.model.eval()

    def _warmup(self):
        """Warmup model for consistent performance."""
        if self.config.verbose:
            print(f"Warming up ({self.config.warmup_runs} runs)...")

        dummy = torch.randn(
            1, 3, self.config.image_size, self.config.image_size,
            device=self.device, dtype=self.dtype,
        )

        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                _ = self.model(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("VLA Inference Engine")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Precision: {self.config.precision}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Action dim: {self.config.action_dim}")
        print(f"Image size: {self.config.image_size}")
        print("=" * 60 + "\n")

    def preprocess(
        self,
        image: Union[np.ndarray, "Image.Image", str, torch.Tensor],
    ) -> torch.Tensor:
        """Preprocess image for model input."""
        # Load from path
        if isinstance(image, str):
            if Image is not None:
                image = np.array(Image.open(image).convert("RGB"))
            elif cv2 is not None:
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ImportError("Need PIL or OpenCV to load images")

        # Convert PIL to numpy
        if Image is not None and isinstance(image, Image.Image):
            image = np.array(image)

        # Already tensor
        if isinstance(image, torch.Tensor):
            return image.to(self.device, dtype=self.dtype)

        # Resize
        if cv2 is not None:
            image = cv2.resize(image, (self.config.image_size, self.config.image_size))

        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.config.normalize_mean)
        std = np.array(self.config.normalize_std)
        image = (image - mean) / std

        # To tensor
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(image).unsqueeze(0)
        return tensor.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, "Image.Image", str, torch.Tensor],
        instruction: str = None,
    ) -> Dict[str, Any]:
        """
        Run inference on single image.

        Returns:
            Dictionary with action and metadata
        """
        start_time = time.perf_counter()

        # Preprocess
        tensor = self.preprocess(image)

        # Inference
        action = self.model(tensor)
        action = action.squeeze(0).cpu().float().numpy()

        # Timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time

        self.inference_times.append(inference_time)
        self.action_history.append(action)

        return {
            "action": action,
            "action_dict": {
                name: float(action[i])
                for i, name in enumerate(self.config.action_names[:len(action)])
            },
            "inference_time_ms": inference_time * 1000,
            "timestamp": datetime.now().isoformat(),
        }

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[np.ndarray, "Image.Image", str]],
        instruction: str = None,
    ) -> Dict[str, Any]:
        """Run inference on batch of images."""
        start_time = time.perf_counter()

        # Preprocess all
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        # Inference
        actions = self.model(batch)
        actions = actions.cpu().float().numpy()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time

        return {
            "actions": actions,
            "batch_size": len(images),
            "inference_time_ms": inference_time * 1000,
            "per_image_ms": inference_time * 1000 / len(images),
        }

    def process_video(
        self,
        video_path: str,
        instruction: str = None,
        output_path: str = None,
    ) -> Dict[str, Any]:
        """Process video file and predict actions for each frame."""
        if cv2 is None:
            raise ImportError("OpenCV required for video processing")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(f"  {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

        # Output writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        actions = []
        times = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Predict
            result = self.predict(frame_rgb, instruction)
            actions.append(result["action"])
            times.append(result["inference_time_ms"])

            # Annotate and write
            if writer:
                annotated = self._annotate_frame(frame, result)
                writer.write(annotated)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames}")

        cap.release()
        if writer:
            writer.release()
            print(f"Saved to: {output_path}")

        return {
            "video_path": video_path,
            "num_frames": len(actions),
            "actions": np.array(actions),
            "mean_inference_ms": np.mean(times),
            "achievable_fps": 1000 / np.mean(times),
        }

    def _annotate_frame(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
    ) -> np.ndarray:
        """Add action annotations to frame."""
        annotated = frame.copy()

        # Action values
        y = 30
        for name, value in result["action_dict"].items():
            text = f"{name}: {value:.3f}"
            cv2.putText(annotated, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20

        # Timing
        time_text = f"{result['inference_time_ms']:.1f}ms"
        cv2.putText(annotated, time_text, (frame.shape[1] - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        print(f"\nBenchmarking ({num_iterations} iterations)...")

        dummy = torch.randn(
            1, 3, self.config.image_size, self.config.image_size,
            device=self.device, dtype=self.dtype,
        )

        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = self.model(dummy)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        times_ms = np.array(times) * 1000

        results = {
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "p50_ms": float(np.percentile(times_ms, 50)),
            "p95_ms": float(np.percentile(times_ms, 95)),
            "p99_ms": float(np.percentile(times_ms, 99)),
            "fps": float(1000 / np.mean(times_ms)),
        }

        print(f"Results:")
        print(f"  Mean:    {results['mean_ms']:.2f} ms")
        print(f"  P95:     {results['p95_ms']:.2f} ms")
        print(f"  P99:     {results['p99_ms']:.2f} ms")
        print(f"  FPS:     {results['fps']:.1f}")

        return results

    def save_session(self):
        """Save inference session results."""
        if not self.action_history:
            return

        session_data = {
            "config": asdict(self.config),
            "num_inferences": len(self.action_history),
            "actions": [a.tolist() for a in self.action_history],
            "inference_times_ms": [t * 1000 for t in self.inference_times],
            "mean_inference_ms": np.mean(self.inference_times) * 1000,
            "timestamp": datetime.now().isoformat(),
        }

        output_path = os.path.join(
            self.config.output_dir,
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_path, "w") as f:
            json.dump(session_data, f, indent=2)

        print(f"Saved session to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VLA Inference Pipeline")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--instruction", type=str, default="Execute task",
                        help="Language instruction")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--action-dim", type=int, default=7, help="Action dimension")

    args = parser.parse_args()

    # Create config
    config = InferenceConfig(
        model_path=args.model,
        device=args.device,
        precision=args.precision,
        action_dim=args.action_dim,
    )

    # Create engine
    engine = VLAInferenceEngine(config)

    if args.image:
        result = engine.predict(args.image, args.instruction)
        print(f"\nAction: {result['action']}")
        print(f"Time: {result['inference_time_ms']:.2f} ms")

    if args.video:
        output = args.output or args.video.replace(".mp4", "_annotated.mp4")
        result = engine.process_video(args.video, args.instruction, output)
        print(f"\nProcessed {result['num_frames']} frames")
        print(f"Achievable FPS: {result['achievable_fps']:.1f}")

    if args.benchmark:
        engine.benchmark()

    # Save session
    engine.save_session()


if __name__ == "__main__":
    main()
