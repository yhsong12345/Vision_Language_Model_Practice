#!/usr/bin/env python3
"""
Inference Demo - Running VLA Models for Deployment

This example demonstrates how to run inference with trained VLA models,
including batch inference, video processing, and real-time control.

Requirements:
    pip install torch transformers pillow opencv-python

Usage:
    python examples/inference_demo.py --demo
    python examples/inference_demo.py --image path/to/image.jpg --instruction "Pick up the cube"
    python examples/inference_demo.py --video path/to/video.mp4 --instruction "Navigate to goal"
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

try:
    from PIL import Image
except ImportError:
    print("Please install pillow: pip install pillow")
    Image = None

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Video processing will be disabled.")
    print("Install with: pip install opencv-python")
    cv2 = None

# Framework imports
from model.utils import get_device, count_parameters


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    image_size: int = 224
    action_dim: int = 7  # Default for manipulation (6 DoF + gripper)
    device: str = "auto"
    precision: str = "fp32"  # fp32, fp16, bf16
    warmup_runs: int = 3
    batch_size: int = 1


class VLAInference:
    """
    High-level inference interface for VLA models.

    Supports:
    - Single image inference
    - Batch inference
    - Video processing
    - Real-time control loop
    """

    def __init__(
        self,
        model: nn.Module = None,
        model_path: str = None,
        config: InferenceConfig = None,
    ):
        self.config = config or InferenceConfig()
        self.device = get_device(self.config.device)

        # Load or use provided model
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            # Create demo model
            self.model = self._create_demo_model()

        self.model.eval()

        # Set precision
        if self.config.precision == "fp16":
            self.model = self.model.half()
        elif self.config.precision == "bf16" and torch.cuda.is_available():
            self.model = self.model.to(torch.bfloat16)

        # Warm up
        self._warmup()

        print(f"VLAInference initialized:")
        print(f"  Device: {self.device}")
        print(f"  Precision: {self.config.precision}")
        print(f"  Parameters: {count_parameters(self.model):,}")

    def _create_demo_model(self) -> nn.Module:
        """Create a simple demo model."""

        class DemoVLA(nn.Module):
            def __init__(self, action_dim=7):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(7),
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 256),
                    nn.ReLU(),
                )
                self.action_head = nn.Linear(256, action_dim)

            def forward(self, image):
                features = self.encoder(image)
                return self.action_head(features)

        return DemoVLA(self.config.action_dim).to(self.device)

    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint."""
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Determine model type and load
        if "model_state_dict" in checkpoint:
            model = self._create_demo_model()
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = self._create_demo_model()
            model.load_state_dict(checkpoint)

        return model

    def _warmup(self):
        """Warm up the model for consistent timing."""
        print(f"Warming up model ({self.config.warmup_runs} runs)...")
        dummy_input = torch.randn(
            1, 3, self.config.image_size, self.config.image_size,
            device=self.device,
        )

        if self.config.precision == "fp16":
            dummy_input = dummy_input.half()
        elif self.config.precision == "bf16":
            dummy_input = dummy_input.to(torch.bfloat16)

        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                _ = self.model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def preprocess_image(
        self,
        image: Union[np.ndarray, "Image.Image", str],
    ) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: numpy array, PIL Image, or path to image file

        Returns:
            Tensor of shape (1, 3, H, W)
        """
        # Load from path if string
        if isinstance(image, str):
            if Image is None:
                raise ImportError("PIL is required for loading images from paths")
            image = Image.open(image).convert("RGB")

        # Convert PIL to numpy
        if Image is not None and isinstance(image, Image.Image):
            image = np.array(image)

        # Resize
        if cv2 is not None:
            image = cv2.resize(
                image,
                (self.config.image_size, self.config.image_size),
            )
        else:
            # Simple resize without OpenCV (lower quality)
            pass

        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # To tensor
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

        if self.config.precision == "fp16":
            tensor = tensor.half()
        elif self.config.precision == "bf16":
            tensor = tensor.to(torch.bfloat16)

        return tensor

    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, "Image.Image", str, torch.Tensor],
        instruction: str = None,
    ) -> np.ndarray:
        """
        Predict action for a single image.

        Args:
            image: Input image (numpy, PIL, path, or tensor)
            instruction: Language instruction (optional)

        Returns:
            Action as numpy array of shape (action_dim,)
        """
        # Preprocess if needed
        if not isinstance(image, torch.Tensor):
            image = self.preprocess_image(image)

        # Forward pass
        action = self.model(image)

        return action.squeeze(0).cpu().float().numpy()

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[np.ndarray, "Image.Image", str]],
        instruction: str = None,
    ) -> np.ndarray:
        """
        Predict actions for a batch of images.

        Args:
            images: List of input images
            instruction: Language instruction (shared for all)

        Returns:
            Actions as numpy array of shape (batch_size, action_dim)
        """
        # Preprocess all images
        tensors = [self.preprocess_image(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        # Forward pass
        actions = self.model(batch)

        return actions.cpu().float().numpy()

    @torch.no_grad()
    def process_video(
        self,
        video_path: str,
        instruction: str = None,
        output_path: str = None,
        fps: int = None,
    ) -> List[np.ndarray]:
        """
        Process a video file and predict actions for each frame.

        Args:
            video_path: Path to input video
            instruction: Language instruction
            output_path: Path to save annotated video (optional)
            fps: Output FPS (default: same as input)

        Returns:
            List of actions for each frame
        """
        if cv2 is None:
            raise ImportError("OpenCV is required for video processing")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {input_fps}")
        print(f"  Total frames: {total_frames}")

        # Setup output video writer if needed
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps or input_fps,
                (width, height),
            )

        actions = []
        frame_times = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predict
            start_time = time.perf_counter()
            action = self.predict(frame, instruction)
            inference_time = time.perf_counter() - start_time

            actions.append(action)
            frame_times.append(inference_time)

            # Annotate frame if saving
            if writer is not None:
                annotated = self._annotate_frame(frame, action, inference_time)
                writer.write(annotated)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved annotated video to: {output_path}")

        # Print statistics
        avg_time = np.mean(frame_times)
        print(f"\nInference Statistics:")
        print(f"  Average time per frame: {avg_time * 1000:.2f} ms")
        print(f"  Achievable FPS: {1 / avg_time:.1f}")

        return actions

    def _annotate_frame(
        self,
        frame: np.ndarray,
        action: np.ndarray,
        inference_time: float,
    ) -> np.ndarray:
        """Annotate frame with action information."""
        annotated = frame.copy()

        # Add action text
        y_offset = 30
        for i, a in enumerate(action):
            text = f"a[{i}]: {a:.3f}"
            cv2.putText(
                annotated, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            y_offset += 25

        # Add inference time
        fps_text = f"Inference: {inference_time * 1000:.1f}ms"
        cv2.putText(
            annotated, fps_text, (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        return annotated

    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            num_iterations: Number of inference iterations

        Returns:
            Dictionary with timing statistics
        """
        print(f"\nBenchmarking ({num_iterations} iterations)...")

        dummy_input = torch.randn(
            1, 3, self.config.image_size, self.config.image_size,
            device=self.device,
        )

        if self.config.precision == "fp16":
            dummy_input = dummy_input.half()
        elif self.config.precision == "bf16":
            dummy_input = dummy_input.to(torch.bfloat16)

        times = []

        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = self.model(dummy_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                times.append(time.perf_counter() - start)

        times = np.array(times) * 1000  # Convert to ms

        results = {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "throughput_fps": 1000 / np.mean(times),
        }

        print(f"Results:")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  Std: {results['std_ms']:.2f} ms")
        print(f"  P50: {results['p50_ms']:.2f} ms")
        print(f"  P95: {results['p95_ms']:.2f} ms")
        print(f"  P99: {results['p99_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_fps']:.1f} FPS")

        return results


def run_demo():
    """Run demonstration of inference capabilities."""
    print("\n" + "=" * 60)
    print("VLA Inference Demo")
    print("=" * 60)

    # Create inference handler
    config = InferenceConfig(action_dim=7)
    infer = VLAInference(config=config)

    # Single image inference
    print("\n1. Single Image Inference")
    print("-" * 40)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    action = infer.predict(dummy_image, "Pick up the red cube")
    print(f"   Input: RGB image (640x480)")
    print(f"   Action: {action}")

    # Batch inference
    print("\n2. Batch Inference")
    print("-" * 40)
    batch_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
    actions = infer.predict_batch(batch_images, "Stack the blocks")
    print(f"   Batch size: {len(batch_images)}")
    print(f"   Actions shape: {actions.shape}")

    # Benchmark
    print("\n3. Benchmark")
    print("-" * 40)
    infer.benchmark(num_iterations=50)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VLA Inference Demo")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--instruction", type=str, default="Execute the task",
                        help="Language instruction")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, help="Output path for annotated video")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"], help="Precision")

    args = parser.parse_args()

    # Default to demo
    if not (args.demo or args.image or args.video or args.benchmark):
        args.demo = True

    if args.demo:
        run_demo()
        return

    # Create inference handler
    config = InferenceConfig(device=args.device, precision=args.precision)
    infer = VLAInference(model_path=args.checkpoint, config=config)

    if args.image:
        print(f"\nProcessing image: {args.image}")
        action = infer.predict(args.image, args.instruction)
        print(f"Instruction: {args.instruction}")
        print(f"Predicted action: {action}")

    if args.video:
        print(f"\nProcessing video: {args.video}")
        actions = infer.process_video(
            args.video,
            args.instruction,
            args.output,
        )
        print(f"Processed {len(actions)} frames")

    if args.benchmark:
        infer.benchmark()


if __name__ == "__main__":
    main()
