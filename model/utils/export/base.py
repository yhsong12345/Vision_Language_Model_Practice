"""
Base Export Configuration and Exporter

Shared configuration and base class for all exporters.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn


@dataclass
class ExportConfig:
    """Configuration for model export."""

    # Output settings
    output_dir: str = "./exported_models"
    model_name: str = "vla_model"

    # Input specifications
    batch_size: int = 1
    image_size: int = 224
    input_channels: int = 3
    action_dim: int = 7

    # Additional input specs for multi-input models
    text_max_length: int = 77
    num_cameras: int = 1

    # ONNX settings
    onnx_opset_version: int = 17
    onnx_dynamic_axes: bool = True
    onnx_simplify: bool = True

    # Quantization settings
    quantization_dtype: str = "int8"  # int8, fp16, bf16
    calibration_samples: int = 100

    # OpenVINO settings
    openvino_precision: str = "FP16"  # FP32, FP16, INT8
    openvino_device: str = "CPU"  # CPU, GPU, MYRIAD, HDDL

    # Triton settings
    triton_max_batch_size: int = 8
    triton_preferred_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    triton_model_version: int = 1
    triton_instance_count: int = 1
    triton_kind: str = "KIND_GPU"  # KIND_CPU, KIND_GPU

    # Optimization settings
    optimize_for_inference: bool = True
    fuse_operations: bool = True

    def get_sample_input(self, device: str = "cpu") -> torch.Tensor:
        """Generate sample input tensor based on config."""
        return torch.randn(
            self.batch_size,
            self.input_channels,
            self.image_size,
            self.image_size,
            device=device,
        )

    def get_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape."""
        return (self.batch_size, self.input_channels, self.image_size, self.image_size)


class BaseExporter(ABC):
    """Base class for all model exporters."""

    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    @abstractmethod
    def export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        output_path: str = None,
        **kwargs,
    ) -> str:
        """
        Export model to target format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor for tracing
            output_path: Output file path
            **kwargs: Additional format-specific options

        Returns:
            Path to exported model
        """
        pass

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for export (eval mode, etc.)."""
        model = model.eval()
        return model

    def _get_sample_input(
        self,
        sample_input: torch.Tensor = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Get or create sample input tensor."""
        if sample_input is not None:
            return sample_input
        return self.config.get_sample_input(device)

    def _get_output_path(self, output_path: str, extension: str) -> str:
        """Get output path with correct extension."""
        if output_path is not None:
            return output_path
        return os.path.join(
            self.config.output_dir,
            f"{self.config.model_name}.{extension}"
        )

    def verify(
        self,
        original_model: nn.Module,
        exported_path: str,
        sample_input: torch.Tensor = None,
        tolerance: float = 1e-5,
    ) -> bool:
        """
        Verify exported model produces same outputs as original.

        Args:
            original_model: Original PyTorch model
            exported_path: Path to exported model
            sample_input: Input to test with
            tolerance: Maximum allowed difference

        Returns:
            True if outputs match within tolerance
        """
        # Default implementation - subclasses should override
        print(f"Verification not implemented for {self.__class__.__name__}")
        return True

    def benchmark(
        self,
        exported_path: str,
        sample_input: torch.Tensor = None,
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark exported model.

        Args:
            exported_path: Path to exported model
            sample_input: Input to benchmark with
            num_iterations: Number of iterations
            warmup: Warmup iterations

        Returns:
            Benchmark results
        """
        # Default implementation - subclasses should override
        print(f"Benchmarking not implemented for {self.__class__.__name__}")
        return {}


def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if a module is available."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        pkg = package_name or module_name
        print(f"Warning: {module_name} not installed. Install with: pip install {pkg}")
        return False
