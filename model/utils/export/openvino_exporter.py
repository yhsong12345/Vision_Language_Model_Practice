"""
OpenVINO Exporter

Export models to OpenVINO Intermediate Representation (IR) for
optimized inference on Intel hardware (CPUs, GPUs, VPUs, FPGAs).
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np

from .base import BaseExporter, ExportConfig, check_import


@dataclass
class OpenVINOConfig:
    """Configuration specific to OpenVINO export."""

    # Model precision
    precision: str = "FP16"  # FP32, FP16, INT8

    # Target device
    device: str = "CPU"  # CPU, GPU, MYRIAD, HDDL, AUTO, MULTI

    # Input shape
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)

    # INT8 calibration
    calibration_samples: int = 100
    calibration_subset_size: int = 300

    # Optimization
    compress_to_fp16: bool = True
    mean_values: Optional[List[float]] = None
    scale_values: Optional[List[float]] = None

    # Performance hints
    performance_hint: str = "LATENCY"  # LATENCY, THROUGHPUT, CUMULATIVE_THROUGHPUT

    # Threading
    num_streams: int = 1
    num_threads: int = 0  # 0 = auto

    # Caching
    cache_dir: Optional[str] = None


class OpenVINOExporter(BaseExporter):
    """
    Export models to OpenVINO IR format.

    OpenVINO (Open Visual Inference and Neural Network Optimization)
    provides optimized inference on Intel hardware:
    - Intel CPUs (with AVX-512, VNNI)
    - Intel integrated/discrete GPUs
    - Intel VPUs (Movidius)
    - Intel FPGAs

    Workflow:
    1. Export PyTorch model to ONNX
    2. Convert ONNX to OpenVINO IR (xml + bin)
    3. Optionally apply INT8 quantization
    """

    def __init__(
        self,
        config: ExportConfig = None,
        openvino_config: OpenVINOConfig = None,
    ):
        super().__init__(config)
        self.ov_config = openvino_config or OpenVINOConfig()
        self._ov_available = check_import("openvino", "openvino")
        self._nncf_available = check_import("nncf", "nncf")

    def export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        output_path: str = None,
        precision: str = None,
        **kwargs,
    ) -> str:
        """
        Export PyTorch model to OpenVINO IR format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input for tracing
            output_path: Output path (without extension)
            precision: Model precision (FP32, FP16, INT8)

        Returns:
            Path to exported OpenVINO model (.xml)
        """
        if not self._ov_available:
            raise ImportError("OpenVINO not installed. Install with: pip install openvino")

        precision = precision or self.ov_config.precision
        model = self._prepare_model(model)
        sample_input = self._get_sample_input(sample_input)

        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"{self.config.model_name}_openvino"
            )

        print(f"Exporting model to OpenVINO IR: {output_path}")
        print(f"  Precision: {precision}")
        print(f"  Input shape: {sample_input.shape}")

        # Method 1: Direct conversion from PyTorch (OpenVINO 2023+)
        try:
            return self._export_direct(model, sample_input, output_path, precision)
        except Exception as e:
            print(f"  Direct conversion failed: {e}")
            print("  Falling back to ONNX conversion...")

        # Method 2: Convert via ONNX
        return self._export_via_onnx(model, sample_input, output_path, precision)

    def _export_direct(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        precision: str,
    ) -> str:
        """Export directly from PyTorch using OpenVINO's convert_model."""
        import openvino as ov

        # Convert PyTorch model
        ov_model = ov.convert_model(model, example_input=sample_input)

        # Compress to FP16 if requested
        if precision == "FP16" and self.ov_config.compress_to_fp16:
            ov_model = self._compress_to_fp16(ov_model)

        # Save
        xml_path = f"{output_path}.xml"
        ov.save_model(ov_model, xml_path, compress_to_fp16=(precision == "FP16"))

        print(f"OpenVINO export successful: {xml_path}")
        return xml_path

    def _export_via_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        precision: str,
    ) -> str:
        """Export via ONNX intermediate format."""
        import openvino as ov

        # First export to ONNX
        onnx_path = f"{output_path}.onnx"
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            opset_version=self.config.onnx_opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        # Convert ONNX to OpenVINO
        ov_model = ov.convert_model(onnx_path)

        # Save
        xml_path = f"{output_path}.xml"
        ov.save_model(ov_model, xml_path, compress_to_fp16=(precision == "FP16"))

        # Clean up ONNX file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

        print(f"OpenVINO export successful: {xml_path}")
        return xml_path

    def _compress_to_fp16(self, ov_model):
        """Compress model weights to FP16."""
        import openvino as ov

        # Use OpenVINO's compression
        from openvino.runtime import serialize
        # Note: compression happens during save_model with compress_to_fp16=True
        return ov_model

    def quantize_int8(
        self,
        model_path: str,
        calibration_data: List[torch.Tensor] = None,
        output_path: str = None,
    ) -> str:
        """
        Apply INT8 quantization using NNCF (Neural Network Compression Framework).

        Args:
            model_path: Path to OpenVINO model (.xml)
            calibration_data: Calibration dataset
            output_path: Output path for quantized model

        Returns:
            Path to quantized model
        """
        if not self._nncf_available:
            raise ImportError("NNCF not installed. Install with: pip install nncf")

        import openvino as ov
        import nncf

        print("Applying INT8 quantization with NNCF...")

        if output_path is None:
            base = model_path.replace(".xml", "")
            output_path = f"{base}_int8.xml"

        # Load model
        core = ov.Core()
        ov_model = core.read_model(model_path)

        # Create calibration dataset
        if calibration_data is None:
            calibration_data = [
                self._get_sample_input() for _ in range(self.ov_config.calibration_samples)
            ]

        def transform_fn(data):
            return {ov_model.input().get_any_name(): data.numpy()}

        calibration_dataset = nncf.Dataset(calibration_data, transform_fn)

        # Quantize
        quantized_model = nncf.quantize(
            ov_model,
            calibration_dataset,
            subset_size=self.ov_config.calibration_subset_size,
        )

        # Save
        ov.save_model(quantized_model, output_path)

        print(f"INT8 quantization complete: {output_path}")
        return output_path

    def benchmark(
        self,
        model_path: str,
        sample_input: torch.Tensor = None,
        num_iterations: int = 100,
        warmup: int = 10,
        device: str = None,
    ) -> Dict[str, float]:
        """Benchmark OpenVINO model."""
        if not self._ov_available:
            print("Cannot benchmark: OpenVINO not installed")
            return {}

        import openvino as ov
        import time

        device = device or self.ov_config.device
        sample_input = self._get_sample_input(sample_input)

        print(f"Benchmarking OpenVINO model on {device}...")

        # Initialize OpenVINO
        core = ov.Core()

        # Configure device
        config = {}
        if self.ov_config.performance_hint:
            config["PERFORMANCE_HINT"] = self.ov_config.performance_hint
        if self.ov_config.num_streams > 0:
            config["NUM_STREAMS"] = str(self.ov_config.num_streams)
        if self.ov_config.cache_dir:
            config["CACHE_DIR"] = self.ov_config.cache_dir

        # Compile model
        compiled_model = core.compile_model(model_path, device, config)
        infer_request = compiled_model.create_infer_request()

        # Prepare input
        input_tensor = ov.Tensor(sample_input.numpy())

        # Warmup
        for _ in range(warmup):
            infer_request.infer({0: input_tensor})

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            infer_request.infer({0: input_tensor})
            times.append(time.perf_counter() - start)

        times_ms = np.array(times) * 1000

        results = {
            "device": device,
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "p50_ms": float(np.percentile(times_ms, 50)),
            "p95_ms": float(np.percentile(times_ms, 95)),
            "p99_ms": float(np.percentile(times_ms, 99)),
            "throughput_fps": float(1000 / np.mean(times_ms)),
        }

        print(f"\nOpenVINO Benchmark ({num_iterations} iterations, {device}):")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  P50:  {results['p50_ms']:.2f} ms")
        print(f"  P95:  {results['p95_ms']:.2f} ms")
        print(f"  FPS:  {results['throughput_fps']:.1f}")

        return results

    def verify(
        self,
        original_model: nn.Module,
        exported_path: str,
        sample_input: torch.Tensor = None,
        tolerance: float = 1e-3,
    ) -> bool:
        """Verify OpenVINO model produces similar outputs."""
        if not self._ov_available:
            print("Cannot verify: OpenVINO not installed")
            return False

        import openvino as ov

        sample_input = self._get_sample_input(sample_input)

        # PyTorch inference
        original_model.eval()
        with torch.no_grad():
            pytorch_output = original_model(sample_input).numpy()

        # OpenVINO inference
        core = ov.Core()
        compiled_model = core.compile_model(exported_path)
        ov_output = compiled_model([sample_input.numpy()])[0]

        # Compare
        max_diff = np.abs(pytorch_output - ov_output).max()
        passed = max_diff < tolerance

        print(f"  Verification: {'PASSED' if passed else 'FAILED'}")
        print(f"  Max difference: {max_diff:.2e}")

        return passed

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available OpenVINO devices."""
        if not self._ov_available:
            return {"error": "OpenVINO not installed"}

        import openvino as ov

        core = ov.Core()
        devices = core.available_devices

        info = {"available_devices": devices}

        for device in devices:
            try:
                device_info = {
                    "full_name": core.get_property(device, "FULL_DEVICE_NAME"),
                }
                info[device] = device_info
            except Exception:
                pass

        return info

    def create_benchmark_app_config(
        self,
        model_path: str,
        output_path: str = None,
    ) -> str:
        """
        Create configuration for OpenVINO benchmark_app tool.

        The benchmark_app is a command-line tool for benchmarking:
        benchmark_app -m model.xml -d CPU -hint latency
        """
        if output_path is None:
            output_path = model_path.replace(".xml", "_benchmark_config.json")

        config = {
            "model": model_path,
            "device": self.ov_config.device,
            "hint": self.ov_config.performance_hint.lower(),
            "nstreams": self.ov_config.num_streams,
            "nthreads": self.ov_config.num_threads,
            "shape": list(self.ov_config.input_shape),
        }

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Benchmark config saved: {output_path}")
        print(f"\nRun with: benchmark_app -m {model_path} -d {config['device']}")

        return output_path
