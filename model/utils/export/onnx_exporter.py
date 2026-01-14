"""
ONNX Exporter

Export PyTorch models to ONNX format for cross-platform deployment.
"""

import os
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn as nn
import numpy as np

from .base import BaseExporter, ExportConfig, check_import


class ONNXExporter(BaseExporter):
    """
    Export PyTorch models to ONNX format.

    ONNX (Open Neural Network Exchange) enables:
    - Cross-platform inference (ONNX Runtime, TensorRT, OpenVINO)
    - Hardware acceleration on various devices
    - Model optimization through ONNX tools
    """

    def __init__(self, config: ExportConfig = None):
        super().__init__(config)
        self._onnx_available = check_import("onnx", "onnx")
        self._ort_available = check_import("onnxruntime", "onnxruntime")

    def export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        output_path: str = None,
        input_names: List[str] = None,
        output_names: List[str] = None,
        dynamic_axes: Dict[str, Dict[int, str]] = None,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor for tracing
            output_path: Output file path
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes configuration

        Returns:
            Path to exported ONNX file
        """
        model = self._prepare_model(model)
        sample_input = self._get_sample_input(sample_input)
        output_path = self._get_output_path(output_path, "onnx")

        # Default names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # Dynamic axes for variable batch size
        if dynamic_axes is None and self.config.onnx_dynamic_axes:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        print(f"Exporting model to ONNX: {output_path}")
        print(f"  Input shape: {sample_input.shape}")
        print(f"  Opset version: {self.config.onnx_opset_version}")

        # Export
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            opset_version=self.config.onnx_opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
        )

        # Simplify if requested
        if self.config.onnx_simplify:
            output_path = self._simplify(output_path)

        # Verify export
        if self._onnx_available:
            self._verify_model(output_path)

        print(f"ONNX export successful: {output_path}")
        return output_path

    def _simplify(self, onnx_path: str) -> str:
        """Simplify ONNX model using onnx-simplifier."""
        try:
            import onnx
            from onnxsim import simplify

            print("  Simplifying ONNX model...")
            model = onnx.load(onnx_path)
            simplified, check = simplify(model)

            if check:
                onnx.save(simplified, onnx_path)
                print("  Simplification successful")
            else:
                print("  Simplification validation failed, keeping original")

        except ImportError:
            print("  Skipping simplification (onnx-simplifier not installed)")

        return onnx_path

    def _verify_model(self, onnx_path: str):
        """Verify ONNX model is valid."""
        import onnx

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("  ONNX model verification passed")

    def verify(
        self,
        original_model: nn.Module,
        exported_path: str,
        sample_input: torch.Tensor = None,
        tolerance: float = 1e-5,
    ) -> bool:
        """Verify ONNX model produces same outputs as PyTorch model."""
        if not self._ort_available:
            print("Cannot verify: onnxruntime not installed")
            return False

        import onnxruntime as ort

        sample_input = self._get_sample_input(sample_input)

        # PyTorch inference
        original_model.eval()
        with torch.no_grad():
            pytorch_output = original_model(sample_input).numpy()

        # ONNX Runtime inference
        session = ort.InferenceSession(exported_path)
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: sample_input.numpy()})[0]

        # Compare
        max_diff = np.abs(pytorch_output - onnx_output).max()
        passed = max_diff < tolerance

        print(f"  Verification: {'PASSED' if passed else 'FAILED'}")
        print(f"  Max difference: {max_diff:.2e}")

        return passed

    def benchmark(
        self,
        exported_path: str,
        sample_input: torch.Tensor = None,
        num_iterations: int = 100,
        warmup: int = 10,
        providers: List[str] = None,
    ) -> Dict[str, float]:
        """Benchmark ONNX model with ONNX Runtime."""
        if not self._ort_available:
            print("Cannot benchmark: onnxruntime not installed")
            return {}

        import onnxruntime as ort
        import time

        sample_input = self._get_sample_input(sample_input)

        # Session options
        if providers is None:
            providers = ort.get_available_providers()

        print(f"Benchmarking ONNX model with providers: {providers}")

        session = ort.InferenceSession(exported_path, providers=providers)
        input_name = session.get_inputs()[0].name
        input_data = {input_name: sample_input.numpy()}

        # Warmup
        for _ in range(warmup):
            session.run(None, input_data)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            session.run(None, input_data)
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
            "throughput_fps": float(1000 / np.mean(times_ms)),
        }

        print(f"\nONNX Runtime Benchmark ({num_iterations} iterations):")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  P50:  {results['p50_ms']:.2f} ms")
        print(f"  P95:  {results['p95_ms']:.2f} ms")
        print(f"  FPS:  {results['throughput_fps']:.1f}")

        return results

    def export_with_external_data(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        output_path: str = None,
        size_threshold: int = 1024,
    ) -> str:
        """
        Export large model with external data files.

        Useful for models > 2GB where weights are stored separately.
        """
        model = self._prepare_model(model)
        sample_input = self._get_sample_input(sample_input)
        output_path = self._get_output_path(output_path, "onnx")

        print(f"Exporting large model with external data: {output_path}")

        # First export normally
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            opset_version=self.config.onnx_opset_version,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )

        # Convert to external data format
        if self._onnx_available:
            import onnx
            from onnx.external_data_helper import convert_model_to_external_data

            onnx_model = onnx.load(output_path)
            convert_model_to_external_data(
                onnx_model,
                all_tensors_to_one_file=True,
                location=f"{self.config.model_name}_weights.bin",
                size_threshold=size_threshold,
            )
            onnx.save(onnx_model, output_path)
            print(f"  External data saved to: {self.config.model_name}_weights.bin")

        return output_path

    def optimize_for_inference(self, onnx_path: str, output_path: str = None) -> str:
        """Apply ONNX Runtime optimizations."""
        if not self._ort_available:
            print("Cannot optimize: onnxruntime not installed")
            return onnx_path

        import onnxruntime as ort

        if output_path is None:
            base, ext = os.path.splitext(onnx_path)
            output_path = f"{base}_optimized{ext}"

        print(f"Optimizing ONNX model: {output_path}")

        sess_options = ort.SessionOptions()
        sess_options.optimized_model_filepath = output_path
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # This creates the optimized model
        ort.InferenceSession(onnx_path, sess_options)

        print(f"Optimized model saved: {output_path}")
        return output_path
