"""
Quantization Exporter

Quantize PyTorch models for efficient inference.
"""

import os
from typing import Dict, List, Optional, Any, Callable

import torch
import torch.nn as nn
import numpy as np

from .base import BaseExporter, ExportConfig


class QuantizationExporter(BaseExporter):
    """
    Quantize models for efficient inference.

    Supports:
    - Dynamic quantization (no calibration data)
    - Static quantization (requires calibration)
    - Quantization-aware training (QAT)
    - FP16/BF16 half precision
    """

    def export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        output_path: str = None,
        method: str = "dynamic",
        calibration_data: List[torch.Tensor] = None,
    ) -> str:
        """
        Export quantized model.

        Args:
            model: PyTorch model
            sample_input: Sample input for tracing
            output_path: Output file path
            method: Quantization method (dynamic, static, fp16, bf16)
            calibration_data: Data for static quantization calibration

        Returns:
            Path to quantized model
        """
        if method == "dynamic":
            quantized = self.quantize_dynamic(model)
        elif method == "static":
            if calibration_data is None:
                calibration_data = [self._get_sample_input(sample_input)]
            quantized = self.quantize_static(model, calibration_data)
        elif method == "fp16":
            quantized = self.to_half_precision(model, dtype="fp16")
        elif method == "bf16":
            quantized = self.to_half_precision(model, dtype="bf16")
        else:
            raise ValueError(f"Unknown quantization method: {method}")

        # Save
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"{self.config.model_name}_{method}.pt"
            )

        torch.save({
            "model_state_dict": quantized.state_dict(),
            "quantization_method": method,
        }, output_path)

        print(f"Quantized model saved: {output_path}")
        return output_path

    def quantize_dynamic(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        layers: List[type] = None,
    ) -> nn.Module:
        """
        Apply dynamic quantization.

        Weights are quantized statically, activations dynamically.
        No calibration data needed.

        Best for: RNNs, Transformers, models with large linear layers.
        """
        print("Applying dynamic quantization...")

        if layers is None:
            layers = {nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.GRUCell}

        quantized = torch.quantization.quantize_dynamic(
            model.cpu(),
            layers,
            dtype=dtype,
        )

        # Report size reduction
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")

        return quantized

    def quantize_static(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        backend: str = "fbgemm",
    ) -> nn.Module:
        """
        Apply static quantization with calibration.

        Both weights and activations are quantized.
        More accurate but requires calibration data.

        Args:
            model: Model to quantize
            calibration_data: Representative input data
            backend: Quantization backend (fbgemm for x86, qnnpack for ARM)
        """
        print("Applying static quantization...")
        print(f"  Backend: {backend}")
        print(f"  Calibration samples: {len(calibration_data)}")

        # Clone model
        model = model.cpu().eval()

        # Set backend
        torch.backends.quantized.engine = backend

        # Configure quantization
        model.qconfig = torch.quantization.get_default_qconfig(backend)

        # Prepare for calibration
        model_prepared = torch.quantization.prepare(model, inplace=False)

        # Calibrate
        print("  Running calibration...")
        with torch.no_grad():
            for data in calibration_data:
                model_prepared(data.cpu())

        # Convert to quantized
        quantized = torch.quantization.convert(model_prepared, inplace=False)

        # Report
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")

        return quantized

    def prepare_qat(
        self,
        model: nn.Module,
        backend: str = "fbgemm",
    ) -> nn.Module:
        """
        Prepare model for quantization-aware training (QAT).

        QAT simulates quantization during training for better accuracy.
        Train the returned model, then convert with convert_qat().
        """
        print("Preparing model for quantization-aware training...")

        model = model.cpu()
        model.train()

        # Configure QAT
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)

        # Prepare
        model_prepared = torch.quantization.prepare_qat(model, inplace=False)

        print("  Model prepared for QAT training")
        print("  Train this model, then call convert_qat()")

        return model_prepared

    def convert_qat(self, model: nn.Module) -> nn.Module:
        """
        Convert QAT-trained model to quantized model.

        Call this after training a QAT-prepared model.
        """
        print("Converting QAT model to quantized...")

        model.eval()
        quantized = torch.quantization.convert(model, inplace=False)

        print("  QAT conversion complete")
        return quantized

    def to_half_precision(
        self,
        model: nn.Module,
        dtype: str = "fp16",
    ) -> nn.Module:
        """
        Convert model to half precision (FP16 or BF16).

        Args:
            model: Model to convert
            dtype: "fp16" or "bf16"
        """
        print(f"Converting to {dtype.upper()}...")

        if dtype == "fp16":
            model_half = model.half()
        elif dtype == "bf16":
            model_half = model.to(dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        # Report
        original_size = self._get_model_size(model)
        half_size = self._get_model_size(model_half)
        reduction = (1 - half_size / original_size) * 100

        print(f"  Original size: {original_size:.2f} MB")
        print(f"  {dtype.upper()} size: {half_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")

        return model_half

    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def benchmark_quantized(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        sample_input: torch.Tensor = None,
        num_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Compare original and quantized model performance."""
        import time

        sample_input = self._get_sample_input(sample_input).cpu()
        original_model = original_model.cpu().eval()
        quantized_model = quantized_model.cpu().eval()

        # Benchmark original
        with torch.no_grad():
            for _ in range(10):  # warmup
                original_model(sample_input)

            original_times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                original_model(sample_input)
                original_times.append(time.perf_counter() - start)

        # Benchmark quantized
        with torch.no_grad():
            for _ in range(10):  # warmup
                quantized_model(sample_input)

            quantized_times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                quantized_model(sample_input)
                quantized_times.append(time.perf_counter() - start)

        original_ms = np.mean(original_times) * 1000
        quantized_ms = np.mean(quantized_times) * 1000
        speedup = original_ms / quantized_ms

        results = {
            "original_ms": original_ms,
            "quantized_ms": quantized_ms,
            "speedup": speedup,
            "original_size_mb": self._get_model_size(original_model),
            "quantized_size_mb": self._get_model_size(quantized_model),
        }

        print(f"\nQuantization Comparison ({num_iterations} iterations):")
        print(f"  Original:  {original_ms:.2f} ms, {results['original_size_mb']:.2f} MB")
        print(f"  Quantized: {quantized_ms:.2f} ms, {results['quantized_size_mb']:.2f} MB")
        print(f"  Speedup:   {speedup:.2f}x")

        return results

    def verify(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        sample_input: torch.Tensor = None,
        tolerance: float = 0.1,
    ) -> Dict[str, float]:
        """
        Verify quantized model accuracy.

        Note: Quantization inherently loses precision, so tolerance is higher.
        """
        sample_input = self._get_sample_input(sample_input).cpu()

        original_model = original_model.cpu().eval()
        quantized_model = quantized_model.cpu().eval()

        with torch.no_grad():
            original_output = original_model(sample_input)
            quantized_output = quantized_model(sample_input)

        # Handle different output types
        if isinstance(original_output, tuple):
            original_output = original_output[0]
            quantized_output = quantized_output[0]

        # Compare
        abs_diff = torch.abs(original_output.float() - quantized_output.float())
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        rel_diff = (abs_diff / (torch.abs(original_output.float()) + 1e-8)).mean().item()

        passed = max_diff < tolerance

        results = {
            "passed": passed,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "rel_diff": rel_diff,
        }

        print(f"\nQuantization Verification:")
        print(f"  Status: {'PASSED' if passed else 'FAILED'}")
        print(f"  Max difference: {max_diff:.4f}")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  Relative difference: {rel_diff:.2%}")

        return results
