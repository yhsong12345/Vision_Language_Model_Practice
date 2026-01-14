"""
TorchScript Exporter

Export PyTorch models to TorchScript for production deployment.
"""

import os
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import numpy as np

from .base import BaseExporter, ExportConfig


class TorchScriptExporter(BaseExporter):
    """
    Export PyTorch models to TorchScript format.

    TorchScript enables:
    - Production deployment without Python
    - Optimization passes for faster inference
    - Cross-language support (C++, mobile)
    """

    def export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        output_path: str = None,
        method: str = "trace",
        strict: bool = True,
    ) -> str:
        """
        Export model to TorchScript.

        Args:
            model: PyTorch model to export
            sample_input: Sample input for tracing
            output_path: Output file path
            method: "trace" or "script"
            strict: Strict mode for scripting

        Returns:
            Path to exported TorchScript file
        """
        if method == "trace":
            return self.export_traced(model, sample_input, output_path)
        else:
            return self.export_scripted(model, output_path, strict)

    def export_traced(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        output_path: str = None,
    ) -> str:
        """
        Export using torch.jit.trace.

        Best for models without data-dependent control flow.
        Faster execution but less flexible.
        """
        model = self._prepare_model(model)
        sample_input = self._get_sample_input(sample_input)
        output_path = self._get_output_path(output_path, "pt").replace(".pt", "_traced.pt")

        print(f"Tracing model to TorchScript: {output_path}")
        print(f"  Input shape: {sample_input.shape}")

        # Trace
        with torch.no_grad():
            traced = torch.jit.trace(model, sample_input)

        # Optimize if requested
        if self.config.optimize_for_inference:
            traced = torch.jit.freeze(traced)
            traced = torch.jit.optimize_for_inference(traced)
            print("  Applied inference optimizations")

        # Save
        torch.jit.save(traced, output_path)
        print(f"TorchScript (traced) export successful: {output_path}")

        return output_path

    def export_scripted(
        self,
        model: nn.Module,
        output_path: str = None,
        strict: bool = True,
    ) -> str:
        """
        Export using torch.jit.script.

        Best for models with control flow (if/else, loops).
        More flexible but may require code modifications.
        """
        model = self._prepare_model(model)
        output_path = self._get_output_path(output_path, "pt").replace(".pt", "_scripted.pt")

        print(f"Scripting model to TorchScript: {output_path}")

        # Script
        scripted = torch.jit.script(model)

        # Optimize if requested
        if self.config.optimize_for_inference:
            scripted = torch.jit.freeze(scripted)
            scripted = torch.jit.optimize_for_inference(scripted)
            print("  Applied inference optimizations")

        # Save
        torch.jit.save(scripted, output_path)
        print(f"TorchScript (scripted) export successful: {output_path}")

        return output_path

    def export_mobile(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        output_path: str = None,
        optimize_for_mobile: bool = True,
    ) -> str:
        """
        Export model optimized for mobile deployment.

        Uses torch.utils.mobile_optimizer for mobile-specific optimizations.
        """
        model = self._prepare_model(model)
        sample_input = self._get_sample_input(sample_input)
        output_path = self._get_output_path(output_path, "ptl")

        print(f"Exporting model for mobile: {output_path}")

        # Trace first
        with torch.no_grad():
            traced = torch.jit.trace(model, sample_input)

        # Mobile optimization
        if optimize_for_mobile:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            traced = optimize_for_mobile(traced)
            print("  Applied mobile optimizations")

        # Save as mobile format
        traced._save_for_lite_interpreter(output_path)
        print(f"Mobile export successful: {output_path}")

        return output_path

    def verify(
        self,
        original_model: nn.Module,
        exported_path: str,
        sample_input: torch.Tensor = None,
        tolerance: float = 1e-5,
    ) -> bool:
        """Verify TorchScript model produces same outputs as original."""
        sample_input = self._get_sample_input(sample_input)

        # Load exported model
        loaded = torch.jit.load(exported_path)
        loaded.eval()

        # Original inference
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(sample_input)

        # TorchScript inference
        with torch.no_grad():
            ts_output = loaded(sample_input)

        # Compare
        max_diff = torch.abs(original_output - ts_output).max().item()
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
        device: str = "cpu",
    ) -> Dict[str, float]:
        """Benchmark TorchScript model."""
        import time

        sample_input = self._get_sample_input(sample_input)
        sample_input = sample_input.to(device)

        # Load model
        loaded = torch.jit.load(exported_path)
        loaded = loaded.to(device)
        loaded.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                loaded(sample_input)

        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                loaded(sample_input)
                if device == "cuda":
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
            "throughput_fps": float(1000 / np.mean(times_ms)),
        }

        print(f"\nTorchScript Benchmark ({num_iterations} iterations, {device}):")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  P50:  {results['p50_ms']:.2f} ms")
        print(f"  P95:  {results['p95_ms']:.2f} ms")
        print(f"  FPS:  {results['throughput_fps']:.1f}")

        return results

    def get_model_info(self, exported_path: str) -> Dict[str, Any]:
        """Get information about exported TorchScript model."""
        loaded = torch.jit.load(exported_path)

        info = {
            "path": exported_path,
            "file_size_mb": os.path.getsize(exported_path) / (1024 * 1024),
            "graph": str(loaded.graph),
            "code": loaded.code,
        }

        return info
