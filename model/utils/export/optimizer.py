"""
Model Optimizer

Common optimization utilities for model deployment:
- Module fusion
- Benchmarking
- Profiling
- Memory analysis
"""

import os
import time
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import numpy as np

from .base import ExportConfig


class ModelOptimizer:
    """
    Optimize models for efficient inference.

    Provides:
    - Module fusion (Conv+BN+ReLU, etc.)
    - Inference optimization
    - Benchmarking across formats
    - Memory profiling
    """

    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()

    def optimize_for_inference(
        self,
        model: nn.Module,
        fuse_modules: bool = True,
        use_channels_last: bool = True,
    ) -> nn.Module:
        """
        Apply common inference optimizations to model.

        Args:
            model: PyTorch model to optimize
            fuse_modules: Fuse Conv+BN+ReLU patterns
            use_channels_last: Use channels-last memory format (faster on modern GPUs)
        """
        model = model.eval()

        # Fuse modules
        if fuse_modules:
            model = self._fuse_modules(model)

        # Channels-last memory format
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)

        return model

    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse common module patterns for faster inference."""
        # Try to use torch.quantization.fuse_modules if available
        try:
            # This requires knowing the module names
            # For generic models, we do a simpler optimization
            model = torch.jit.freeze(torch.jit.script(model.eval()))
            model = torch.jit.optimize_for_inference(model)
            print("  Applied JIT fusion optimizations")
        except Exception:
            # For models that can't be scripted, skip fusion
            pass

        return model

    def benchmark(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        num_iterations: int = 100,
        warmup: int = 10,
        device: str = None,
    ) -> Dict[str, float]:
        """
        Benchmark model inference performance.

        Args:
            model: Model to benchmark
            sample_input: Sample input tensor
            num_iterations: Number of iterations
            warmup: Warmup iterations
            device: Device to benchmark on
        """
        if sample_input is None:
            sample_input = self.config.get_sample_input()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device).eval()
        sample_input = sample_input.to(device)

        print(f"Benchmarking on {device}...")

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(sample_input)

        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(sample_input)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        times_ms = np.array(times) * 1000

        results = {
            "device": device,
            "batch_size": sample_input.shape[0],
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "p50_ms": float(np.percentile(times_ms, 50)),
            "p95_ms": float(np.percentile(times_ms, 95)),
            "p99_ms": float(np.percentile(times_ms, 99)),
            "throughput_fps": float(1000 / np.mean(times_ms)),
        }

        print(f"\nBenchmark Results ({num_iterations} iterations):")
        print(f"  Device: {device}")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  Std:  {results['std_ms']:.2f} ms")
        print(f"  P50:  {results['p50_ms']:.2f} ms")
        print(f"  P95:  {results['p95_ms']:.2f} ms")
        print(f"  P99:  {results['p99_ms']:.2f} ms")
        print(f"  FPS:  {results['throughput_fps']:.1f}")

        return results

    def benchmark_batch_sizes(
        self,
        model: nn.Module,
        batch_sizes: List[int] = None,
        num_iterations: int = 50,
        device: str = None,
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark model across different batch sizes."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        results = {}

        print(f"Benchmarking batch sizes on {device}...")

        for bs in batch_sizes:
            try:
                sample_input = torch.randn(
                    bs,
                    self.config.input_channels,
                    self.config.image_size,
                    self.config.image_size,
                    device=device,
                )

                result = self.benchmark(
                    model,
                    sample_input,
                    num_iterations=num_iterations,
                    warmup=5,
                    device=device,
                )
                results[bs] = result

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Batch size {bs}: OOM")
                    break
                raise

        # Summary
        print("\n" + "=" * 60)
        print("Batch Size Comparison:")
        print("=" * 60)
        print(f"{'Batch':<8} {'Latency (ms)':<15} {'Throughput (FPS)':<18}")
        print("-" * 60)
        for bs, r in results.items():
            print(f"{bs:<8} {r['mean_ms']:<15.2f} {r['throughput_fps']:<18.1f}")

        return results

    def profile_memory(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """Profile model memory usage."""
        if sample_input is None:
            sample_input = self.config.get_sample_input()

        # Parameter memory
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers())

        results = {
            "parameters_mb": param_mem / (1024 * 1024),
            "buffers_mb": buffer_mem / (1024 * 1024),
            "total_mb": (param_mem + buffer_mem) / (1024 * 1024),
        }

        # GPU memory if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            sample_input = sample_input.to(device)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # Forward pass
            with torch.no_grad():
                _ = model(sample_input)

            results["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            results["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            results["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

        print("\nMemory Profile:")
        print(f"  Parameters: {results['parameters_mb']:.2f} MB")
        print(f"  Buffers: {results['buffers_mb']:.2f} MB")
        print(f"  Total model: {results['total_mb']:.2f} MB")
        if "gpu_peak_mb" in results:
            print(f"  GPU peak: {results['gpu_peak_mb']:.2f} MB")

        return results

    def profile_flops(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """Profile model FLOPs (requires thop or fvcore)."""
        if sample_input is None:
            sample_input = self.config.get_sample_input()

        results = {}

        # Try thop
        try:
            from thop import profile as thop_profile

            flops, params = thop_profile(model, inputs=(sample_input,), verbose=False)
            results["flops"] = flops
            results["gflops"] = flops / 1e9
            results["params"] = params
            results["mparams"] = params / 1e6

            print("\nFLOPs Profile (thop):")
            print(f"  FLOPs: {results['gflops']:.2f} G")
            print(f"  Params: {results['mparams']:.2f} M")

            return results

        except ImportError:
            pass

        # Try fvcore
        try:
            from fvcore.nn import FlopCountAnalysis

            flops = FlopCountAnalysis(model, sample_input)
            results["flops"] = flops.total()
            results["gflops"] = flops.total() / 1e9

            print("\nFLOPs Profile (fvcore):")
            print(f"  FLOPs: {results['gflops']:.2f} G")

            return results

        except ImportError:
            pass

        print("FLOPs profiling requires 'thop' or 'fvcore' package")
        return results

    def compare_formats(
        self,
        model: nn.Module,
        formats: List[str] = None,
        sample_input: torch.Tensor = None,
        num_iterations: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model performance across different export formats.

        Args:
            model: PyTorch model
            formats: List of formats to compare
            sample_input: Sample input tensor
            num_iterations: Benchmark iterations
        """
        if formats is None:
            formats = ["pytorch", "torchscript"]

        if sample_input is None:
            sample_input = self.config.get_sample_input()

        results = {}

        # PyTorch baseline
        if "pytorch" in formats:
            results["pytorch"] = self.benchmark(
                model, sample_input, num_iterations=num_iterations
            )

        # TorchScript
        if "torchscript" in formats:
            try:
                traced = torch.jit.trace(model.eval(), sample_input)
                traced = torch.jit.freeze(traced)
                results["torchscript"] = self.benchmark(
                    traced, sample_input, num_iterations=num_iterations
                )
            except Exception as e:
                print(f"TorchScript benchmark failed: {e}")

        # Print comparison
        print("\n" + "=" * 60)
        print("Format Comparison:")
        print("=" * 60)
        print(f"{'Format':<15} {'Mean (ms)':<12} {'P95 (ms)':<12} {'FPS':<10}")
        print("-" * 60)
        for fmt, r in results.items():
            print(f"{fmt:<15} {r['mean_ms']:<12.2f} {r['p95_ms']:<12.2f} {r['throughput_fps']:<10.1f}")

        return results

    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive model information."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

        # Layer counts
        layer_counts = {}
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            layer_counts[class_name] = layer_counts.get(class_name, 0) + 1

        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "model_size_mb": (param_size + buffer_size) / (1024 * 1024),
            "layer_counts": layer_counts,
        }

        print("\nModel Information:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Model size: {info['model_size_mb']:.2f} MB")
        print(f"  Layers: {sum(layer_counts.values())}")

        return info
