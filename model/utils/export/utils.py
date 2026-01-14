"""
Export Utility Functions

Convenience functions for exporting models to multiple formats.
"""

import os
import json
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from .base import ExportConfig
from .onnx_exporter import ONNXExporter
from .torchscript_exporter import TorchScriptExporter
from .quantization_exporter import QuantizationExporter
from .openvino_exporter import OpenVINOExporter, OpenVINOConfig
from .triton_exporter import TritonExporter, TritonConfig


def export_model(
    model: nn.Module,
    output_dir: str = "./exported_models",
    model_name: str = "vla_model",
    formats: List[str] = None,
    config: ExportConfig = None,
    sample_input: torch.Tensor = None,
) -> Dict[str, str]:
    """
    Export model to multiple formats.

    Args:
        model: PyTorch model to export
        output_dir: Output directory for exported models
        model_name: Base name for exported files
        formats: List of formats to export
            - "onnx": ONNX format
            - "torchscript": TorchScript (traced)
            - "torchscript_scripted": TorchScript (scripted)
            - "quantized": Dynamic INT8 quantization
            - "fp16": Half precision
            - "openvino": OpenVINO IR format
            - "triton": Triton model repository
        config: Export configuration
        sample_input: Sample input tensor for tracing

    Returns:
        Dictionary mapping format to exported file path
    """
    if formats is None:
        formats = ["torchscript"]

    if config is None:
        config = ExportConfig(output_dir=output_dir, model_name=model_name)
    else:
        config.output_dir = output_dir
        config.model_name = model_name

    os.makedirs(output_dir, exist_ok=True)

    exported = {}
    model = model.eval()

    print("=" * 60)
    print(f"Exporting model: {model_name}")
    print(f"Formats: {formats}")
    print("=" * 60)

    # ONNX
    if "onnx" in formats:
        try:
            exporter = ONNXExporter(config)
            exported["onnx"] = exporter.export(model, sample_input)
        except Exception as e:
            print(f"ONNX export failed: {e}")

    # TorchScript (traced)
    if "torchscript" in formats:
        try:
            exporter = TorchScriptExporter(config)
            exported["torchscript"] = exporter.export_traced(model, sample_input)
        except Exception as e:
            print(f"TorchScript (traced) export failed: {e}")

    # TorchScript (scripted)
    if "torchscript_scripted" in formats:
        try:
            exporter = TorchScriptExporter(config)
            exported["torchscript_scripted"] = exporter.export_scripted(model)
        except Exception as e:
            print(f"TorchScript (scripted) export failed: {e}")

    # Mobile
    if "mobile" in formats:
        try:
            exporter = TorchScriptExporter(config)
            exported["mobile"] = exporter.export_mobile(model, sample_input)
        except Exception as e:
            print(f"Mobile export failed: {e}")

    # Quantized (INT8)
    if "quantized" in formats:
        try:
            exporter = QuantizationExporter(config)
            quantized = exporter.quantize_dynamic(model)
            path = os.path.join(output_dir, f"{model_name}_quantized.pt")
            torch.save(quantized.state_dict(), path)
            exported["quantized"] = path
        except Exception as e:
            print(f"Quantization failed: {e}")

    # FP16
    if "fp16" in formats:
        try:
            exporter = QuantizationExporter(config)
            fp16_model = exporter.to_half_precision(model, dtype="fp16")
            path = os.path.join(output_dir, f"{model_name}_fp16.pt")
            torch.save(fp16_model.state_dict(), path)
            exported["fp16"] = path
        except Exception as e:
            print(f"FP16 conversion failed: {e}")

    # BF16
    if "bf16" in formats:
        try:
            exporter = QuantizationExporter(config)
            bf16_model = exporter.to_half_precision(model, dtype="bf16")
            path = os.path.join(output_dir, f"{model_name}_bf16.pt")
            torch.save(bf16_model.state_dict(), path)
            exported["bf16"] = path
        except Exception as e:
            print(f"BF16 conversion failed: {e}")

    # OpenVINO
    if "openvino" in formats:
        try:
            ov_config = OpenVINOConfig()
            exporter = OpenVINOExporter(config, ov_config)
            exported["openvino"] = exporter.export(model, sample_input)
        except Exception as e:
            print(f"OpenVINO export failed: {e}")

    # Triton
    if "triton" in formats:
        try:
            triton_config = TritonConfig(
                model_repository=os.path.join(output_dir, "triton_models"),
                model_name=model_name,
            )
            exporter = TritonExporter(config, triton_config)
            exported["triton"] = exporter.export(model, sample_input)
        except Exception as e:
            print(f"Triton export failed: {e}")

    # Save export info
    _save_export_info(config, exported)

    print("\n" + "=" * 60)
    print(f"Exported {len(exported)} format(s) to {output_dir}")
    print("=" * 60)

    return exported


def export_all_formats(
    model: nn.Module,
    output_dir: str = "./exported_models",
    model_name: str = "vla_model",
    config: ExportConfig = None,
    sample_input: torch.Tensor = None,
    skip_formats: List[str] = None,
) -> Dict[str, str]:
    """
    Export model to all supported formats.

    Args:
        model: PyTorch model
        output_dir: Output directory
        model_name: Model name
        config: Export configuration
        sample_input: Sample input tensor
        skip_formats: Formats to skip

    Returns:
        Dictionary mapping format to path
    """
    all_formats = [
        "onnx",
        "torchscript",
        "torchscript_scripted",
        "mobile",
        "quantized",
        "fp16",
        "openvino",
        "triton",
    ]

    if skip_formats:
        all_formats = [f for f in all_formats if f not in skip_formats]

    return export_model(
        model=model,
        output_dir=output_dir,
        model_name=model_name,
        formats=all_formats,
        config=config,
        sample_input=sample_input,
    )


def _save_export_info(config: ExportConfig, exported: Dict[str, str]):
    """Save export information to JSON file."""
    info_path = os.path.join(config.output_dir, "export_info.json")

    info = {
        "model_name": config.model_name,
        "exported_formats": list(exported.keys()),
        "paths": exported,
        "config": {
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "input_channels": config.input_channels,
            "action_dim": config.action_dim,
            "onnx_opset_version": config.onnx_opset_version,
        },
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nExport info saved: {info_path}")


def load_exported_model(
    model_path: str,
    format: str = None,
    device: str = "cpu",
) -> Any:
    """
    Load an exported model for inference.

    Args:
        model_path: Path to exported model
        format: Model format (auto-detected if None)
        device: Device to load model on

    Returns:
        Loaded model ready for inference
    """
    if format is None:
        format = _detect_format(model_path)

    print(f"Loading {format} model from {model_path}")

    if format == "torchscript":
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model

    elif format == "onnx":
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if device == "cuda":
                providers = ["CUDAExecutionProvider"] + providers

            session = ort.InferenceSession(model_path, providers=providers)
            return session

        except ImportError:
            raise ImportError("onnxruntime not installed")

    elif format == "openvino":
        try:
            import openvino as ov

            core = ov.Core()
            compiled = core.compile_model(model_path, device.upper())
            return compiled

        except ImportError:
            raise ImportError("openvino not installed")

    else:
        raise ValueError(f"Unknown format: {format}")


def _detect_format(model_path: str) -> str:
    """Detect model format from file extension."""
    ext = os.path.splitext(model_path)[1].lower()

    format_map = {
        ".onnx": "onnx",
        ".pt": "torchscript",
        ".pth": "torchscript",
        ".ptl": "mobile",
        ".xml": "openvino",
        ".plan": "tensorrt",
    }

    return format_map.get(ext, "unknown")


def benchmark_exported_models(
    exported_paths: Dict[str, str],
    sample_input: torch.Tensor = None,
    num_iterations: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark all exported models and compare performance.

    Args:
        exported_paths: Dictionary of format -> path
        sample_input: Sample input tensor
        num_iterations: Number of benchmark iterations

    Returns:
        Dictionary of format -> benchmark results
    """
    import time
    import numpy as np

    if sample_input is None:
        sample_input = torch.randn(1, 3, 224, 224)

    results = {}

    for format_name, path in exported_paths.items():
        if not os.path.exists(path):
            continue

        print(f"\nBenchmarking {format_name}...")

        try:
            model = load_exported_model(path, format_name)
            input_data = sample_input.numpy()

            # Warmup
            for _ in range(10):
                if format_name == "torchscript":
                    with torch.no_grad():
                        model(sample_input)
                elif format_name == "onnx":
                    input_name = model.get_inputs()[0].name
                    model.run(None, {input_name: input_data})
                elif format_name == "openvino":
                    model([input_data])

            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()

                if format_name == "torchscript":
                    with torch.no_grad():
                        model(sample_input)
                elif format_name == "onnx":
                    input_name = model.get_inputs()[0].name
                    model.run(None, {input_name: input_data})
                elif format_name == "openvino":
                    model([input_data])

                times.append(time.perf_counter() - start)

            times_ms = np.array(times) * 1000

            results[format_name] = {
                "mean_ms": float(np.mean(times_ms)),
                "std_ms": float(np.std(times_ms)),
                "p50_ms": float(np.percentile(times_ms, 50)),
                "p95_ms": float(np.percentile(times_ms, 95)),
                "throughput_fps": float(1000 / np.mean(times_ms)),
            }

        except Exception as e:
            print(f"  Failed: {e}")
            results[format_name] = {"error": str(e)}

    # Print comparison
    print("\n" + "=" * 70)
    print("Benchmark Comparison:")
    print("=" * 70)
    print(f"{'Format':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'FPS':<10}")
    print("-" * 70)

    for fmt, r in results.items():
        if "error" in r:
            print(f"{fmt:<20} {'ERROR':<12}")
        else:
            print(f"{fmt:<20} {r['mean_ms']:<12.2f} {r['p95_ms']:<12.2f} {r['throughput_fps']:<10.1f}")

    return results


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export VLA model")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="./exported", help="Output directory")
    parser.add_argument("--name", type=str, default="vla_model", help="Model name")
    parser.add_argument(
        "--formats",
        type=str,
        default="torchscript,onnx",
        help="Comma-separated formats",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")

    args = parser.parse_args()

    # Create demo model if no checkpoint provided
    if args.model is None:
        print("No model provided, creating demo model...")

        class DemoModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.bn = nn.BatchNorm2d(64)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 7)

            def forward(self, x):
                x = torch.relu(self.bn(self.conv(x)))
                x = self.pool(x).flatten(1)
                return self.fc(x)

        model = DemoModel()
    else:
        # Load model from checkpoint
        checkpoint = torch.load(args.model, map_location="cpu")
        # Would need to instantiate correct model architecture
        raise NotImplementedError("Model loading from checkpoint requires model class")

    # Export
    formats = args.formats.split(",")
    exported = export_model(
        model=model,
        output_dir=args.output,
        model_name=args.name,
        formats=formats,
    )

    # Benchmark if requested
    if args.benchmark and exported:
        benchmark_exported_models(exported)
