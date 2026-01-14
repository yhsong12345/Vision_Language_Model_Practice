"""
NVIDIA Triton Inference Server Exporter

Export models for deployment on NVIDIA Triton Inference Server.
Supports multiple backends: TensorRT, ONNX Runtime, PyTorch, TensorFlow.
"""

import os
import json
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from .base import BaseExporter, ExportConfig, check_import


@dataclass
class TritonConfig:
    """Configuration for Triton Inference Server deployment."""

    # Model repository settings
    model_repository: str = "./triton_models"
    model_name: str = "vla_model"
    model_version: int = 1

    # Backend selection
    backend: str = "onnxruntime"  # onnxruntime, tensorrt, pytorch, openvino

    # Batching configuration
    max_batch_size: int = 8
    preferred_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    max_queue_delay_microseconds: int = 100

    # Instance configuration
    instance_count: int = 1
    instance_kind: str = "KIND_GPU"  # KIND_CPU, KIND_GPU, KIND_MODEL
    instance_gpus: List[int] = field(default_factory=lambda: [0])

    # Input/Output configuration
    input_name: str = "input"
    input_dims: List[int] = field(default_factory=lambda: [3, 224, 224])
    input_dtype: str = "TYPE_FP32"

    output_name: str = "output"
    output_dims: List[int] = field(default_factory=lambda: [7])
    output_dtype: str = "TYPE_FP32"

    # Performance settings
    response_cache_enable: bool = False
    sequence_batching_enable: bool = False

    # TensorRT specific
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    tensorrt_max_workspace_size: int = 1 << 30  # 1GB


class TritonExporter(BaseExporter):
    """
    Export models for NVIDIA Triton Inference Server.

    Triton Inference Server provides:
    - Multi-framework support (TensorRT, ONNX, PyTorch, TensorFlow)
    - Dynamic batching for higher throughput
    - Model ensembles and pipelines
    - GPU/CPU inference
    - Metrics and monitoring
    - gRPC and HTTP endpoints

    Model Repository Structure:
    ```
    model_repository/
    └── model_name/
        ├── config.pbtxt
        └── 1/
            └── model.onnx (or model.pt, model.plan, etc.)
    ```
    """

    def __init__(
        self,
        config: ExportConfig = None,
        triton_config: TritonConfig = None,
    ):
        super().__init__(config)
        self.triton_config = triton_config or TritonConfig()
        self._tritonclient_available = check_import(
            "tritonclient", "tritonclient[all]"
        )

    def export(
        self,
        model: nn.Module,
        sample_input: torch.Tensor = None,
        output_path: str = None,
        backend: str = None,
        **kwargs,
    ) -> str:
        """
        Export model for Triton Inference Server.

        Creates complete model repository structure with config.pbtxt.

        Args:
            model: PyTorch model to export
            sample_input: Sample input for tracing
            output_path: Model repository path (optional)
            backend: Backend to use (onnxruntime, tensorrt, pytorch, openvino)

        Returns:
            Path to model repository
        """
        backend = backend or self.triton_config.backend
        model = self._prepare_model(model)
        sample_input = self._get_sample_input(sample_input)

        # Create model repository structure
        repo_path = output_path or self.triton_config.model_repository
        model_path = os.path.join(repo_path, self.triton_config.model_name)
        version_path = os.path.join(model_path, str(self.triton_config.model_version))
        os.makedirs(version_path, exist_ok=True)

        print(f"Creating Triton model repository: {model_path}")
        print(f"  Backend: {backend}")
        print(f"  Version: {self.triton_config.model_version}")

        # Export model based on backend
        if backend == "onnxruntime":
            model_file = self._export_onnx(model, sample_input, version_path)
        elif backend == "tensorrt":
            model_file = self._export_tensorrt(model, sample_input, version_path)
        elif backend == "pytorch":
            model_file = self._export_pytorch(model, sample_input, version_path)
        elif backend == "openvino":
            model_file = self._export_openvino(model, sample_input, version_path)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Generate config.pbtxt
        config_path = self._generate_config(model_path, backend)

        print(f"\nTriton model repository created:")
        print(f"  Model: {model_file}")
        print(f"  Config: {config_path}")

        return model_path

    def _export_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        version_path: str,
    ) -> str:
        """Export model as ONNX for onnxruntime backend."""
        output_path = os.path.join(version_path, "model.onnx")

        torch.onnx.export(
            model,
            sample_input,
            output_path,
            opset_version=self.config.onnx_opset_version,
            input_names=[self.triton_config.input_name],
            output_names=[self.triton_config.output_name],
            dynamic_axes={
                self.triton_config.input_name: {0: "batch_size"},
                self.triton_config.output_name: {0: "batch_size"},
            },
        )

        print(f"  Exported ONNX model: {output_path}")
        return output_path

    def _export_tensorrt(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        version_path: str,
    ) -> str:
        """
        Export model as TensorRT engine.

        Note: Requires TensorRT installation.
        Alternative: Export ONNX and let Triton convert to TensorRT.
        """
        # First export to ONNX
        onnx_path = os.path.join(version_path, "model.onnx")
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            opset_version=self.config.onnx_opset_version,
            input_names=[self.triton_config.input_name],
            output_names=[self.triton_config.output_name],
        )

        # Try to convert to TensorRT
        try:
            plan_path = os.path.join(version_path, "model.plan")
            self._convert_onnx_to_tensorrt(onnx_path, plan_path)
            os.remove(onnx_path)  # Remove intermediate ONNX
            print(f"  Exported TensorRT engine: {plan_path}")
            return plan_path
        except Exception as e:
            print(f"  TensorRT conversion failed: {e}")
            print("  Using ONNX model (Triton will convert to TensorRT)")
            return onnx_path

    def _convert_onnx_to_tensorrt(self, onnx_path: str, plan_path: str):
        """Convert ONNX to TensorRT engine."""
        try:
            import tensorrt as trt

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # Parse ONNX
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("ONNX parsing failed")

            # Build config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                self.triton_config.tensorrt_max_workspace_size,
            )

            if self.triton_config.tensorrt_precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)

            # Build engine
            engine = builder.build_serialized_network(network, config)

            with open(plan_path, "wb") as f:
                f.write(engine)

        except ImportError:
            raise ImportError("TensorRT not installed")

    def _export_pytorch(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        version_path: str,
    ) -> str:
        """Export model as TorchScript for pytorch backend."""
        output_path = os.path.join(version_path, "model.pt")

        with torch.no_grad():
            traced = torch.jit.trace(model, sample_input)
            traced = torch.jit.freeze(traced)

        torch.jit.save(traced, output_path)
        print(f"  Exported TorchScript model: {output_path}")
        return output_path

    def _export_openvino(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        version_path: str,
    ) -> str:
        """Export model as OpenVINO IR for openvino backend."""
        try:
            import openvino as ov

            xml_path = os.path.join(version_path, "model.xml")
            ov_model = ov.convert_model(model, example_input=sample_input)
            ov.save_model(ov_model, xml_path)

            print(f"  Exported OpenVINO model: {xml_path}")
            return xml_path

        except ImportError:
            # Fallback to ONNX
            print("  OpenVINO not installed, exporting as ONNX")
            return self._export_onnx(model, sample_input, version_path)

    def _generate_config(self, model_path: str, backend: str) -> str:
        """Generate Triton config.pbtxt file."""
        config_path = os.path.join(model_path, "config.pbtxt")

        # Map backend to platform
        platform_map = {
            "onnxruntime": "onnxruntime_onnx",
            "tensorrt": "tensorrt_plan",
            "pytorch": "pytorch_libtorch",
            "openvino": "openvino",
        }
        platform = platform_map.get(backend, "onnxruntime_onnx")

        config_content = f'''name: "{self.triton_config.model_name}"
platform: "{platform}"
max_batch_size: {self.triton_config.max_batch_size}

input [
  {{
    name: "{self.triton_config.input_name}"
    data_type: {self.triton_config.input_dtype}
    dims: {self.triton_config.input_dims}
  }}
]

output [
  {{
    name: "{self.triton_config.output_name}"
    data_type: {self.triton_config.output_dtype}
    dims: {self.triton_config.output_dims}
  }}
]

instance_group [
  {{
    count: {self.triton_config.instance_count}
    kind: {self.triton_config.instance_kind}
'''

        if self.triton_config.instance_kind == "KIND_GPU":
            gpus = ", ".join(str(g) for g in self.triton_config.instance_gpus)
            config_content += f"    gpus: [{gpus}]\n"

        config_content += "  }\n]\n"

        # Dynamic batching
        if self.triton_config.max_batch_size > 1:
            preferred = ", ".join(str(s) for s in self.triton_config.preferred_batch_sizes)
            config_content += f'''
dynamic_batching {{
  preferred_batch_size: [{preferred}]
  max_queue_delay_microseconds: {self.triton_config.max_queue_delay_microseconds}
}}
'''

        # Response cache
        if self.triton_config.response_cache_enable:
            config_content += "\nresponse_cache { enable: true }\n"

        with open(config_path, "w") as f:
            f.write(config_content)

        return config_path

    def create_ensemble(
        self,
        models: List[Dict[str, Any]],
        ensemble_name: str = "vla_ensemble",
        output_path: str = None,
    ) -> str:
        """
        Create a Triton ensemble model (pipeline).

        Args:
            models: List of model configs with name, input_map, output_map
            ensemble_name: Name for the ensemble
            output_path: Model repository path

        Example:
            models = [
                {
                    "name": "preprocessor",
                    "input_map": {"raw_image": "input"},
                    "output_map": {"processed": "output"},
                },
                {
                    "name": "vla_model",
                    "input_map": {"input": "processed"},
                    "output_map": {"output": "action"},
                },
            ]
        """
        repo_path = output_path or self.triton_config.model_repository
        ensemble_path = os.path.join(repo_path, ensemble_name)
        version_path = os.path.join(ensemble_path, "1")
        os.makedirs(version_path, exist_ok=True)

        # Generate ensemble config
        steps = []
        for model in models:
            step = f'''
    {{
      model_name: "{model['name']}"
      model_version: -1
      input_map {{
'''
            for src, dst in model.get("input_map", {}).items():
                step += f'        key: "{src}"\n        value: "{dst}"\n'
            step += "      }\n      output_map {\n"
            for src, dst in model.get("output_map", {}).items():
                step += f'        key: "{src}"\n        value: "{dst}"\n'
            step += "      }\n    }"
            steps.append(step)

        config_content = f'''name: "{ensemble_name}"
platform: "ensemble"
max_batch_size: {self.triton_config.max_batch_size}

input [
  {{
    name: "raw_input"
    data_type: {self.triton_config.input_dtype}
    dims: {self.triton_config.input_dims}
  }}
]

output [
  {{
    name: "final_output"
    data_type: {self.triton_config.output_dtype}
    dims: {self.triton_config.output_dims}
  }}
]

ensemble_scheduling {{
  step [
{",".join(steps)}
  ]
}}
'''

        config_path = os.path.join(ensemble_path, "config.pbtxt")
        with open(config_path, "w") as f:
            f.write(config_content)

        print(f"Ensemble model created: {ensemble_path}")
        return ensemble_path

    def benchmark(
        self,
        model_name: str = None,
        server_url: str = "localhost:8000",
        num_iterations: int = 100,
        concurrency: int = 1,
    ) -> Dict[str, float]:
        """
        Benchmark model on running Triton server.

        Requires: Triton server running with the model loaded.
        """
        if not self._tritonclient_available:
            print("Cannot benchmark: tritonclient not installed")
            return {}

        import tritonclient.http as httpclient
        import time

        model_name = model_name or self.triton_config.model_name

        print(f"Benchmarking {model_name} on Triton server...")

        try:
            client = httpclient.InferenceServerClient(url=server_url)

            # Check model is ready
            if not client.is_model_ready(model_name):
                print(f"Model {model_name} not ready on server")
                return {}

            # Prepare input
            sample_input = self._get_sample_input().numpy()
            inputs = [
                httpclient.InferInput(
                    self.triton_config.input_name,
                    sample_input.shape,
                    "FP32",
                )
            ]
            inputs[0].set_data_from_numpy(sample_input)

            outputs = [
                httpclient.InferRequestedOutput(self.triton_config.output_name)
            ]

            # Warmup
            for _ in range(10):
                client.infer(model_name, inputs, outputs=outputs)

            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                client.infer(model_name, inputs, outputs=outputs)
                times.append(time.perf_counter() - start)

            times_ms = np.array(times) * 1000

            results = {
                "mean_ms": float(np.mean(times_ms)),
                "std_ms": float(np.std(times_ms)),
                "p50_ms": float(np.percentile(times_ms, 50)),
                "p95_ms": float(np.percentile(times_ms, 95)),
                "p99_ms": float(np.percentile(times_ms, 99)),
                "throughput_fps": float(1000 / np.mean(times_ms)),
            }

            print(f"\nTriton Benchmark ({num_iterations} iterations):")
            print(f"  Mean: {results['mean_ms']:.2f} ms")
            print(f"  P95:  {results['p95_ms']:.2f} ms")
            print(f"  FPS:  {results['throughput_fps']:.1f}")

            return results

        except Exception as e:
            print(f"Benchmark failed: {e}")
            return {}

    def generate_client_code(
        self,
        language: str = "python",
        output_path: str = None,
    ) -> str:
        """Generate sample client code for Triton inference."""
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"triton_client.{language}"
            )

        if language == "python":
            code = f'''#!/usr/bin/env python3
"""
Triton Inference Client for {self.triton_config.model_name}

Usage:
    python triton_client.py --image path/to/image.jpg
"""

import numpy as np
import tritonclient.http as httpclient
from PIL import Image


def preprocess(image_path: str) -> np.ndarray:
    """Preprocess image for inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(({self.triton_config.input_dims[1]}, {self.triton_config.input_dims[2]}))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, ...]  # Add batch dim


def infer(
    image_path: str,
    server_url: str = "localhost:8000",
    model_name: str = "{self.triton_config.model_name}",
) -> np.ndarray:
    """Run inference on Triton server."""
    client = httpclient.InferenceServerClient(url=server_url)

    # Prepare input
    input_data = preprocess(image_path)
    inputs = [
        httpclient.InferInput("{self.triton_config.input_name}", input_data.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(input_data)

    # Inference
    outputs = [httpclient.InferRequestedOutput("{self.triton_config.output_name}")]
    result = client.infer(model_name, inputs, outputs=outputs)

    return result.as_numpy("{self.triton_config.output_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--server", default="localhost:8000", help="Triton server URL")
    args = parser.parse_args()

    action = infer(args.image, args.server)
    print(f"Predicted action: {{action}}")
'''
        else:
            code = f"// Client code generation for {language} not implemented"

        with open(output_path, "w") as f:
            f.write(code)

        print(f"Client code generated: {output_path}")
        return output_path

    def get_server_status(self, server_url: str = "localhost:8000") -> Dict[str, Any]:
        """Get Triton server status and loaded models."""
        if not self._tritonclient_available:
            return {"error": "tritonclient not installed"}

        import tritonclient.http as httpclient

        try:
            client = httpclient.InferenceServerClient(url=server_url)

            status = {
                "server_live": client.is_server_live(),
                "server_ready": client.is_server_ready(),
                "models": {},
            }

            # Get model repository
            repo = client.get_model_repository_index()
            for model in repo:
                name = model["name"]
                status["models"][name] = {
                    "ready": client.is_model_ready(name),
                }

            return status

        except Exception as e:
            return {"error": str(e)}
