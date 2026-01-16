# VLA Model Deployment

This document covers exporting and deploying trained VLA models for production inference, including ONNX, TorchScript, OpenVINO, Triton, and quantization.

## Export Formats Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Model Export Formats                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐   │
│  │     ONNX       │     │  TorchScript   │     │   OpenVINO     │   │
│  │ Cross-platform │     │ PyTorch native │     │  Intel CPUs    │   │
│  │ ONNX Runtime   │     │  C++/Mobile    │     │  Optimized     │   │
│  └───────┬────────┘     └───────┬────────┘     └───────┬────────┘   │
│          │                      │                      │            │
│          └──────────┬───────────┴──────────────────────┘            │
│                     │                                                │
│                     ▼                                                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Inference Servers                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │ │
│  │  │ NVIDIA Triton│  │ TorchServe   │  │ ONNX Runtime │          │ │
│  │  │   Server     │  │              │  │    Server    │          │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Optimization                                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │ │
│  │  │ Quantization │  │  Pruning     │  │ Graph Optim  │          │ │
│  │  │  INT8/FP16   │  │              │  │              │          │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

| Format | Use Case | Platform | Notes |
|--------|----------|----------|-------|
| **ONNX** | Cross-platform | Any | Best for flexibility |
| **TorchScript** | PyTorch ecosystem | PyTorch/C++ | Native PyTorch support |
| **OpenVINO** | Intel hardware | Intel CPUs/GPUs | Intel-optimized |
| **Triton** | Production serving | NVIDIA GPUs | High-throughput serving |
| **Quantized** | Edge/mobile | Any | Reduced size/latency |

---

## ONNX Export

ONNX (Open Neural Network Exchange) enables cross-platform deployment with broad runtime support.

### Basic Export

```python
from model.utils.export import ONNXExporter, ExportConfig

# Configure export
config = ExportConfig(
    model_name="vla_driving",
    onnx_opset_version=17,
    onnx_simplify=True,
    onnx_dynamic_axes=True,
)

# Create exporter
exporter = ONNXExporter(config)

# Export model
model = load_trained_model("./output/best_model")
sample_input = torch.randn(1, 3, 224, 224)  # Example input

output_path = exporter.export(
    model=model,
    sample_input=sample_input,
    output_path="./exported/model.onnx",
    input_names=["image"],
    output_names=["action"],
)
```

### Multi-Input VLA Export

```python
from model.utils.export import ONNXExporter

exporter = ONNXExporter()

# VLA with image and text inputs
class VLAWrapper(nn.Module):
    def __init__(self, vla_model):
        super().__init__()
        self.vla = vla_model

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.vla(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs["predicted_actions"]

wrapper = VLAWrapper(model)

# Export with multiple inputs
sample_inputs = (
    torch.randn(1, 3, 224, 224),  # pixel_values
    torch.randint(0, 1000, (1, 32)),  # input_ids
    torch.ones(1, 32),  # attention_mask
)

exporter.export(
    model=wrapper,
    sample_input=sample_inputs,
    output_path="./exported/vla.onnx",
    input_names=["pixel_values", "input_ids", "attention_mask"],
    output_names=["action"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "action": {0: "batch"},
    },
)
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

# Create session
session = ort.InferenceSession(
    "./exported/model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Inference
image = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: image})[0]
print(f"Action: {result}")
```

### Verification and Benchmarking

```python
# Verify export correctness
passed = exporter.verify(
    original_model=model,
    exported_path="./exported/model.onnx",
    sample_input=sample_input,
    tolerance=1e-5,
)

# Benchmark performance
results = exporter.benchmark(
    exported_path="./exported/model.onnx",
    sample_input=sample_input,
    num_iterations=100,
    providers=["CUDAExecutionProvider"],
)
print(f"Mean latency: {results['mean_ms']:.2f} ms")
print(f"Throughput: {results['throughput_fps']:.1f} FPS")
```

---

## TorchScript Export

TorchScript enables deployment without Python, supporting C++ and mobile platforms.

### Traced Export

Best for models without data-dependent control flow:

```python
from model.utils.export import TorchScriptExporter, ExportConfig

config = ExportConfig(
    model_name="vla_traced",
    optimize_for_inference=True,
)

exporter = TorchScriptExporter(config)

# Trace model
output_path = exporter.export_traced(
    model=model,
    sample_input=sample_input,
    output_path="./exported/model_traced.pt",
)
```

### Scripted Export

Best for models with control flow (if/else, loops):

```python
output_path = exporter.export_scripted(
    model=model,
    output_path="./exported/model_scripted.pt",
    strict=True,
)
```

### Mobile Export

Optimized for mobile deployment:

```python
output_path = exporter.export_mobile(
    model=model,
    sample_input=sample_input,
    output_path="./exported/model_mobile.ptl",
    optimize_for_mobile=True,
)
```

### TorchScript Inference

```python
import torch

# Load model
model = torch.jit.load("./exported/model_traced.pt")
model.eval()

# Inference
with torch.no_grad():
    image = torch.randn(1, 3, 224, 224)
    action = model(image)
    print(f"Action: {action}")
```

### C++ Inference

```cpp
#include <torch/script.h>

int main() {
    // Load model
    torch::jit::script::Module model;
    model = torch::jit::load("model_traced.pt");
    model.eval();

    // Create input
    torch::Tensor input = torch::randn({1, 3, 224, 224});

    // Inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    at::Tensor output = model.forward(inputs).toTensor();

    return 0;
}
```

---

## OpenVINO Export

OpenVINO provides optimized inference on Intel hardware.

```python
from model.utils.export import OpenVINOExporter, ExportConfig

config = ExportConfig(
    model_name="vla_openvino",
    openvino_precision="FP16",  # FP32, FP16, or INT8
)

exporter = OpenVINOExporter(config)

# Export to OpenVINO IR format
output_path = exporter.export(
    model=model,
    sample_input=sample_input,
    output_path="./exported/model_openvino",
)
```

### OpenVINO Inference

```python
from openvino.runtime import Core

# Create inference engine
core = Core()
model = core.read_model("./exported/model_openvino/model.xml")
compiled_model = core.compile_model(model, "CPU")

# Inference
input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = compiled_model([input_tensor])[0]
```

---

## Triton Inference Server

NVIDIA Triton provides high-throughput model serving.

### Export for Triton

```python
from model.utils.export import TritonExporter, ExportConfig

config = ExportConfig(
    model_name="vla_triton",
    triton_backend="onnxruntime",  # or "pytorch", "tensorrt"
    triton_max_batch_size=16,
)

exporter = TritonExporter(config)

# Create Triton model repository
exporter.export(
    model=model,
    sample_input=sample_input,
    output_path="./model_repository/vla_driving",
)
```

### Triton Model Repository Structure

```
model_repository/
└── vla_driving/
    ├── config.pbtxt           # Model configuration
    ├── 1/                     # Version 1
    │   └── model.onnx         # Model file
    └── labels.txt             # Optional labels
```

### config.pbtxt Example

```protobuf
name: "vla_driving"
platform: "onnxruntime_onnx"
max_batch_size: 16

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "action"
    data_type: TYPE_FP32
    dims: [ 3 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

### Running Triton Server

```bash
# Start Triton server
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.04-py3 \
    tritonserver --model-repository=/models
```

### Triton Client

```python
import tritonclient.grpc as grpcclient
import numpy as np

# Create client
client = grpcclient.InferenceServerClient(url="localhost:8001")

# Create input
image = np.random.randn(1, 3, 224, 224).astype(np.float32)
inputs = [grpcclient.InferInput("image", image.shape, "FP32")]
inputs[0].set_data_from_numpy(image)

# Inference
outputs = [grpcclient.InferRequestedOutput("action")]
result = client.infer("vla_driving", inputs, outputs=outputs)
action = result.as_numpy("action")
```

---

## Quantization

Reduce model size and latency through quantization.

### Dynamic Quantization

```python
from model.utils.export import QuantizationExporter, ExportConfig

config = ExportConfig(
    model_name="vla_quantized",
    quantization_dtype="int8",
    quantization_mode="dynamic",
)

exporter = QuantizationExporter(config)

output_path = exporter.export_dynamic(
    model=model,
    output_path="./exported/model_dynamic_int8.pt",
)
```

### Static Quantization (PTQ)

```python
# Calibration dataset required
from torch.utils.data import DataLoader

calibration_loader = DataLoader(calibration_dataset, batch_size=32)

output_path = exporter.export_static(
    model=model,
    calibration_loader=calibration_loader,
    output_path="./exported/model_static_int8.pt",
    num_calibration_batches=100,
)
```

### QAT (Quantization-Aware Training)

```python
from model.utils.export import QuantizationExporter

exporter = QuantizationExporter()

# Prepare model for QAT
qat_model = exporter.prepare_qat(model)

# Fine-tune with quantization simulation
optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-5)
for batch in train_loader:
    outputs = qat_model(batch["image"])
    loss = criterion(outputs, batch["action"])
    loss.backward()
    optimizer.step()

# Convert to quantized model
quantized_model = exporter.convert_qat(qat_model)
torch.save(quantized_model.state_dict(), "./exported/model_qat.pt")
```

### ONNX Quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="./exported/model.onnx",
    model_output="./exported/model_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

---

## Complete Export Pipeline

### Shell Script

```bash
#!/bin/bash
# export_model.sh

MODEL_PATH="./output/best_model"
OUTPUT_DIR="./exported"

# Create output directory
mkdir -p $OUTPUT_DIR

# ONNX export
python -c "
from model.utils.export import ONNXExporter
import torch

model = torch.load('$MODEL_PATH/model.pt')
exporter = ONNXExporter()
exporter.export(
    model=model,
    sample_input=torch.randn(1, 3, 224, 224),
    output_path='$OUTPUT_DIR/model.onnx',
)
"

# TorchScript export
python -c "
from model.utils.export import TorchScriptExporter
import torch

model = torch.load('$MODEL_PATH/model.pt')
exporter = TorchScriptExporter()
exporter.export_traced(
    model=model,
    sample_input=torch.randn(1, 3, 224, 224),
    output_path='$OUTPUT_DIR/model_traced.pt',
)
"

# Quantized export
python -c "
from model.utils.export import QuantizationExporter
import torch

model = torch.load('$MODEL_PATH/model.pt')
exporter = QuantizationExporter()
exporter.export_dynamic(
    model=model,
    output_path='$OUTPUT_DIR/model_int8.pt',
)
"

echo "Export complete!"
ls -la $OUTPUT_DIR
```

### Python Export Script

```python
#!/usr/bin/env python
"""Complete model export pipeline."""

import torch
from pathlib import Path

from model.utils.export import (
    ONNXExporter,
    TorchScriptExporter,
    OpenVINOExporter,
    QuantizationExporter,
    ExportConfig,
)


def export_all(model_path: str, output_dir: str):
    """Export model to all formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = torch.load(model_path)
    model.eval()

    sample_input = torch.randn(1, 3, 224, 224)

    # ONNX
    print("\n1. Exporting to ONNX...")
    onnx_exporter = ONNXExporter()
    onnx_path = onnx_exporter.export(
        model=model,
        sample_input=sample_input,
        output_path=str(output_dir / "model.onnx"),
    )
    onnx_exporter.verify(model, onnx_path, sample_input)
    onnx_exporter.benchmark(onnx_path, sample_input)

    # TorchScript
    print("\n2. Exporting to TorchScript...")
    ts_exporter = TorchScriptExporter()
    ts_path = ts_exporter.export_traced(
        model=model,
        sample_input=sample_input,
        output_path=str(output_dir / "model_traced.pt"),
    )
    ts_exporter.verify(model, ts_path, sample_input)
    ts_exporter.benchmark(ts_path, sample_input)

    # Quantized
    print("\n3. Exporting quantized model...")
    quant_exporter = QuantizationExporter()
    quant_path = quant_exporter.export_dynamic(
        model=model,
        output_path=str(output_dir / "model_int8.pt"),
    )

    print("\nExport complete!")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--output", default="./exported", help="Output directory")
    args = parser.parse_args()

    export_all(args.model, args.output)
```

---

## ROS Integration

For real robot deployment with ROS/ROS2:

```python
from integration.ros_bridge import ROSVLANode

# Create ROS node
node = ROSVLANode(
    model_path="./exported/model_traced.pt",
    image_topic="/camera/image_raw",
    action_topic="/robot/action_cmd",
    rate=30,
)

# Run
node.spin()
```

### ROS Launch File

```xml
<launch>
    <node name="vla_inference" pkg="vla_ros" type="vla_node.py">
        <param name="model_path" value="$(find vla_ros)/models/model_traced.pt"/>
        <param name="device" value="cuda"/>
        <param name="rate" value="30"/>
        <remap from="image" to="/camera/image_raw"/>
        <remap from="action" to="/robot/action_cmd"/>
    </node>
</launch>
```

---

## Safety for Real Robot Deployment

### Safety Shield Integration

```python
from model.safety import SafetyShield

# Wrap model with safety
safe_model = SafetyShield(
    policy=model,
    max_velocity=1.0,
    max_acceleration=2.0,
    collision_threshold=0.3,
    emergency_stop_enabled=True,
)

# Safe inference
action = safe_model(observation)  # Action is filtered for safety
```

### Constraint Handling

```python
from model.safety import ConstraintHandler

handler = ConstraintHandler(
    action_limits={
        "steering": (-1.0, 1.0),
        "throttle": (0.0, 1.0),
        "brake": (0.0, 1.0),
    },
    jerk_limit=0.1,
    rate=30,
)

# Apply constraints
safe_action = handler.apply(raw_action)
```

---

## Performance Benchmarking

### Comprehensive Benchmark Script

```python
from model.utils.export import ONNXExporter, TorchScriptExporter
import torch
import time

def benchmark_all_formats(model_path, sample_input, num_iterations=100):
    """Benchmark all export formats."""
    results = {}

    # Load model
    model = torch.load(model_path)
    model.eval()

    # PyTorch baseline
    print("Benchmarking PyTorch...")
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            model(sample_input)
            times.append(time.perf_counter() - start)
    results["pytorch"] = {
        "mean_ms": sum(times) / len(times) * 1000,
    }

    # ONNX
    print("Benchmarking ONNX...")
    onnx_exporter = ONNXExporter()
    onnx_path = onnx_exporter.export(model, sample_input, "temp.onnx")
    results["onnx"] = onnx_exporter.benchmark(onnx_path, sample_input)

    # TorchScript
    print("Benchmarking TorchScript...")
    ts_exporter = TorchScriptExporter()
    ts_path = ts_exporter.export_traced(model, sample_input, "temp.pt")
    results["torchscript"] = ts_exporter.benchmark(ts_path, sample_input)

    # Print comparison
    print("\n=== Benchmark Results ===")
    for format_name, metrics in results.items():
        print(f"{format_name}: {metrics.get('mean_ms', 0):.2f} ms")

    return results
```

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Complete training pipeline
- [Training Datasets](training_datasets.md) - Dataset documentation
- [Reinforcement Learning](training_reinforcement_learning.md) - RL methods
- [World Model Training](training_world_model.md) - Dynamics learning
