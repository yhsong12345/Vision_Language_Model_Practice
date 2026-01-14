"""
Model Utilities

Shared utilities for VLA models:
- layers: Common neural network layers (PositionalEncoding, MLP)
- parameter_utils: Module freezing, parameter counting
- device_utils: Device detection and management
- checkpoint_utils: Model saving and loading
- export: Model export (ONNX, TorchScript, quantization, OpenVINO, Triton)
"""

from .layers import (
    PositionalEncoding,
    MLP,
    build_mlp,
)

from .parameter_utils import (
    freeze_module,
    unfreeze_module,
    count_parameters,
    count_trainable_parameters,
    get_parameter_stats,
    set_requires_grad,
)

from .device_utils import (
    get_device,
    move_to_device,
)

from .checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    ModelCheckpoint,
)

# Export utilities from export folder
from .export import (
    # Config
    ExportConfig,
    # Base
    BaseExporter,
    # Exporters
    ONNXExporter,
    TorchScriptExporter,
    QuantizationExporter,
    OpenVINOExporter,
    OpenVINOConfig,
    TritonExporter,
    TritonConfig,
    # Optimizer
    ModelOptimizer,
    # Utilities
    export_model,
    export_all_formats,
)

__all__ = [
    # Layers
    "PositionalEncoding",
    "MLP",
    "build_mlp",
    # Parameter utilities
    "freeze_module",
    "unfreeze_module",
    "count_parameters",
    "count_trainable_parameters",
    "get_parameter_stats",
    "set_requires_grad",
    # Device utilities
    "get_device",
    "move_to_device",
    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint",
    "ModelCheckpoint",
    # Export utilities
    "ExportConfig",
    "BaseExporter",
    "ONNXExporter",
    "TorchScriptExporter",
    "QuantizationExporter",
    "OpenVINOExporter",
    "OpenVINOConfig",
    "TritonExporter",
    "TritonConfig",
    "ModelOptimizer",
    "export_model",
    "export_all_formats",
]
