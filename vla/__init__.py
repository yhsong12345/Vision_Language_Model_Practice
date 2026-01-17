"""
VLA - Vision-Language-Action Model Training Framework

DEPRECATED: This module has been renamed to 'core'.
This file is kept for backward compatibility.
Please use 'from core import ...' instead.
"""

import warnings

warnings.warn(
    "The 'vla' module has been renamed to 'core'. "
    "Please update your imports to use 'from core import ...' instead. "
    "This backward compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from core for backward compatibility
from core import (
    __version__,
    VLAError,
    ConfigurationError,
    CheckpointError,
    ModelError,
    TrainingError,
    DatasetError,
    ExportError,
    InferenceError,
)

__all__ = [
    "__version__",
    "VLAError",
    "ConfigurationError",
    "CheckpointError",
    "ModelError",
    "TrainingError",
    "DatasetError",
    "ExportError",
    "InferenceError",
]
