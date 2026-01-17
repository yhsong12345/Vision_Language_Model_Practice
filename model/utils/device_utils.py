"""
Device Utilities

DEPRECATED: This module has been moved to 'core.device_utils'.
This file is kept for backward compatibility.
"""

import warnings

warnings.warn(
    "model.utils.device_utils has been moved to core.device_utils. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from core for backward compatibility
from core.device_utils import (
    get_device,
    move_to_device,
    get_device_info,
    print_device_info,
)

__all__ = [
    "get_device",
    "move_to_device",
    "get_device_info",
    "print_device_info",
]
