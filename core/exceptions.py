"""
Core Framework Custom Exceptions

Provides a hierarchy of exceptions for better error handling and debugging.
"""

from typing import Optional, Any


class VLAError(Exception):
    """
    Base exception for all VLA framework errors.

    All custom exceptions in the VLA framework should inherit from this class.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ConfigurationError(VLAError):
    """
    Raised when there is an invalid configuration.

    Examples:
        - Invalid model configuration parameters
        - Missing required configuration fields
        - Incompatible configuration combinations
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        expected: Optional[str] = None,
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if invalid_value is not None:
            details["invalid_value"] = invalid_value
        if expected:
            details["expected"] = expected
        super().__init__(message, details)
        self.config_key = config_key
        self.invalid_value = invalid_value
        self.expected = expected


class CheckpointError(VLAError):
    """
    Raised when there is an error loading or saving checkpoints.

    Examples:
        - Checkpoint file not found
        - Corrupted checkpoint data
        - Incompatible checkpoint version
        - Missing keys in state dict
    """

    def __init__(
        self,
        message: str,
        checkpoint_path: Optional[str] = None,
        missing_keys: Optional[list] = None,
        unexpected_keys: Optional[list] = None,
    ):
        details = {}
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        if missing_keys:
            details["missing_keys"] = len(missing_keys)
        if unexpected_keys:
            details["unexpected_keys"] = len(unexpected_keys)
        super().__init__(message, details)
        self.checkpoint_path = checkpoint_path
        self.missing_keys = missing_keys or []
        self.unexpected_keys = unexpected_keys or []


class ModelError(VLAError):
    """
    Raised when there is an error with model creation or forward pass.

    Examples:
        - Invalid model architecture
        - Shape mismatch during forward pass
        - Unsupported model type
        - Failed model initialization
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        component: Optional[str] = None,
    ):
        details = {}
        if model_name:
            details["model_name"] = model_name
        if component:
            details["component"] = component
        super().__init__(message, details)
        self.model_name = model_name
        self.component = component


class TrainingError(VLAError):
    """
    Raised when there is an error during training.

    Examples:
        - NaN loss detected
        - Gradient explosion
        - Out of memory
        - Dataset loading failure
    """

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss_value: Optional[float] = None,
    ):
        details = {}
        if epoch is not None:
            details["epoch"] = epoch
        if step is not None:
            details["step"] = step
        if loss_value is not None:
            details["loss_value"] = loss_value
        super().__init__(message, details)
        self.epoch = epoch
        self.step = step
        self.loss_value = loss_value


class DatasetError(VLAError):
    """
    Raised when there is an error with dataset loading or processing.

    Examples:
        - Dataset not found
        - Invalid data format
        - Corrupted data files
    """

    def __init__(
        self,
        message: str,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
    ):
        details = {}
        if dataset_name:
            details["dataset_name"] = dataset_name
        if split:
            details["split"] = split
        super().__init__(message, details)
        self.dataset_name = dataset_name
        self.split = split


class ExportError(VLAError):
    """
    Raised when there is an error exporting the model.

    Examples:
        - Unsupported export format
        - ONNX conversion failure
        - Quantization error
    """

    def __init__(
        self,
        message: str,
        export_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        details = {}
        if export_format:
            details["export_format"] = export_format
        if output_path:
            details["output_path"] = output_path
        super().__init__(message, details)
        self.export_format = export_format
        self.output_path = output_path


class InferenceError(VLAError):
    """
    Raised when there is an error during inference.

    Examples:
        - Invalid input format
        - Model not loaded
        - Device mismatch
    """

    def __init__(
        self,
        message: str,
        input_type: Optional[str] = None,
    ):
        details = {}
        if input_type:
            details["input_type"] = input_type
        super().__init__(message, details)
        self.input_type = input_type
