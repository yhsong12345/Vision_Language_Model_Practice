"""
Core Framework Model Registry

Provides a centralized registry for models, action heads, vision encoders,
and other components. This enables flexible component discovery and instantiation.
"""

from typing import Dict, Type, TypeVar, Optional, Callable, Any, List
from dataclasses import dataclass
from functools import wraps

from core.exceptions import ConfigurationError


T = TypeVar("T")


@dataclass
class RegistryEntry:
    """Entry in a component registry."""
    cls: Type
    name: str
    description: str
    default_config: Optional[dict] = None
    aliases: Optional[List[str]] = None


class Registry:
    """
    A generic registry for component classes.

    Supports registration via decorator or direct method call,
    and provides lookup by name or alias.

    Example:
        >>> model_registry = Registry("models")
        >>>
        >>> @model_registry.register("vla-base")
        ... class VLAModel:
        ...     pass
        >>>
        >>> model_cls = model_registry.get("vla-base")
    """

    def __init__(self, name: str):
        """
        Initialize a new registry.

        Args:
            name: Name of the registry (for error messages)
        """
        self.name = name
        self._entries: Dict[str, RegistryEntry] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        description: str = "",
        default_config: Optional[dict] = None,
        aliases: Optional[List[str]] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a class.

        Args:
            name: Unique name for the component
            description: Human-readable description
            default_config: Default configuration dict
            aliases: Alternative names for lookup

        Returns:
            Decorator function

        Example:
            >>> @registry.register("my-model", description="My custom model")
            ... class MyModel:
            ...     pass
        """
        def decorator(cls: Type[T]) -> Type[T]:
            self._register_class(
                cls=cls,
                name=name,
                description=description,
                default_config=default_config,
                aliases=aliases,
            )
            return cls
        return decorator

    def _register_class(
        self,
        cls: Type,
        name: str,
        description: str = "",
        default_config: Optional[dict] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Internal method to register a class."""
        if name in self._entries:
            raise ConfigurationError(
                f"Component '{name}' is already registered in {self.name} registry",
                config_key=name,
            )

        entry = RegistryEntry(
            cls=cls,
            name=name,
            description=description,
            default_config=default_config,
            aliases=aliases,
        )
        self._entries[name] = entry

        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in self._aliases or alias in self._entries:
                    raise ConfigurationError(
                        f"Alias '{alias}' conflicts with existing name in {self.name} registry",
                        config_key=alias,
                    )
                self._aliases[alias] = name

    def get(self, name: str) -> Type:
        """
        Get a registered class by name or alias.

        Args:
            name: Name or alias of the component

        Returns:
            The registered class

        Raises:
            ConfigurationError: If component is not found
        """
        # Check aliases first
        if name in self._aliases:
            name = self._aliases[name]

        if name not in self._entries:
            available = list(self._entries.keys())
            raise ConfigurationError(
                f"Component '{name}' not found in {self.name} registry. "
                f"Available: {available}",
                config_key=name,
            )

        return self._entries[name].cls

    def get_entry(self, name: str) -> RegistryEntry:
        """Get the full registry entry for a component."""
        if name in self._aliases:
            name = self._aliases[name]

        if name not in self._entries:
            raise ConfigurationError(
                f"Component '{name}' not found in {self.name} registry",
                config_key=name,
            )

        return self._entries[name]

    def create(self, name: str, **kwargs) -> Any:
        """
        Create an instance of a registered component.

        Args:
            name: Name or alias of the component
            **kwargs: Arguments to pass to the constructor

        Returns:
            Instance of the component
        """
        cls = self.get(name)
        entry = self.get_entry(name)

        # Merge default config with provided kwargs
        if entry.default_config:
            merged_kwargs = {**entry.default_config, **kwargs}
        else:
            merged_kwargs = kwargs

        return cls(**merged_kwargs)

    def list(self) -> List[str]:
        """List all registered component names."""
        return list(self._entries.keys())

    def list_with_descriptions(self) -> Dict[str, str]:
        """List all components with their descriptions."""
        return {name: entry.description for name, entry in self._entries.items()}

    def __contains__(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._entries or name in self._aliases

    def __len__(self) -> int:
        """Return number of registered components."""
        return len(self._entries)


# ============================================
# Global Registries
# ============================================

# Model registry for VLA models
MODEL_REGISTRY = Registry("models")

# Vision encoder registry
VISION_ENCODER_REGISTRY = Registry("vision_encoders")

# Action head registry
ACTION_HEAD_REGISTRY = Registry("action_heads")

# Trainer registry
TRAINER_REGISTRY = Registry("trainers")

# Dataset registry
DATASET_REGISTRY = Registry("datasets")


# ============================================
# Convenience Functions
# ============================================

def register_model(
    name: str,
    description: str = "",
    default_config: Optional[dict] = None,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a model class."""
    return MODEL_REGISTRY.register(name, description, default_config, aliases)


def register_vision_encoder(
    name: str,
    description: str = "",
    default_config: Optional[dict] = None,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a vision encoder class."""
    return VISION_ENCODER_REGISTRY.register(name, description, default_config, aliases)


def register_action_head(
    name: str,
    description: str = "",
    default_config: Optional[dict] = None,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register an action head class."""
    return ACTION_HEAD_REGISTRY.register(name, description, default_config, aliases)


def register_trainer(
    name: str,
    description: str = "",
    default_config: Optional[dict] = None,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a trainer class."""
    return TRAINER_REGISTRY.register(name, description, default_config, aliases)


def register_dataset(
    name: str,
    description: str = "",
    default_config: Optional[dict] = None,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a dataset class."""
    return DATASET_REGISTRY.register(name, description, default_config, aliases)


def list_available_models() -> List[str]:
    """List all available model names."""
    return MODEL_REGISTRY.list()


def list_available_vision_encoders() -> List[str]:
    """List all available vision encoder names."""
    return VISION_ENCODER_REGISTRY.list()


def list_available_action_heads() -> List[str]:
    """List all available action head names."""
    return ACTION_HEAD_REGISTRY.list()


def create_model(name: str, **kwargs) -> Any:
    """Create a model instance by name."""
    return MODEL_REGISTRY.create(name, **kwargs)


def create_vision_encoder(name: str, **kwargs) -> Any:
    """Create a vision encoder instance by name."""
    return VISION_ENCODER_REGISTRY.create(name, **kwargs)


def create_action_head(name: str, **kwargs) -> Any:
    """Create an action head instance by name."""
    return ACTION_HEAD_REGISTRY.create(name, **kwargs)
