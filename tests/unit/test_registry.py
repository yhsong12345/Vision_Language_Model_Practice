"""
Unit Tests for Model Registry

Tests the registry pattern for component registration and lookup.
"""

import pytest
import torch.nn as nn

from vla.registry import (
    Registry,
    MODEL_REGISTRY,
    ACTION_HEAD_REGISTRY,
    register_model,
    register_action_head,
)
from vla.exceptions import ConfigurationError


class TestRegistry:
    """Tests for the Registry class."""

    def test_register_and_get(self):
        """Test basic registration and retrieval."""
        registry = Registry("test")

        @registry.register("test-model", description="A test model")
        class TestModel(nn.Module):
            pass

        cls = registry.get("test-model")
        assert cls == TestModel

    def test_register_with_aliases(self):
        """Test registration with aliases."""
        registry = Registry("test")

        @registry.register("full-name", aliases=["short", "abbrev"])
        class AliasedModel(nn.Module):
            pass

        assert registry.get("full-name") == AliasedModel
        assert registry.get("short") == AliasedModel
        assert registry.get("abbrev") == AliasedModel

    def test_duplicate_registration_raises_error(self):
        """Test that duplicate registration raises error."""
        registry = Registry("test")

        @registry.register("duplicate")
        class Model1(nn.Module):
            pass

        with pytest.raises(ConfigurationError):
            @registry.register("duplicate")
            class Model2(nn.Module):
                pass

    def test_alias_conflict_raises_error(self):
        """Test that alias conflicts raise error."""
        registry = Registry("test")

        @registry.register("model1", aliases=["alias"])
        class Model1(nn.Module):
            pass

        with pytest.raises(ConfigurationError):
            @registry.register("model2", aliases=["alias"])
            class Model2(nn.Module):
                pass

    def test_get_unknown_raises_error(self):
        """Test that unknown name raises error."""
        registry = Registry("test")

        with pytest.raises(ConfigurationError):
            registry.get("unknown")

    def test_list_registered(self):
        """Test listing registered components."""
        registry = Registry("test")

        @registry.register("model-a")
        class ModelA(nn.Module):
            pass

        @registry.register("model-b")
        class ModelB(nn.Module):
            pass

        names = registry.list()
        assert "model-a" in names
        assert "model-b" in names
        assert len(names) == 2

    def test_contains(self):
        """Test the __contains__ method."""
        registry = Registry("test")

        @registry.register("registered", aliases=["alias"])
        class Model(nn.Module):
            pass

        assert "registered" in registry
        assert "alias" in registry
        assert "unregistered" not in registry

    def test_len(self):
        """Test the __len__ method."""
        registry = Registry("test")

        assert len(registry) == 0

        @registry.register("model")
        class Model(nn.Module):
            pass

        assert len(registry) == 1

    def test_create_instance(self):
        """Test creating an instance from registry."""
        registry = Registry("test")

        @registry.register("configurable")
        class ConfigurableModel(nn.Module):
            def __init__(self, hidden_dim=64):
                super().__init__()
                self.hidden_dim = hidden_dim

        # Create with default
        model1 = registry.create("configurable")
        assert model1.hidden_dim == 64

        # Create with custom kwargs
        model2 = registry.create("configurable", hidden_dim=128)
        assert model2.hidden_dim == 128

    def test_create_with_default_config(self):
        """Test creating instance with default config from registry."""
        registry = Registry("test")

        @registry.register(
            "with-defaults",
            default_config={"hidden_dim": 256}
        )
        class ModelWithDefaults(nn.Module):
            def __init__(self, hidden_dim=64):
                super().__init__()
                self.hidden_dim = hidden_dim

        # Should use default from registry
        model = registry.create("with-defaults")
        assert model.hidden_dim == 256

        # Explicit kwargs should override
        model2 = registry.create("with-defaults", hidden_dim=512)
        assert model2.hidden_dim == 512

    def test_get_entry(self):
        """Test getting full registry entry."""
        registry = Registry("test")

        @registry.register("detailed", description="A detailed model")
        class DetailedModel(nn.Module):
            pass

        entry = registry.get_entry("detailed")
        assert entry.name == "detailed"
        assert entry.description == "A detailed model"
        assert entry.cls == DetailedModel

    def test_list_with_descriptions(self):
        """Test listing with descriptions."""
        registry = Registry("test")

        @registry.register("model-a", description="First model")
        class ModelA(nn.Module):
            pass

        @registry.register("model-b", description="Second model")
        class ModelB(nn.Module):
            pass

        descriptions = registry.list_with_descriptions()
        assert descriptions["model-a"] == "First model"
        assert descriptions["model-b"] == "Second model"


class TestGlobalRegistries:
    """Tests for global registry instances."""

    def test_model_registry_exists(self):
        """Test that MODEL_REGISTRY exists."""
        assert MODEL_REGISTRY is not None
        assert isinstance(MODEL_REGISTRY, Registry)

    def test_action_head_registry_exists(self):
        """Test that ACTION_HEAD_REGISTRY exists."""
        assert ACTION_HEAD_REGISTRY is not None
        assert isinstance(ACTION_HEAD_REGISTRY, Registry)


class TestDecoratorHelpers:
    """Tests for convenience decorator functions."""

    def test_register_model_decorator(self):
        """Test register_model convenience decorator."""
        # Create a temporary registry for testing
        original_entries = dict(MODEL_REGISTRY._entries)
        original_aliases = dict(MODEL_REGISTRY._aliases)

        try:
            @register_model("test-decorator-model")
            class TestDecoratorModel(nn.Module):
                pass

            assert "test-decorator-model" in MODEL_REGISTRY
        finally:
            # Restore original state
            MODEL_REGISTRY._entries = original_entries
            MODEL_REGISTRY._aliases = original_aliases
