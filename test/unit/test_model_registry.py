"""Tests for ModelRegistry singleton.

Tests the singleton pattern, model adapter registration, provider registration,
model configuration overrides, and model creation.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from hawi.models.registry import (
    ModelRegistry,
    ModelConfig,
    ModelProviderConfig,
    UnknownModelError,
    model_registry,
)
from hawi.models import (
    OpenAIModel,
    Model,
    create_model,
    get_model_adapter,
    get_model_config,
    list_models,
    list_providers,
    load_config,
)


class TestSingletonPattern:
    """Tests for singleton behavior."""

    def test_singleton_returns_same_instance(self):
        """Test that ModelRegistry is a true singleton."""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        assert registry1 is registry2

    def test_global_instance_exists(self):
        """Test that global model_registry instance exists."""
        assert model_registry is not None
        assert isinstance(model_registry, ModelRegistry)

    def test_thread_safety_initialization(self):
        """Test that singleton handles concurrent initialization."""
        # This is a basic test - real thread safety would need multi-threading test
        instances = [ModelRegistry() for _ in range(5)]
        assert all(instance is instances[0] for instance in instances)


class TestAdapterRegistration:
    """Tests for Model adapter (class) registration."""

    def test_builtin_adapters_registered(self):
        """Test that built-in model adapters are auto-registered."""
        adapters = model_registry.list_model_adapters()
        assert "OpenAIModel" in adapters
        assert "AnthropicModel" in adapters
        assert "DeepSeekOpenAIModel" in adapters
        assert "DeepSeekAnthropicModel" in adapters

    def test_get_model_adapter_returns_class(self):
        """Test retrieving registered model class by name."""
        cls = model_registry.get_model_adapter("OpenAIModel")
        assert cls is OpenAIModel

    def test_get_model_adapter_returns_none_for_unknown(self):
        """Test that unknown class name returns None."""
        cls = model_registry.get_model_adapter("NonExistentModel")
        assert cls is None

    def test_register_adapter_adds_new_class(self):
        """Test registering a custom model class."""
        registry = ModelRegistry()
        registry.clear()
        # Use a real model class as mock
        registry.register_adapter("MockModel", OpenAIModel, quiet=True)

        assert "MockModel" in registry.list_model_adapters()
        assert registry.get_model_adapter("MockModel") is OpenAIModel

    def test_register_adapter_overrides_existing(self):
        """Test that new registration overrides old (new overrides old rule)."""
        from hawi.models.anthropic import AnthropicModel

        registry = ModelRegistry()
        registry.clear()
        registry.register_adapter("TestModel", OpenAIModel, quiet=True)
        registry.register_adapter("TestModel", AnthropicModel, quiet=True)

        assert registry.get_model_adapter("TestModel") is AnthropicModel


class TestProviderRegistration:
    """Tests for provider registration."""

    def test_register_provider_adds_provider(self):
        """Test registering a new provider."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "test-provider",
            "OpenAIModel",
            ["gpt-4", "gpt-3.5"],
            {"api_key": "test-key"},
            quiet=True,
        )

        assert registry.has_provider("test-provider")
        providers = registry.get_provider("test-provider")
        assert providers is not None
        assert len(providers) == 1
        assert providers[0].adapter == "OpenAIModel"

    def test_list_providers_returns_all_providers(self):
        """Test listing all registered providers."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "provider-a", "OpenAIModel", ["model-a"], {}, quiet=True
        )
        registry.register_provider(
            "provider-b", "AnthropicModel", ["model-b"], {}, quiet=True
        )

        providers = registry.list_providers()
        assert "provider-a" in providers
        assert "provider-b" in providers

    def test_unregister_provider_removes_provider(self):
        """Test unregistering a provider."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "temp-provider", "OpenAIModel", ["model"], {}, quiet=True
        )
        assert registry.has_provider("temp-provider")

        result = registry.unregister_provider("temp-provider")
        assert result is True
        assert not registry.has_provider("temp-provider")

    def test_unregister_nonexistent_provider_returns_false(self):
        """Test unregistering unknown provider returns False."""
        registry = ModelRegistry()
        registry.clear()

        result = registry.unregister_provider("nonexistent")
        assert result is False


class TestModelConfigOverride:
    """Tests for model configuration override registration."""

    def test_register_model_config_override_adds_config(self):
        """Test registering a model config override."""
        registry = ModelRegistry()
        registry.clear()

        # First register a provider
        registry.register_provider(
            "test-provider",
            "OpenAIModel",
            ["gpt-4"],
            {"api_key": "test"},
            quiet=True,
        )

        # Then register an override
        registry.register_model_config_override(
            "test-provider/gpt-4",
            {"temperature": 0.7},
            quiet=True,
        )

        assert "test-provider/gpt-4" in registry.list_models()

    def test_get_model_config_returns_config(self):
        """Test retrieving model configuration."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "test-provider",
            "OpenAIModel",
            ["gpt-4"],
            {"api_key": "test-key", "base_url": "https://api.example.com"},
            quiet=True,
        )

        config = registry.get_model_config("test-provider/gpt-4")
        assert config is not None
        assert isinstance(config, ModelConfig)
        assert config.adapter == "OpenAIModel"
        assert config.properties["api_key"] == "test-key"

    def test_get_model_config_with_override(self):
        """Test that overrides are applied to model config."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "test-provider",
            "OpenAIModel",
            ["gpt-4"],
            {"api_key": "test", "temperature": 0.5},
            quiet=True,
        )
        registry.register_model_config_override(
            "test-provider/gpt-4",
            {"temperature": 0.9},
            quiet=True,
        )

        config = registry.get_model_config("test-provider/gpt-4")
        assert config is not None
        assert config.properties["temperature"] == 0.9

    def test_unregister_model_config_override_removes_config(self):
        """Test unregistering a model config override."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "test-provider", "OpenAIModel", ["gpt-4"], {}, quiet=True
        )
        registry.register_model_config_override(
            "test-provider/gpt-4", {}, quiet=True
        )
        assert "test-provider/gpt-4" in registry.list_models()

        result = registry.unregister_model_config_override("test-provider/gpt-4")
        assert result is True

    def test_has_model_checks_existence(self):
        """Test has_model method."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "test-provider", "OpenAIModel", ["gpt-4"], {}, quiet=True
        )

        assert registry.has_model("test-provider/gpt-4")
        assert not registry.has_model("test-provider/nonexistent")


class TestCreateModel:
    """Tests for create_model method."""

    def test_create_model_from_config(self):
        """Test creating model instance from config."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"model_id": "gpt-4", "api_key": "test-key"},
            quiet=True,
        )

        model = registry.create_model("openai/gpt-4")
        assert isinstance(model, OpenAIModel)
        assert model.model_id == "gpt-4"

    def test_create_model_unknown_raises(self):
        """Test that unknown model raises UnknownModelError."""
        registry = ModelRegistry()
        registry.clear()

        with pytest.raises(UnknownModelError):
            registry.create_model("nonexistent/model")

    def test_create_model_with_overrides(self):
        """Test creating model with argument overrides."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"model_id": "gpt-4", "api_key": "test-key"},
            quiet=True,
        )

        model = registry.create_model("openai/gpt-4", model_id="gpt-3.5")
        assert model.model_id == "gpt-3.5"


class TestConfigLoading:
    """Tests for YAML configuration loading."""

    def test_load_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        registry = ModelRegistry()
        registry.clear()

        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
providers:
  - name: test-provider
    adapter: OpenAIModel
    model_ids:
      - gpt-4
    properties:
      api_key: test-key
""")

        registry.load_config(config_file, quiet=True)

        # Check provider was created
        assert registry.has_provider("test-provider")
        providers = registry.get_provider("test-provider")
        assert providers[0].adapter == "OpenAIModel"

    def test_load_config_with_model_configs(self, tmp_path):
        """Test loading config with model_configs section."""
        registry = ModelRegistry()
        registry.clear()

        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
providers:
  - name: my-provider
    adapter: OpenAIModel
    model_ids:
      - gpt-4
    properties:
      api_key: test-key
      temperature: 0.5

model_configs:
  my-provider/gpt-4:
    temperature: 0.9
""")

        registry.load_config(config_file, quiet=True)

        config = registry.get_model_config("my-provider/gpt-4")
        assert config is not None
        assert config.properties["temperature"] == 0.9

    def test_load_nonexistent_file_is_noop(self, tmp_path):
        """Test loading non-existent file does nothing."""
        registry = ModelRegistry()
        registry.clear()

        nonexistent = tmp_path / "nonexistent.yaml"
        # Should not raise
        registry.load_config(nonexistent, quiet=True)

    def test_load_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML raises ValueError."""
        registry = ModelRegistry()
        registry.clear()

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError):
            registry.load_config(config_file)

    def test_has_model_triggers_auto_load(self, tmp_path):
        """Test has_model() triggers auto-loading from the default project config."""
        registry = ModelRegistry()
        registry.clear()

        config_dir = tmp_path / ".hawi"
        config_dir.mkdir()
        config_file = config_dir / "models.yaml"
        config_file.write_text("""
providers:
  - name: auto-provider
    adapter: OpenAIModel
    model_ids:
      - gpt-4
    properties:
      api_key: test-key
""")

        with patch("hawi.models.registry.Path.cwd", return_value=tmp_path):
            assert registry.has_model("auto-provider/gpt-4")

    def test_list_models_triggers_auto_load(self, tmp_path):
        """Test list_models() triggers auto-loading from the default project config."""
        registry = ModelRegistry()
        registry.clear()

        config_dir = tmp_path / ".hawi"
        config_dir.mkdir()
        config_file = config_dir / "models.yaml"
        config_file.write_text("""
providers:
  - name: auto-provider
    adapter: OpenAIModel
    model_ids:
      - gpt-4
    properties:
      api_key: test-key
""")

        with patch("hawi.models.registry.Path.cwd", return_value=tmp_path):
            assert "auto-provider/gpt-4" in registry.list_models()

    def test_get_model_config_triggers_auto_load(self, tmp_path):
        """Test get_model_config() triggers auto-loading from the default project config."""
        registry = ModelRegistry()
        registry.clear()

        config_dir = tmp_path / ".hawi"
        config_dir.mkdir()
        config_file = config_dir / "models.yaml"
        config_file.write_text("""
providers:
  - name: auto-provider
    adapter: OpenAIModel
    model_ids:
      - gpt-4
    properties:
      api_key: test-key
""")

        with patch("hawi.models.registry.Path.cwd", return_value=tmp_path):
            config = registry.get_model_config("auto-provider/gpt-4")

        assert config is not None
        assert config.adapter == "OpenAIModel"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_model_convenience_function(self):
        """Test create_model() convenience function uses global registry."""
        model_registry.clear()
        model_registry.register_provider(
            "test-provider",
            "OpenAIModel",
            ["gpt-4"],
            {"model_id": "gpt-4", "api_key": "test"},
            quiet=True,
        )

        model = create_model("test-provider/gpt-4")
        assert isinstance(model, OpenAIModel)

    def test_get_model_adapter_convenience_function(self):
        """Test get_model_adapter() convenience function."""
        model_registry.clear()
        # Register a test class first
        model_registry.register_adapter("TestModel", OpenAIModel, quiet=True)

        cls = get_model_adapter("TestModel")
        assert cls is OpenAIModel

    def test_get_model_config_convenience_function(self):
        """Test get_model_config() convenience function."""
        model_registry.clear()
        model_registry.register_provider(
            "test-provider",
            "OpenAIModel",
            ["gpt-4"],
            {"api_key": "test"},
            quiet=True,
        )

        config = get_model_config("test-provider/gpt-4")
        assert config is not None
        assert config.adapter == "OpenAIModel"

    def test_list_models_convenience_function(self):
        """Test list_models() convenience function."""
        model_registry.clear()
        model_registry.register_provider(
            "test-provider", "OpenAIModel", ["gpt-4"], {}, quiet=True
        )

        models = list_models()
        assert "test-provider/gpt-4" in models

    def test_list_providers_convenience_function(self):
        """Test list_providers() convenience function."""
        model_registry.clear()
        model_registry.register_provider(
            "test-provider", "OpenAIModel", ["model"], {}, quiet=True
        )

        providers = list_providers()
        assert "test-provider" in providers

    def test_load_config_convenience_function(self, tmp_path):
        """Test load_config() convenience function."""
        model_registry.clear()

        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
providers:
  - name: test-provider
    adapter: OpenAIModel
    model_ids:
      - gpt-4
    properties:
      api_key: test
""")

        load_config(config_file, quiet=True)
        assert "test-provider" in model_registry.list_providers()


class TestClear:
    """Tests for registry clear functionality."""

    def test_clear_removes_models(self):
        """Test that clear() removes all providers and models."""
        registry = ModelRegistry()
        registry.clear()

        with patch.dict(os.environ, {"HAWI_NO_AUTO_LOAD": "1"}):
            registry.register_provider(
                "test", "OpenAIModel", ["gpt-4"], {}, quiet=True
            )
            assert registry.list_models()

            registry.clear()
            assert registry.list_models() == []
            assert registry.list_providers() == []

    def test_clear_re_registers_builtin_adapters(self):
        """Test that clear() re-registers built-in adapters."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_adapter("OpenAIModel", MagicMock, quiet=True)
        assert registry.get_model_adapter("OpenAIModel") is not OpenAIModel

        registry.clear()
        assert registry.get_model_adapter("OpenAIModel") is OpenAIModel


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating ModelConfig."""
        config = ModelConfig(
            adapter="OpenAIModel",
            properties={"model_id": "gpt-4", "api_key": "test"},
        )
        assert config.adapter == "OpenAIModel"
        assert config.properties["model_id"] == "gpt-4"

    def test_model_provider_config_creation(self):
        """Test creating ModelProviderConfig."""
        config = ModelProviderConfig(
            name="test-provider",
            adapter="OpenAIModel",
            model_ids=["gpt-4", "gpt-3.5"],
            properties={"api_key": "test"},
        )
        assert config.name == "test-provider"
        assert config.adapter == "OpenAIModel"
        assert "gpt-4" in config.model_ids
