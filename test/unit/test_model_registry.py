"""Tests for ModelRegistry singleton.

Tests the singleton pattern, model adapter registration, provider registration,
model configuration overrides, and model creation.
"""

import os
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from hawi.errors import ConfigurationError
from hawi.models.registry import (
    ModelRegistry,
    ModelConfig,
    ModelProviderConfig,
    UnknownModelError,
    model_registry,
)
from hawi.models.config_persistence import persist_provider_properties
from hawi.models import (
    Message,
    MessageResponse,
    OpenAIModel,
    Model,
    TextPart,
    create_model,
    get_model_adapter,
    get_model_config,
    list_models,
    list_providers,
    load_config,
)


class RefreshableModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"
    refreshed_ids = ["remote-a", "remote-b"]

    def __init__(self, *, model_id: str, **params):
        self._model_id = model_id
        self.params = params

    @property
    def model_id(self) -> str:
        return self._model_id

    def _prepare_request_impl(self, request):
        return {}

    def _parse_response_impl(self, response):
        raise NotImplementedError

    def _invoke_impl(self, request):
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return list(self.refreshed_ids)


class HookedInvokeModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self, *, model_id: str, **params):
        super().__init__()
        self._model_id = model_id
        self.params = params

    @property
    def model_id(self) -> str:
        return self._model_id

    def _prepare_request_impl(self, request):
        return {}

    def _parse_response_impl(self, response):
        raise NotImplementedError

    def _invoke_impl(self, request):
        return MessageResponse(
            id="hooked",
            content=[TextPart(type="text", text="ok")],
            stop_reason="end_turn",
            usage=None,
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

    def test_refresh_provider_models_merges_remote_ids(self):
        """Refreshing a provider should add remote model IDs in memory."""
        registry = ModelRegistry()
        registry.clear()
        registry.register_adapter("RefreshableModel", RefreshableModel, quiet=True)
        registry.register_provider(
            "dynamic",
            "RefreshableModel",
            ["local-a"],
            {"api_key": "test-key", "max_context_tokens": 1000},
            quiet=True,
        )

        models = registry.refresh_provider_models("dynamic")

        assert models == [
            "dynamic/local-a",
            "dynamic/remote-a",
            "dynamic/remote-b",
        ]
        assert registry.has_model("dynamic/remote-a")
        config = registry.get_model_config("dynamic/remote-a")
        assert config is not None
        assert config.properties == {"api_key": "test-key"}

    def test_refresh_provider_models_rejects_unknown_provider(self):
        registry = ModelRegistry()
        registry.clear()

        with pytest.raises(Exception, match="Provider 'missing' not found"):
            registry.refresh_provider_models("missing")

    def test_provider_config_overrides_update_provider_properties(self):
        registry = ModelRegistry()
        registry.clear()
        registry.register_provider(
            "editable",
            "OpenAIModel",
            ["local-a"],
            {"api_key": "old-key", "base_url": "https://old.test"},
            quiet=True,
        )

        registry.apply_provider_config_overrides({
            "editable": {
                "api_key": "new-key",
                "timeout": 30,
            }
        })

        config = registry.get_model_config("editable/local-a")

        assert config is not None
        assert config.properties == {
            "api_key": "new-key",
            "base_url": "https://old.test",
            "timeout": 30,
        }

    def test_persist_provider_properties_preserves_yaml_order(self, tmp_path):
        config_path = tmp_path / "models.yaml"
        config_path.write_text(
            "# keep header\n"
            "providers:\n"
            "  - name: first\n"
            "    adapter: OpenAIModel\n"
            "    model_ids:\n"
            "      - a\n"
            "    properties:\n"
            "      api_key: old-first\n"
            "  - name: editable\n"
            "    adapter: OpenAIModel\n"
            "    model_ids:\n"
            "      - b\n"
            "    properties:\n"
            "      api_key: old-key\n"
            "      base_url: https://old.test\n"
            "  - name: last\n"
            "    adapter: OpenAIModel\n"
            "    model_ids:\n"
            "      - c\n"
            "    properties:\n"
            "      api_key: old-last\n",
            encoding="utf-8",
        )

        written = persist_provider_properties(
            "editable",
            {"api_key": "new-key", "timeout": 30},
            [config_path],
        )

        text = config_path.read_text(encoding="utf-8")
        assert written == config_path
        assert text.startswith("# keep header\n")
        assert text.index("name: first") < text.index("name: editable") < text.index("name: last")
        assert text.index("api_key: new-key") < text.index("base_url: https://old.test") < text.index("timeout: 30")
        assert "api_key: old-key" not in text

    def test_persist_provider_properties_uses_first_config_containing_provider(self, tmp_path):
        missing_path = tmp_path / "workspace.yaml"
        target_path = tmp_path / "home.yaml"
        missing_path.write_text(
            "providers:\n"
            "  - name: other\n"
            "    adapter: OpenAIModel\n"
            "    model_ids: [a]\n"
            "    properties:\n"
            "      base_url: https://old.test\n",
            encoding="utf-8",
        )
        target_path.write_text(
            "providers:\n"
            "  - name: editable\n"
            "    adapter: OpenAIModel\n"
            "    model_ids: [b]\n"
            "    properties:\n"
            "      base_url: https://old.test\n",
            encoding="utf-8",
        )

        written = persist_provider_properties(
            "editable",
            {"base_url": "https://new.test"},
            [missing_path, target_path],
        )

        assert written == target_path
        assert "base_url: https://old.test" in missing_path.read_text(encoding="utf-8")
        assert "base_url: https://new.test" in target_path.read_text(encoding="utf-8")


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

    def test_get_model_config_with_steer_merge_mode_override(self):
        """Test that steer merge mode override is applied to model config."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "test-provider",
            "OpenAIModel",
            ["gpt-4"],
            {"api_key": "test"},
            quiet=True,
        )
        registry.register_model_config_override(
            "test-provider/gpt-4",
            {
                "steer_merge_mode": "user_message_template",
            },
            quiet=True,
        )

        config = registry.get_model_config("test-provider/gpt-4")
        assert config is not None
        assert config.steer_merge_mode == "user_message_template"

    def test_wildcard_model_config_matches_multiple_providers(self):
        """Wildcard model configs should match full provider/model names."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4", "o3-mini"],
            {"api_key": "test"},
            quiet=True,
        )
        registry.register_provider(
            "azure",
            "OpenAIModel",
            ["gpt-4o"],
            {"api_key": "test"},
            quiet=True,
        )
        registry.register_model_config_override(
            "*/gpt-*",
            {"temperature": 0.2},
            quiet=True,
        )

        openai_config = registry.get_model_config("openai/gpt-4")
        azure_config = registry.get_model_config("azure/gpt-4o")
        o3_config = registry.get_model_config("openai/o3-mini")

        assert openai_config is not None
        assert azure_config is not None
        assert o3_config is not None
        assert openai_config.properties["temperature"] == 0.2
        assert azure_config.properties["temperature"] == 0.2
        assert "temperature" not in o3_config.properties

    def test_wildcard_model_config_matches_model_ids_with_slashes(self):
        """Wildcard provider/* should match model ids that contain slashes."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "siliconflow",
            "OpenAIModel",
            ["Pro/deepseek-ai/DeepSeek-V3.2"],
            {"api_key": "test"},
            quiet=True,
        )
        registry.register_model_config_override(
            "siliconflow/*",
            {"temperature": 0.3, "max_context_tokens": 64_000},
            quiet=True,
        )

        config = registry.get_model_config("siliconflow/Pro/deepseek-ai/DeepSeek-V3.2")

        assert config is not None
        assert config.properties["temperature"] == 0.3
        assert config.max_context_tokens == 64_000
        assert registry.has_model("siliconflow/Pro/deepseek-ai/DeepSeek-V3.2")

    def test_explicit_registration_suppresses_default_auto_load(self, tmp_path):
        """Programmatic test configs should not merge user/project auto configs."""
        registry = ModelRegistry()
        registry.clear()

        config_dir = tmp_path / ".hawi"
        config_dir.mkdir()
        (config_dir / "models.yaml").write_text("""
providers:
  - name: openai
    adapter: OpenAIModel
    model_ids:
      - gpt-4
    properties:
      api_key: auto-key
      max_context_tokens: 131072
""")

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"api_key": "explicit-key", "max_context_tokens": 32_000},
            quiet=True,
        )

        with patch("hawi.models.registry.Path.cwd", return_value=tmp_path):
            config = registry.get_model_config("openai/gpt-4")

        assert config is not None
        assert config.properties["api_key"] == "explicit-key"
        assert config.max_context_tokens == 32_000

    def test_wildcard_model_configs_merge_in_registration_order(self):
        """Multiple wildcard matches should merge in registration order."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"api_key": "test", "temperature": 0.1},
            quiet=True,
        )
        registry.register_model_config_override(
            "*/*",
            {"temperature": 0.2, "top_p": 0.8},
            quiet=True,
        )
        registry.register_model_config_override(
            "openai/*",
            {"temperature": 0.9},
            quiet=True,
        )

        config = registry.get_model_config("openai/gpt-4")

        assert config is not None
        assert config.properties["temperature"] == 0.9
        assert config.properties["top_p"] == 0.8

    def test_exact_model_config_overrides_wildcard_config(self):
        """Exact model configs should apply after all wildcard configs."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"api_key": "test"},
            quiet=True,
        )
        registry.register_model_config_override(
            "openai/*",
            {"temperature": 0.9, "max_context_tokens": 32_000},
            quiet=True,
        )
        registry.register_model_config_override(
            "openai/gpt-4",
            {"temperature": 0.4, "max_context_tokens": 64_000},
            quiet=True,
        )

        config = registry.get_model_config("openai/gpt-4")

        assert config is not None
        assert config.properties["temperature"] == 0.4
        assert config.max_context_tokens == 64_000

    def test_wildcard_model_config_does_not_add_models(self):
        """Wildcard configs should not create new provider/model entries."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"api_key": "test"},
            quiet=True,
        )
        registry.register_model_config_override(
            "missing/*",
            {"temperature": 0.2},
            quiet=True,
        )

        models = registry.list_models()
        assert "openai/gpt-4" in models
        assert "missing/gpt-4" not in models
        assert registry.get_model_config("missing/gpt-4") is None

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

    def test_create_model_applies_steer_merge_mode_to_instance(self):
        """Test creating model applies steer merge mode without polluting params."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"model_id": "gpt-4", "api_key": "test-key"},
            quiet=True,
        )
        registry.register_model_config_override(
            "openai/gpt-4",
            {
                "steer_merge_mode": "user_message_template",
            },
            quiet=True,
        )

        model = registry.create_model("openai/gpt-4")

        assert isinstance(model, OpenAIModel)
        assert model.get_configured_steer_merge_mode() == "user_message_template"
        assert "steer_merge_mode" not in model.params

    def test_create_model_accepts_steer_merge_mode_override_kwarg(self):
        """Test create_model() can override steer merge mode directly."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"model_id": "gpt-4", "api_key": "test-key"},
            quiet=True,
        )

        model = registry.create_model(
            "openai/gpt-4",
            steer_merge_mode="user_message_template",
        )

        assert isinstance(model, OpenAIModel)
        assert model.get_configured_steer_merge_mode() == "user_message_template"
        assert "steer_merge_mode" not in model.params

    def test_create_model_rejects_invalid_steer_merge_mode_override(self):
        """Invalid steer merge modes should fail before model invocation."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"model_id": "gpt-4", "api_key": "test-key"},
            quiet=True,
        )

        with pytest.raises(ConfigurationError, match="Invalid steer_merge_mode"):
            registry.create_model(
                "openai/gpt-4",
                steer_merge_mode="missing_default",
            )

    def test_create_model_requires_model_declared_steer_merge_mode(self):
        """A concrete model class must declare a mode or be configured."""
        class UndeclaredModel(Model):
            def __init__(self, *, model_id: str, **params):
                super().__init__()
                self._model_id = model_id
                self.params = params

            @property
            def model_id(self) -> str:
                return self._model_id

            def _prepare_request_impl(self, request):
                return {}

            def _parse_response_impl(self, response) -> MessageResponse:
                return MessageResponse(
                    id="undeclared",
                    content=[TextPart(type="text", text="ok")],
                    stop_reason="end_turn",
                    usage=None,
                )

            def _invoke_impl(self, request) -> MessageResponse:
                return self._parse_response_impl({})

        registry = ModelRegistry()
        registry.clear()
        registry.register_adapter("UndeclaredModel", UndeclaredModel, quiet=True)
        registry.register_provider(
            "test",
            "UndeclaredModel",
            ["model"],
            {},
            quiet=True,
        )

        with pytest.raises(ConfigurationError, match="does not declare default_steer_merge_mode"):
            registry.create_model("test/model")

    def test_create_model_applies_max_context_tokens_metadata(self):
        """max_context_tokens should configure Hawi metadata, not provider params."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {
                "model_id": "gpt-4",
                "api_key": "test-key",
                "max_context_tokens": 32_000,
            },
            quiet=True,
        )

        model = registry.create_model("openai/gpt-4")

        assert isinstance(model, OpenAIModel)
        assert model.get_max_context_tokens() == 32_000
        assert "max_context_tokens" not in model.params

    def test_create_model_accepts_max_context_tokens_override_kwarg(self):
        """create_model() should allow CLI/runtime context-window overrides."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_provider(
            "openai",
            "OpenAIModel",
            ["gpt-4"],
            {"model_id": "gpt-4", "api_key": "test-key"},
            quiet=True,
        )
        registry.register_model_config_override(
            "openai/*",
            {"max_context_tokens": 32_000},
            quiet=True,
        )

        model = registry.create_model("openai/gpt-4", max_context_tokens=64_000)

        assert isinstance(model, OpenAIModel)
        assert model.get_max_context_tokens() == 64_000
        assert "max_context_tokens" not in model.params


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
        assert providers is not None
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

    def test_load_config_with_model_steer_merge_mode(self, tmp_path):
        """Test loading config with top-level steer merge mode."""
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

model_configs:
  my-provider/gpt-4:
    steer_merge_mode: user_message_template
""")

        registry.load_config(config_file, quiet=True)

        config = registry.get_model_config("my-provider/gpt-4")
        assert config is not None
        assert config.steer_merge_mode == "user_message_template"

    def test_load_config_with_model_max_context_tokens(self, tmp_path):
        """Test loading Hawi-only max_context_tokens metadata."""
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

model_configs:
  my-provider/gpt-4:
    max_context_tokens: 64000
""")

        registry.load_config(config_file, quiet=True)

        config = registry.get_model_config("my-provider/gpt-4")
        assert config is not None
        assert config.max_context_tokens == 64_000
        assert "max_context_tokens" not in config.properties

        model = registry.create_model("my-provider/gpt-4")
        assert isinstance(model, OpenAIModel)
        assert model.get_max_context_tokens() == 64_000
        assert "max_context_tokens" not in model.params

    def test_load_config_with_model_before_connect_hook(self, tmp_path):
        """Model lifecycle hooks should be Hawi metadata, not provider params."""
        registry = ModelRegistry()
        registry.clear()
        registry.register_adapter("HookedInvokeModel", HookedInvokeModel, quiet=True)

        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
providers:
  - name: my-provider
    adapter: HookedInvokeModel
    model_ids:
      - gpt-4
    properties:
      temperature: 0.5

model_configs:
  "my-provider/*":
    temperature: 0.9
    hooks:
      before_connect: "ember --app-launch --start"
""")

        registry.load_config(config_file, quiet=True)

        config = registry.get_model_config("my-provider/gpt-4")
        assert config is not None
        assert config.properties["temperature"] == 0.9
        assert config.hooks == {
            "before_connect": "ember --app-launch --start",
        }

        model = registry.create_model("my-provider/gpt-4")
        assert isinstance(model, HookedInvokeModel)
        assert "hooks" not in model.params

    def test_before_connect_hook_runs_once_per_model_instance(self, tmp_path):
        """before_connect should run before the first model call only."""
        registry = ModelRegistry()
        registry.clear()
        registry.register_adapter("HookedInvokeModel", HookedInvokeModel, quiet=True)

        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
providers:
  - name: my-provider
    adapter: HookedInvokeModel
    model_ids:
      - gpt-4
    properties: {}

model_configs:
  my-provider/gpt-4:
    hooks:
      before_connect: "ember --app-launch --start"
""")
        registry.load_config(config_file, quiet=True)
        model = registry.create_model("my-provider/gpt-4")
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        message: Message = {
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
            "name": None,
            "metadata": None,
        }
        with patch("hawi.models.model.subprocess.run", side_effect=fake_run):
            model.invoke([message])
            model.invoke([message])

        assert [call[0] for call in calls] == ["ember --app-launch --start"]
        assert calls[0][1]["shell"] is True
        assert calls[0][1]["capture_output"] is True

    def test_provider_before_connect_hook_runs_before_refresh(self):
        """Provider hooks should run before querying remote model lists."""
        registry = ModelRegistry()
        registry.clear()
        registry.register_adapter("RefreshableModel", RefreshableModel, quiet=True)
        registry.register_provider(
            "my-provider",
            "RefreshableModel",
            ["seed"],
            {},
            hooks={"before_connect": "ember --app-launch --start"},
            quiet=True,
        )
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch("hawi.models.model.subprocess.run", side_effect=fake_run):
            models = registry.refresh_provider_models("my-provider")

        assert calls[0][0] == "ember --app-launch --start"
        assert models == ["my-provider/seed", "my-provider/remote-a", "my-provider/remote-b"]

    def test_load_config_with_wildcard_model_configs(self, tmp_path):
        """Test loading wildcard model_configs from YAML in declaration order."""
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
      temperature: 0.1

model_configs:
  "*/*":
    temperature: 0.2
    top_p: 0.8
  "my-provider/*":
    temperature: 0.9
    max_context_tokens: 32000
  my-provider/gpt-4:
    temperature: 0.4
""")

        registry.load_config(config_file, quiet=True)

        config = registry.get_model_config("my-provider/gpt-4")
        assert config is not None
        assert config.properties["temperature"] == 0.4
        assert config.properties["top_p"] == 0.8
        assert config.max_context_tokens == 32_000

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

    def test_auto_load_uses_git_root_from_nested_cwd(self, tmp_path):
        """Test auto-loading searches the Git root when cwd is nested."""
        registry = ModelRegistry()
        registry.clear()

        repo = tmp_path / "repo"
        nested = repo / "src" / "package"
        config_dir = repo / ".hawi"
        nested.mkdir(parents=True)
        (repo / ".git").mkdir()
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

        with patch("hawi.models.registry.Path.cwd", return_value=nested):
            assert registry.has_model("auto-provider/gpt-4")


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
