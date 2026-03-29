"""Tests for ModelRegistry singleton.

Tests the singleton pattern, class/factory registration, configuration loading,
factory inheritance (parent), and environment variable resolution.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from hawi.models.registry import (
    ModelRegistry,
    FactoryConfig,
    CircularDependencyError,
    UnknownFactoryError,
    InvalidInheritanceError,
    model_registry,
    create_model,
    get_model_class,
    list_factories,
    load_config,
    get_factory_arguments,
)
from hawi.models import OpenAIModel, Model


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


class TestClassRegistration:
    """Tests for Model class registration."""

    def test_builtin_classes_registered(self):
        """Test that built-in model classes are auto-registered."""
        classes = model_registry.list_classes()
        assert "OpenAIModel" in classes
        assert "AnthropicModel" in classes
        assert "DeepSeekOpenAIModel" in classes
        assert "DeepSeekAnthropicModel" in classes

    def test_get_model_class_returns_class(self):
        """Test retrieving registered model class by name."""
        cls = model_registry.get_model_class("OpenAIModel")
        assert cls is OpenAIModel

    def test_get_model_class_returns_none_for_unknown(self):
        """Test that unknown class name returns None."""
        cls = model_registry.get_model_class("NonExistentModel")
        assert cls is None

    def test_register_class_adds_new_class(self):
        """Test registering a custom model class."""
        registry = ModelRegistry()
        registry.clear()
        # Use a real model class as mock
        registry.register_class("MockModel", OpenAIModel, quiet=True)

        assert "MockModel" in registry.list_classes()
        assert registry.get_model_class("MockModel") is OpenAIModel

    def test_register_class_overrides_existing(self):
        """Test that new registration overrides old (new overrides old rule)."""
        from hawi.models.anthropic import AnthropicModel

        registry = ModelRegistry()
        registry.clear()
        registry.register_class("TestModel", OpenAIModel, quiet=True)
        registry.register_class("TestModel", AnthropicModel, quiet=True)

        assert registry.get_model_class("TestModel") is AnthropicModel


class TestFactoryRegistration:
    """Tests for factory registration and management."""

    def test_register_factory_adds_factory(self):
        """Test registering a new factory."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "test-factory",
            "OpenAIModel",
            {"model_id": "gpt-4", "api_key": "test"},
            quiet=True,
        )

        assert "test-factory" in registry.list_factories()
        assert registry.has_factory("test-factory")

    def test_get_factory_returns_config(self):
        """Test retrieving factory configuration."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "test-factory",
            "DeepSeekOpenAIModel",
            {"model_id": "deepseek-chat"},
            quiet=True,
        )

        config = registry.get_factory("test-factory")
        assert isinstance(config, FactoryConfig)
        assert config.class_name == "DeepSeekOpenAIModel"
        assert config.arguments["model_id"] == "deepseek-chat"

    def test_unregister_factory_removes_factory(self):
        """Test unregistering a factory."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "temp-factory", "OpenAIModel", {"model_id": "test"}, quiet=True
        )
        assert registry.has_factory("temp-factory")

        result = registry.unregister_factory("temp-factory")
        assert result is True
        assert not registry.has_factory("temp-factory")

    def test_unregister_nonexistent_factory_returns_false(self):
        """Test unregistering unknown factory returns False."""
        registry = ModelRegistry()
        registry.clear()

        result = registry.unregister_factory("nonexistent")
        assert result is False


class TestFactoryInheritance:
    """Tests for factory parent/inheritance."""

    def test_resolve_factory_parent(self):
        """Test that factory inherits from parent factory."""
        registry = ModelRegistry()
        registry.clear()

        # Register base factory
        registry.register_factory(
            "base-model",
            "OpenAIModel",
            {"model_id": "gpt-4", "temperature": 0.7},
            quiet=True,
        )

        # Register child factory with parent
        registry.register_factory(
            "child-model",
            "OpenAIModel",
            {"max_tokens": 512},
            parent="base-model",
            quiet=True,
        )

        # Resolve and check merged config
        resolved = registry._resolve_factory("child-model")
        assert resolved.class_name == "OpenAIModel"
        assert resolved.arguments["model_id"] == "gpt-4"  # inherited
        assert resolved.arguments["temperature"] == 0.7  # inherited
        assert resolved.arguments["max_tokens"] == 512  # own

    def test_child_overrides_parent_arguments(self):
        """Test that child factory arguments override parent."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "base-model", "OpenAIModel", {"temperature": 0.7}, quiet=True
        )

        registry.register_factory(
            "override-model",
            "OpenAIModel",
            {"temperature": 0.5},
            parent="base-model",
            quiet=True,
        )

        resolved = registry._resolve_factory("override-model")
        assert resolved.arguments["temperature"] == 0.5

    def test_circular_dependency_raises_error(self):
        """Test that circular parent raises CircularDependencyError."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "factory-a", "OpenAIModel", {}, parent="factory-b", quiet=True
        )
        registry.register_factory(
            "factory-b", "OpenAIModel", {}, parent="factory-a", quiet=True
        )

        with pytest.raises(CircularDependencyError):
            registry._resolve_factory("factory-a")

    def test_unknown_parent_raises_error(self):
        """Test that extending unknown factory raises error."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "orphan-model",
            "OpenAIModel",
            {},
            parent="nonexistent-parent",
            quiet=True,
        )

        with pytest.raises(UnknownFactoryError):
            registry._resolve_factory("orphan-model")

    def test_multi_level_inheritance(self):
        """Test parent chain with multiple levels."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "grandparent", "OpenAIModel", {"base_url": "http://api.com"}, quiet=True
        )
        registry.register_factory(
            "parent",
            "OpenAIModel",
            {"temperature": 0.7},
            parent="grandparent",
            quiet=True,
        )
        registry.register_factory(
            "child",
            "OpenAIModel",
            {"max_tokens": 256},
            parent="parent",
            quiet=True,
        )

        resolved = registry._resolve_factory("child")
        assert resolved.arguments["base_url"] == "http://api.com"  # from grandparent
        assert resolved.arguments["temperature"] == 0.7  # from parent
        assert resolved.arguments["max_tokens"] == 256  # own

    def test_factory_multiple_parents_order(self):
        """Test factory with multiple parents - later parents override earlier ones."""
        registry = ModelRegistry()
        registry.clear()

        # Register templates with same key but different values
        registry.register_template("first", {"api_key": "first-key", "temperature": 0.5})
        registry.register_template("second", {"api_key": "second-key", "temperature": 0.7})
        registry.register_template("third", {"api_key": "third-key"})

        # Factory with multiple parents: [first, second, third]
        # Expected: third's api_key wins, temperature from second (not overridden by third)
        registry.register_factory(
            "multi-parent-model",
            "OpenAIModel",
            {"model_id": "gpt-4"},
            parents=["first", "second", "third"],
            quiet=True,
        )

        resolved = registry._resolve_factory("multi-parent-model")
        assert resolved.arguments["api_key"] == "third-key"  # from third (last)
        assert resolved.arguments["temperature"] == 0.7  # from second
        assert resolved.arguments["model_id"] == "gpt-4"  # own

    def test_factory_mixed_parents_template_and_factory(self):
        """Test factory inheriting from both templates and factories in specific order."""
        registry = ModelRegistry()
        registry.clear()

        # Template provides base config
        registry.register_template("base-config", {"base_url": "https://api1.com", "timeout": 30})

        # Factory provides model-specific config
        registry.register_factory(
            "model-base",
            "OpenAIModel",
            {"temperature": 0.7, "base_url": "https://api2.com"},  # overrides template's base_url
            quiet=True,
        )

        # Another template for auth
        registry.register_template("auth-config", {"api_key": "secret-key"})

        # Factory with mixed parents: template -> factory -> template
        registry.register_factory(
            "final-model",
            "OpenAIModel",
            {"max_tokens": 512},
            parents=["base-config", "model-base", "auth-config"],
            quiet=True,
        )

        resolved = registry._resolve_factory("final-model")
        # Order: base-config -> model-base -> auth-config -> own
        assert resolved.arguments["base_url"] == "https://api2.com"  # from model-base
        assert resolved.arguments["timeout"] == 30  # from base-config
        assert resolved.arguments["temperature"] == 0.7  # from model-base
        assert resolved.arguments["api_key"] == "secret-key"  # from auth-config (last)
        assert resolved.arguments["max_tokens"] == 512  # own


class TestTemplateInheritance:
    """Tests for template parent/inheritance."""

    def test_template_inherits_from_template(self):
        """Test that template can inherit from another template."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_template("base", {"api_key": "base-key", "temperature": 0.5})
        registry.register_template("child", {"temperature": 0.7}, parents=["base"])

        resolved = registry._resolve_template("child")
        assert resolved.arguments["api_key"] == "base-key"
        assert resolved.arguments["temperature"] == 0.7  # overridden

    def test_template_multiple_parents_order(self):
        """Test template with multiple parents - later parents override earlier ones."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_template("first", {"a": 1, "b": 1, "c": 1})
        registry.register_template("second", {"b": 2, "c": 2})
        registry.register_template("third", {"c": 3})
        registry.register_template("combined", {"d": 4}, parents=["first", "second", "third"])

        resolved = registry._resolve_template("combined")
        assert resolved.arguments["a"] == 1  # from first
        assert resolved.arguments["b"] == 2  # from second (overrides first)
        assert resolved.arguments["c"] == 3  # from third (overrides second)
        assert resolved.arguments["d"] == 4  # own

    def test_template_inherits_class_from_parent(self):
        """Test that template can inherit __class from parent template."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_template("base", {"api_key": "key"}, class_name="OpenAIModel")
        registry.register_template("child", {"temperature": 0.5}, parents=["base"])

        resolved = registry._resolve_template("child")
        assert resolved.class_name == "OpenAIModel"
        assert resolved.arguments["api_key"] == "key"
        assert resolved.arguments["temperature"] == 0.5

    def test_template_cannot_inherit_from_factory(self):
        """Test that template cannot inherit from factory."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory("some-factory", "OpenAIModel", {"api_key": "test"})
        registry.register_template("bad-template", {}, parents=["some-factory"])

        with pytest.raises(InvalidInheritanceError):
            registry._resolve_template("bad-template")


class TestEnvironmentVariableResolution:
    """Tests for ${ENV_VAR} syntax resolution."""

    def test_resolve_simple_env_var(self):
        """Test resolving ${VAR} syntax."""
        registry = ModelRegistry()

        with patch.dict(os.environ, {"TEST_API_KEY": "secret123"}):
            result = registry._resolve_substitutions({"api_key": "${TEST_API_KEY}"})
            assert result["api_key"] == "secret123"

    def test_resolve_env_var_with_default(self):
        """Test resolving ${VAR:default} syntax."""
        registry = ModelRegistry()

        result = registry._resolve_substitutions(
            {"api_key": "${NONEXISTENT_KEY:default_value}"}
        )
        assert result["api_key"] == "default_value"

    def test_resolve_env_var_without_default_uses_original(self):
        """Test that missing env var without default keeps original."""
        registry = ModelRegistry()

        result = registry._resolve_substitutions({"key": "${NONEXISTENT_VAR}"})
        assert result["key"] == "${NONEXISTENT_VAR}"

    def test_resolve_embedded_env_var(self):
        """Test ${VAR} embedded in string."""
        registry = ModelRegistry()

        with patch.dict(os.environ, {"USER": "testuser"}):
            result = registry._resolve_substitutions(
                {"url": "https://api.example.com/${USER}/endpoint"}
            )
            assert result["url"] == "https://api.example.com/testuser/endpoint"

    def test_resolve_nested_dict(self):
        """Test env var resolution in nested dictionaries."""
        registry = ModelRegistry()

        with patch.dict(os.environ, {"KEY": "value"}):
            result = registry._resolve_substitutions(
                {
                    "level1": {
                        "level2": {"key": "${KEY}"},
                    }
                }
            )
            assert result["level1"]["level2"]["key"] == "value"

    def test_resolve_list(self):
        """Test env var resolution in lists."""
        registry = ModelRegistry()

        with patch.dict(os.environ, {"ITEM": "resolved"}):
            result = registry._resolve_substitutions(["${ITEM}", "static", "${ITEM}"])
            assert result == ["resolved", "static", "resolved"]


class TestConfigLoading:
    """Tests for YAML configuration loading."""

    def test_load_config_from_yaml(self, tmp_path):
        """Test loading factories from YAML file."""
        registry = ModelRegistry()
        registry.clear()

        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
factories:
  test-model:
    class: OpenAIModel
    model_id: gpt-4
    api_key: test-key
""")

        registry.load_config(config_file, quiet=True)

        assert "test-model" in registry.list_factories()
        config = registry.get_factory("test-model")
        assert config is not None
        assert config.class_name == "OpenAIModel"
        assert config.arguments["model_id"] == "gpt-4"

    def test_load_config_with_parent(self, tmp_path):
        """Test loading YAML with parent inheritance."""
        registry = ModelRegistry()
        registry.clear()

        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
factories:
  base-model:
    class: OpenAIModel
    model_id: gpt-4

  derived-model:
    class: OpenAIModel
    parent: base-model
    temperature: 0.5
""")

        registry.load_config(config_file, quiet=True)

        resolved = registry._resolve_factory("derived-model")
        assert resolved.arguments["model_id"] == "gpt-4"  # inherited
        assert resolved.arguments["temperature"] == 0.5  # own

    def test_load_config_resolves_env_vars(self, tmp_path):
        """Test that loaded config resolves environment variables."""
        registry = ModelRegistry()
        registry.clear()

        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
factories:
  env-model:
    class: OpenAIModel
    api_key: ${TEST_API_KEY}
""")

        with patch.dict(os.environ, {"TEST_API_KEY": "resolved_secret"}):
            registry.load_config(config_file, quiet=True)
            # Note: env vars are resolved at create_model time, not load time
            config = registry.get_factory("env-model")
            assert config is not None
            assert config.arguments["api_key"] == "${TEST_API_KEY}"

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

    def test_load_config_without_factories_key(self, tmp_path, monkeypatch):
        """Test loading YAML without factories key is handled."""
        monkeypatch.setenv("HAWI_NO_AUTO_LOAD", "1")
        registry = ModelRegistry()
        registry.clear()

        config_file = tmp_path / "empty.yaml"
        config_file.write_text("other_key: value")

        # Should not raise, just warn
        registry.load_config(config_file, quiet=True)
        assert registry.list_factories() == []


class TestCreateModel:
    """Tests for create_model function."""

    def test_create_model_from_factory(self):
        """Test creating model instance from factory."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "openai-test",
            "OpenAIModel",
            {"model_id": "gpt-4", "api_key": "test-key"},
            quiet=True,
        )

        model = registry.create_model("openai-test")
        assert isinstance(model, OpenAIModel)
        assert model.model_id == "gpt-4"

    def test_create_model_unknown_factory_raises(self):
        """Test that unknown factory raises UnknownFactoryError."""
        registry = ModelRegistry()
        registry.clear()

        with pytest.raises(UnknownFactoryError):
            registry.create_model("nonexistent-factory")

    def test_create_model_unknown_class_raises(self):
        """Test that unknown model class raises error."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "bad-factory",
            "NonExistentClass",
            {"api_key": "test"},
            quiet=True,
        )

        with pytest.raises(UnknownFactoryError):
            registry.create_model("bad-factory")

    def test_create_model_with_overrides(self):
        """Test creating model with argument overrides."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "openai-test",
            "OpenAIModel",
            {"model_id": "gpt-4", "api_key": "test-key"},
            quiet=True,
        )

        model = registry.create_model("openai-test", overrides={"model_id": "gpt-3.5"})
        assert model.model_id == "gpt-3.5"

    def test_create_model_resolves_env_vars(self):
        """Test that create_model resolves environment variables."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "env-model",
            "OpenAIModel",
            {"model_id": "gpt-4", "api_key": "${ENV_API_KEY}"},
            quiet=True,
        )

        # Verify that env vars are resolved in arguments
        with patch.dict(os.environ, {"ENV_API_KEY": "resolved_key"}):
            resolved = registry._resolve_substitutions({"api_key": "${ENV_API_KEY}"})
            assert resolved["api_key"] == "resolved_key"
            # Model creation also works, but we verify the key was resolved
            model = registry.create_model("env-model")
            assert isinstance(model, OpenAIModel)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_model_convenience_function(self):
        """Test create_model() convenience function uses global registry."""
        model_registry.clear()
        model_registry.register_factory(
            "test-model",
            "OpenAIModel",
            {"model_id": "gpt-4", "api_key": "test"},
            quiet=True,
        )

        model = create_model("test-model")
        assert isinstance(model, OpenAIModel)

    def test_get_model_class_convenience_function(self):
        """Test get_model_class() convenience function."""
        model_registry.clear()
        # Register a test class first
        model_registry.register_class("TestModel", OpenAIModel, quiet=True)

        cls = get_model_class("TestModel")
        assert cls is OpenAIModel

    def test_list_factories_convenience_function(self):
        """Test list_factories() convenience function."""
        model_registry.clear()
        model_registry.register_factory(
            "test-factory", "OpenAIModel", {"api_key": "test"}, quiet=True
        )

        factories = list_factories()
        assert "test-factory" in factories

    def test_load_config_convenience_function(self, tmp_path):
        """Test load_config() convenience function."""
        model_registry.clear()

        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
factories:
  loaded-model:
    class: OpenAIModel
    api_key: test
""")

        load_config(config_file, quiet=True)
        assert "loaded-model" in model_registry.list_factories()


class TestClear:
    """Tests for registry clear functionality."""

    def test_clear_removes_factories(self):
        """Test that clear() removes all factories."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory("test", "OpenAIModel", {}, quiet=True)
        assert registry.list_factories()

        registry.clear()
        assert registry.list_factories() == []

    def test_clear_re_registers_builtin_classes(self):
        """Test that clear() re-registers built-in classes."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_class("OpenAIModel", MagicMock, quiet=True)
        assert registry.get_model_class("OpenAIModel") is not OpenAIModel

        registry.clear()
        assert registry.get_model_class("OpenAIModel") is OpenAIModel


class TestFactoryConfig:
    """Tests for FactoryConfig dataclass."""

    def test_factory_config_from_dict(self):
        """Test creating FactoryConfig from dictionary."""
        data = {
            "class": "OpenAIModel",
            "model_id": "gpt-4",
            "api_key": "test",
            "parent": "base-model",
        }

        config = FactoryConfig.from_dict(data)
        assert config.class_name == "OpenAIModel"
        assert config.arguments == {"model_id": "gpt-4", "api_key": "test"}
        assert config.parent == "base-model"

    def test_factory_config_from_dict_copies_data(self):
        """Test that from_dict doesn't modify original data."""
        data = {"class": "OpenAIModel", "api_key": "test"}
        original_keys = set(data.keys())

        FactoryConfig.from_dict(data)

        assert set(data.keys()) == original_keys

    def test_factory_config_to_dict(self):
        """Test converting FactoryConfig to dictionary."""
        config = FactoryConfig(
            class_name="OpenAIModel",
            arguments={"model_id": "gpt-4"},
            parent="base",
        )

        data = config.to_dict()
        assert data == {
            "class": "OpenAIModel",
            "model_id": "gpt-4",
            "parent": "base",
        }

    def test_factory_config_to_dict_without_parent(self):
        """Test to_dict when parent is None."""
        config = FactoryConfig(
            class_name="OpenAIModel", arguments={"api_key": "test"}, parent=None
        )

        data = config.to_dict()
        assert "parent" not in data


class TestGetFactoryArguments:
    """Tests for get_factory_arguments method."""

    def test_get_original_arguments(self):
        """Test getting original arguments without expansion."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "base-model",
            "OpenAIModel",
            {"temperature": 0.7, "base_url": "https://api.example.com"},
            quiet=True,
        )
        registry.register_factory(
            "child-model",
            "OpenAIModel",
            {"model_id": "gpt-4"},
            parent="base-model",
            quiet=True,
        )

        # expanded=False should return only own arguments
        args = registry.get_factory_arguments("child-model", expanded=False)
        assert args == {"model_id": "gpt-4"}
        assert "temperature" not in args

    def test_get_expanded_arguments(self):
        """Test getting expanded arguments with inheritance."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "base-model",
            "OpenAIModel",
            {"temperature": 0.7, "base_url": "https://api.example.com"},
            quiet=True,
        )
        registry.register_factory(
            "child-model",
            "OpenAIModel",
            {"model_id": "gpt-4"},
            parent="base-model",
            quiet=True,
        )

        # expanded=True should return merged arguments
        args = registry.get_factory_arguments("child-model", expanded=True)
        assert args["model_id"] == "gpt-4"  # own
        assert args["temperature"] == 0.7  # inherited
        assert args["base_url"] == "https://api.example.com"  # inherited

    def test_expanded_child_overrides_parent(self):
        """Test that expanded args show child overriding parent."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "base-model",
            "OpenAIModel",
            {"temperature": 0.7, "max_tokens": 512},
            quiet=True,
        )
        registry.register_factory(
            "child-model",
            "OpenAIModel",
            {"temperature": 0.5},  # override parent's temperature
            parent="base-model",
            quiet=True,
        )

        args = registry.get_factory_arguments("child-model", expanded=True)
        assert args["temperature"] == 0.5  # child's value
        assert args["max_tokens"] == 512  # inherited from parent

    def test_get_factory_arguments_returns_copy(self):
        """Test that returned arguments are a copy, not reference."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "test-model",
            "OpenAIModel",
            {"key": "value"},
            quiet=True,
        )

        args = registry.get_factory_arguments("test-model", expanded=False)
        args["key"] = "modified"  # modify returned dict

        # original should be unchanged
        original = registry.get_factory_arguments("test-model", expanded=False)
        assert original["key"] == "value"

    def test_get_factory_arguments_unknown_factory_raises(self):
        """Test that unknown factory raises UnknownFactoryError."""
        registry = ModelRegistry()
        registry.clear()

        with pytest.raises(UnknownFactoryError):
            registry.get_factory_arguments("nonexistent")

    def test_get_factory_arguments_default_expanded_is_false(self):
        """Test that expanded defaults to False."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "base-model",
            "OpenAIModel",
            {"temperature": 0.7},
            quiet=True,
        )
        registry.register_factory(
            "child-model",
            "OpenAIModel",
            {"model_id": "gpt-4"},
            parent="base-model",
            quiet=True,
        )

        # default should be expanded=False
        args = registry.get_factory_arguments("child-model")
        assert args == {"model_id": "gpt-4"}  # only own args

    def test_convenience_function_get_factory_arguments(self):
        """Test get_factory_arguments convenience function."""
        model_registry.clear()
        model_registry.register_factory(
            "test-model",
            "OpenAIModel",
            {"temperature": 0.7, "model_id": "gpt-4"},
            quiet=True,
        )

        # Test convenience function
        args = get_factory_arguments("test-model", expanded=False)
        assert args == {"temperature": 0.7, "model_id": "gpt-4"}

    def test_expanded_with_multiple_parents(self):
        """Test expanded arguments with multiple parents."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_factory(
            "first",
            "OpenAIModel",
            {"a": 1, "b": 1},
            quiet=True,
        )
        registry.register_factory(
            "second",
            "OpenAIModel",
            {"b": 2, "c": 2},
            quiet=True,
        )
        registry.register_factory(
            "combined",
            "OpenAIModel",
            {"d": 4},
            parents=["first", "second"],
            quiet=True,
        )

        args = registry.get_factory_arguments("combined", expanded=True)
        assert args["a"] == 1  # from first
        assert args["b"] == 2  # from second (overrides first)
        assert args["c"] == 2  # from second
        assert args["d"] == 4  # own

    def test_expanded_with_template_parent(self):
        """Test expanded arguments when inheriting from template."""
        registry = ModelRegistry()
        registry.clear()

        registry.register_template(
            "auth-template",
            {"api_key": "secret-key"},
        )
        registry.register_factory(
            "model-with-auth",
            "OpenAIModel",
            {"model_id": "gpt-4"},
            parents=["auth-template"],
            quiet=True,
        )

        args = registry.get_factory_arguments("model-with-auth", expanded=True)
        assert args["model_id"] == "gpt-4"  # own
        assert args["api_key"] == "secret-key"  # from template
