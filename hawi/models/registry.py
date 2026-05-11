"""Model Registry - 单例模式，管理 Model 类和 Model 注册。

设计原则：
- 单例模式：全局唯一实例
- 职责：Model 类注册表 + Model 注册表 + Provider 模板注册表
- 无对象池：每次 create_model 创建新实例
- 自动配置加载：首次使用时自动加载默认配置路径

配置格式（新格式）：
providers: 服务提供商配置列表
  - name: provider 名称
    adapter: Model 类名
    parent: 继承的 provider（可选）
    model_ids: 支持的模型 ID 列表
    properties: 附加属性参数集
      field1: value1
      field2: value2
      ...
model_configs: 模型特定配置
  provider_name/model_id: 配置项（也支持 Shell glob 通配符，如 "provider/*"）
    field1: value1
    field2: value2
    steer_merge_mode: tool_result_assistant_template_and_user_message
"""

from __future__ import annotations

import os
import re
from fnmatch import fnmatchcase
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union

from pydantic import BaseModel

from hawi.models.model import Model
from hawi.utils.config_loader import ConfigLoaderError, load_config_file

__all__ = [
    "ModelRegistry",
    "model_registry",
    "create_model",
    "get_model_adapter",
    "get_model_config",
    "load_config",
    "list_models",
    "list_providers",
    "refresh_provider_models",
    "ModelConfig",
    "ModelOverrideConfig",
    "CircularDependencyError",
    "UnknownModelError",
    "UnknownTemplateError",
    "InvalidInheritanceError",
]


class CircularDependencyError(Exception):
    """Model/Template 循环继承错误"""
    pass


class UnknownModelError(Exception):
    """未知 Model 错误"""
    pass


class UnknownTemplateError(Exception):
    """未知 Template 错误"""
    pass


class InvalidInheritanceError(Exception):
    """无效的继承关系错误"""
    pass

class ModelProviderConfig(BaseModel):
    """Provider 配置对象"""
    name: str
    adapter: str
    model_ids: list[str]
    properties: dict[str,Any]

class ModelOverrideConfig(BaseModel):
    provider: str
    model_id: str
    properties: dict[str,Any]
    steer_merge_mode: str | None = None
    max_context_tokens: int | None = None

class ModelConfig(BaseModel):
    adapter: str
    properties: dict[str, Any]
    steer_merge_mode: str | None = None
    max_context_tokens: int | None = None

class ModelRegistry:
    """Model Registry 单例类

    管理：
    1. Model 类注册表（类名 -> Model 类）
    2. Model Provider 注册（provider名 -> ModelProviderConfig）

    使用方式：
        from hawi.models import model_registry

        # 获取 Model 类
        cls = model_registry.get_model_adapter("DeepSeekOpenAIModel")

        # 创建模型实例
        model = model_registry.create_model("deepseek-chat")

        # 手动加载配置
        model_registry.load_config(Path("/path/to/custom/models.yaml"))
    """

    _instance: Optional[ModelRegistry] = None
    _lock: Lock = Lock()

    def __new__(cls) -> ModelRegistry:
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """初始化（仅执行一次）"""
        self._classes: dict[str, type[Model]] = {}
        self._providers: list[ModelProviderConfig] = []
        self._provider_groups: dict[str, list[ModelProviderConfig]] = {}
        self._model_config_overrides: dict[str, ModelOverrideConfig] = {}
        self._model_config_wildcard_overrides: list[ModelOverrideConfig] = []

        self._auto_load_needed: bool = True

        # 注册内置 Model 类
        self._register_builtin_classes()

    def _register_builtin_classes(self) -> None:
        """注册 Hawi 内置的 Model 类"""
        # 延迟导入避免循环依赖
        from hawi.models.anthropic import AnthropicModel
        from hawi.models.deepseek import (
            DeepSeekAnthropicModel,
            DeepSeekModel,
            DeepSeekOpenAIModel,
        )
        from hawi.models.kimi import KimiAnthropicModel, KimiModel, KimiOpenAIModel
        from hawi.models.minimax import (
            MiniMaxAnthropicModel,
            MiniMaxModel,
            MiniMaxOpenAIModel,
        )
        from hawi.models.openai import OpenAIModel
        from hawi.models.strands import StrandsModel

        builtin_classes = [
            OpenAIModel,
            AnthropicModel,
            DeepSeekModel,
            DeepSeekOpenAIModel,
            DeepSeekAnthropicModel,
            KimiModel,
            KimiOpenAIModel,
            KimiAnthropicModel,
            MiniMaxModel,
            MiniMaxOpenAIModel,
            MiniMaxAnthropicModel,
            StrandsModel,
        ]

        for cls in builtin_classes:
            self._classes[cls.__name__] = cls

    # ========================================================================
    # Model 类管理
    # ========================================================================

    def register_adapter(
        self, name: str, cls: type[Model], quiet: bool = False
    ) -> None:
        """注册 Model 类

        Args:
            name: 类名（用于 model 配置中的 class 字段）
            cls: Model 类
            quiet: 是否静默（不输出覆盖日志）

        Note:
            如果类名已存在，新注册的类会覆盖旧的。
        """
        if name in self._classes and not quiet:
            existing = self._classes[name]
            print(
                f"[ModelRegistry] Model class '{name}' overridden: "
                f"{existing.__module__} -> {cls.__module__}"
            )
        self._classes[name] = cls
    
    def get_model_adapter(self, name: str) -> Optional[type[Model]]:
        """通过类名获取 Model 类"""
        return self._classes.get(name)

    def list_model_adapters(self) -> list[str]:
        """列出所有已注册的类名"""
        return list(self._classes.keys())

    # ========================================================================
    # Template 管理
    # ========================================================================

    def register_provider(
        self,
        name: str,
        adapter: str,
        model_ids: list[str],
        properties: dict[str,Any],
        quiet: bool = False,
    ) -> None:
        """注册 Template

        Args:
            name: Provider 名称
            adapter: Model Class 名称
            model_ids: provider支持的model ids
            properties: provider创建Model时附带的参数
            quiet: 是否静默（不输出覆盖警告）
        """
        self._auto_load_needed = False
        provider = ModelProviderConfig(
            name=name,
            adapter=adapter,
            model_ids=model_ids,
            properties=properties,
        )
        self._register_provider(provider)
    
    def _register_provider(self, provider:ModelProviderConfig, quiet: bool = False):
        # Initialize provider group if not exists
        if provider.name not in self._provider_groups:
            self._provider_groups[provider.name] = []
        
        # Check for duplicate model_ids in existing providers
        existing_model_ids = []
        for existing_provider in self._provider_groups[provider.name]:
            for model_id in provider.model_ids:
                if model_id in existing_provider.model_ids:
                    existing_model_ids.append(model_id)
        if existing_model_ids and not quiet:
            print(f"[ModelRegistry] Model Ids ({','.join(existing_model_ids)}) of provider '{provider.name}' overridden")
        
        # Remove duplicate model_ids from existing providers (before adding new provider)
        for existing_provider in self._provider_groups[provider.name]:
            existing_provider.model_ids = [i for i in existing_provider.model_ids if i not in provider.model_ids]
        
        # Now add the new provider
        self._providers.append(provider)
        self._provider_groups[provider.name].append(provider)

    def unregister_provider(self, name: str) -> bool:
        """注销 Template"""
        if name in self._provider_groups:
            for p in self._provider_groups[name]:
                self._providers.remove(p)
            del self._provider_groups[name]
            return True
        return False

    def get_provider(self, name: str) -> list[ModelProviderConfig] | None:
        """获取 Template 配置"""
        return self._provider_groups.get(name)

    def list_providers(self) -> list[str]:
        """列出所有已注册的 provider 名称"""
        return list(p.name for p in self._providers)

    def has_provider(self, name: str) -> bool:
        """检查 provider 是否存在"""
        return name in self._provider_groups

    # ========================================================================
    # Model 管理
    # ========================================================================

    def register_model_config_override(
        self,
        name: str,
        properties: dict[str, Any],
        quiet: bool = False,
    ) -> None:
        """注册 Model

        Args:
            name: Model 名称 (Provider/ModelId)
            properties: 实例化参数
            quiet: 是否静默（不输出覆盖警告）
        """
        provider, model_id = self._split_model_name(name)
        self._auto_load_needed = False

        is_wildcard = self._is_wildcard_model_config_name(name)
        if not is_wildcard and name in self._model_config_overrides and not quiet:
            print(f"[ModelRegistry] Model '{name}' overridden")

        normalized_properties, steer_merge_mode, max_context_tokens = self._normalize_model_override(properties)

        override = ModelOverrideConfig(
            provider=provider,
            model_id=model_id,
            properties=normalized_properties,
            steer_merge_mode=steer_merge_mode,
            max_context_tokens=max_context_tokens,
        )
        if is_wildcard:
            self._model_config_wildcard_overrides.append(override)
        else:
            self._model_config_overrides[name] = override

    @staticmethod
    def _split_model_name(name: str) -> tuple[str, str]:
        """Split provider/model while allowing slashes inside model ids."""
        if "/" not in name:
            raise ValueError("Model config name must use provider/model format")
        provider, model_id = name.split("/", 1)
        if not provider or not model_id:
            raise ValueError("Model config name must use provider/model format")
        return provider, model_id

    @staticmethod
    def _is_wildcard_model_config_name(name: str) -> bool:
        """Whether a model config name uses shell-style glob syntax."""
        return any(ch in name for ch in "*?[")

    def _normalize_model_override(
        self,
        override: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None, int | None]:
        """Split a model override into provider params and Hawi metadata."""
        if (
            "properties" not in override
            and "steer_merge_mode" not in override
            and "max_context_tokens" not in override
        ):
            properties = dict(override)
            max_context_tokens = self._pop_max_context_tokens(properties)
            return properties, None, max_context_tokens

        properties = dict(override.get("properties") or {})
        steer_merge_mode = Model.coerce_steer_merge_mode(
            override.get("steer_merge_mode"),
            source="models.yaml model_configs.steer_merge_mode",
        )
        max_context_tokens = self._coerce_max_context_tokens(
            override.get("max_context_tokens")
        )
        for key, value in override.items():
            if key not in {"properties", "steer_merge_mode", "max_context_tokens"}:
                properties[key] = value
        if max_context_tokens is None:
            max_context_tokens = self._pop_max_context_tokens(properties)
        return properties, steer_merge_mode, max_context_tokens

    @staticmethod
    def _coerce_max_context_tokens(value: Any) -> int | None:
        """Normalize max_context_tokens values from YAML or CLI overrides."""
        if value is None:
            return None
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            raise ValueError("max_context_tokens must be a positive integer")
        if tokens <= 0:
            raise ValueError("max_context_tokens must be a positive integer")
        return tokens

    def _pop_max_context_tokens(self, properties: dict[str, Any]) -> int | None:
        """Remove Hawi context metadata from provider params."""
        if "max_context_tokens" not in properties:
            return None
        return self._coerce_max_context_tokens(properties.pop("max_context_tokens"))

    def unregister_model_config_override(self, name: str) -> bool:
        """注销 Model"""
        if name in self._model_config_overrides:
            del self._model_config_overrides[name]
            return True
        before = len(self._model_config_wildcard_overrides)
        self._model_config_wildcard_overrides = [
            override
            for override in self._model_config_wildcard_overrides
            if self._override_name(override) != name
        ]
        if len(self._model_config_wildcard_overrides) != before:
            return True
        return False

    def list_models(self) -> list[str]:
        """列出所有可用的 model 名称

        从 providers 的 model_ids 动态生成所有 provider_name/model_id 组合，
        并合并 model_configs 中定义的额外模型。
        """
        self._ensure_auto_load()

        models = []
        seen: set[str] = set()
        for provider in self._providers:
            for model_id in provider.model_ids:
                name = f"{provider.name}/{model_id}"
                if name not in seen:
                    models.append(name)
                    seen.add(name)
        return models

    def refresh_provider_models(self, name: str) -> list[str]:
        """Query a provider's remote model list and merge it into this registry.

        The refresh is in-memory only. It updates the current provider configs so
        newly discovered ``provider/model_id`` names can be selected without
        writing back to ``models.yaml``.
        """
        self._ensure_auto_load()

        providers = list(self._provider_groups.get(name) or [])
        if not providers:
            raise UnknownTemplateError(f"Provider '{name}' not found")

        refreshed_any = False
        errors: list[str] = []
        for provider in providers:
            try:
                model_ids = self._fetch_provider_model_ids(provider)
            except Exception as exc:
                errors.append(f"{provider.adapter}: {exc}")
                continue
            normalized = self._normalize_refreshed_model_ids(name, model_ids)
            if not normalized:
                continue
            provider.model_ids = self._merge_model_ids(
                provider.model_ids,
                normalized,
            )
            refreshed_any = True

        if not refreshed_any and errors:
            raise RuntimeError(
                f"Failed to refresh provider '{name}': " + "; ".join(errors)
            )

        prefix = f"{name}/"
        return [model for model in self.list_models() if model.startswith(prefix)]

    def _fetch_provider_model_ids(self, provider: ModelProviderConfig) -> list[str]:
        adapter_class = self.get_model_adapter(provider.adapter)
        if adapter_class is None:
            raise UnknownModelError(f"Model adapter '{provider.adapter}' not found")

        properties = dict(provider.properties)
        self._pop_max_context_tokens(properties)
        properties["model_id"] = provider.model_ids[0] if provider.model_ids else ""

        model = adapter_class(**properties)
        try:
            return model.list_models()
        finally:
            model.reset()

    @staticmethod
    def _merge_model_ids(current: list[str], incoming: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for model_id in [*current, *incoming]:
            if model_id not in seen:
                merged.append(model_id)
                seen.add(model_id)
        return merged

    @staticmethod
    def _normalize_refreshed_model_ids(
        provider_name: str,
        model_ids: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        prefix = f"{provider_name}/"
        for value in model_ids:
            model_id = str(value).strip()
            if model_id.startswith(prefix):
                model_id = model_id[len(prefix):]
            if not model_id or model_id in seen:
                continue
            normalized.append(model_id)
            seen.add(model_id)
        return normalized

    def has_model(self, name: str) -> bool:
        """检查 model 是否存在"""
        self._ensure_auto_load()

        provider, model_id = self._split_model_name(name)
        for p in self._provider_groups.get(provider, []):
            if model_id in p.model_ids:
                return True
        return False

    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """获取 Model 配置（动态构建）"""
        self._ensure_auto_load()
        
        provider, model_id = self._split_model_name(name)

        adapter = ""
        properties = {}
        max_context_tokens = None
        for p in self._provider_groups.get(provider, []):
            if model_id in p.model_ids:
                adapter = p.adapter
                properties = dict(p.properties)  # Make a copy
                max_context_tokens = self._pop_max_context_tokens(properties)
                break
        else:
            return None
    
        steer_merge_mode = None
        for override in self._matching_wildcard_overrides(name):
            properties, steer_merge_mode, max_context_tokens = self._merge_model_override(
                properties,
                steer_merge_mode,
                max_context_tokens,
                override,
            )

        if name in self._model_config_overrides:
            override = self._model_config_overrides[name]
            properties, steer_merge_mode, max_context_tokens = self._merge_model_override(
                properties,
                steer_merge_mode,
                max_context_tokens,
                override,
            )

        return ModelConfig(
            adapter=adapter,
            properties=properties,
            steer_merge_mode=steer_merge_mode,
            max_context_tokens=max_context_tokens,
        )

    @staticmethod
    def _override_name(override: ModelOverrideConfig) -> str:
        """Return the full provider/model key represented by an override."""
        return f"{override.provider}/{override.model_id}"

    def _matching_wildcard_overrides(self, name: str) -> list[ModelOverrideConfig]:
        """Return wildcard overrides matching a concrete provider/model name."""
        return [
            override
            for override in self._model_config_wildcard_overrides
            if fnmatchcase(name, self._override_name(override))
        ]

    def _merge_model_override(
        self,
        properties: dict[str, Any],
        steer_merge_mode: str | None,
        max_context_tokens: int | None,
        override: ModelOverrideConfig,
    ) -> tuple[dict[str, Any], str | None, int | None]:
        """Apply one normalized override over the accumulated config."""
        merged_properties = dict(properties)
        merged_properties.update(override.properties)

        merged_steer_merge_mode = steer_merge_mode
        if override.steer_merge_mode is not None:
            merged_steer_merge_mode = override.steer_merge_mode

        merged_max_context_tokens = max_context_tokens
        if override.max_context_tokens is not None:
            merged_max_context_tokens = override.max_context_tokens
        else:
            property_max_context_tokens = self._pop_max_context_tokens(merged_properties)
            if property_max_context_tokens is not None:
                merged_max_context_tokens = property_max_context_tokens

        return merged_properties, merged_steer_merge_mode, merged_max_context_tokens

    # ========================================================================
    # 模型创建
    # ========================================================================
    def create_model(
        self, name: str, **overrides: Any
    ) -> Model:
        """通过 Model 名称创建 Model 实例

        Args:
            name: Model 名称，格式为 "provider_name/model_id"
            **overrides: 覆盖参数

        Returns:
            Model 实例

        Raises:
            UnknownModelError: model 不存在
            UnknownModelError: Model 类未注册
        """
        # 自动加载配置（首次使用，除非被显式禁用）
        self._ensure_auto_load()

        # 动态构建 model 配置
        model_config = self.get_model_config(name)
        if model_config is None:
            raise UnknownModelError(f"Model '{name}' not found")

        adapter_class = self.get_model_adapter(model_config.adapter)
        assert adapter_class
        
        # Parse provider and model_id from name
        provider_name, model_id = self._split_model_name(name)
        
        # Apply overrides (model_id from config name takes precedence)
        override_steer_merge_mode = Model.coerce_steer_merge_mode(
            overrides.pop("steer_merge_mode", None),
            source=f"create_model({name}) steer_merge_mode override",
        )
        override_max_context_tokens = self._coerce_max_context_tokens(
            overrides.pop("max_context_tokens", None)
        )
        properties = dict(model_config.properties)
        property_max_context_tokens = self._pop_max_context_tokens(properties)
        properties['model_id'] = model_id
        properties.update(overrides)
        override_property_max_context_tokens = self._pop_max_context_tokens(properties)
        
        model = adapter_class(**properties)
        model.configure_steer_merge_mode(
            override_steer_merge_mode or model_config.steer_merge_mode
        )
        model.configure_max_context_tokens(
            override_max_context_tokens
            or override_property_max_context_tokens
            or property_max_context_tokens
            or model_config.max_context_tokens
        )
        model.validate_steer_merge_mode_config(source=f"create_model({name})")
        return model
    # ========================================================================
    # 配置加载
    # ========================================================================

    def _ensure_auto_load(self) -> None:
        """确保自动加载已处理（加载完成或被显式禁用）"""
        if not self._auto_load_needed:
            return  # 已经处理过，不需要再加载

        # 检查是否显式禁用自动加载
        if os.environ.get("HAWI_NO_AUTO_LOAD"):
            self._auto_load_needed = False
            return

        # 执行自动加载
        # 1. 用户级配置
        user_config = Path.home() / ".hawi" / "models.yaml"
        if user_config.exists():
            self._load_config_file(user_config, quiet=True)

        # 2. 项目级配置
        project_config = Path.cwd() / ".hawi" / "models.yaml"
        if project_config.exists():
            self._load_config_file(project_config, quiet=True)

        self._auto_load_needed = False

    def load_config(
        self, path: Union[str, Path], quiet: bool = False
    ) -> None:
        """从 YAML 文件加载配置

        Args:
            path: 配置文件路径
            quiet: 是否静默（不输出日志）

        合并策略：
        - Model 级别覆盖，后加载的同名 model 完全替换先加载的
        - 如果尚未自动加载，会先尝试自动加载（除非被显式禁用）
        """
        # 先处理自动加载（如果还需要的话）
        self._ensure_auto_load()

        # 然后加载用户指定的配置
        self._load_config_file(path, quiet)

    def _substitute_env_vars(self, obj: Any) -> Any:
        """递归替换字符串中的环境变量。
        
        支持格式：
        - ${ENV_VAR} - 使用环境变量值，未设置则返回空字符串
        - ${ENV_VAR:default} - 使用环境变量值，未设置则使用默认值
        """
        import os
        import re
        
        pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
        
        def replace(val: str) -> str:
            def replacer(match: re.Match) -> str:
                var_name = match.group(1)
                default = match.group(2) if match.group(2) is not None else ""
                return os.environ.get(var_name, default)
            return pattern.sub(replacer, val)
        
        if isinstance(obj, str):
            return replace(obj)
        elif isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        return obj

    def _load_config_file(self, path: Union[str, Path], quiet: bool = False) -> None:
        """实际加载配置文件（不包含自动加载逻辑）

        新配置格式：
        providers:
          - name: provider_name
            adapter: ModelClassName
            parent: parent_provider (可选)
            model_ids: [model_id1, model_id2]
            properties:
              - key: value

        model_configs:
          provider_name/model_id:
            key: value
            steer_merge_mode: tool_result_assistant_template_and_user_message
        """
        path = Path(path)
        if not path.exists():
            if not quiet:
                print(f"[ModelRegistry] Config file not found: {path}")
            return

        try:
            data = load_config_file(path)
        except ConfigLoaderError as e:
            raise ValueError(f"Invalid model config in {path}: {e}") from e

        # 递归替换环境变量
        data = self._substitute_env_vars(data)

        for provider_data in data.get('providers', []):
            provider = ModelProviderConfig.model_validate(provider_data)
            self._register_provider(provider, quiet)
        
        for model_name, properties in data.get('model_configs', {}).items():
            self.register_model_config_override(model_name, properties, quiet)

    def clear(self) -> None:
        """清空所有注册（主要用于测试）"""
        self._classes.clear()
        self._providers.clear()
        self._provider_groups.clear()
        self._model_config_overrides.clear()
        self._model_config_wildcard_overrides.clear()
        self._auto_load_needed = True
        self._register_builtin_classes()


# 全局单例实例
model_registry = ModelRegistry()


def create_model(name: str, **overrides: Any) -> Model:
    """Create a model instance using the global registry."""
    return model_registry.create_model(name, **overrides)


def get_model_adapter(name: str) -> type[Model] | None:
    """Get a model adapter class using the global registry."""
    return model_registry.get_model_adapter(name)


def get_model_config(name: str) -> ModelConfig | None:
    """Get model configuration using the global registry."""
    return model_registry.get_model_config(name)


def load_config(path: Union[str, Path], quiet: bool = False) -> None:
    """Load model configuration using the global registry."""
    model_registry.load_config(path, quiet=quiet)


def list_models() -> list[str]:
    """List all available models using the global registry."""
    return model_registry.list_models()


def list_providers() -> list[str]:
    """List all registered providers using the global registry."""
    return model_registry.list_providers()


def refresh_provider_models(provider: str) -> list[str]:
    """Refresh one provider's model list using the global registry."""
    return model_registry.refresh_provider_models(provider)
