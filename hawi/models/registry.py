"""Model Registry - 单例模式，管理 Model 类和 Factory 注册。

设计原则：
- 单例模式：全局唯一实例
- 职责：Model 类注册表 + Factory 注册表
- 无对象池：每次 create_model 创建新实例
- 自动配置加载：首次使用时自动加载默认配置路径
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union

from hawi.models.model import Model

__all__ = [
    "ModelRegistry",
    "model_registry",
    "create_model",
    "get_model_class",
    "load_config",
    "list_factories",
    "CircularDependencyError",
    "UnknownFactoryError",
    "UnknownApiKeyAliasError",
]


class CircularDependencyError(Exception):
    """Factory 循环继承错误"""
    pass


class UnknownFactoryError(Exception):
    """未知 Factory 错误"""
    pass


class UnknownApiKeyAliasError(Exception):
    """未知 API Key 别名错误"""
    pass


class FactoryConfig:
    """Factory 配置对象"""

    def __init__(
        self,
        class_name: str,
        arguments: dict[str, Any],
        parent: Optional[str] = None,
    ):
        self.class_name = class_name
        self.arguments = arguments
        self.parent = parent

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactoryConfig:
        """从配置字典创建"""
        data = data.copy()
        class_name = data.pop("class")
        parent = data.pop("parent", None)
        # 剩余字段都是 arguments
        return cls(class_name=class_name, arguments=data, parent=parent)

    def to_dict(self) -> dict[str, Any]:
        """转换为配置字典"""
        result = {"class": self.class_name, **self.arguments}
        if self.parent:
            result["parent"] = self.parent
        return result


class ModelRegistry:
    """Model Registry 单例类

    管理：
    1. Model 类注册表（类名 -> Model 类）
    2. Factory 注册表（factory 名 -> FactoryConfig）

    使用方式：
        from hawi.models import model_registry

        # 获取 Model 类
        cls = model_registry.get_model_class("DeepSeekOpenAIModel")

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
        self._factories: dict[str, FactoryConfig] = {}
        self._api_keys: dict[str, str] = {}
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

    def register_class(
        self, name: str, cls: type[Model], quiet: bool = False
    ) -> None:
        """注册 Model 类

        Args:
            name: 类名（用于 factory 配置中的 class 字段）
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

    def get_model_class(self, name: str) -> Optional[type[Model]]:
        """通过类名获取 Model 类"""
        return self._classes.get(name)

    def list_classes(self) -> list[str]:
        """列出所有已注册的类名"""
        return list(self._classes.keys())

    # ========================================================================
    # Factory 管理
    # ========================================================================

    def register_factory(
        self,
        name: str,
        class_name: str,
        arguments: dict[str, Any],
        parent: Optional[str] = None,
        quiet: bool = False,
    ) -> None:
        """注册 Factory

        Args:
            name: Factory 名称
            class_name: Model 类名
            arguments: 实例化参数
            parent: 继承的 factory 名称
            quiet: 是否静默（不输出覆盖警告）
        """
        if name in self._factories and not quiet:
            print(f"[ModelRegistry] Factory '{name}' overridden")

        self._factories[name] = FactoryConfig(
            class_name=class_name, arguments=arguments, parent=parent
        )

    def unregister_factory(self, name: str) -> bool:
        """注销 Factory"""
        if name in self._factories:
            del self._factories[name]
            return True
        return False

    def get_factory(self, name: str) -> Optional[FactoryConfig]:
        """获取 Factory 配置"""
        return self._factories.get(name)

    def list_factories(self) -> list[str]:
        """列出所有已注册的 factory 名称"""
        return list(self._factories.keys())

    def has_factory(self, name: str) -> bool:
        """检查 factory 是否存在"""
        return name in self._factories

    # ========================================================================
    # 模型创建
    # ========================================================================

    def create_model(
        self, name: str, overrides: Optional[dict[str, Any]] = None
    ) -> Model:
        """通过 Factory 名称创建 Model 实例

        Args:
            name: Factory 名称
            overrides: 覆盖参数

        Returns:
            Model 实例

        Raises:
            UnknownFactoryError: factory 不存在
            UnknownFactoryError: Model 类未注册
        """
        # 自动加载配置（首次使用，除非被显式禁用）
        self._ensure_auto_load()

        if name not in self._factories:
            raise UnknownFactoryError(f"Factory '{name}' not found")

        # 解析配置（处理 parent 继承）
        resolved = self._resolve_factory(name)

        # 获取 Model 类
        model_class = self._classes.get(resolved.class_name)
        if model_class is None:
            raise UnknownFactoryError(
                f"Model class '{resolved.class_name}' not registered for factory '{name}'"
            )

        # 合并参数：factory arguments < overrides
        arguments = resolved.arguments.copy()
        if overrides:
            arguments.update(overrides)

        # 解析占位符替换（api_key 别名 + 环境变量）
        arguments = self._resolve_substitutions(arguments)

        return model_class(**arguments)

    def _resolve_factory(
        self, name: str, visited: Optional[set[str]] = None
    ) -> FactoryConfig:
        """解析 factory 配置，处理 parent 继承"""
        if visited is None:
            visited = set()

        if name in visited:
            raise CircularDependencyError(
                f"Circular parent detected: {' -> '.join(visited)} -> {name}"
            )

        config = self._factories.get(name)
        if config is None:
            raise UnknownFactoryError(f"Factory '{name}' not found")

        if config.parent:
            visited.add(name)
            # 递归解析父配置
            parent = self._resolve_factory(config.parent, visited)
            # 合并：父配置 + 当前配置
            merged_args = {**parent.arguments, **config.arguments}
            return FactoryConfig(
                class_name=config.class_name or parent.class_name,
                arguments=merged_args,
                parent=None,  # 已解析，不需要保留
            )

        return config

    def _resolve_substitutions(self, value: Any) -> Any:
        """递归解析所有占位符替换（API Keys 别名 + 环境变量）

        支持语法：
        - ${api_keys.ALIAS}   -> 从 api_keys 配置中查找
        - ${ENV_VAR}          -> 从环境变量中查找
        - ${ENV_VAR:default}  -> 环境变量带默认值

        优先级：api_keys 别名 > 环境变量

        Raises:
            UnknownApiKeyAliasError: 当 ${api_keys.ALIAS} 找不到对应别名时
        """
        if isinstance(value, str):
            # 完整字符串是单一占位符
            if value.startswith("${") and value.endswith("}"):
                inner = value[2:-1]

                # 1. 处理 ${api_keys.ALIAS}
                prefix = "api_keys."
                if inner.startswith(prefix):
                    alias = inner[len(prefix):]  # 提取 ALIAS
                    if alias in self._api_keys:
                        return self._api_keys[alias]
                    # 别名不存在，抛出明确错误
                    raise UnknownApiKeyAliasError(
                        f"Unknown API key alias '${{api_keys.{alias}}}'. "
                        f"Available aliases: {list(self._api_keys.keys())}"
                    )

                # 2. 处理 ${ENV_VAR} 或 ${ENV_VAR:default}
                if ":" in inner:
                    env_var, default = inner.split(":", 1)
                    return os.environ.get(env_var, default)
                return os.environ.get(inner, value)

            # 处理嵌入的占位符
            # 先处理 api_keys 别名
            api_keys_pattern = r"\$\{api_keys\.([^}]+)\}"

            def replace_api_keys(match: re.Match) -> str:
                alias = match.group(1)
                if alias not in self._api_keys:
                    raise UnknownApiKeyAliasError(
                        f"Unknown API key alias '${{api_keys.{alias}}}'. "
                        f"Available aliases: {list(self._api_keys.keys())}"
                    )
                return self._api_keys[alias]

            value = re.sub(api_keys_pattern, replace_api_keys, value)

            # 再处理环境变量
            env_pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

            def replace_env_var(match: re.Match) -> str:
                var_name = match.group(1)
                default_val = match.group(2)
                env_val = os.environ.get(var_name)
                if env_val is not None:
                    return env_val
                if default_val is not None:
                    return default_val
                return match.group(0) or ""

            return re.sub(env_pattern, replace_env_var, value)

        elif isinstance(value, dict):
            return {k: self._resolve_substitutions(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._resolve_substitutions(item) for item in value]

        return value

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
        self._do_auto_load()
        self._auto_load_needed = False

    def _do_auto_load(self) -> None:
        """执行实际的自动加载（不包含标志设置）"""
        try:
            import yaml
        except ImportError:
            print("[ModelRegistry] PyYAML not installed, skipping config auto-load")
            return

        # 1. 用户级配置
        user_config = Path.home() / ".hawi" / "models.yaml"
        if user_config.exists():
            self._load_config_file(user_config, quiet=True)

        # 2. 项目级配置
        project_config = Path.cwd() / ".hawi" / "models.yaml"
        if project_config.exists():
            self._load_config_file(project_config, quiet=True)

    def load_config(
        self, path: Union[str, Path], quiet: bool = False
    ) -> None:
        """从 YAML 文件加载配置

        Args:
            path: 配置文件路径
            quiet: 是否静默（不输出日志）

        合并策略：
        - Factory 级别覆盖，后加载的同名 factory 完全替换先加载的
        - 如果尚未自动加载，会先尝试自动加载（除非被显式禁用）
        """
        # 先处理自动加载（如果还需要的话）
        if self._auto_load_needed:
            if not os.environ.get("HAWI_NO_AUTO_LOAD"):
                self._do_auto_load()
            self._auto_load_needed = False

        # 然后加载用户指定的配置
        self._load_config_file(path, quiet)

    def _load_config_file(self, path: Union[str, Path], quiet: bool = False) -> None:
        """实际加载配置文件（不包含自动加载逻辑）"""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load config files. Install with: pip install pyyaml"
            )

        path = Path(path)
        if not path.exists():
            if not quiet:
                print(f"[ModelRegistry] Config file not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # 加载 API Key
            if "api_keys" in data:
                self._api_keys.update(data["api_keys"])
                if not quiet:
                    print(f"[ModelRegistry] Loaded {len(data['api_keys'])} API keys from {path}")

            if not data or "factories" not in data:
                if not quiet:
                    print(f"[ModelRegistry] No factories found in {path}")
                return

            factories = data["factories"]
            for name, config in factories.items():
                factory_config = FactoryConfig.from_dict(config)
                self.register_factory(
                    name=name,
                    class_name=factory_config.class_name,
                    arguments=factory_config.arguments,
                    parent=factory_config.parent,
                    quiet=quiet,
                )

            if not quiet:
                print(f"[ModelRegistry] Loaded {len(factories)} factories from {path}")

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

    def clear(self) -> None:
        """清空所有注册（主要用于测试）"""
        self._factories.clear()
        self._classes.clear()
        self._api_keys.clear()
        self._auto_load_needed = True
        self._register_builtin_classes()


# 全局单例实例
model_registry = ModelRegistry()


# 便捷函数（直接通过模块调用）
def create_model(
    name: str, overrides: Optional[dict[str, Any]] = None
) -> Model:
    """通过 Factory 名称创建 Model 实例"""
    return model_registry.create_model(name, overrides)


def get_model_class(name: str) -> Optional[type[Model]]:
    """通过类名获取 Model 类"""
    return model_registry.get_model_class(name)


def load_config(path: Union[str, Path], quiet: bool = False) -> None:
    """加载配置文件"""
    return model_registry.load_config(path, quiet)


def list_factories() -> list[str]:
    """列出所有 factory 名称"""
    return model_registry.list_factories()
