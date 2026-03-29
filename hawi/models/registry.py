"""Model Registry - 单例模式，管理 Model 类和 Factory 注册。

设计原则：
- 单例模式：全局唯一实例
- 职责：Model 类注册表 + Factory 注册表 + Template 注册表
- 无对象池：每次 create_model 创建新实例
- 自动配置加载：首次使用时自动加载默认配置路径

配置格式：
- 特殊字段使用 __ 前缀避免冲突：__class, __parent
- 也支持不加前缀的别名：class, parent（向后兼容）
- __class: 指定 Model 类名（factory 必需）
- __parent: 继承的模板或工厂名称（字符串或列表）
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
    "get_factory_arguments",
    "load_config",
    "list_factories",
    "list_templates",
    "CircularDependencyError",
    "UnknownFactoryError",
    "UnknownTemplateError",
    "InvalidInheritanceError",
]


class CircularDependencyError(Exception):
    """Factory/Template 循环继承错误"""
    pass


class UnknownFactoryError(Exception):
    """未知 Factory 错误"""
    pass


class UnknownTemplateError(Exception):
    """未知 Template 错误"""
    pass


class InvalidInheritanceError(Exception):
    """无效的继承关系错误"""
    pass


# 特殊字段名（使用 __ 前缀避免冲突，也支持无前缀的别名）
CLASS_FIELD = "__class"
CLASS_FIELD_ALIAS = "class"
PARENT_FIELD = "__parent"
PARENT_FIELD_ALIAS = "parent"


def _get_special_field(data: dict[str, Any], field: str, alias: str) -> tuple[Any, bool]:
    """获取特殊字段值，优先使用 @_ 前缀的字段
    
    Returns:
        (值, 是否找到)
    """
    if field in data:
        return data[field], True
    if alias in data:
        return data[alias], True
    return None, False


def _pop_special_field(data: dict[str, Any], field: str, alias: str) -> Any:
    """弹出特殊字段值，优先使用 @_ 前缀的字段"""
    if field in data:
        return data.pop(field)
    if alias in data:
        return data.pop(alias)
    return None


def _normalize_parent(parent: Union[str, list, None]) -> list[str]:
    """将 parent 字段规范化为列表
    
    - 字符串 -> [字符串]
    - None -> []
    - 列表 -> 保持原样
    """
    if parent is None:
        return []
    if isinstance(parent, str):
        return [parent]
    if isinstance(parent, list):
        return parent
    raise ValueError(f"Invalid parent type: {type(parent)}, expected str or list")


class TemplateConfig:
    """Template 配置对象
    
    Template 是属性集合，可以被 factory 或其他 template 继承。
    - 可以有 @class 字段（作为预设的类名）
    - 可以有 @parent 字段（只能继承其他 template）
    - 其他字段都是属性
    """

    def __init__(
        self,
        arguments: dict[str, Any],
        class_name: Optional[str] = None,
        parents: Optional[list[str]] = None,
    ):
        self.arguments = arguments
        self.class_name = class_name  # template 可以有预设的 class
        self.parents = parents or []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateConfig:
        """从配置字典创建"""
        data = data.copy()
        class_name = _pop_special_field(data, CLASS_FIELD, CLASS_FIELD_ALIAS)
        parent = _pop_special_field(data, PARENT_FIELD, PARENT_FIELD_ALIAS)
        parents = _normalize_parent(parent)
        # 剩余字段都是 arguments
        return cls(class_name=class_name, arguments=data, parents=parents)

    def to_dict(self, use_prefix: bool = False) -> dict[str, Any]:
        """转换为配置字典
        
        Args:
            use_prefix: 是否使用 @_ 前缀，默认 False
        """
        class_key = CLASS_FIELD if use_prefix else CLASS_FIELD_ALIAS
        parent_key = PARENT_FIELD if use_prefix else PARENT_FIELD_ALIAS
        
        result = {}
        if self.class_name is not None:
            result[class_key] = self.class_name
        if self.parents:
            if len(self.parents) == 1:
                result[parent_key] = self.parents[0]
            else:
                result[parent_key] = self.parents
        result.update(self.arguments)
        return result


class FactoryConfig:
    """Factory 配置对象"""

    def __init__(
        self,
        class_name: str,
        arguments: dict[str, Any],
        parent: Optional[Union[str, list[str]]] = None,
    ):
        self.class_name = class_name
        self.arguments = arguments
        # 内部统一使用列表存储
        self._parents = _normalize_parent(parent)

    @property
    def parent(self) -> Optional[Union[str, list[str]]]:
        """向后兼容的 parent 属性
        
        - 无 parent 时返回 None
        - 单 parent 时返回字符串（向后兼容）
        - 多 parent 时返回列表
        """
        if not self._parents:
            return None
        if len(self._parents) == 1:
            return self._parents[0]
        return self._parents

    @property
    def parents(self) -> list[str]:
        """返回 parent 列表"""
        return self._parents

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactoryConfig:
        """从配置字典创建"""
        data = data.copy()
        class_name = _pop_special_field(data, CLASS_FIELD, CLASS_FIELD_ALIAS)
        # class_name 可以为 None，如果 parent 会提供它
        parent = _pop_special_field(data, PARENT_FIELD, PARENT_FIELD_ALIAS)
        # 剩余字段都是 arguments
        return cls(
            class_name=class_name or "",  # 空字符串表示需要继承
            arguments=data,
            parent=parent
        )

    def to_dict(self, use_prefix: bool = False) -> dict[str, Any]:
        """转换为配置字典
        
        Args:
            use_prefix: 是否使用 @_ 前缀，默认 False（向后兼容）
        """
        class_key = CLASS_FIELD if use_prefix else CLASS_FIELD_ALIAS
        parent_key = PARENT_FIELD if use_prefix else PARENT_FIELD_ALIAS
        
        result = {class_key: self.class_name, **self.arguments}
        if self._parents:
            if len(self._parents) == 1:
                result[parent_key] = self._parents[0]
            else:
                result[parent_key] = self._parents
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
        self._templates: dict[str, TemplateConfig] = {}
        self._factories: dict[str, FactoryConfig] = {}

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
    # Template 管理
    # ========================================================================

    def register_template(
        self,
        name: str,
        arguments: dict[str, Any],
        class_name: Optional[str] = None,
        parents: Optional[list[str]] = None,
        quiet: bool = False,
    ) -> None:
        """注册 Template

        Args:
            name: Template 名称
            arguments: 属性字典
            class_name: 可选的预设 Model 类名
            parents: 继承的 template 名称列表
            quiet: 是否静默（不输出覆盖警告）
        """
        if name in self._templates and not quiet:
            print(f"[ModelRegistry] Template '{name}' overridden")

        self._templates[name] = TemplateConfig(
            class_name=class_name, arguments=arguments, parents=parents or []
        )

    def unregister_template(self, name: str) -> bool:
        """注销 Template"""
        if name in self._templates:
            del self._templates[name]
            return True
        return False

    def get_template(self, name: str) -> Optional[TemplateConfig]:
        """获取 Template 配置"""
        return self._templates.get(name)

    def list_templates(self) -> list[str]:
        """列出所有已注册的 template 名称"""
        return list(self._templates.keys())

    def has_template(self, name: str) -> bool:
        """检查 template 是否存在"""
        return name in self._templates

    # ========================================================================
    # Factory 管理
    # ========================================================================

    def register_factory(
        self,
        name: str,
        class_name: str,
        arguments: dict[str, Any],
        parent: Optional[Union[str, list[str]]] = None,
        parents: Optional[list[str]] = None,
        quiet: bool = False,
    ) -> None:
        """注册 Factory

        Args:
            name: Factory 名称
            class_name: Model 类名
            arguments: 实例化参数
            parent: 继承的 template/factory 名称（字符串或列表，向后兼容）
            parents: 继承的 template/factory 名称列表（与 parent 互斥）
            quiet: 是否静默（不输出覆盖警告）
        """
        if name in self._factories and not quiet:
            print(f"[ModelRegistry] Factory '{name}' overridden")

        # 处理 parent/parents 参数
        if parent is not None and parents is not None:
            raise ValueError("Cannot specify both 'parent' and 'parents'")
        
        if parents is not None:
            parent_list = parents
        elif parent is not None:
            parent_list = _normalize_parent(parent)
        else:
            parent_list = []

        self._factories[name] = FactoryConfig(
            class_name=class_name, arguments=arguments, parent=parent_list
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

    def get_factory_arguments(self, name: str, expanded: bool = False) -> dict[str, Any]:
        """获取 Factory 的参数

        Args:
            name: Factory 名称
            expanded: 是否展开继承链，获取合并后的完整参数
                     - False (默认): 返回原始参数（不包含继承的参数）
                     - True: 返回展开后的完整参数（包含从 parent/template 继承合并后的参数）

        Returns:
            Factory 的参数字典

        Raises:
            UnknownFactoryError: factory 不存在

        Example:
            # 获取原始参数
            args = registry.get_factory_arguments("deepseek-chat")
            # {"model_id": "deepseek-chat"}

            # 获取展开后的完整参数（继承链已合并）
            full_args = registry.get_factory_arguments("deepseek-chat", expanded=True)
            # {"model_id": "deepseek-chat", "api_key": "...", "temperature": 0.7}
        """
        # 自动加载配置（首次使用，除非被显式禁用）
        self._ensure_auto_load()

        if name not in self._factories:
            raise UnknownFactoryError(f"Factory '{name}' not found")

        if expanded:
            # 解析继承链，获取合并后的完整参数
            resolved = self._resolve_factory(name)
            return resolved.arguments.copy()
        else:
            # 返回原始参数（不包含继承的参数）
            return self._factories[name].arguments.copy()

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

        # 解析占位符替换（环境变量）
        arguments = self._resolve_substitutions(arguments)

        return model_class(**arguments)

    def _resolve_template(
        self, name: str, visited: Optional[set[str]] = None
    ) -> TemplateConfig:
        """解析 template 配置，处理 parent 继承
        
        Template 只能以其他 template 为 parent。
        多 parent 时按列表顺序合并属性（后面的覆盖前面的）。
        """
        if visited is None:
            visited = set()

        if name in visited:
            raise CircularDependencyError(
                f"Circular parent detected: {' -> '.join(visited)} -> {name}"
            )

        config = self._templates.get(name)
        if config is None:
            raise UnknownTemplateError(f"Template '{name}' not found")

        if not config.parents:
            return config

        # 按顺序合并所有 parent
        visited.add(name)
        merged_args: dict[str, Any] = {}
        merged_class: Optional[str] = None

        for parent_name in config.parents:
            if parent_name in self._templates:
                # 只能继承 template
                parent = self._resolve_template(parent_name, visited.copy())
                merged_class = parent.class_name or merged_class
                merged_args.update(parent.arguments)
            else:
                raise InvalidInheritanceError(
                    f"Template '{name}' cannot inherit from '{parent_name}': "
                    f"template can only inherit from template"
                )

        # 当前配置覆盖 parent 配置
        merged_class = config.class_name or merged_class
        merged_args.update(config.arguments)

        return TemplateConfig(
            class_name=merged_class,
            arguments=merged_args,
            parents=[],  # 已解析，不需要保留
        )

    def _resolve_factory(
        self, name: str, visited: Optional[set[str]] = None
    ) -> FactoryConfig:
        """解析 factory 配置，处理 parent 继承
        
        Factory 可以以 template 或 factory 为 parent。
        多 parent 时按列表顺序合并属性（后面的覆盖前面的）。
        """
        if visited is None:
            visited = set()

        if name in visited:
            raise CircularDependencyError(
                f"Circular parent detected: {' -> '.join(visited)} -> {name}"
            )

        config = self._factories.get(name)
        if config is None:
            raise UnknownFactoryError(f"Factory '{name}' not found")

        if not config.parents:
            return config

        # 按顺序合并所有 parent
        visited.add(name)
        merged_args: dict[str, Any] = {}
        merged_class: Optional[str] = None

        for parent_name in config.parents:
            if parent_name in self._templates:
                # 继承 template
                parent = self._resolve_template(parent_name, visited.copy())
                merged_class = parent.class_name or merged_class
                merged_args.update(parent.arguments)
            elif parent_name in self._factories:
                # 继承 factory
                parent = self._resolve_factory(parent_name, visited.copy())
                merged_class = parent.class_name or merged_class
                merged_args.update(parent.arguments)
            else:
                raise UnknownFactoryError(
                    f"Factory '{name}' parent '{parent_name}' not found "
                    f"(neither template nor factory)"
                )

        # 当前配置覆盖 parent 配置
        merged_class = config.class_name or merged_class
        merged_args.update(config.arguments)

        if not merged_class:
            raise InvalidInheritanceError(
                f"Factory '{name}' has no class specified and no parent provides one"
            )

        return FactoryConfig(
            class_name=merged_class,
            arguments=merged_args,
            parent=[],  # 已解析，不需要保留
        )

    def _resolve_substitutions(self, value: Any) -> Any:
        """递归解析占位符替换（环境变量）

        支持语法：
        - ${ENV_VAR}          -> 从环境变量中查找
        - ${ENV_VAR:default}  -> 环境变量带默认值
        """
        if isinstance(value, str):
            # 完整字符串是单一占位符
            if value.startswith("${") and value.endswith("}"):
                inner = value[2:-1]
                # 处理 ${ENV_VAR} 或 ${ENV_VAR:default}
                if ":" in inner:
                    env_var, default = inner.split(":", 1)
                    return os.environ.get(env_var, default)
                return os.environ.get(inner, value)

            # 处理嵌入的占位符
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
                data = yaml.safe_load(f) or {}

            # 加载 Templates
            if "templates" in data and data["templates"]:
                templates = data["templates"]
                for name, config in templates.items():
                    template_config = TemplateConfig.from_dict(config)
                    self.register_template(
                        name=name,
                        class_name=template_config.class_name,
                        arguments=template_config.arguments,
                        parents=template_config.parents,
                        quiet=quiet,
                    )
                if not quiet:
                    print(f"[ModelRegistry] Loaded {len(templates)} templates from {path}")

            # 加载 Factories
            if "factories" in data:
                factories = data["factories"]
                for name, config in factories.items():
                    factory_config = FactoryConfig.from_dict(config)
                    self.register_factory(
                        name=name,
                        class_name=factory_config.class_name,
                        arguments=factory_config.arguments,
                        parents=factory_config.parents,
                        quiet=quiet,
                    )
                if not quiet:
                    print(f"[ModelRegistry] Loaded {len(factories)} factories from {path}")

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

    def clear(self) -> None:
        """清空所有注册（主要用于测试）"""
        self._factories.clear()
        self._templates.clear()
        self._classes.clear()
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


def get_factory_arguments(name: str, expanded: bool = False) -> dict[str, Any]:
    """获取 Factory 的参数

    Args:
        name: Factory 名称
        expanded: 是否展开继承链，获取合并后的完整参数

    Returns:
        Factory 的参数字典
    """
    return model_registry.get_factory_arguments(name, expanded)


def load_config(path: Union[str, Path], quiet: bool = False) -> None:
    """加载配置文件"""
    return model_registry.load_config(path, quiet)


def list_factories() -> list[str]:
    """列出所有 factory 名称"""
    return model_registry.list_factories()


def list_templates() -> list[str]:
    """列出所有 template 名称"""
    return model_registry.list_templates()
