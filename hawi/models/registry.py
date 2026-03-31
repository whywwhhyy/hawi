"""Model Registry - 单例模式，管理 Model 类和 Model 注册。

设计原则：
- 单例模式：全局唯一实例
- 职责：Model 类注册表 + Model 注册表 + Template 注册表
- 无对象池：每次 create_model 创建新实例
- 自动配置加载：首次使用时自动加载默认配置路径

配置格式：
- 特殊字段使用 __ 前缀避免冲突：__class, __template
- 也支持不加前缀的别名：class, template（向后兼容）
- __class: 指定 Model 类名（model 必需）
- __template: 继承的模板或 model 名称（字符串或列表）
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
    "get_model_arguments",
    "load_config",
    "list_models",
    "list_templates",
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


# 特殊字段名（使用 __ 前缀避免冲突，也支持无前缀的别名）
CLASS_FIELD = "__class"
CLASS_FIELD_ALIAS = "class"
TEMPLATE_FIELD = "__template"
TEMPLATE_FIELD_ALIAS = "template"
# 向后兼容旧格式
PARENT_FIELD = "__parent"
PARENT_FIELD_ALIAS = "parent"


def _pop_special_field(data: dict[str, Any], field: str, alias: str) -> Any:
    """弹出特殊字段值，优先使用 @_ 前缀的字段"""
    if field in data:
        return data.pop(field)
    if alias in data:
        return data.pop(alias)
    return None


def _normalize_template(template: Union[str, list, None]) -> list[str]:
    """将 template 字段规范化为列表

    - 字符串 -> [字符串]
    - None -> []
    - 列表 -> 保持原样
    """
    if template is None:
        return []
    if isinstance(template, str):
        return [template]
    if isinstance(template, list):
        return template
    raise ValueError(f"Invalid template type: {type(template)}, expected str or list")


class TemplateConfig:
    """Template 配置对象

    Template 是属性集合，可以被 model 或其他 template 继承。
    - 可以有 @class 字段（作为预设的类名）
    - 可以有 @template 字段（只能继承其他 template）
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
        template = _pop_special_field(data, TEMPLATE_FIELD, TEMPLATE_FIELD_ALIAS)
        parents = _normalize_template(template)
        # 剩余字段都是 arguments
        return cls(class_name=class_name, arguments=data, parents=parents)

    def to_dict(self, use_prefix: bool = False) -> dict[str, Any]:
        """转换为配置字典

        Args:
            use_prefix: 是否使用 @_ 前缀，默认 False
        """
        class_key = CLASS_FIELD if use_prefix else CLASS_FIELD_ALIAS
        template_key = TEMPLATE_FIELD if use_prefix else TEMPLATE_FIELD_ALIAS

        result = {}
        if self.class_name is not None:
            result[class_key] = self.class_name
        if self.parents:
            if len(self.parents) == 1:
                result[template_key] = self.parents[0]
            else:
                result[template_key] = self.parents
        result.update(self.arguments)
        return result


class ModelConfig:
    """Model 配置对象"""

    def __init__(
        self,
        class_name: str,
        arguments: dict[str, Any],
        template: Optional[Union[str, list[str]]] = None,
    ):
        self.class_name = class_name
        self.arguments = arguments
        # 内部统一使用列表存储
        self._templates = _normalize_template(template)

    @property
    def template(self) -> Optional[Union[str, list[str]]]:
        """向后兼容的 template 属性

        - 无 template 时返回 None
        - 单 template 时返回字符串（向后兼容）
        - 多 template 时返回列表
        """
        if not self._templates:
            return None
        if len(self._templates) == 1:
            return self._templates[0]
        return self._templates

    @property
    def templates(self) -> list[str]:
        """返回 template 列表"""
        return self._templates

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """从配置字典创建"""
        data = data.copy()
        class_name = _pop_special_field(data, CLASS_FIELD, CLASS_FIELD_ALIAS)
        # class_name 可以为 None，如果 template 会提供它
        template = _pop_special_field(data, TEMPLATE_FIELD, TEMPLATE_FIELD_ALIAS)
        # 剩余字段都是 arguments
        return cls(
            class_name=class_name or "",  # 空字符串表示需要继承
            arguments=data,
            template=template
        )

    def to_dict(self, use_prefix: bool = False) -> dict[str, Any]:
        """转换为配置字典

        Args:
            use_prefix: 是否使用 @_ 前缀，默认 False（向后兼容）
        """
        class_key = CLASS_FIELD if use_prefix else CLASS_FIELD_ALIAS
        template_key = TEMPLATE_FIELD if use_prefix else TEMPLATE_FIELD_ALIAS

        result = {class_key: self.class_name, **self.arguments}
        if self._templates:
            if len(self._templates) == 1:
                result[template_key] = self._templates[0]
            else:
                result[template_key] = self._templates
        return result


class ModelRegistry:
    """Model Registry 单例类

    管理：
    1. Model 类注册表（类名 -> Model 类）
    2. Model 注册（model 名 -> ModelConfig）

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
        self._models: dict[str, ModelConfig] = {}

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
    # Model 管理
    # ========================================================================

    def register_model(
        self,
        name: str,
        class_name: str,
        arguments: dict[str, Any],
        template: Optional[Union[str, list[str]]] = None,
        templates: Optional[list[str]] = None,
        quiet: bool = False,
    ) -> None:
        """注册 Model

        Args:
            name: Model 名称
            class_name: Model 类名
            arguments: 实例化参数
            template: 继承的 template/model 名称（字符串或列表，向后兼容）
            templates: 继承的 template/model 名称列表（与 template 互斥）
            quiet: 是否静默（不输出覆盖警告）
        """
        if name in self._models and not quiet:
            print(f"[ModelRegistry] Model '{name}' overridden")

        # 处理 template/templates 参数
        if template is not None and templates is not None:
            raise ValueError("Cannot specify both 'template' and 'templates'")

        if templates is not None:
            template_list = templates
        elif template is not None:
            template_list = _normalize_template(template)
        else:
            template_list = []

        self._models[name] = ModelConfig(
            class_name=class_name, arguments=arguments, template=template_list
        )

    def get_model_config(self, name: str) -> dict | None:
        """获取 Model 配置（向后兼容）

        Returns:
            Model 配置字典或 None
        """
        config = self._models.get(name)
        if config is None:
            return None
        return config.to_dict()

    def unregister_model(self, name: str) -> bool:
        """注销 Model"""
        if name in self._models:
            del self._models[name]
            return True
        return False

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """获取 Model 配置"""
        return self._models.get(name)

    def list_models(self) -> list[str]:
        """列出所有已注册的 model 名称"""
        return list(self._models.keys())

    def has_model(self, name: str) -> bool:
        """检查 model 是否存在"""
        return name in self._models

    def get_model_arguments(self, name: str, expanded: bool = False) -> dict[str, Any]:
        """获取 Model 的参数

        Args:
            name: Model 名称
            expanded: 是否展开继承链，获取合并后的完整参数
                     - False (默认): 返回原始参数（不包含继承的参数）
                     - True: 返回展开后的完整参数（包含从 template/model 继承合并后的参数）

        Returns:
            Model 的参数字典

        Raises:
            UnknownModelError: model 不存在

        Example:
            # 获取原始参数
            args = registry.get_model_arguments("deepseek-chatting")
            # {"model_id": "deepseek-chatting"}

            # 获取展开后的完整参数（继承链已合并）
            full_args = registry.get_model_arguments("deepseek-chatting", expanded=True)
            # {"model_id": "deepseek-chatting", "api_key": "...", "temperature": 0.7}
        """
        # 自动加载配置（首次使用，除非被显式禁用）
        self._ensure_auto_load()

        if name not in self._models:
            raise UnknownModelError(f"Model '{name}' not found")

        if expanded:
            #；解析继承链，获取合并后的完整参数
            resolved = self._resolve_model(name)
            return resolved.arguments.copy()
        else:
            # 返回原始参数（不包含继承的参数）
            return self._models[name].arguments.copy()

    # ========================================================================
    # 模型创建
    # ========================================================================

    def create_model(
        self, name: str, **overrides: Any
    ) -> Model:
        """通过 Model 名称创建 Model 实例

        Args:
            name: Model 名称，支持 "TEMPLATE/MODEL_ID" 格式
                  如果 models.yaml 中没有定义，会自动使用 TEMPLATE 模板创建，
                  并将 model_id 设置为 MODEL_ID
            **overrides: 覆盖参数

        Returns:
            Model 实例

        Raises:
            UnknownModelError: model 不存在且无法从模板自动创建
            UnknownModelError: Model 类未注册

        Example:
            # 使用预定义的 model
            model = registry.create_model("deepseek-chat")

            # 使用模板自动创建（假设 "openai" 模板已注册）
            model = registry.create_model("openai/gpt-4")
            # 等价于使用 openai 模板，model_id="gpt-4"

            # 覆盖参数
            model = registry.create_model("deepseek-chat", temperature=0.5)
        """
        # 自动加载配置（首次使用，除非被显式禁用）
        self._ensure_auto_load()

        # 如果 model 不存在，尝试从 "TEMPLATE/MODEL_ID" 格式自动创建
        if name not in self._models:
            if "/" in name:
                template_name, model_id = name.split("/", 1)
                if template_name in self._templates:
                    # 自动创建 model 配置
                    self._create_model_from_template(name, template_name, model_id)
                else:
                    raise UnknownModelError(
                        f"Model '{name}' not found and template '{template_name}' does not exist"
                    )
            else:
                raise UnknownModelError(f"Model '{name}' not found")

        # 解析配置（处理 template 继承）
        resolved = self._resolve_model(name)

        # 获取 Model 类
        model_class = self._classes.get(resolved.class_name)
        if model_class is None:
            raise UnknownModelError(
                f"Model class '{resolved.class_name}' not registered for model '{name}'"
            )

        # 合并参数：model arguments < overrides
        arguments = resolved.arguments.copy()
        if overrides:
            arguments.update(overrides)

        # 解析占位符替换（环境变量）
        arguments = self._resolve_substitutions(arguments)

        return model_class(**arguments)

    def _create_model_from_template(
        self, name: str, template_name: str, model_id: str
    ) -> None:
        """从模板自动创建 Model 配置

        Args:
            name: Model 名称（如 "openai/gpt-4"）
            template_name: 模板名称（如 "openai"）
            model_id: 模型 ID（如 "gpt-4"）
        """
        template = self._templates.get(template_name)
        if template is None:
            raise UnknownTemplateError(f"Template '{template_name}' not found")

        # 确定 class_name：优先使用模板的 class_name
        class_name = template.class_name or ""

        # 创建 model 配置
        # 格式：{__template: template_name, model_id: model_id, ...template_args}
        self._models[name] = ModelConfig(
            class_name=class_name,
            arguments={"model_id": model_id, **template.arguments},
            template=[template_name],
        )

    def _resolve_template(
        self, name: str, visited: Optional[set[str]] = None
    ) -> TemplateConfig:
        """解析 template 配置，处理 template 继承

        Template 只能以其他 template 为 template。
        多 template 时按列表顺序合并属性（后面的覆盖前面的）。
        """
        if visited is None:
            visited = set()

        if name in visited:
            raise CircularDependencyError(
                f"Circular template detected: {' -> '.join(visited)} -> {name}"
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

    def _resolve_model(
        self, name: str, visited: Optional[set[str]] = None
    ) -> ModelConfig:
        """解析 model 配置，处理 template 继承

        Model 可以以 template 或 model 为 template。
        多 template 时按列表顺序合并属性（后面的覆盖前面的）。
        """
        if visited is None:
            visited = set()

        if name in visited:
            raise CircularDependencyError(
                f"Circular template detected: {' -> '.join(visited)} -> {name}"
            )

        config = self._models.get(name)
        if config is None:
            raise UnknownModelError(f"Model '{name}' not found")

        if not config.templates:
            return config

        # 按顺序合并所有 template
        visited.add(name)
        merged_args: dict[str, Any] = {}
        merged_class: Optional[str] = None

        for template_name in config.templates:
            if template_name in self._templates:
                # 继承 template
                parent = self._resolve_template(template_name, visited.copy())
                merged_class = parent.class_name or merged_class
                merged_args.update(parent.arguments)
            elif template_name in self._models:
                # 继承 model
                parent = self._resolve_model(template_name, visited.copy())
                merged_class = parent.class_name or merged_class
                merged_args.update(parent.arguments)
            else:
                raise UnknownModelError(
                    f"Model '{name}' template '{template_name}' not found "
                    f"(neither template nor model)"
                )

        # 当前配置覆盖 template 配置
        merged_class = config.class_name or merged_class
        merged_args.update(config.arguments)

        if not merged_class:
            raise InvalidInheritanceError(
                f"Model '{name}' has no class specified and no template provides one"
            )

        return ModelConfig(
            class_name=merged_class,
            arguments=merged_args,
            template=[],  # 已解析，不需要保留
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
        - Model 级别覆盖，后加载的同名 model 完全替换先加载的
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

            # 加载 Models
            if "models" in data:
                models = data["models"]
                for name, config in models.items():
                    model_config = ModelConfig.from_dict(config)
                    self.register_model(
                        name=name,
                        class_name=model_config.class_name,
                        arguments=model_config.arguments,
                        templates=model_config.templates,
                        quiet=quiet,
                    )
                if not quiet:
                    print(f"[ModelRegistry] Loaded {len(models)} models from {path}")

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

    def clear(self) -> None:
        """清空所有注册（主要用于测试）"""
        self._models.clear()
        self._templates.clear()
        self._classes.clear()
        self._auto_load_needed = True
        self._register_builtin_classes()


# 全局单例实例
model_registry = ModelRegistry()


# 便捷函数（直接通过模块调用）
def create_model(
    name: str, **overrides: Any
) -> Model:
    """通过 Model 名称创建 Model 实例

    Args:
        name: Model 名称，支持 "TEMPLATE/MODEL_ID" 格式
        **overrides: 覆盖参数

    Returns:
        Model 实例
    """
    return model_registry.create_model(name, **overrides)


def get_model_class(name: str) -> Optional[type[Model]]:
    """通过类名获取 Model 类"""
    return model_registry.get_model_class(name)


def get_model_arguments(name: str, expanded: bool = False) -> dict[str, Any]:
    """获取 Model 的参数

    Args:
        name: Model 名称
        expanded: 是否展开继承链，获取合并后的完整参数

    Returns:
        Model 的参数字典
    """
    return model_registry.get_model_arguments(name, expanded)


def load_config(path: Union[str, Path], quiet: bool = False) -> None:
    """加载配置文件"""
    return model_registry.load_config(path, quiet)


def list_models() -> list[str]:
    """列出所有 model 名称"""
    return model_registry.list_models()


def list_templates() -> list[str]:
    """列出所有 template 名称"""
    return model_registry.list_templates()
