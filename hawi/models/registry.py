"""
Model注册表（工厂），可以通过名字找到对应的Model类，也可以通过名字+参数dict直接创建Model
1. 名字可以是类名（自动识别），也可以是别名（用户设定）
2. 手动创建的注册表为空，全局单例预置内置模型
3. 支持手动注册新的模型

Example:
    # 使用全局单例（预置内置模型）
    from hawi.models import model_registry

    model = model_registry.create("OpenAIModel", {"model_id": "gpt-4", "api_key": "..."})

    # 手动创建独立的空注册表
    from hawi.models import ModelRegistry
    registry = ModelRegistry()
    registry.register(OpenAIModel)
"""

from typing import Dict, Type, Optional, TypeVar, Any
from threading import Lock

from .model import Model

ModelType = TypeVar("ModelType", bound=Model)


class ModelRegistry:
    """
    Model 注册表，支持类注册、别名设置和实例创建。

    手动创建的实例为空，不预置任何模型。
    如需带内置模型的单例，使用模块级 `get_global_registry()`。
    """

    def __init__(self) -> None:
        """初始化空的注册表。"""
        self._classes: Dict[str, Type[Model]] = {}
        self._aliases: Dict[str, str] = {}  # 别名 -> 真实类名
        self._defaults: Dict[str, Dict[str, Any]] = {}  # 类名 -> 默认参数
        self._alias_defaults: Dict[str, Dict[str, Any]] = {}  # 别名 -> 默认参数
        self._template_providers: Dict[str, list[str]] = {}  # template -> [provider_names]

    def register(
        self,
        model_class: Type[ModelType],
        alias: Optional[str] = None,
        aliases: Optional[list[str]] = None,
    ) -> "ModelRegistry":
        """
        注册一个Model类。

        Args:
            model_class: Model类（继承自Model）
            alias: 单个别名（可选）
            aliases: 多个别名列表（可选）

        Returns:
            self，支持链式调用

        Example:
            registry.register(OpenAIModel)
            registry.register(OpenAIModel, alias="gpt")
            registry.register(OpenAIModel, aliases=["gpt", "openai"])
        """
        class_name = model_class.__name__

        # 注册类名
        self._classes[class_name] = model_class

        # 注册单个别名
        if alias:
            self._aliases[alias] = class_name

        # 注册多个别名
        if aliases:
            for a in aliases:
                self._aliases[a] = class_name

        return self

    def unregister(self, name: str) -> bool:
        """
        取消注册一个Model类或别名。

        Args:
            name: 类名或别名

        Returns:
            是否成功移除
        """
        # 如果是类名，从_classes移除，并清理相关别名
        if name in self._classes:
            del self._classes[name]
            # 清理指向该类名的别名
            self._aliases = {k: v for k, v in self._aliases.items() if v != name}
            return True

        # 如果是别名，仅从_aliases移除
        if name in self._aliases:
            del self._aliases[name]
            return True

        return False

    def alias(self, name: str, alias_name: str) -> "ModelRegistry":
        """
        为已注册的类添加别名。

        Args:
            name: 已注册的类名
            alias_name: 别名

        Returns:
            self，支持链式调用

        Raises:
            KeyError: 如果name未注册
        """
        if name not in self._classes:
            raise KeyError(f"Model class '{name}' not registered")

        self._aliases[alias_name] = name
        return self

    def set_defaults(self, name: str, defaults: Dict[str, Any]) -> "ModelRegistry":
        """
        为指定模型类设置全局默认参数。

        Args:
            name: 类名或别名
            defaults: 默认参数字典

        Returns:
            self，支持链式调用

        Raises:
            KeyError: 如果name未注册

        Example:
            registry.set_defaults("OpenAIModel", {"temperature": 0.7, "max_tokens": 2048})
        """
        class_name = self._resolve_name(name)
        if class_name is None:
            raise KeyError(f"Model '{name}' not registered")

        self._defaults[class_name] = defaults.copy()
        return self

    def get_defaults(self, name: str) -> Dict[str, Any]:
        """
        获取指定模型类的全局默认参数。

        Args:
            name: 类名或别名

        Returns:
            默认参数字典，如果未设置返回空字典

        Raises:
            KeyError: 如果name未注册
        """
        class_name = self._resolve_name(name)
        if class_name is None:
            raise KeyError(f"Model '{name}' not registered")

        return self._defaults.get(class_name, {}).copy()

    def set_alias_defaults(self, alias: str, defaults: Dict[str, Any]) -> "ModelRegistry":
        """
        为指定别名设置默认参数（别名级别，优先级高于类级别）。

        Args:
            alias: 别名
            defaults: 默认参数字典

        Returns:
            self，支持链式调用

        Raises:
            KeyError: 如果别名未注册

        Example:
            registry.set_alias_defaults("deepseek-openai", {"temperature": 0.7})
        """
        if alias not in self._aliases:
            raise KeyError(f"Alias '{alias}' not registered")

        self._alias_defaults[alias] = defaults.copy()
        return self

    def get_alias_defaults(self, alias: str) -> Dict[str, Any]:
        """
        获取指定别名的默认参数。

        Args:
            alias: 别名

        Returns:
            默认参数字典，如果未设置返回空字典

        Raises:
            KeyError: 如果别名未注册
        """
        if alias not in self._aliases:
            raise KeyError(f"Alias '{alias}' not registered")

        return self._alias_defaults.get(alias, {}).copy()

    def clear_defaults(self, name: Optional[str] = None) -> "ModelRegistry":
        """
        清除全局默认参数。

        Args:
            name: 类名或别名，如果为None则清除所有默认参数

        Returns:
            self，支持链式调用
        """
        if name is None:
            self._defaults.clear()
            self._alias_defaults.clear()
        else:
            # 尝试作为类名清除
            if name in self._defaults:
                del self._defaults[name]
            # 尝试作为别名清除
            if name in self._alias_defaults:
                del self._alias_defaults[name]
            # 如果是别名，也清除指向该别名的类 defaults
            class_name = self._resolve_name(name)
            if class_name and class_name in self._defaults:
                del self._defaults[class_name]
        return self

    def _resolve_name(self, name: str) -> Optional[str]:
        """将类名或别名解析为真实类名"""
        if name in self._classes:
            return name
        if name in self._aliases:
            return self._aliases[name]
        return None

    def get_class(self, name: str) -> Optional[Type[Model]]:
        """
        通过名字获取Model类。

        Args:
            name: 类名或别名

        Returns:
            Model类，如果未找到返回None
        """
        # 直接是类名
        if name in self._classes:
            return self._classes[name]

        # 是别名，解析为类名
        if name in self._aliases:
            class_name = self._aliases[name]
            return self._classes.get(class_name)

        return None

    def create(self, name: str, params: Optional[Dict[str, Any]] = None) -> Model:
        """
        通过名字和参数创建Model实例。

        创建时会自动合并全局默认参数（优先级：传入参数 > 默认参数）。

        Args:
            name: 类名或别名
            params: 实例化参数字典

        Returns:
            Model实例

        Raises:
            KeyError: 如果name未注册
            TypeError: 如果params包含无效参数
        """
        model_class = self.get_class(name)
        if model_class is None:
            raise KeyError(f"Model '{name}' not found in registry")

        class_name = model_class.__name__
        params = params or {}

        # 合并参数（优先级从低到高）：
        # 1. 类级别默认参数
        # 2. 别名级别默认参数（覆盖类级别）
        # 3. 传入参数（最高优先级）
        class_defaults = self._defaults.get(class_name, {})
        alias_defaults = self._alias_defaults.get(name, {}) if name in self._aliases else {}
        merged_params = {**class_defaults, **alias_defaults, **params}

        return model_class(**merged_params)

    def list_models(self) -> Dict[str, Type[Model]]:
        """
        获取所有注册的Model类（副本）。

        Returns:
            类名到Model类的字典
        """
        return self._classes.copy()

    def list_aliases(self) -> Dict[str, str]:
        """
        获取所有别名映射（副本）。

        Returns:
            别名到类名的字典
        """
        return self._aliases.copy()

    def is_registered(self, name: str) -> bool:
        """
        检查名字是否已注册。

        Args:
            name: 类名或别名

        Returns:
            是否已注册
        """
        return name in self._classes or name in self._aliases

    def register_template_provider(self, template: str, provider: str) -> "ModelRegistry":
        """
        注册 template 与 provider 的关联。

        用于记录哪个 provider（来自 apikey.yaml）为某个 template 提供了 API key。

        Args:
            template: 模板名（alias）
            provider: provider 名称

        Returns:
            self，支持链式调用
        """
        if template not in self._template_providers:
            self._template_providers[template] = []
        if provider not in self._template_providers[template]:
            self._template_providers[template].append(provider)
        return self

    def get_template_providers(self, template: str) -> list[str]:
        """
        获取指定 template 对应的所有 providers。

        Args:
            template: 模板名（alias）

        Returns:
            provider 名称列表，如果没有关联返回空列表
        """
        return self._template_providers.get(template, []).copy()

    def list_templates_with_providers(self) -> Dict[str, list[str]]:
        """
        获取所有 template 及其对应的 providers。

        Returns:
            template 到 provider 列表的字典

        Example:
            >>> registry.list_templates_with_providers()
            {'kimi-openai': ['moonshot', 'moonshot-bao'], 'deepseek-openai': ['deepseek']}
        """
        return dict(self._template_providers)

    def clear(self) -> "ModelRegistry":
        """
        清空所有注册的类、别名和默认参数。

        Returns:
            self，支持链式调用
        """
        self._classes.clear()
        self._aliases.clear()
        self._defaults.clear()
        self._alias_defaults.clear()
        self._template_providers.clear()
        return self

    def __contains__(self, name: str) -> bool:
        """支持 'in' 操作符："OpenAIModel" in registry"""
        return self.is_registered(name)

    def __len__(self) -> int:
        """返回注册的类数量（不包括别名）"""
        return len(self._classes)

    def __repr__(self) -> str:
        classes = list(self._classes.keys())
        aliases = dict(self._aliases)
        defaults = {k: list(v.keys()) for k, v in self._defaults.items()}
        return f"ModelRegistry(classes={classes}, aliases={aliases}, defaults={defaults})"


# =============================================================================
# 框架级别的全局单例（不属于 ModelRegistry 类的核心职责）
# =============================================================================

# 全局单例锁
_singleton_lock: Lock = Lock()
_global_registry: Optional[ModelRegistry] = None


def _register_builtin_models(registry: ModelRegistry) -> None:
    """向注册表注册Hawi内置的所有模型类"""
    # 延迟导入，避免循环依赖
    from .openai import OpenAIModel
    from .anthropic import AnthropicModel
    from .deepseek import DeepSeekModel, DeepSeekOpenAIModel, DeepSeekAnthropicModel
    from .kimi import KimiModel, KimiOpenAIModel, KimiAnthropicModel
    from .minimax import MiniMaxModel, MiniMaxOpenAIModel, MiniMaxAnthropicModel
    from .strands import StrandsModel

    builtin_models = [
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

    for model_class in builtin_models:
        registry.register(model_class)


def get_global_registry() -> ModelRegistry:
    """
    获取全局单例注册表（线程安全，延迟初始化）。

    全局单例预置了Hawi所有内置模型（OpenAI、Anthropic、DeepSeek等）。

    Returns:
        预置了所有内置模型的全局ModelRegistry实例
    """
    global _global_registry

    if _global_registry is None:
        with _singleton_lock:
            # 双重检查，避免重复初始化
            if _global_registry is None:
                _global_registry = ModelRegistry()
                _register_builtin_models(_global_registry)

    return _global_registry


# 模块级别的全局单例（预置所有内置模型）
model_registry: ModelRegistry = get_global_registry()


__all__ = ["ModelRegistry", "model_registry", "get_global_registry"]
