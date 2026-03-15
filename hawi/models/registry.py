"""
Model注册表（工厂 + 对象池），支持类注册、实例创建/复用。

核心功能：
1. 名字可以是类名（自动识别），也可以是别名（用户设定）
2. 手动创建的注册表为空，全局单例预置内置模型
3. 支持手动注册新的模型类
4. **对象池模式**: 相同参数的模型实例可复用，减少资源消耗

    对象池设计原理：
    - 所有 Model 实现支持单线程异步并发（ainvoke/astream），因此异步调用可安全复用实例
    - 同步调用（invoke/stream）会阻塞事件循环，需要独占实例
    - `async_only=True`（默认）: 从对象池获取或创建实例，可被多个异步调用复用
    - `async_only=False`: 总是创建新实例，供同步调用独占使用

推荐用法（对象池模式）：
    from hawi.models import model_registry

    # 获取异步实例（可复用，推荐用于 ainvoke/astream）
    model = model_registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"})

    # 再次调用返回同一实例
    model2 = model_registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"})
    assert model is model2  # True

    # 获取同步实例（独占新实例，用于 invoke/stream）
    model3 = model_registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"}, async_only=False)

传统用法（每次都创建新实例）：
    model = model_registry.create("OpenAIModel", {"model_id": "gpt-4", "api_key": "..."})

手动创建注册表：
    from hawi.models import ModelRegistry
    registry = ModelRegistry()
    registry.register(OpenAIModel)
"""

import json
from typing import Dict, Type, Optional, TypeVar, Any
from threading import Lock

from .model import Model

ModelType = TypeVar("ModelType", bound=Model)


class ModelRegistry:
    """
    Model 注册表，支持类注册、别名设置和实例创建/复用。

    手动创建的实例为空，不预置任何模型。
    如需带内置模型的单例，使用模块级 `get_global_registry()`。

    支持对象池模式：
    - `obtain_model(name, args, async_only=True)`: 获取异步实例（可复用）
    - `obtain_model(name, args, async_only=False)`: 获取同步实例（独占新实例）
    - `release_model(name, args)`: 释放共享实例
    - `clear_pool()`: 清空对象池

    设计原理：
    - 所有 Model 实现都支持单线程异步并发（ainvoke/astream），因此 async_only=True 时可安全复用实例
    - 同步调用（invoke/stream）会阻塞事件循环，需要独占实例，因此 async_only=False 时创建新实例
    - 这是性能与安全的权衡：异步调用复用减少连接开销，同步调用隔离避免阻塞问题
    """

    def __init__(self) -> None:
        """初始化空的注册表。"""
        self._classes: Dict[str, Type[Model]] = {}
        self._aliases: Dict[str, str] = {}  # 别名 -> 真实类名
        self._defaults: Dict[str, Dict[str, Any]] = {}  # 类名 -> 默认参数
        self._alias_defaults: Dict[str, Dict[str, Any]] = {}  # 别名 -> 默认参数
        self._template_providers: Dict[str, list[str]] = {}  # template -> [provider_names]
        self._model_object_pool: Dict[str, Model] = {}  # 对象池: key -> Model实例
        self._model_object_pool_lock: Lock = Lock()  # 对象池锁

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

        merged_params = self._get_merged_params(name, params)
        return model_class(**merged_params)

    def obtain_model(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        async_only: bool = True,
    ) -> Model:
        """
        获取 Model 实例，支持对象池复用（推荐使用）。

        这是 `create()` 的增强版本，根据使用模式决定实例复用策略：
        - async_only=True:  从对象池获取或创建实例（异步调用支持并发复用，默认）
        - async_only=False: 总是创建新实例（同步调用需要独占实例）

        设计原理（重要）：
        1. 所有 Model 实现都支持单线程异步并发（ainvoke/astream 使用 async/await），
           因此 async_only=True 时可以安全复用同一实例
        2. 同步调用（invoke/stream）会阻塞事件循环，如果复用实例会导致其他任务等待，
           因此 async_only=False 时创建独占的新实例
        3. 这是性能与安全的权衡：异步复用减少连接开销，同步隔离避免阻塞问题

        Args:
            name: 类名或别名
            args: 实例化参数字典
            async_only: 是否仅用于异步调用（默认True，可复用实例）

        Returns:
            Model实例

        Raises:
            KeyError: 如果name未注册
            TypeError: 如果params包含无效参数

        Example:
            # 异步调用场景：获取可复用实例（推荐）
            # 所有异步调用可共享此实例，提高性能
            model = registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"})
            # await model.ainvoke(...)

            # 再次调用返回同一实例（节省连接资源）
            model2 = registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"})
            assert model is model2  # True

            # 同步调用场景：必须获取独占新实例
            # 同步调用会阻塞，不可与其他调用共享实例
            model3 = registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"}, async_only=False)
            # model3.invoke(...)  # 不会阻塞其他异步任务
            assert model is not model3  # True
        """
        if not async_only:
            # 同步调用需要独占实例，创建新的
            instance = self.create(name, args)
            instance._async_only = False  # 标记为同步实例
            return instance

        # 获取合并后的参数（用于生成pool key）
        merged_params = self._get_merged_params(name, args)

        # 生成对象池key
        pool_key = self._make_pool_key(name, merged_params)

        with self._model_object_pool_lock:
            # 检查对象池
            if pool_key in self._model_object_pool:
                return self._model_object_pool[pool_key]

            # 创建新实例并放入对象池（直接使用create，它会再次合并参数）
            instance = self.create(name, args)
            instance._async_only = True  # 标记为异步专用实例
            self._model_object_pool[pool_key] = instance
            return instance

    def _make_pool_key(self, name: str, merged_params: Dict[str, Any]) -> str:
        """生成对象池key（基于name和已合并的params）。

        Args:
            name: 类名或别名
            merged_params: 已合并的参数（由_get_merged_params生成）

        Returns:
            对象池key字符串
        """
        # 序列化所有值（不可序列化的使用str()）
        serializable = {}
        for k, v in sorted(merged_params.items()):  # 排序确保一致性
            if isinstance(v, (str, int, float, bool, type(None))):
                serializable[k] = v
            else:
                # 不可序列化的值使用str()表示
                serializable[k] = f"<obj:{str(v)}>"

        param_str = json.dumps(serializable, separators=(',', ':'))
        return f"{name}:{param_str}"

    def _get_merged_params(self, name: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """获取合并后的参数（与create()逻辑一致）。

        合并优先级（从低到高）：
        1. 类级别默认参数
        2. 别名级别默认参数
        3. 传入参数（最高优先级）
        """
        model_class = self.get_class(name)
        if model_class is None:
            return params or {}

        class_name = model_class.__name__
        params = params or {}

        class_defaults = self._defaults.get(class_name, {})
        alias_defaults = self._alias_defaults.get(name, {}) if name in self._aliases else {}

        return {**class_defaults, **alias_defaults, **params}

    def release_model(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        从对象池释放共享实例。

        Args:
            name: 类名或别名
            args: 实例化参数字典

        Returns:
            是否成功释放

        Example:
            registry.release_model("deepseek-openai", {"model_id": "deepseek-chat"})
        """
        merged_params = self._get_merged_params(name, args)
        pool_key = self._make_pool_key(name, merged_params)

        with self._model_object_pool_lock:
            if pool_key in self._model_object_pool:
                del self._model_object_pool[pool_key]
                return True
            return False

    def clear_pool(self) -> "ModelRegistry":
        """
        清空对象池中的所有共享实例。

        Returns:
            self，支持链式调用

        Example:
            registry.clear_pool()
        """
        with self._model_object_pool_lock:
            self._model_object_pool.clear()
        return self

    def get_pool_info(self) -> Dict[str, Any]:
        """
        获取对象池信息。

        Returns:
            包含池大小和keys的字典

        Example:
            >>> registry.get_pool_info()
            {'size': 2, 'keys': ['deepseek-openai:{"model_id":"deepseek-chat"}', ...]}
        """
        with self._model_object_pool_lock:
            return {
                'size': len(self._model_object_pool),
                'keys': list(self._model_object_pool.keys()),
            }

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
        清空所有注册的类、别名、默认参数和对象池。

        Returns:
            self，支持链式调用
        """
        self._classes.clear()
        self._aliases.clear()
        self._defaults.clear()
        self._alias_defaults.clear()
        self._template_providers.clear()
        self.clear_pool()
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
        pool_info = self.get_pool_info()
        return f"ModelRegistry(classes={classes}, aliases={aliases}, defaults={defaults}, pool_size={pool_info['size']})"


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
