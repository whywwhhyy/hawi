"""
Hawi 配置加载模块

支持从 YAML 文件加载模型模板并注册到 ModelRegistry。

配置层级（优先级从低到高）：
1. 内置配置: hawi/config/models.yaml
2. 项目配置: ./models.yaml（完全覆盖内置，null/~ 表示删除）

Example:
    # 在 main.py 中初始化
    from hawi.config import init_registry_from_yaml
    from hawi.models import model_registry

    # 自动加载内置 + 项目配置
    init_registry_from_yaml()

    # 现在可以通过模板名创建模型
    model = model_registry.create("deepseek-openai", {"model_id": "deepseek-chat"})
"""

from pathlib import Path
from typing import Any

import yaml

from hawi.models import ModelRegistry, model_registry


def _get_builtin_config_path() -> Path:
    """获取内置配置文件路径"""
    return Path(__file__).parent / "models.yaml"


def _get_project_config_path() -> Path:
    """获取项目配置文件路径"""
    return Path.cwd() / "models.yaml"


def load_yaml_config(path: Path) -> dict[str, Any]:
    """
    加载 YAML 配置文件。

    Args:
        path: 配置文件路径

    Returns:
        配置字典，文件不存在或解析失败返回空字典
    """
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, IOError):
        return {}


def merge_templates(
    builtin: dict[str, Any], project: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """
    合并内置和项目模板配置。

    合并规则：
    - 项目配置中的模板完全覆盖内置模板
    - 项目配置中值为 null/~ 的模板表示删除

    Args:
        builtin: 内置配置字典
        project: 项目配置字典

    Returns:
        合并后的模板字典
    """
    # 从内置配置获取模板
    templates = dict(builtin.get("templates", {}))

    # 应用项目配置
    project_templates = project.get("templates", {})
    for name, config in project_templates.items():
        if config is None or config == "~":
            # 删除模板
            templates.pop(name, None)
        else:
            # 完全覆盖
            templates[name] = config

    return templates


def _register_builtin_model_classes(registry: ModelRegistry) -> None:
    """注册 Hawi 内置的模型类到 registry。"""
    # 延迟导入避免循环依赖
    from hawi.models import (
        OpenAIModel,
        AnthropicModel,
        DeepSeekModel,
        KimiModel,
        MiniMaxModel,
        StrandsModel,
    )

    builtin_models = [
        OpenAIModel,
        AnthropicModel,
        DeepSeekModel,
        KimiModel,
        MiniMaxModel,
        StrandsModel,
    ]

    for model_class in builtin_models:
        registry.register(model_class)


def init_registry_from_yaml(
    registry: ModelRegistry | None = None,
    builtin_path: Path | None = None,
    project_path: Path | None = None,
) -> ModelRegistry:
    """
    从 YAML 配置文件初始化 ModelRegistry。

    加载内置配置和项目配置（如果存在），将模板注册到 registry。

    Args:
        registry: 要初始化的注册表，默认使用全局 model_registry
        builtin_path: 内置配置文件路径，默认使用 hawi/config/models.yaml
        project_path: 项目配置文件路径，默认使用 ./models.yaml

    Returns:
        初始化后的注册表

    Example:
        # 基本用法
        init_registry_from_yaml()
        model = model_registry.create("kimi-openai", {"model_id": "kimi-k2.5"})

        # 使用自定义注册表
        custom_registry = ModelRegistry()
        init_registry_from_yaml(registry=custom_registry)
    """
    global _template_configs, _providers_config

    registry = registry or model_registry

    # 0. 清空之前的状态（防止多次调用时状态残留）
    _template_configs.clear()
    _providers_config.clear()

    # 1. 注册内置模型类
    _register_builtin_model_classes(registry)

    # 确定配置路径
    builtin_path = builtin_path or _get_builtin_config_path()
    project_path = project_path or _get_project_config_path()

    # 加载配置
    builtin_config = load_yaml_config(builtin_path)
    project_config = load_yaml_config(project_path)

    # 合并模板
    templates = merge_templates(builtin_config, project_config)

    # 注册到 registry
    for template_name, config in templates.items():
        _register_template(registry, template_name, config)

    return registry


# 存储 template 的原始配置（包含 model_ids）
_template_configs: dict[str, dict[str, Any]] = {}

# 存储所有 provider 的完整配置（支持多 provider 查询）
# 格式: {provider_name: {"apikey": "...", "templates": [...], "website": "..."}}
_providers_config: dict[str, dict[str, Any]] = {}


def _register_template(
    registry: ModelRegistry, name: str, config: dict[str, Any]
) -> None:
    """
    将单个模板注册到 registry。

    Args:
        registry: ModelRegistry 实例
        name: 模板名称（注册为 alias）
        config: 模板配置字典

    Raises:
        ValueError: 配置缺少必要字段或 model class 不存在
    """
    # 获取 model class
    class_name = config.get("class")
    if not class_name:
        raise ValueError(f"Template '{name}' missing required 'class' field")

    model_class = registry.get_class(class_name)
    if model_class is None:
        raise ValueError(f"Unknown model class '{class_name}' in template '{name}'")

    # 注册模板为 alias
    registry.register(model_class, alias=name)

    # 设置默认参数（排除元数据字段）
    # class: Model 类名（已用于查找类）
    # model_ids: 支持的 model_id 列表（用于选择，不传递给模型）
    excluded_fields = ("class", "model_ids")
    defaults = {k: v for k, v in config.items() if k not in excluded_fields}
    if defaults:
        # 使用别名级别 defaults，确保不同模板的配置互不干扰
        registry.set_alias_defaults(name, defaults)

    # 保存原始配置（包含 model_ids 供后续选择）
    _template_configs[name] = config


def get_template_config(name: str) -> dict[str, Any] | None:
    """
    获取模板的原始配置（包含 model_ids）。

    Args:
        name: 模板名称

    Returns:
        模板配置字典，不存在返回 None
    """
    return _template_configs.get(name)


def get_available_model_ids(name: str) -> list[str]:
    """
    获取模板支持的 model_id 列表。

    Args:
        name: 模板名称

    Returns:
        model_id 列表，模板不存在或无配置返回空列表

    Example:
        >>> get_available_model_ids("deepseek-openai")
        ['deepseek-chat', 'deepseek-reasoner']
    """
    config = _template_configs.get(name)
    if not config:
        return []

    model_ids = config.get("model_ids", [])
    return model_ids if isinstance(model_ids, list) else []


def select_model_id(name: str, argv: list[str] | None = None) -> str | None:
    """
    从 argv 或交互式选择 model_id。

    Args:
        name: 模板名称
        argv: 命令行参数列表，检查其中是否包含可用的 model_id

    Returns:
        选中的 model_id，用户取消返回 None

    Example:
        >>> select_model_id("deepseek-openai", ["deepseek-chat", "hello"])
        'deepseek-chat'
    """
    from hawi.utils.terminal import user_select

    available = get_available_model_ids(name)
    if not available:
        return None

    if len(available) == 1:
        return available[0]

    # 检查 argv
    if argv:
        for arg in argv:
            if arg in available:
                return arg

    # 交互式选择
    return user_select(available, f"Select model for {name}:")


# =============================================================================
# Provider 配置查询接口
# =============================================================================

def list_providers() -> list[str]:
    """
    获取所有已加载的 provider 名称列表。

    Returns:
        provider 名称列表

    Example:
        >>> from hawi.config import list_providers
        >>> list_providers()
        ['deepseek', 'moonshot', 'moonshot-bao', 'kimi-code']
    """
    return list(_providers_config.keys())


def get_provider_config(provider_name: str) -> dict[str, Any] | None:
    """
    获取指定 provider 的完整配置。

    Args:
        provider_name: provider 名称

    Returns:
        provider 配置字典，包含 apikey、templates、website 等
        不存在返回 None

    Example:
        >>> from hawi.config import get_provider_config
        >>> get_provider_config('moonshot')
        {'apikey': 'sk-xxx', 'templates': ['kimi-openai'], 'website': '...'}
    """
    return _providers_config.get(provider_name)


def get_provider_for_template(template_name: str) -> list[str]:
    """
    获取支持指定 template 的所有 provider 名称。

    Args:
        template_name: 模板名称

    Returns:
        provider 名称列表

    Example:
        >>> from hawi.config import get_provider_for_template
        >>> get_provider_for_template('kimi-openai')
        ['moonshot', 'moonshot-bao']
    """
    providers = []
    for name, config in _providers_config.items():
        templates = config.get("templates", [])
        if template_name in templates:
            providers.append(name)
    return providers


def get_template_providers_with_configs(template_name: str) -> list[tuple[str, dict[str, Any]]]:
    """
    获取指定 template 对应的所有 provider 及其配置。

    用于显示所有可选的 provider 及其详细信息（如 website）。

    Args:
        template_name: 模板名称

    Returns:
        (provider_name, config) 元组列表

    Example:
        >>> from hawi.config import get_template_providers_with_configs
        >>> get_template_providers_with_configs('kimi-openai')
        [
            ('moonshot', {'apikey': 'sk-xxx', 'templates': ['kimi-openai'], 'website': '...'}),
            ('moonshot-bao', {'apikey': 'sk-yyy', 'templates': ['kimi-openai'], 'website': '...'}),
        ]
    """
    result = []
    for name, config in _providers_config.items():
        templates = config.get("templates", [])
        if template_name in templates:
            result.append((name, config.copy()))
    return result


def list_all_template_provider_mappings() -> dict[str, list[str]]:
    """
    获取所有 template 到 providers 的映射。

    Returns:
        template 名称到 provider 名称列表的字典

    Example:
        >>> from hawi.config import list_all_template_provider_mappings
        >>> list_all_template_provider_mappings()
        {
            'kimi-openai': ['moonshot', 'moonshot-bao'],
            'deepseek-openai': ['deepseek'],
            'deepseek-anthropic': ['deepseek'],
        }
    """
    mappings: dict[str, list[str]] = {}
    for provider_name, config in _providers_config.items():
        for template_name in config.get("templates", []):
            if template_name not in mappings:
                mappings[template_name] = []
            mappings[template_name].append(provider_name)
    return mappings


def load_apikey_config(path: Path | None = None) -> dict[str, Any]:
    """
    加载 apikey.yaml 配置（新格式）。

    格式：
    ```yaml
    providers:
      provider_name:
        apikey: sk-xxx
        templates: [template1, template2]
    ```

    Args:
        path: 配置文件路径，默认使用 ./apikey.yaml

    Returns:
        配置字典，文件不存在或解析失败返回空字典
    """
    path = path or (Path.cwd() / "apikey.yaml")

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
            return {}
    except (yaml.YAMLError, IOError):
        return {}


def inject_api_keys(
    registry: ModelRegistry,
    apikey_config: dict[str, Any],
) -> None:
    """
    将 apikey.yaml 中的密钥注入到 registry 的默认参数中。

    同时保存所有 provider 配置到模块级存储，支持多 provider 查询。

    Args:
        registry: ModelRegistry 实例
        apikey_config: apikey.yaml 解析后的配置（新格式）

    Example:
        apikey_config = load_apikey_config()
        inject_api_keys(model_registry, apikey_config)

        # 查询所有 provider
        from hawi.config import list_providers, get_provider_config
        providers = list_providers()  # ['moonshot', 'moonshot-bao', ...]
        config = get_provider_config('moonshot')  # {'apikey': '...', 'templates': [...]}
    """
    global _providers_config

    providers = apikey_config.get("providers", {})

    # 保存完整 provider 配置（用于后续查询）
    for provider_name, provider_config in providers.items():
        if provider_config.get("apikey"):  # 只保存有 apikey 的
            _providers_config[provider_name] = provider_config.copy()

    # 为每个模板注入 api_key（最后一个生效，保持向后兼容）
    for provider_name, provider_config in providers.items():
        apikey = provider_config.get("apikey")
        if not apikey:
            continue

        templates = provider_config.get("templates", [])
        for template_name in templates:
            if template_name not in registry.list_aliases():
                continue

            # 记录 template 与 provider 的关联
            registry.register_template_provider(template_name, provider_name)

            defaults = registry.get_alias_defaults(template_name)
            defaults["api_key"] = apikey
            registry.set_alias_defaults(template_name, defaults)


def setup_registry(
    registry: ModelRegistry | None = None,
    apikey_path: Path | None = None,
    models_path: Path | None = None,
) -> ModelRegistry:
    """
    完整初始化 ModelRegistry，加载所有配置和密钥。

    这是高层封装函数，一次性完成：
    1. 加载模型模板配置（内置 + 项目）
    2. 加载 apikey 配置
    3. 注入密钥到 registry

    Args:
        registry: 要使用的注册表，默认使用全局 model_registry
        apikey_path: apikey.yaml 路径，默认使用 ./apikey.yaml
        models_path: 项目级 models.yaml 路径，默认使用 ./models.yaml

    Returns:
        初始化完成的注册表

    Example:
        from hawi.config import setup_registry
        from hawi.models import model_registry

        # 完整初始化
        setup_registry()

        # 现在可以直接创建模型
        model = model_registry.create("deepseek-openai", {"model_id": "deepseek-chat"})
    """
    registry = registry or model_registry

    # 1. 加载模型模板配置
    init_registry_from_yaml(registry=registry, project_path=models_path)

    # 2. 加载并注入 apikey
    apikey_config = load_apikey_config(apikey_path)
    inject_api_keys(registry, apikey_config)

    return registry


__all__ = [
    "setup_registry",
    "init_registry_from_yaml",
    "load_apikey_config",
    "inject_api_keys",
    "load_yaml_config",
    "merge_templates",
    "get_template_config",
    "get_available_model_ids",
    "select_model_id",
    # Provider 查询接口
    "list_providers",
    "get_provider_config",
    "get_provider_for_template",
    "get_template_providers_with_configs",
    "list_all_template_provider_mappings",
]
