import sys
import warnings

# 过滤 Anthropic SDK 内部的 Pydantic 序列化警告
warnings.filterwarnings(
    "ignore",
    message="PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)

from strands import tool
from hawi.utils.terminal import user_select

# 使用 model_compatibility 模块中的 DeepSeekModel
from hawi.agent.models import DeepSeekModel, KimiOpenAIModel, KimiAnthropicModel
from hawi.agent import Agent,CachePointHook
from hawi.agent_tools.python_interpreter import PythonInterpreter

ds_model = DeepSeekModel(
    client_args={
        "api_key": "sk-22c80777f606402aa416fe3a325c1c66",
        "base_url": "https://api.deepseek.com",
    },
    model_id="deepseek-chat",
    params={
        "temperature": 1,
        "max_tokens": 4096,
    },
)

# 使用 KimiOpenAIModel 修复 thinking 模式下的 tool calling 问题

# Kimi OpenAI API - 禁用 thinking 模式 (工具调用更稳定)
kimi_openai_model = KimiOpenAIModel(
    client_args={
        "api_key": "sk-EIUw4OOpex1kaEDigO7SqT6avmP2h2EDXyAbIWQxz9VUpAdS",
        "base_url": "https://api.moonshot.cn/v1",
    },
    model_id="kimi-k2.5",
    params={
        # 禁用 thinking 模式时，temperature 必须是 0.6
        "temperature": 0.6,
        "thinking": {"type": "disabled"},
    },
)

# Kimi OpenAI API - 启用 thinking 模式 (会有 reasoning_content)
kimi_openai_thinking_model = KimiOpenAIModel(
    client_args={
        "api_key": "sk-EIUw4OOpex1kaEDigO7SqT6avmP2h2EDXyAbIWQxz9VUpAdS",
        "base_url": "https://api.moonshot.cn/v1",
    },
    model_id="kimi-k2.5",
    # 默认启用 thinking 模式，不需要额外参数
)

# 使用 KimiAnthropicModel 修复 Pydantic 序列化警告
kimi_subscript_model = KimiAnthropicModel(
    client_args={
        "api_key": "sk-kimi-hU7a0PKq60BKCh5kiBLrVIeCH33s2AmB6E7ySGjW3bxDJzx6SHtmaRrcficPegVC",
        "base_url": "https://api.kimi.com/coding/",
    },
    max_tokens=4096,
    model_id="kimi-k2.5",
)

LLM_PROVIDERS = {
    'deepseek': ds_model,
    'kimi-oai': kimi_openai_model,
    'kimi-oai-thinking': kimi_openai_thinking_model,
    'kimi-subscript': kimi_subscript_model,
}

def create_agent(llm_provider:str) -> Agent:
    executor = PythonInterpreter(work_dir=".python_vm")
    model = LLM_PROVIDERS.get(llm_provider, ds_model)

    return Agent(
        name=llm_provider,
        model=model,
        tools=[tool(t) for t in executor.get_tools()],
        system_prompt="你是一个专门用来管理当前python环境的Agent，通过execute工具，你可以完成任何任务。你也可以使用restart_server重启解释器来清空状态，或使用install_dependency安装依赖包。",
        hooks=[CachePointHook()],
    )

def main():
    argv = sys.argv[1:]
    for llm_provider in LLM_PROVIDERS:
        if llm_provider in argv:
            argv.remove(llm_provider)
            llm_provider = llm_provider
            break
    else:
        llm_provider = user_select(list(LLM_PROVIDERS.keys()), "Select a provider")
    assert isinstance(llm_provider, str)
    agent = create_agent(llm_provider)
    
    if argv:
        agent(argv[0])
    else:
        import readline
        while True:
            try:
                prompt = input(">>> ")
                if prompt.lower() in ['exit', 'quit', 'q']:
                    break
                agent(prompt)
                print()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nExiting...")
                break


if __name__ == "__main__":
    main()