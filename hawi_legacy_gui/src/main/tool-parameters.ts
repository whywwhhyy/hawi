export const TOOL_CALL_PURPOSE_PARAMETER_DIRECTIVE = {
  name: "tool_call_purpose",
  schema: {
    type: "string",
    default: null,
    description: "【必填】用一句话说明本次工具调用的目的；允许与其他调用重复，会显示在工具标题旁边。未指定时工具仍会执行，但结果会附加错误提示，说明这会导致用户误解并影响自动审核 agent 的判断准确度。"
  },
  required: true
} as const;

export function toolCallPurposeEngineArgs(enabled: boolean): string[] {
  return enabled
    ? ["--extra-tool-parameter-json", JSON.stringify(TOOL_CALL_PURPOSE_PARAMETER_DIRECTIVE)]
    : [];
}
