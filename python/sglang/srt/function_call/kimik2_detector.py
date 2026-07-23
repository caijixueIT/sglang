import json
import logging
import re
from typing import List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import (
    BaseFormatDetector,
    StructuralTag,
    get_model_structural_tag,
)
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.tool_call_metrics import (
    begin_stream_tool,
    close_open_stream_tools,
    fail_stream_tool,
    forget_stream_tool,
    observe_nonstandard_tool_id,
    observe_undefined_function,
    reset_stream_tool_metrics,
    succeed_stream_tool,
)
from sglang.srt.function_call.utils import _is_complete_json

logger = logging.getLogger(__name__)

_KIMI_K2_SPECIAL_TOKENS = [
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
]

_KIMI_NON_STRICT_ARGUMENTS_SCHEMA = {"type": "object"}


def _strip_special_tokens(text: str) -> str:
    """Remove all Kimi-K2 tool-call special tokens from text."""
    for token in _KIMI_K2_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


class KimiK2Detector(BaseFormatDetector):
    """
    Detector for Kimi K2 / K2.5 model function call format.

    Format Structure (standard):
    ```
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{func_name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    <|tool_calls_section_end|>
    ```

    Format Structure (bare counter — model omits function name):
    ```
    <|tool_call_begin|>{counter}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    ```

    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
    """

    def __init__(self):
        super().__init__()

        self.bot_token: str = "<|tool_calls_section_begin|>"
        self.eot_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_argument_begin_token: str = "<|tool_call_argument_begin|>"

        # Capture tool_call_id broadly: the model may emit standard IDs
        # like "functions.ReadFile:0" or bare call counters like "3".
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^\s<|]+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^\s<|]+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)",
            re.DOTALL,
        )

        self._last_arguments = ""
        self._current_stream_function_name: str | None = None
        self._last_invalid_function_id: str | None = None
        self._last_invalid_function_name: str | None = None
        self._last_invalid_function_index: int | None = None
        self._last_invalid_reason: str | None = None
        self._last_tool_call_resolution_reason: str = "standard"
        self._current_stream_resolution_reason: str = "standard"
        self._tool_metrics_parser = "kimi_k2"
        reset_stream_tool_metrics(self)

        # Tool call ID with optional `functions.` / `functions_` prefix and
        # `:` or `_` separator before the call index. Accepted forms:
        #   "functions.search:0"  "functions_search_0"
        #   "functions.search_0"  "functions_search:0"
        #   "search:0"            "search_0"
        # The name part is non-greedy so a trailing "_<digits>" is recognized
        # as the index even when the name itself contains underscores
        # (e.g. "get_weather_0" → name="get_weather", index=0).
        self.tool_call_id_regex = re.compile(
            r"^(?:functions[._])?(?P<name>[\w.\-]+?)[:_](?P<index>\d+)$"
        )
        # Bare call counter: "0", "3" (model uses auto-incrementing counter)
        self.tool_call_id_counter_regex = re.compile(r"^\d+$")

        # Fallback recovery for malformed IDs. Only used when the standard
        # regex fails or its name is not in the tools list, and the recovered
        # name MUST exist in the tools table — otherwise we keep treating it as
        # an undefined function (never forward hallucinated calls).
        #
        # Case D — parenthesized / float index:  "functions(file_read):2.0"
        self.tool_call_id_paren_regex = re.compile(
            r"^functions?\s*\(\s*(?P<name>[\w.\-]+?)\s*\)\s*[:_]?\s*"
            r"(?P<index>\d+)(?:\.\d+)?$"
        )
        # Case G — missing separator between name and index: "functions.bash30",
        # "functions.file_edit152". Name is non-greedy so the trailing run of
        # digits is treated as the index; the result is only accepted when the
        # name is a real tool (see _recover_malformed_tool_call_id).
        self.tool_call_id_nosep_regex = re.compile(
            r"^(?:functions[._])?(?P<name>[A-Za-z][\w.\-]*?)(?P<index>\d+)$"
        )

    def _parse_tool_call_id(
        self, function_id: str, tools: List[Tool], function_args: str = None
    ):
        """Parse a tool call ID into (function_name, call_index).

        Standard format: "functions.ReadFile:0" → ("ReadFile", 0)
        Bare counter:    "3" → call_index=3, infer name from arguments.

        The bare counter is a conversation-level auto-increment, NOT an index
        into the tools list. The function name is inferred by matching argument
        keys against tool parameter schemas.
        """
        tool_indices = self._get_tool_indices(tools)
        available_tools = list(tool_indices.keys())
        self._last_invalid_function_id = None
        self._last_invalid_function_name = None
        self._last_invalid_function_index = None
        self._last_invalid_reason = None
        self._last_tool_call_resolution_reason = "standard"

        m = self.tool_call_id_regex.match(function_id)
        if m:
            function_name = m.group("name")
            call_index = int(m.group("index"))
            if function_name in tool_indices:
                return function_name, call_index
            # Standard parse succeeded but name is unknown — try malformed
            # recovery (e.g. trailing ".0" left an odd name) before giving up.
            recovered = self._recover_malformed_tool_call_id(
                function_id, tool_indices
            )
            if recovered is not None:
                function_name, call_index, reason = recovered
                observe_nonstandard_tool_id(
                    self._tool_metrics_parser, reason, function_id, function_name
                )
                self._last_tool_call_resolution_reason = reason
                return function_name, call_index
            metric_function_name = self._metric_tool_name(
                function_id, function_name, tool_indices
            )
            self._last_invalid_function_id = function_id
            self._last_invalid_function_name = metric_function_name
            self._last_invalid_function_index = call_index
            self._last_invalid_reason = "tool_name_not_in_tool_set"
            logger.warning(
                "tool_call_parse_error | parser=%s reason=%s "
                "function_id=%r parsed_name=%r metric_func_name=%r "
                "parsed_index=%s metric_func_required_params=%s "
                "available_tools=%s available_tool_schemas=%s forward_unknown=%s "
                "args_snippet=%.300r",
                self._tool_metrics_parser,
                self._last_invalid_reason,
                function_id,
                function_name,
                metric_function_name,
                call_index,
                self._tool_required_params(tools, metric_function_name),
                available_tools,
                self._tool_schema_snapshot(tools),
                envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get(),
                function_args[:300] if function_args else "",
            )
            if envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return function_name, call_index
            return None, call_index

        if self.tool_call_id_counter_regex.match(function_id):
            call_index = int(function_id)
            name = self._infer_tool_name(tools, function_args)
            if name:
                observe_nonstandard_tool_id(
                    self._tool_metrics_parser,
                    "argument_schema_inference",
                    function_id,
                    name,
                )
                self._last_tool_call_resolution_reason = "argument_schema_inference"
                return name, call_index
            self._last_invalid_function_id = function_id
            self._last_invalid_function_name = function_id or "<empty>"
            self._last_invalid_function_index = call_index
            self._last_invalid_reason = "argument_schema_inference_failed"
            return None, call_index

        # Standard regex did not match at all — attempt malformed recovery
        # (Case D parenthesized form, Case G missing separator).
        recovered = self._recover_malformed_tool_call_id(function_id, tool_indices)
        if recovered is not None:
            function_name, call_index, reason = recovered
            observe_nonstandard_tool_id(
                self._tool_metrics_parser, reason, function_id, function_name
            )
            self._last_tool_call_resolution_reason = reason
            return function_name, call_index

        metric_function_name = self._metric_tool_name(
            function_id, None, tool_indices
        )
        self._last_invalid_function_id = function_id
        self._last_invalid_function_name = metric_function_name
        self._last_invalid_function_index = 0
        self._last_invalid_reason = (
            "tool_name_not_in_tool_set"
            if self._extract_tool_name(function_id, None) is not None
            else "malformed_tool_call_id"
        )
        logger.warning(
            "tool_call_parse_error | parser=%s reason=%s "
            "function_id=%r metric_func_name=%r metric_func_required_params=%s "
            "available_tools=%s available_tool_schemas=%s args_snippet=%.300r",
            self._tool_metrics_parser,
            self._last_invalid_reason,
            function_id,
            metric_function_name,
            self._tool_required_params(tools, metric_function_name),
            available_tools,
            self._tool_schema_snapshot(tools),
            function_args[:300] if function_args else "",
        )
        return None, 0

    def _extract_tool_name(self, function_id: str, parsed_name: str | None):
        if parsed_name:
            return parsed_name
        for pattern in (
            self.tool_call_id_regex,
            self.tool_call_id_paren_regex,
            self.tool_call_id_nosep_regex,
        ):
            m = pattern.match(function_id)
            if m and "name" in m.groupdict():
                return m.group("name")
        return None

    def _metric_tool_name(self, function_id: str, parsed_name: str | None, tool_indices):
        """Return a low-cardinality best-effort tool name for invalid target metrics."""
        name = self._extract_tool_name(function_id, parsed_name)
        if not name:
            return function_id or "<empty>"
        for cand in (name, name.replace("-", "_"), name.replace("_", "-")):
            if cand in tool_indices:
                return cand
        stripped = re.sub(r"\d+$", "", name)
        if stripped:
            for cand in (stripped, stripped.replace("-", "_"), stripped.replace("_", "-")):
                if cand in tool_indices:
                    return cand
            alias_base = stripped
            if alias_base.endswith("_file"):
                file_alias = "file_" + alias_base[: -len("_file")]
                if file_alias in tool_indices:
                    return file_alias
                return file_alias
            return stripped
        return name

    def _tool_schema_snapshot(self, tools: List[Tool], max_chars: int = 4000):
        schemas = []
        for tool in tools or []:
            function = getattr(tool, "function", None)
            name = getattr(function, "name", None)
            if not name:
                continue
            schemas.append(
                {
                    "name": name,
                    "parameters": getattr(function, "parameters", None),
                }
            )
        try:
            text = json.dumps(schemas, ensure_ascii=False, default=str)
        except Exception:
            text = repr(schemas)
        if len(text) > max_chars:
            return text[:max_chars] + "...<truncated>"
        return text

    def _tool_required_params(self, tools: List[Tool], tool_name: str | None):
        if not tool_name:
            return None
        for tool in tools or []:
            function = getattr(tool, "function", None)
            if getattr(function, "name", None) != tool_name:
                continue
            parameters = getattr(function, "parameters", None)
            if isinstance(parameters, dict):
                required = parameters.get("required")
                if isinstance(required, list):
                    return required
            return []
        return None

    def _argument_names(self, function_args: str | None):
        if not function_args:
            return ["<none>"]
        try:
            parsed_args = json.loads(self._trim_argument_payload(function_args))
        except (json.JSONDecodeError, TypeError):
            return ["<unparseable>"]
        if isinstance(parsed_args, dict):
            argument_names = sorted(str(key) for key in parsed_args.keys())
            return [" @ ".join(argument_names)] if argument_names else ["<none>"]
        return [type(parsed_args).__name__]

    def _trim_argument_payload(self, function_args: str | None) -> str:
        if not function_args:
            return ""
        end_positions = [
            position
            for token in (self.tool_call_end_token, self.eot_token)
            if (position := function_args.find(token)) >= 0
        ]
        if end_positions:
            return function_args[: min(end_positions)]
        return function_args

    def _observe_invalid_tool_target(
        self,
        reason: str,
        parsed_tool_name: str,
        function_args: str | None,
    ):
        for argument_name in self._argument_names(function_args):
            observe_undefined_function(
                self._tool_metrics_parser,
                parsed_tool_name,
                reason,
                argument_name,
            )

    def incomplete_stream_reason(self) -> str:
        current_text = getattr(self, "_buffer", "") or ""
        if self.tool_call_start_token in current_text:
            if self.tool_call_argument_begin_token not in current_text:
                return "missing_argument_begin"

            match = self.stream_tool_call_portion_regex.search(current_text)
            if not match:
                return "incomplete_tool_call_structure"

            function_args = match.group("function_arguments")
            parsed_args = self._trim_argument_payload(function_args)
            has_tool_call_end = self.tool_call_end_token in function_args

            if _is_complete_json(parsed_args):
                if not has_tool_call_end:
                    return "missing_tool_call_end_with_complete_json"
                return "incomplete_stream"

            if has_tool_call_end:
                return "malformed_arguments_with_end_token"
            return "incomplete_arguments_json"

        if getattr(self, "current_tool_name_sent", False):
            return "incomplete_tool_call_structure"
        return "incomplete_stream"

    def _recover_malformed_tool_call_id(self, function_id: str, tool_indices):
        """Best-effort recovery of malformed tool_call_ids.

        Handles a few observed model glitches, but ONLY returns a result when
        the recovered name is a real tool in ``tool_indices``. Hallucinated /
        unknown names still fall through to the undefined-function path.

        Case D — parenthesized form with optional float index:
            "functions(file_read):2.0" -> ("file_read", 2)
        Case G — missing separator between name and index:
            "functions.bash30"      -> ("bash", 30)
            "functions.file_edit152" -> ("file_edit", 152)
        Case A — hyphen used instead of underscore in the name:
            "functions.pending-tracker:30" -> ("pending_tracker", 30)
            "functions.file-edit:177"      -> ("file_edit", 177)
        """
        # Case A: standard shape parses, but the name uses '-' where the real
        # tool uses '_' (or vice versa). Only accept when the normalized name is
        # a real tool, so PowerShell-style hallucinations like "Start-Sleep"
        # (no matching tool) correctly stay undefined.
        m = self.tool_call_id_regex.match(function_id)
        if m and m.group("name") not in tool_indices:
            name = m.group("name")
            recovered_alias = self._recover_tool_name_alias(name, tool_indices)
            if recovered_alias is not None:
                function_name, reason = recovered_alias
                return function_name, int(m.group("index")), reason
            stripped = re.sub(r"\d+$", "", name)
            if stripped and stripped in tool_indices:
                return stripped, int(m.group("index")), "strip_trailing_digits"
            recovered_alias = self._recover_tool_name_alias(
                stripped or name, tool_indices
            )
            if recovered_alias is not None:
                function_name, reason = recovered_alias
                return function_name, int(m.group("index")), reason

        # Case D: functions(name):index(.0)
        m = self.tool_call_id_paren_regex.match(function_id)
        if m:
            name = m.group("name")
            if name in tool_indices:
                return name, int(m.group("index")), "parenthesized_id"
            recovered_alias = self._recover_tool_name_alias(name, tool_indices)
            if recovered_alias is not None:
                function_name, reason = recovered_alias
                return function_name, int(m.group("index")), reason

        # Case G: name glued to index. The regex name part is non-greedy, so we
        # progressively shrink the trailing digit run until the name matches a
        # real tool. This avoids mis-splitting legitimate names ending in digits
        # (e.g. "s3_bucket2") because we only accept a known tool.
        m = self.tool_call_id_nosep_regex.match(function_id)
        if m:
            name = m.group("name")
            index = m.group("index")
            if name in tool_indices:
                return name, int(index), "missing_separator"
            recovered_alias = self._recover_tool_name_alias(name, tool_indices)
            if recovered_alias is not None:
                function_name, reason = recovered_alias
                return function_name, int(index), reason
            # Shift digits from the front of the index back onto the name and
            # retry, e.g. "file_edit15" + "2" -> "file_edit1" + "52" -> ...
            while index:
                name = name + index[0]
                index = index[1:]
                if index and name in tool_indices:
                    return name, int(index), "missing_separator"
                recovered_alias = self._recover_tool_name_alias(name, tool_indices)
                if index and recovered_alias is not None:
                    function_name, reason = recovered_alias
                    return function_name, int(index), reason

        return None

    def _recover_tool_name_alias(self, name: str, tool_indices):
        for cand in (name.replace("-", "_"), name.replace("_", "-")):
            if cand in tool_indices:
                return cand, "dash_underscore_alias"
        if name.endswith("_file"):
            file_alias = "file_" + name[: -len("_file")]
            if file_alias in tool_indices:
                return file_alias, "file_prefix_alias"
        return None

    def _infer_tool_name(self, tools: List[Tool], function_args: str = None):
        """Infer function name when the model omits it (bare counter ID).

        Matches argument keys against tool parameter schemas, preferring the
        tool whose declared properties best match the actual arguments.
        """
        if not tools:
            return None
        if len(tools) == 1:
            return tools[0].function.name

        if not function_args:
            logger.debug(
                "No function_args for tool name inference with %d tools", len(tools)
            )
            return None

        try:
            arg_keys = set(json.loads(function_args).keys())
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                "Could not parse function_args for tool name inference "
                "(may be partial JSON in streaming)"
            )
            return None

        # Pick the tool whose properties best match the argument keys.
        best_name = None
        best_score = None
        for tool in tools:
            params = tool.function.parameters or {}
            props = set(params.get("properties", {}).keys())
            if not props:
                continue
            overlap = len(arg_keys & props)
            if overlap == 0:
                continue
            extra = len(arg_keys - props)
            score = overlap - extra
            if best_score is None or score > best_score:
                best_score = score
                best_name = tool.function.name

        return best_name

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a KimiK2 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: StreamingParseResult with normal_text (content before tool calls) and calls (parsed items).
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])
        try:
            function_call_tuples = self.tool_call_regex.findall(text)

            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []
            for match in function_call_tuples:
                function_id, function_args = match
                function_args = self._trim_argument_payload(function_args)
                function_name, function_idx = self._parse_tool_call_id(
                    function_id, tools, function_args
                )
                if function_name is None:
                    self._observe_invalid_tool_target(
                        self._last_invalid_reason or "unknown_function_id",
                        self._last_invalid_function_name or function_id,
                        function_args,
                    )
                    continue

                logger.debug(f"function_name {function_name}")

                tool_calls.append(
                    ToolCallItem(
                        tool_index=function_idx,
                        name=function_name,
                        parameters=function_args,
                    )
                )

            content = text[: text.find(self.bot_token)]
            return StreamingParseResult(normal_text=content, calls=tool_calls)

        except Exception as e:
            logger.error("Error in detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for KimiK2 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        has_tool_call = (
            self.bot_token in current_text or self.tool_call_start_token in current_text
        )

        if not has_tool_call:
            self._buffer = ""
            normal_text = _strip_special_tokens(new_text)
            return StreamingParseResult(normal_text=normal_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            match = self.stream_tool_call_portion_regex.search(current_text)
            if match:
                function_id = match.group("tool_call_id")
                function_args = match.group("function_arguments")

                # Reuse cached name for current tool call to avoid repeated
                # json.loads on partial JSON in _infer_tool_name.
                if self._current_stream_function_name is not None:
                    function_name = self._current_stream_function_name
                else:
                    function_name, function_idx = self._parse_tool_call_id(
                        function_id, tools, self._trim_argument_payload(function_args)
                    )
                if function_name is None:
                    # bare counter 场景：JSON 不完整时推断不出函数名，继续缓冲等待更多 chunk。
                    # 只有在整个 tool call 已完整（含 end token）但仍无法推断时才打 fail。
                    if (
                        self.tool_call_end_token in current_text
                        or self.eot_token in current_text
                    ):
                        tool_id = self.current_tool_id if self.current_tool_id >= 0 else 0
                        failure_reason = (
                            self._last_invalid_reason or "unknown_function_id"
                        )
                        if fail_stream_tool(
                            self,
                            self._tool_metrics_parser,
                            tool_id,
                            failure_reason,
                        ):
                            available_tools = list(self._get_tool_indices(tools).keys())
                            metric_function_name = (
                                self._last_invalid_function_name
                                or self._metric_tool_name(
                                    function_id, None, self._get_tool_indices(tools)
                                )
                            )
                            logger.warning(
                                "tool_call_parse_error | parser=%s reason=%s "
                                "tool_id=%s function_id=%r metric_func_name=%r "
                                "parsed_index=%s metric_func_required_params=%s "
                                "available_tools=%s available_tool_schemas=%s "
                                "args_snippet=%.300r buffer=%.300r",
                                self._tool_metrics_parser,
                                failure_reason,
                                tool_id,
                                function_id,
                                metric_function_name,
                                function_idx,
                                self._tool_required_params(tools, metric_function_name),
                                available_tools,
                                self._tool_schema_snapshot(tools),
                                function_args[:300] if function_args else "",
                                current_text,
                            )
                            self._observe_invalid_tool_target(
                                failure_reason,
                                metric_function_name,
                                function_args,
                            )
                        end_match = re.search(
                            r"<\|tool_call_begin\|>.*?(?:<\|tool_call_end\|>|<\|tool_calls_section_end\|>)",
                            current_text,
                            re.DOTALL,
                        )
                        self._buffer = (
                            current_text[end_match.end() :] if end_match else ""
                        )
                        self.current_tool_id = tool_id
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        self._current_stream_function_name = None
                        forget_stream_tool(self, tool_id)
                    return StreamingParseResult(normal_text="", calls=calls)

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=function_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self._current_stream_function_name = function_name
                    self._current_stream_resolution_reason = (
                        self._last_tool_call_resolution_reason
                    )
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": function_name,
                        "arguments": {},
                    }
                    begin_stream_tool(self, self._tool_metrics_parser, self.current_tool_id, "name_sent")

                argument_payload = self._trim_argument_payload(function_args)
                argument_diff = (
                    argument_payload[len(self._last_arguments) :]
                    if argument_payload.startswith(self._last_arguments)
                    else argument_payload
                )

                parsed_args_diff = argument_diff

                if parsed_args_diff:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=parsed_args_diff,
                        )
                    )
                    self._last_arguments += parsed_args_diff
                    self.streamed_args_for_tool[self.current_tool_id] += parsed_args_diff

                parsed_args = argument_payload
                if _is_complete_json(parsed_args):
                    has_tool_call_end = self.tool_call_end_token in function_args
                    has_section_end = self.eot_token in function_args
                    if not has_tool_call_end and not has_section_end:
                        return StreamingParseResult(normal_text="", calls=calls)

                    try:
                        parsed_args = json.loads(parsed_args)
                        self.prev_tool_call_arr[self.current_tool_id][
                            "arguments"
                        ] = parsed_args
                    except json.JSONDecodeError:
                        pass

                    # Find the end of the current tool call and remove only that part from buffer
                    tool_call_end_pattern = (
                        r"<\|tool_call_begin\|>.*?(?:<\|tool_call_end\|>|<\|tool_calls_section_end\|>)"
                    )
                    end_match = re.search(
                        tool_call_end_pattern, current_text, re.DOTALL
                    )
                    if end_match:
                        self._buffer = current_text[end_match.end() :]
                    else:
                        self._buffer = ""

                    result = StreamingParseResult(normal_text="", calls=calls)
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False
                    self._current_stream_function_name = None
                    success_reason = (
                        "missing_tool_call_end_with_complete_json"
                        if not has_tool_call_end
                        else (
                            "nonstandard_tool_id"
                            if self._current_stream_resolution_reason != "standard"
                            else "parsed"
                        )
                    )
                    self._current_stream_resolution_reason = "standard"
                    succeed_stream_tool(
                        self,
                        self._tool_metrics_parser,
                        self.current_tool_id - 1,
                        reason=success_reason,
                    )
                    return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            tool_id = self.current_tool_id if self.current_tool_id >= 0 else 0
            func_name_log = (
                self.prev_tool_call_arr[tool_id].get("name", "unknown")
                if tool_id < len(self.prev_tool_call_arr)
                else "unknown"
            )
            logger.error(
                "tool_call_parse_error | parser=%s reason=exception "
                "tool_id=%s func_name=%s error=%s buffer=%.300r",
                self._tool_metrics_parser, tool_id, func_name_log, e, current_text,
                exc_info=True,
            )
            close_open_stream_tools(self, self._tool_metrics_parser, "exception")
            return StreamingParseResult(normal_text=_strip_special_tokens(current_text))

    def structure_info(self) -> _GetInfoFunc:
        """Return function that creates StructureInfo for guided generation."""

        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>",
                end="<|tool_call_end|><|tool_calls_section_end|>",
                trigger="<|tool_calls_section_begin|>",
            )

        return get_info

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
    ) -> Optional[StructuralTag]:
        if not (
            tools and (tool_choice == "required" or isinstance(tool_choice, ToolChoice))
        ):
            return super().get_structural_tag(
                tools=tools, tool_choice=tool_choice, thinking_mode=thinking_mode
            )
        if get_model_structural_tag is None:
            return None

        converted_tools = []
        for tool in tools:
            converted_tool = tool.model_dump()
            function = converted_tool["function"]
            if not function.get("strict", False):
                # Kimi's parser accepts only object-shaped tool arguments. XGrammar
                # treats strict=False arguments as unconstrained JSON, which can
                # generate strings/arrays/numbers that Kimi cannot parse. Keep
                # non-strict semantics loose by constraining only the outer type.
                function["strict"] = True
                function["parameters"] = _KIMI_NON_STRICT_ARGUMENTS_SCHEMA
            converted_tools.append(converted_tool)

        converted_tool_choice = (
            tool_choice.model_dump()
            if isinstance(tool_choice, ToolChoice)
            else tool_choice
        )
        return get_model_structural_tag(
            model="kimi",
            tools=converted_tools,
            tool_choice=converted_tool_choice,
            reasoning=thinking_mode,
        )

    def get_structural_tag_name(self) -> str:
        return "kimi"
