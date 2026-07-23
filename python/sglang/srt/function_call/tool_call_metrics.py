import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)

_tool_call_parse_total = None
_tool_call_parse_total_init_failed = False
_tool_call_parse_total_lock = threading.Lock()
_tool_choice_ctx = ContextVar("sglang_tool_call_tool_choice", default="auto")


def _normalize_tool_choice(tool_choice) -> str:
    if tool_choice in (None, "", "auto"):
        return "auto"
    if tool_choice == "required":
        return "required"
    if tool_choice == "named":
        return "named"
    if tool_choice == "none":
        return "none"
    if hasattr(tool_choice, "function"):
        return "named"
    return "unknown"


@contextmanager
def tool_call_context(tool_choice):
    token = _tool_choice_ctx.set(_normalize_tool_choice(tool_choice))
    try:
        yield
    finally:
        _tool_choice_ctx.reset(token)


def _get_current_tool_choice() -> str:
    return _normalize_tool_choice(_tool_choice_ctx.get())


def _get_tool_call_parse_counter():
    """Return the Prometheus counter, creating it lazily.

    SGLang initializes Prometheus multiprocess settings before metrics are used,
    but parser modules may be imported earlier.  Creating the Counter at import
    time can bind it to the wrong registry in multiprocess deployments, so keep
    the import and Counter construction on the first observation path.
    """
    global _tool_call_parse_total, _tool_call_parse_total_init_failed

    if _tool_call_parse_total is not None or _tool_call_parse_total_init_failed:
        return _tool_call_parse_total

    with _tool_call_parse_total_lock:
        if _tool_call_parse_total is not None or _tool_call_parse_total_init_failed:
            return _tool_call_parse_total
        try:
            from prometheus_client import Counter, REGISTRY

            metric_name = "sglang_tool_call_parse_total"
            names_to_collectors = getattr(REGISTRY, "_names_to_collectors", {})
            existing = names_to_collectors.get(metric_name)
            if existing is None and metric_name.endswith("_total"):
                existing = names_to_collectors.get(metric_name[: -len("_total")])
            if existing is not None:
                _tool_call_parse_total = existing
            else:
                _tool_call_parse_total = Counter(
                    metric_name,
                    "Tool-call parser events.",
                    ["parser", "mode", "result", "reason", "tool_choice"],
                )
        except Exception:
            _tool_call_parse_total_init_failed = True
            logger.debug("Failed to initialize tool-call parser metric", exc_info=True)
        return _tool_call_parse_total


def observe_tool_call_parse(
    parser: str,
    mode: str,
    result: str,
    reason: str,
    count: int = 1,
    tool_choice: str | None = None,
) -> None:
    counter = _get_tool_call_parse_counter()
    if counter is None:
        return
    try:
        counter.labels(
            parser,
            mode,
            result,
            reason,
            _normalize_tool_choice(tool_choice)
            if tool_choice is not None
            else _get_current_tool_choice(),
        ).inc(count)
    except Exception:
        logger.debug("Failed to record tool-call parser metric", exc_info=True)


def _stream_metrics_state(owner):
    state = getattr(owner, "_tool_call_metrics_stream_state", None)
    if not isinstance(state, dict):
        state = {"open": set(), "closed": set()}
        setattr(owner, "_tool_call_metrics_stream_state", state)
    return state


def reset_stream_tool_metrics(owner) -> None:
    setattr(owner, "_tool_call_metrics_stream_state", {"open": set(), "closed": set()})


def _stream_tool_key(tool_index):
    if tool_index is None:
        return 0
    if isinstance(tool_index, (int, str)):
        return tool_index
    return str(tool_index)


def begin_stream_tool(
    owner,
    parser: str,
    tool_index,
    reason: str,
    tool_choice: str | None = None,
) -> None:
    state = _stream_metrics_state(owner)
    key = _stream_tool_key(tool_index)
    if key in state["open"] or key in state["closed"]:
        return
    observe_tool_call_parse(parser, "stream", "triggered", reason, tool_choice=tool_choice)
    state["open"].add(key)


def succeed_stream_tool(
    owner,
    parser: str,
    tool_index,
    tool_choice: str | None = None,
    reason: str = "parsed",
) -> bool:
    state = _stream_metrics_state(owner)
    key = _stream_tool_key(tool_index)
    if key in state["closed"]:
        return False
    if key not in state["open"]:
        begin_stream_tool(owner, parser, key, "tool_marker", tool_choice=tool_choice)
    if key in state["open"]:
        observe_tool_call_parse(parser, "stream", "success", reason, tool_choice=tool_choice)
        state["open"].discard(key)
        state["closed"].add(key)
        return True
    return False


def fail_stream_tool(
    owner,
    parser: str,
    tool_index,
    reason: str,
    tool_choice: str | None = None,
) -> bool:
    state = _stream_metrics_state(owner)
    key = _stream_tool_key(tool_index)
    if key in state["closed"]:
        return False
    if key not in state["open"]:
        begin_stream_tool(owner, parser, key, "tool_marker", tool_choice=tool_choice)
    if key in state["open"]:
        observe_tool_call_parse(parser, "stream", "failure", reason, tool_choice=tool_choice)
        state["open"].discard(key)
        state["closed"].add(key)
        return True
    return False


def forget_stream_tool(owner, tool_index) -> None:
    state = _stream_metrics_state(owner)
    key = _stream_tool_key(tool_index)
    state["open"].discard(key)
    state["closed"].discard(key)


def close_open_stream_tools(
    owner,
    parser: str,
    reason: str = "incomplete_stream",
    tool_choice: str | None = None,
    result: str = "failure",
) -> None:
    state = _stream_metrics_state(owner)
    for key in list(state["open"]):
        observe_tool_call_parse(parser, "stream", result, reason, tool_choice=tool_choice)
        state["open"].discard(key)
        state["closed"].add(key)


# --- Invalid tool identifier observability ---------------------------------
_UNDEFINED_FUNC_MAX_LABELS = 500
_UNDEFINED_FUNC_OVERFLOW = "_other"
_UNDEFINED_FUNC_NAME_MAX_LEN = 128

_undefined_function_total = None
_undefined_function_total_init_failed = False
_undefined_function_total_lock = threading.Lock()

_undefined_func_seen = set()
_undefined_func_seen_lock = threading.Lock()


def _get_undefined_function_counter():
    global _undefined_function_total, _undefined_function_total_init_failed

    if _undefined_function_total is not None or _undefined_function_total_init_failed:
        return _undefined_function_total

    with _undefined_function_total_lock:
        if _undefined_function_total is not None or _undefined_function_total_init_failed:
            return _undefined_function_total
        try:
            from prometheus_client import Counter, REGISTRY

            metric_name = "sglang_tool_call_undefined_function_total"
            names_to_collectors = getattr(REGISTRY, "_names_to_collectors", {})
            existing = names_to_collectors.get(metric_name)
            if existing is None and metric_name.endswith("_total"):
                existing = names_to_collectors.get(metric_name[: -len("_total")])
            if existing is not None:
                _undefined_function_total = existing
            else:
                _undefined_function_total = Counter(
                    metric_name,
                    (
                        "Invalid tool call targets grouped by parser, reason, "
                        "best-effort parsed tool name, and generated argument "
                        "name. Cardinality bounded per process; overflow rolls "
                        "up to parsed_tool_name="
                        + repr(_UNDEFINED_FUNC_OVERFLOW)
                        + " and argument_name="
                        + repr(_UNDEFINED_FUNC_OVERFLOW)
                        + "."
                    ),
                    ["parser", "reason", "parsed_tool_name", "argument_name"],
                )
        except Exception:
            _undefined_function_total_init_failed = True
            logger.debug("Failed to init invalid-tool-identifier metric", exc_info=True)
        return _undefined_function_total


def observe_undefined_function(
    parser: str,
    parsed_tool_name: str,
    reason: str = "unknown_function_id",
    argument_name: str = "<unknown>",
) -> None:
    if not parsed_tool_name:
        parsed_tool_name = "<empty>"
    if not reason:
        reason = "<empty>"
    if not argument_name:
        argument_name = "<none>"
    parsed_tool_name = parsed_tool_name[:_UNDEFINED_FUNC_NAME_MAX_LEN]
    reason = reason[:_UNDEFINED_FUNC_NAME_MAX_LEN]
    argument_name = argument_name[:_UNDEFINED_FUNC_NAME_MAX_LEN]
    counter = _get_undefined_function_counter()
    if counter is None:
        return
    label_parsed_tool_name = parsed_tool_name
    label_argument_name = argument_name
    key = (parser, reason, parsed_tool_name, argument_name)
    with _undefined_func_seen_lock:
        if key not in _undefined_func_seen:
            if len(_undefined_func_seen) < _UNDEFINED_FUNC_MAX_LABELS:
                _undefined_func_seen.add(key)
            else:
                label_parsed_tool_name = _UNDEFINED_FUNC_OVERFLOW
                label_argument_name = _UNDEFINED_FUNC_OVERFLOW
    try:
        counter.labels(
            parser,
            reason,
            label_parsed_tool_name,
            label_argument_name,
        ).inc()
    except Exception:
        logger.debug("Failed to record invalid-tool-identifier metric", exc_info=True)


# --- Non-standard tool identifier observability -----------------------------
_NONSTANDARD_TOOL_ID_MAX_LABELS = 500
_NONSTANDARD_TOOL_ID_OVERFLOW = "_other"
_NONSTANDARD_TOOL_ID_MAX_LEN = 128

_nonstandard_tool_id_total = None
_nonstandard_tool_id_total_init_failed = False
_nonstandard_tool_id_total_lock = threading.Lock()

_nonstandard_tool_id_seen = set()
_nonstandard_tool_id_seen_lock = threading.Lock()


def _get_nonstandard_tool_id_counter():
    global _nonstandard_tool_id_total, _nonstandard_tool_id_total_init_failed

    if (
        _nonstandard_tool_id_total is not None
        or _nonstandard_tool_id_total_init_failed
    ):
        return _nonstandard_tool_id_total

    with _nonstandard_tool_id_total_lock:
        if (
            _nonstandard_tool_id_total is not None
            or _nonstandard_tool_id_total_init_failed
        ):
            return _nonstandard_tool_id_total
        try:
            from prometheus_client import Counter, REGISTRY

            metric_name = "sglang_tool_call_nonstandard_tool_id_total"
            names_to_collectors = getattr(REGISTRY, "_names_to_collectors", {})
            existing = names_to_collectors.get(metric_name)
            if existing is None and metric_name.endswith("_total"):
                existing = names_to_collectors.get(metric_name[: -len("_total")])
            if existing is not None:
                _nonstandard_tool_id_total = existing
            else:
                _nonstandard_tool_id_total = Counter(
                    metric_name,
                    (
                        "Tool-call IDs resolved by non-standard Kimi fallback "
                        "paths or argument-schema inference. Cardinality is "
                        "bounded per process; overflow raw_tool_id rolls up to "
                        "raw_tool_id="
                        + repr(_NONSTANDARD_TOOL_ID_OVERFLOW)
                        + "."
                    ),
                    ["parser", "reason", "raw_tool_id", "func_name"],
                )
        except Exception:
            _nonstandard_tool_id_total_init_failed = True
            logger.debug("Failed to init nonstandard-tool-id metric", exc_info=True)
        return _nonstandard_tool_id_total


def observe_nonstandard_tool_id(
    parser: str, reason: str, raw_tool_id: str, func_name: str
) -> None:
    if not raw_tool_id:
        raw_tool_id = "<empty>"
    if not func_name:
        func_name = "<empty>"
    raw_tool_id = raw_tool_id[:_NONSTANDARD_TOOL_ID_MAX_LEN]
    func_name = func_name[:_NONSTANDARD_TOOL_ID_MAX_LEN]
    counter = _get_nonstandard_tool_id_counter()
    if counter is None:
        return
    label_raw_tool_id = raw_tool_id
    key = (parser, reason, raw_tool_id, func_name)
    with _nonstandard_tool_id_seen_lock:
        if key not in _nonstandard_tool_id_seen:
            if len(_nonstandard_tool_id_seen) < _NONSTANDARD_TOOL_ID_MAX_LABELS:
                _nonstandard_tool_id_seen.add(key)
            else:
                label_raw_tool_id = _NONSTANDARD_TOOL_ID_OVERFLOW
    try:
        counter.labels(parser, reason, label_raw_tool_id, func_name).inc()
    except Exception:
        logger.debug("Failed to record nonstandard-tool-id metric", exc_info=True)
