"""Qwen2.5-specific conversion utilities for Apertus-structured SFT rows.

This module contains only pure parsing, validation, schema resolution, and
message conversion. Dataset loading, tokenization, padding, and loss masking
remain in :mod:`verl.utils.dataset.multiturn_sft_dataset`.
"""

import json
from typing import Any

from jsonschema import SchemaError
from jsonschema.validators import validator_for


def _conversion_error(
    message: str, *, row_index: int | None = None, message_index: int | None = None, block_index: int | None = None
) -> ValueError:
    """Build an error that identifies the source row and structured-message location."""
    location = []
    if row_index is not None:
        location.append(f"row {row_index}")
    if message_index is not None:
        location.append(f"message {message_index}")
    if block_index is not None:
        location.append(f"block {block_index}")
    prefix = ", ".join(location)
    return ValueError(f"{prefix}: {message}" if prefix else message)


def parse_json_cell(value: Any, *, field: str, row_index: int | None = None) -> Any:
    """Decode JSON-string parquet cells while accepting already-decoded values."""
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise _conversion_error(f"invalid JSON in {field}: {exc}", row_index=row_index) from exc


_JSON_SCHEMA_ANNOTATIONS = {"$comment", "description", "examples", "title"}


def _functional_schema(value: Any, *, parent_key: str | None = None) -> Any:
    """Return a comparable JSON-schema value without descriptive annotations."""
    if isinstance(value, dict):
        return {
            key: _functional_schema(nested, parent_key=key)
            for key, nested in value.items()
            if key not in _JSON_SCHEMA_ANNOTATIONS
        }
    if isinstance(value, list):
        normalized = [_functional_schema(nested) for nested in value]
        # Ordering has no semantic meaning for these JSON Schema keywords.
        if parent_key in {"enum", "required", "type"}:
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
        return normalized
    return value


def _parse_tool_call_identity(
    call: Any,
    *,
    row_index: int | None,
    message_index: int,
    block_index: int | None,
    call_index: int,
) -> tuple[str, dict[str, Any], bool]:
    """Extract a call name, object-valued arguments, and an empty-string marker."""
    if not isinstance(call, dict):
        raise _conversion_error(
            f"tool call {call_index} must be an object",
            row_index=row_index,
            message_index=message_index,
            block_index=block_index,
        )
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise _conversion_error(
            f"tool call {call_index} has no valid name",
            row_index=row_index,
            message_index=message_index,
            block_index=block_index,
        )
    arguments = function.get("arguments", {})
    had_empty_arguments = False
    if isinstance(arguments, str):
        if not arguments.strip():
            # Some Apertus rows encode legitimate no-argument calls as an
            # empty string. Conversion validates the candidate empty object
            # against the selected function schema before accepting it.
            arguments = {}
            had_empty_arguments = True
        else:
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise _conversion_error(
                    f"invalid JSON arguments for tool {name!r}: {exc}",
                    row_index=row_index,
                    message_index=message_index,
                    block_index=block_index,
                ) from exc
    if not isinstance(arguments, dict):
        raise _conversion_error(
            f"arguments for tool {name!r} must decode to an object",
            row_index=row_index,
            message_index=message_index,
            block_index=block_index,
        )
    return name, arguments, had_empty_arguments


def _collect_qwen_tool_call_arguments(messages: Any, *, row_index: int | None) -> dict[str, list[dict[str, Any]]]:
    """Collect calls from native or Apertus assistant containers for tool resolution."""
    if not isinstance(messages, list):
        raise _conversion_error("messages must be a list", row_index=row_index)

    arguments_by_name: dict[str, list[dict[str, Any]]] = {}
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        call_groups: list[tuple[Any, int | None]] = []
        if message.get("tool_calls") is not None:
            call_groups.append((message["tool_calls"], None))
        content = message.get("content")
        if isinstance(content, dict) and isinstance(content.get("blocks"), list):
            call_groups.extend(
                (block.get("calls"), block_index)
                for block_index, block in enumerate(content["blocks"])
                if isinstance(block, dict) and block.get("type") == "tool_calls"
            )

        for calls, block_index in call_groups:
            if not isinstance(calls, list) or not calls:
                raise _conversion_error(
                    "tool_calls must contain at least one call",
                    row_index=row_index,
                    message_index=message_index,
                    block_index=block_index,
                )
            for call_index, call in enumerate(calls):
                name, arguments, _ = _parse_tool_call_identity(
                    call,
                    row_index=row_index,
                    message_index=message_index,
                    block_index=block_index,
                    call_index=call_index,
                )
                arguments_by_name.setdefault(name, []).append(arguments)
    return arguments_by_name


def _schema_accepts_all_calls(function: dict[str, Any], calls: list[dict[str, Any]], *, row_index: int | None) -> bool:
    schema = function.get("parameters", {"type": "object"})
    if not isinstance(schema, dict):
        raise _conversion_error(
            f"tool definition {function['name']!r} parameters must be an object", row_index=row_index
        )
    validator_cls = validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except SchemaError as exc:
        raise _conversion_error(
            f"tool definition {function['name']!r} has invalid parameter schema: {exc.message}",
            row_index=row_index,
        ) from exc
    validator = validator_cls(schema)
    return all(validator.is_valid(arguments) for arguments in calls)


def normalize_qwen_tools(
    tools: Any, *, messages: Any = None, row_index: int | None = None
) -> tuple[list[dict[str, Any]], set[str]]:
    """Convert direct Apertus function definitions to Qwen/OpenAI function tools.

    Apertus rows store ``{"name", "description", "parameters"}``, whereas
    Qwen2.5 was post-trained with ``{"type": "function", "function": {...}}``.
    Already-normalized definitions are accepted too. Duplicate names with the
    same functional schema are collapsed, ignoring JSON Schema annotations such
    as descriptions and titles. If schemas conflict, exactly one must validate
    every call to that name in the conversation; otherwise conversion fails.

    This helper is invoked only by :class:`Qwen2_5SFTDataset`. Apertus keeps its
    original definitions and rendering behavior.
    """
    if tools in (None, "", []):
        return [], set()
    if not isinstance(tools, list):
        raise _conversion_error("tools must be a list", row_index=row_index)

    definitions_by_name: dict[str, list[dict[str, Any]]] = {}
    name_order: list[str] = []
    for tool_index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise _conversion_error(f"tool {tool_index} must be an object", row_index=row_index)
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise _conversion_error(f"tool {tool_index} has no valid function name", row_index=row_index)
        name = function["name"]
        if name not in definitions_by_name:
            name_order.append(name)
            definitions_by_name[name] = []
        definitions_by_name[name].append(dict(function))

    schemas_by_name: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
    for name in name_order:
        schemas: list[tuple[Any, dict[str, Any]]] = []
        for definition in definitions_by_name[name]:
            signature = _functional_schema(definition)
            if not any(signature == prior_signature for prior_signature, _ in schemas):
                schemas.append((signature, definition))
        schemas_by_name[name] = schemas

    conflicting_names = {name for name, schemas in schemas_by_name.items() if len(schemas) > 1}
    calls_by_name = _collect_qwen_tool_call_arguments(messages, row_index=row_index) if conflicting_names else {}
    normalized = []
    for name in name_order:
        schemas = schemas_by_name[name]

        if len(schemas) == 1:
            selected = schemas[0][1]
        else:
            calls = calls_by_name.get(name, [])
            matching = [
                definition
                for _, definition in schemas
                if calls and _schema_accepts_all_calls(definition, calls, row_index=row_index)
            ]
            if len(matching) != 1:
                reason = (
                    "no calls are available to disambiguate"
                    if not calls
                    else f"{len(matching)} schemas match all calls"
                )
                raise _conversion_error(f"ambiguous duplicate tool definition {name!r}: {reason}", row_index=row_index)
            selected = matching[0]
        normalized.append({"type": "function", "function": selected})
    names = set(definitions_by_name)
    return normalized, names


def qwen_tool_schemas(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index the selected normalized Qwen function schemas by tool name."""
    return {tool["function"]["name"]: tool["function"].get("parameters", {"type": "object"}) for tool in tools}


def _qwen_tool_calls(
    calls: Any,
    *,
    defined_tools: set[str],
    tool_schemas: dict[str, dict[str, Any]] | None,
    row_index: int | None,
    message_index: int,
    block_index: int | None,
) -> list[dict[str, Any]]:
    """Normalize calls and parse Apertus' JSON-string arguments for Qwen."""
    if not isinstance(calls, list) or not calls:
        raise _conversion_error(
            "tool_calls must contain at least one call",
            row_index=row_index,
            message_index=message_index,
            block_index=block_index,
        )
    converted = []
    for call_index, call in enumerate(calls):
        name, arguments, had_empty_arguments = _parse_tool_call_identity(
            call,
            row_index=row_index,
            message_index=message_index,
            block_index=block_index,
            call_index=call_index,
        )
        if name not in defined_tools:
            raise _conversion_error(
                f"tool call refers to undefined tool {name!r}",
                row_index=row_index,
                message_index=message_index,
                block_index=block_index,
            )
        if had_empty_arguments:
            schema = tool_schemas.get(name) if tool_schemas is not None else None
            if schema is None or not _schema_accepts_all_calls(
                {"name": name, "parameters": schema}, [arguments], row_index=row_index
            ):
                raise _conversion_error(
                    f"empty arguments for tool {name!r} are valid only when its schema accepts an empty object",
                    row_index=row_index,
                    message_index=message_index,
                    block_index=block_index,
                )
        converted.append({"type": "function", "function": {"name": name, "arguments": arguments}})
    return converted


def _text_content(content: Any, *, role: str, row_index: int | None, message_index: int) -> str:
    """Translate Apertus system/user content into text-only Qwen content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, dict):
        raise _conversion_error(
            f"{role} content must be a string or object", row_index=row_index, message_index=message_index
        )
    if role == "system":
        text = content.get("text", "")
        if not isinstance(text, str):
            raise _conversion_error(
                "system content.text must be a string", row_index=row_index, message_index=message_index
            )
        return text

    parts = content.get("parts")
    if parts is None:
        text = content.get("text", "")
        if not isinstance(text, str):
            raise _conversion_error(
                "user content.text must be a string", row_index=row_index, message_index=message_index
            )
        return text
    if not isinstance(parts, list):
        raise _conversion_error("user content.parts must be a list", row_index=row_index, message_index=message_index)
    texts = []
    for part_index, part in enumerate(parts):
        if not isinstance(part, dict) or part.get("type", "text") != "text" or not isinstance(part.get("text"), str):
            raise _conversion_error(
                f"unsupported non-text user part {part_index}", row_index=row_index, message_index=message_index
            )
        texts.append(part["text"])
    return "".join(texts)


def convert_apertus_messages_to_qwen(
    messages: Any,
    *,
    defined_tools: set[str],
    tool_schemas: dict[str, dict[str, Any]] | None = None,
    allow_thinking: bool,
    row_index: int | None = None,
) -> list[dict[str, Any]]:
    """Translate Apertus structured messages to Qwen2.5's role-based schema.

    Apertus may keep response text, thoughts, tool calls, and tool outputs as an
    ordered list of blocks inside one assistant message. Qwen expresses the same
    interaction as assistant messages with optional ``tool_calls``, followed by
    one ``tool`` message per output. Consequently a single Apertus assistant
    container can become an ``assistant -> tool -> assistant`` sequence. Parallel
    calls and outputs are paired by their existing order; neither format carries
    a call id in this dataset.

    Thinking is rendered canonically as::

        <think>\nreasoning\n</think>\n\nanswer

    Tool-output text is preserved but moved to masked ``role=tool`` messages.
    """
    if not isinstance(messages, list) or not messages:
        raise _conversion_error("messages must be a non-empty list", row_index=row_index)

    converted: list[dict[str, Any]] = []
    pending_calls: list[str] = []

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise _conversion_error("message must be an object", row_index=row_index, message_index=message_index)
        role = message.get("role")

        if role == "tool":
            if not pending_calls:
                raise _conversion_error(
                    "tool output has no pending call", row_index=row_index, message_index=message_index
                )
            content = message.get("content", "")
            if not isinstance(content, str):
                raise _conversion_error(
                    "tool content must be a string", row_index=row_index, message_index=message_index
                )
            converted.append({"role": "tool", "name": pending_calls.pop(0), "content": content})
            continue

        if pending_calls:
            raise _conversion_error(
                f"missing outputs for tool calls {pending_calls!r}", row_index=row_index, message_index=message_index
            )

        if role in {"system", "user"}:
            converted.append(
                {
                    "role": role,
                    "content": _text_content(
                        message.get("content", ""), role=role, row_index=row_index, message_index=message_index
                    ),
                }
            )
            continue
        if role == "developer":
            raise _conversion_error(
                "developer messages are preprocessing-only and must not reach SFT parquet",
                row_index=row_index,
                message_index=message_index,
            )
        if role != "assistant":
            raise _conversion_error(f"unsupported role {role!r}", row_index=row_index, message_index=message_index)

        content = message.get("content", "")
        if isinstance(content, str):
            assistant = {"role": "assistant", "content": content}
            if message.get("tool_calls"):
                calls = _qwen_tool_calls(
                    message["tool_calls"],
                    defined_tools=defined_tools,
                    tool_schemas=tool_schemas,
                    row_index=row_index,
                    message_index=message_index,
                    block_index=None,
                )
                assistant["tool_calls"] = calls
                pending_calls = [call["function"]["name"] for call in calls]
            converted.append(assistant)
            continue
        if not isinstance(content, dict):
            raise _conversion_error(
                "assistant content must be a string or object", row_index=row_index, message_index=message_index
            )
        blocks = content.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            text = content.get("text")
            if isinstance(text, str):
                converted.append({"role": "assistant", "content": text})
                continue
            raise _conversion_error("assistant content has no blocks", row_index=row_index, message_index=message_index)

        chunks: list[str] = []
        last_content_kind: str | None = None
        block_pending_calls: list[str] = []

        def flush_assistant(calls: list[dict[str, Any]] | None = None) -> None:
            nonlocal chunks, last_content_kind
            output: dict[str, Any] = {"role": "assistant", "content": "".join(chunks)}
            if calls:
                output["tool_calls"] = calls
            converted.append(output)
            chunks = []
            last_content_kind = None

        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                raise _conversion_error(
                    "assistant block must be an object",
                    row_index=row_index,
                    message_index=message_index,
                    block_index=block_index,
                )
            block_type = block.get("type")
            if block_pending_calls and block_type != "tool_outputs":
                raise _conversion_error(
                    f"missing outputs for tool calls {block_pending_calls!r} before {block_type!r}",
                    row_index=row_index,
                    message_index=message_index,
                    block_index=block_index,
                )
            if block_type == "thoughts":
                if not allow_thinking:
                    raise _conversion_error(
                        "thoughts block requires the SFT1 thinking tokenizer",
                        row_index=row_index,
                        message_index=message_index,
                        block_index=block_index,
                    )
                text = block.get("text", "")
                if not isinstance(text, str):
                    raise _conversion_error(
                        "thoughts text must be a string",
                        row_index=row_index,
                        message_index=message_index,
                        block_index=block_index,
                    )
                chunks.append(f"<think>\n{text}\n</think>")
                last_content_kind = "thoughts"
            elif block_type == "response":
                text = block.get("text", "")
                if not isinstance(text, str):
                    raise _conversion_error(
                        "response text must be a string",
                        row_index=row_index,
                        message_index=message_index,
                        block_index=block_index,
                    )
                if last_content_kind == "thoughts":
                    chunks.append("\n\n")
                chunks.append(text)
                last_content_kind = "response"
            elif block_type == "tool_calls":
                if block_pending_calls:
                    raise _conversion_error(
                        "new tool calls appeared before prior outputs",
                        row_index=row_index,
                        message_index=message_index,
                        block_index=block_index,
                    )
                calls = _qwen_tool_calls(
                    block.get("calls"),
                    defined_tools=defined_tools,
                    tool_schemas=tool_schemas,
                    row_index=row_index,
                    message_index=message_index,
                    block_index=block_index,
                )
                flush_assistant(calls)
                block_pending_calls = [call["function"]["name"] for call in calls]
            elif block_type == "tool_outputs":
                outputs = block.get("outputs")
                if not block_pending_calls:
                    raise _conversion_error(
                        "tool_outputs has no preceding tool_calls block",
                        row_index=row_index,
                        message_index=message_index,
                        block_index=block_index,
                    )
                if not isinstance(outputs, list) or len(outputs) != len(block_pending_calls):
                    raise _conversion_error(
                        f"tool call/output counts differ ({len(block_pending_calls)} calls)",
                        row_index=row_index,
                        message_index=message_index,
                        block_index=block_index,
                    )
                for output_index, (name, output) in enumerate(zip(block_pending_calls, outputs, strict=True)):
                    if not isinstance(output, dict) or not isinstance(output.get("output"), str):
                        raise _conversion_error(
                            f"tool output {output_index} must contain string output",
                            row_index=row_index,
                            message_index=message_index,
                            block_index=block_index,
                        )
                    output_name = output.get("name")
                    if output_name not in (None, "", name):
                        raise _conversion_error(
                            f"tool output name {output_name!r} does not match call {name!r}",
                            row_index=row_index,
                            message_index=message_index,
                            block_index=block_index,
                        )
                    converted.append({"role": "tool", "name": name, "content": output["output"]})
                block_pending_calls = []
            else:
                raise _conversion_error(
                    f"unsupported assistant block type {block_type!r}",
                    row_index=row_index,
                    message_index=message_index,
                    block_index=block_index,
                )

        if chunks:
            flush_assistant()
        if block_pending_calls:
            pending_calls = block_pending_calls

    if pending_calls and pending_calls != ["final_answer"]:
        raise _conversion_error(f"conversation ends with missing outputs for {pending_calls!r}", row_index=row_index)
    return converted
