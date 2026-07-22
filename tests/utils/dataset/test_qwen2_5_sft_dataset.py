"""Tests for translating Apertus-structured SFT rows to Qwen2.5."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import AddedToken, AutoTokenizer

from verl.utils.dataset.multiturn_sft_dataset import Qwen2_5SFTDataset
from verl.utils.dataset.qwen2_5_sft_utils import (
    convert_apertus_messages_to_qwen,
    normalize_qwen_tools,
    qwen_tool_schemas,
)

TOOLS = [
    {
        "name": "calculate",
        "description": "Calculate a value",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    },
    {
        "name": "lookup",
        "description": "Look up a value",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
    },
    {
        "name": "final_answer",
        "description": "Return the final answer",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}},
    },
]


def _assistant_call(name, arguments):
    return [
        {
            "role": "assistant",
            "content": {
                "blocks": [{"type": "tool_calls", "calls": [{"name": name, "arguments": json.dumps(arguments)}]}]
            },
        }
    ]


def test_qwen_tool_normalization_collapses_functionally_equivalent_duplicates():
    tools = [
        {
            "name": "convert_currency",
            "description": "Convert one currency to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert"},
                    "currency": {"type": "string"},
                },
                "required": ["amount", "currency"],
            },
        },
        {
            "name": "convert_currency",
            "description": "Convert currency",
            "parameters": {
                "type": "object",
                "title": "Conversion input",
                "properties": {
                    "amount": {"type": "number", "description": "The amount"},
                    "currency": {"type": "string"},
                },
                "required": ["currency", "amount"],
            },
        },
    ]

    normalized, names = normalize_qwen_tools(tools, row_index=12)

    assert names == {"convert_currency"}
    assert len(normalized) == 1
    assert normalized[0]["function"] == tools[0]


def test_qwen_tool_normalization_selects_unique_schema_matching_calls():
    tools = [
        {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
                "required": ["keywords"],
                "additionalProperties": False,
            },
        },
    ]

    normalized, _ = normalize_qwen_tools(
        tools, messages=_assistant_call("search", {"keywords": ["qwen", "sft"]}), row_index=7
    )

    assert normalized == [{"type": "function", "function": tools[1]}]


def test_qwen_tool_normalization_uses_empty_object_to_select_zero_argument_schema():
    tools = [
        {"name": "quote", "parameters": {}},
        {
            "name": "quote",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    ]

    messages = _assistant_call("quote", {})
    messages[0]["content"]["blocks"][0]["calls"][0]["arguments"] = ""
    normalized, _ = normalize_qwen_tools(tools, messages=messages, row_index=8)

    assert normalized == [{"type": "function", "function": tools[0]}]


@pytest.mark.parametrize(
    ("tools", "messages", "match"),
    [
        (
            [
                {"name": "search", "parameters": {"type": "object"}},
                {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ],
            _assistant_call("search", {"query": "qwen"}),
            "2 schemas match all calls",
        ),
        (
            [
                {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"keywords": {"type": "array"}},
                        "required": ["keywords"],
                        "additionalProperties": False,
                    },
                },
            ],
            _assistant_call("search", {"limit": 5}),
            "0 schemas match all calls",
        ),
        (
            [
                {"name": "search", "parameters": {"type": "object"}},
                {"name": "search", "parameters": {"type": "object", "additionalProperties": False}},
            ],
            [],
            "no calls are available to disambiguate",
        ),
    ],
)
def test_qwen_tool_normalization_rejects_ambiguous_conflicts(tools, messages, match):
    with pytest.raises(ValueError, match=match):
        normalize_qwen_tools(tools, messages=messages, row_index=19)


def test_conversion_preserves_thoughts_and_tool_sequence():
    normalized_tools, names = normalize_qwen_tools(TOOLS)
    messages = [
        {"role": "system", "content": {"text": ""}},
        {"role": "user", "content": {"parts": [{"type": "text", "text": "Solve it"}]}},
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {"type": "thoughts", "text": "First reason"},
                    {"type": "response", "text": "I will use tools."},
                    {
                        "type": "tool_calls",
                        "calls": [
                            {"name": "calculate", "arguments": '{"x": 2}'},
                            {"name": "lookup", "arguments": '{"key": "square"}'},
                        ],
                    },
                    {
                        "type": "tool_outputs",
                        "outputs": [
                            {"name": "calculate", "output": "4"},
                            {"name": "lookup", "output": "four"},
                        ],
                    },
                    {"type": "response", "text": "The answer is 4."},
                ]
            },
        },
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {
                        "type": "tool_calls",
                        "calls": [{"name": "final_answer", "arguments": '{"answer": "4"}'}],
                    }
                ]
            },
        },
    ]

    converted = convert_apertus_messages_to_qwen(messages, defined_tools=names, allow_thinking=True, row_index=3)

    assert normalized_tools[0] == {"type": "function", "function": TOOLS[0]}
    assert converted[2]["content"] == "<think>\nFirst reason\n</think>\n\nI will use tools."
    assert converted[2]["tool_calls"][0]["function"]["arguments"] == {"x": 2}
    assert converted[2]["tool_calls"][1]["function"]["arguments"] == {"key": "square"}
    assert converted[3:5] == [
        {"role": "tool", "name": "calculate", "content": "4"},
        {"role": "tool", "name": "lookup", "content": "four"},
    ]
    assert converted[5] == {"role": "assistant", "content": "The answer is 4."}
    assert converted[6]["tool_calls"][0]["function"]["name"] == "final_answer"


@pytest.mark.parametrize(
    ("tools", "calls", "error"),
    [
        (TOOLS, [{"name": "missing", "arguments": "{}"}], "undefined tool"),
        (TOOLS, [{"name": "calculate", "arguments": "not-json"}], "invalid JSON arguments"),
    ],
)
def test_conversion_rejects_invalid_calls(tools, calls, error):
    _, names = normalize_qwen_tools(tools)
    messages = [
        {"role": "user", "content": {"parts": [{"type": "text", "text": "x"}]}},
        {"role": "assistant", "content": {"blocks": [{"type": "tool_calls", "calls": calls}]}},
    ]
    with pytest.raises(ValueError, match=error):
        convert_apertus_messages_to_qwen(messages, defined_tools=names, allow_thinking=True)


def test_conversion_normalizes_schema_valid_empty_arguments():
    tools = [{"name": "generate_random_quote", "parameters": {}}]
    normalized, names = normalize_qwen_tools(tools)
    messages = [
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {
                        "type": "tool_calls",
                        "calls": [{"name": "generate_random_quote", "arguments": ""}],
                    },
                    {"type": "tool_outputs", "outputs": [{"name": "", "output": "A quote"}]},
                ]
            },
        }
    ]

    converted = convert_apertus_messages_to_qwen(
        messages,
        defined_tools=names,
        tool_schemas=qwen_tool_schemas(normalized),
        allow_thinking=False,
        row_index=21,
    )

    assert converted[0]["tool_calls"][0]["function"]["arguments"] == {}


def test_conversion_rejects_empty_arguments_when_schema_requires_values():
    tools = [
        {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    normalized, names = normalize_qwen_tools(tools)
    messages = _assistant_call("search", {})
    messages[0]["content"]["blocks"][0]["calls"][0]["arguments"] = "  "

    with pytest.raises(ValueError, match="schema accepts an empty object"):
        convert_apertus_messages_to_qwen(
            messages,
            defined_tools=names,
            tool_schemas=qwen_tool_schemas(normalized),
            allow_thinking=False,
            row_index=22,
        )


def test_conversion_supports_multiple_tool_cycles():
    _, names = normalize_qwen_tools(TOOLS)
    messages = [
        {"role": "user", "content": {"parts": [{"type": "text", "text": "x"}]}},
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {
                        "type": "tool_calls",
                        "calls": [{"name": "calculate", "arguments": '{"x": 2}'}],
                    },
                    {
                        "type": "tool_outputs",
                        "outputs": [{"name": "calculate", "output": "4"}],
                    },
                    {"type": "response", "text": "Checking once more."},
                    {
                        "type": "tool_calls",
                        "calls": [{"name": "calculate", "arguments": '{"x": 4}'}],
                    },
                    {
                        "type": "tool_outputs",
                        "outputs": [{"name": "calculate", "output": "16"}],
                    },
                    {"type": "response", "text": "Done."},
                ]
            },
        },
    ]
    converted = convert_apertus_messages_to_qwen(messages, defined_tools=names, allow_thinking=True)
    assert [message["role"] for message in converted] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert converted[2]["content"] == "4"
    assert converted[3]["content"] == "Checking once more."
    assert converted[4]["content"] == "16"
    assert converted[5]["content"] == "Done."


def test_conversion_rejects_mismatched_call_output_counts():
    _, names = normalize_qwen_tools(TOOLS)
    messages = [
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {
                        "type": "tool_calls",
                        "calls": [
                            {"name": "calculate", "arguments": '{"x": 2}'},
                            {"name": "lookup", "arguments": '{"key": "x"}'},
                        ],
                    },
                    {
                        "type": "tool_outputs",
                        "outputs": [{"name": "calculate", "output": "4"}],
                    },
                ]
            },
        }
    ]
    with pytest.raises(ValueError, match="counts differ"):
        convert_apertus_messages_to_qwen(messages, defined_tools=names, allow_thinking=True)


def _local_qwen_tokenizer():
    path = Path("/users/msantelmo/scratch/checkpoints/Qwen2.5-7B-Instruct")
    if not path.exists():
        pytest.skip("local Qwen2.5 tokenizer snapshot is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.add_tokens(
        [
            AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=False)
            for token in ("<think>", "</think>")
        ],
        special_tokens=False,
    )
    template = Path(__file__).parents[4] / "scripts/tokenizers/qwen2_5_sft1_chat_template.jinja"
    # Tests can run from the verl_sft repository or the outer SSFT repository.
    if not template.exists():
        template = Path(__file__).parents[5] / "scripts/tokenizers/qwen2_5_sft1_chat_template.jinja"
    tokenizer.chat_template = template.read_text()
    return tokenizer


def test_dataset_masks_tool_outputs_and_supports_rollouts(tmp_path):
    tokenizer = _local_qwen_tokenizer()
    assert tokenizer.encode("<think>", add_special_tokens=False) == [151665]
    assert tokenizer.decode([151665, 151666], skip_special_tokens=True) == "<think></think>"

    messages = [
        {"role": "system", "content": {"text": ""}},
        {"role": "user", "content": {"parts": [{"type": "text", "text": "Solve it"}]}},
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {"type": "thoughts", "text": "Reason carefully"},
                    {
                        "type": "tool_calls",
                        "calls": [{"name": "calculate", "arguments": '{"x": 2}'}],
                    },
                    {
                        "type": "tool_outputs",
                        "outputs": [{"name": "calculate", "output": "MASKED_TOOL_OUTPUT"}],
                    },
                    {"type": "response", "text": "The answer is 4."},
                ]
            },
        },
    ]
    train_path = tmp_path / "train.parquet"
    pd.DataFrame([{"messages": json.dumps(messages), "tools": json.dumps(TOOLS), "enable_thinking": True}]).to_parquet(
        train_path
    )
    dataset = Qwen2_5SFTDataset(
        str(train_path),
        tokenizer,
        {"max_length": 768, "truncation": "error", "apply_chat_template_kwargs": {"truncation": True}},
    )
    item = dataset[0]
    masked_text = tokenizer.decode(item["responses"][item["response_mask"].bool()], skip_special_tokens=False)
    full_text = tokenizer.decode(item["input_ids"][item["attention_mask"].bool()], skip_special_tokens=False)
    assert "<think>\nReason carefully\n</think>" in masked_text
    assert '"arguments": {"x": 2}' in masked_text
    assert "The answer is 4." in masked_text
    assert "MASKED_TOOL_OUTPUT" in full_text
    assert "MASKED_TOOL_OUTPUT" not in masked_text
    assert item["responses"].shape == item["response_mask"].shape == torch.Size([767])

    rollout_path = tmp_path / "rollout.parquet"
    pd.DataFrame(
        [
            {
                "messages": json.dumps(messages[:2]),
                "tools": "",
                "enable_thinking": True,
                "rollout_params": json.dumps({"task_name": "test", "sampling_params": {"temperature": 0.0}}),
            }
        ]
    ).to_parquet(rollout_path)
    rollout = Qwen2_5SFTDataset(str(rollout_path), tokenizer, {"max_length": 128, "add_generation_prompt": True})[0]
    rollout_text = tokenizer.decode(rollout["input_ids"][rollout["attention_mask"].bool()])
    assert "Deliberation: enabled" in rollout_text
    assert rollout_text.endswith("<|im_start|>assistant\n")
    assert rollout["response_mask"].sum().item() == 0
    assert rollout["rollout_params"]["task_name"] == "test"


def test_dataset_prefilters_conversion_invalid_rows(tmp_path, caplog):
    tokenizer = _local_qwen_tokenizer()
    valid_messages = [
        {"role": "user", "content": {"parts": [{"type": "text", "text": "Hello"}]}},
        {"role": "assistant", "content": {"blocks": [{"type": "response", "text": "Hi"}]}},
    ]
    invalid_messages = [
        {"role": "user", "content": {"parts": [{"type": "text", "text": "Calculate BMI"}]}},
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {"type": "tool_calls", "calls": [{"name": "calculate_bmi", "arguments": ""}]},
                    {"type": "tool_outputs", "outputs": [{"name": "", "output": "22"}]},
                ]
            },
        },
    ]
    bmi_tool = {
        "name": "calculate_bmi",
        "parameters": {
            "type": "object",
            "properties": {"height": {"type": "number"}, "weight": {"type": "number"}},
            "required": ["height", "weight"],
        },
    }
    path = tmp_path / "filtered.parquet"
    pd.DataFrame(
        [
            {"messages": json.dumps(valid_messages), "tools": "", "enable_thinking": False},
            {
                "messages": json.dumps(invalid_messages),
                "tools": json.dumps([bmi_tool]),
                "enable_thinking": False,
            },
        ]
    ).to_parquet(path)

    with caplog.at_level("WARNING"):
        dataset = Qwen2_5SFTDataset(
            str(path),
            tokenizer,
            {
                "max_length": 64,
                "truncation": "error",
                "apply_chat_template_kwargs": {"truncation": True},
                "invalid_row_policy": "filter",
            },
        )

    assert len(dataset) == 1
    assert dataset._source_indices.tolist() == [0]
    assert dataset[0]["response_mask"].sum().item() > 0
    assert "filtered 1 of 2 conversion-invalid rows" in caplog.text


def test_sft0_tokenizer_rejects_thoughts():
    path = Path("/users/msantelmo/scratch/checkpoints/Qwen2.5-7B-Instruct")
    if not path.exists():
        pytest.skip("local Qwen2.5 tokenizer snapshot is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(path)
    assert len(tokenizer.encode("<think>", add_special_tokens=False)) > 1
    with pytest.raises(ValueError, match="SFT1 thinking tokenizer"):
        convert_apertus_messages_to_qwen(
            [{"role": "assistant", "content": {"blocks": [{"type": "thoughts", "text": "x"}]}}],
            defined_tools=set(),
            allow_thinking=False,
        )
