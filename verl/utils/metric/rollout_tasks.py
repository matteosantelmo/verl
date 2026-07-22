"""
Rollout evaluation tasks for training-time metrics.

Each task is a callable that computes metrics from model outputs.
Tasks are registered in TASK_REGISTRY and can be referenced by ID in rollout_params.
"""
from __future__ import annotations
from typing import Any, Callable, Optional, Union, Dict, List
import os
import logging
import re
from functools import lru_cache
from json import loads
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

def _load_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return loads(value)
    except Exception:
        return value


def _infer_task_id(data: dict[str, Any]) -> str:
    task_id = data.get("id")
    if task_id:
        return str(task_id)

    task_name = str(data.get("task_name") or "")
    data_source = str(data.get("data_source") or "")
    if task_name in TASK_REGISTRY:
        return task_name
    if task_name in {"math500", "aime2024", "aime2025"}:
        return "math"
    if task_name in {"gpqa_diamond", "mmlu"}:
        return "mcq"

    data_source_map = {
        "openai/gsm8k": "gsm8k",
        "HuggingFaceH4/MATH-500": "math",
        "aime2024": "math",
        "aime2025": "math",
        "gpqa_diamond": "mcq",
        "mmlu": "mcq",
        "humaneval": "humaneval",
        "google/IFEval": "ifeval",
        "allenai/IFBench_test": "ifbench",
    }
    return data_source_map.get(data_source, "")


@dataclass
class RolloutParams:
    """Dataclass for task rollout parameters."""
    id: str
    task_name: str
    answer: Any = None
    use_tool: bool = True
    sampling_params: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> RolloutParams:
        # Extract known fields
        known_fields = {
            "id",
            "data_source",
            "task_name",
            "answer",
            "ground_truth",
            "extra_info",
            "use_tool",
            "sampling_params",
            "kwargs",
        }
        kwargs = dict(data.get("kwargs", {}))

        # Put any extra fields into kwargs
        for key, value in data.items():
            if key not in known_fields:
                kwargs[key] = value

        ground_truth = data.get("ground_truth")
        extra_info = data.get("extra_info")
        if isinstance(extra_info, dict):
            kwargs.setdefault("extra_info", extra_info)
            for key, value in extra_info.items():
                if value is not None:
                    kwargs.setdefault(key, value)

        parsed_ground_truth = _load_json_maybe(ground_truth)
        if isinstance(parsed_ground_truth, dict):
            for key, value in parsed_ground_truth.items():
                kwargs.setdefault(key, value)
            if "instruction_id" in parsed_ground_truth:
                kwargs["instruction_id_list"] = parsed_ground_truth["instruction_id"]
            if "kwargs" in parsed_ground_truth:
                kwargs["instruction_kwargs"] = parsed_ground_truth["kwargs"]

        if isinstance(kwargs.get("instruction_id_list"), str):
            kwargs["instruction_id_list"] = _load_json_maybe(kwargs["instruction_id_list"])
        if isinstance(kwargs.get("instruction_kwargs"), str):
            kwargs["instruction_kwargs"] = _load_json_maybe(kwargs["instruction_kwargs"])

        answer = data.get("answer")
        if answer is None:
            answer = ground_truth

        return cls(
            id=_infer_task_id(data),
            task_name=data.get("task_name", ""),
            answer=answer,
            use_tool=data.get("use_tool", True),
            sampling_params=data.get("sampling_params", {}),
            kwargs=kwargs,
        )


try:
    import evaluate as hf_evaluate
except ImportError:
    hf_evaluate = None

from .utils import compute_text_ttr, compute_token_ttr
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def extract_answer_from_tool_call(output_text: str) -> str | None:
    """
    Extract answer from tool call in model output.

    Looks for display_answers tool calls in the format:
    <|tools_prefix|>[{"display_answers": {"answers": ["..."]}}]<|tools_suffix|>

    Returns:
        The first answer string if found, None otherwise.
    """
    if "<|tools_prefix|>" not in output_text:
        return None

    try:
        tool_calls_str = output_text.split("<|tools_prefix|>")[1].split("<|tools_suffix|>")[0]
        tool_calls = loads(tool_calls_str)
        for tool_call in tool_calls:
            if "display_answers" in tool_call:
                arguments = tool_call["display_answers"]
                if "answers" in arguments:
                    answers = arguments["answers"]
                    if answers and len(answers) > 0:
                        return str(answers[0])
        return None
    except Exception:
        # Try flexible parsing for malformed JSON
        try:
            tool_call = loads(output_text.split("<|tools_prefix|>[")[1])
            if "display_answers" in tool_call:
                arguments = tool_call["display_answers"]
                if "answers" in arguments:
                    answers = arguments["answers"]
                    if answers and len(answers) > 0:
                        return str(answers[0])
        except Exception:
            pass
        return None


def extract_answer(output_text: str) -> str | None:
    """Return a displayed answer when present, otherwise the plain response.

    ``display_answers`` remains supported for old evaluation prompts, but it is
    not a format requirement. This lets the same verifier score ordinary chat
    completions and tool-formatted completions.
    """
    tool_answer = extract_answer_from_tool_call(output_text)
    if tool_answer is not None:
        return tool_answer.strip() or None
    return output_text.strip() or None


def _aggregate_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate a list of metric dicts by averaging values."""
    aggregated = {}
    for metric_dict in all_metrics:
        for key, value in metric_dict.items():
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(value)

    return {key: sum(values) / len(values) for key, values in aggregated.items()}


# =============================================================================
# Default Metrics - TTR and length (computed for all tasks)
# =============================================================================

def compute_default_metrics(output: dict[str, Any]) -> dict[str, float]:
    """
    Compute default metrics for a single output.

    Returns:
        - token_ttr: 1-gram token type-ratio
        - token_3gram_ttr: 3-gram token type-ratio
        - text_ttr: word-level type-ratio
        - length: output length in tokens
    """
    output_ids = output["output_ids"]
    output_text = output["text"]
    return {
        "token_ttr": compute_token_ttr(output_ids),
        "token_3gram_ttr": compute_token_ttr(output_ids, 3),
        "text_ttr": compute_text_ttr(output_text),
        "length": len(output_ids),
    }


# =============================================================================
# Math Tasks - Using math_verify
# =============================================================================

@lru_cache(maxsize=1)
def _get_math_verify_objects():
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from math_verify.parser import parse

    return parse, verify, LatexExtractionConfig, ExprExtractionConfig


def _math_verify_score(model_output: str, ground_truth: str) -> float:
    """Score a mathematical response with Hugging Face math_verify."""
    parse, verify, LatexExtractionConfig, ExprExtractionConfig = _get_math_verify_objects()
    ground_truth_boxed = ground_truth if "\\boxed" in ground_truth else f"\\boxed{{{ground_truth}}}"
    extracted_gold = parse(ground_truth_boxed, (LatexExtractionConfig(),))
    extracted_pred = parse(model_output, (ExprExtractionConfig(), LatexExtractionConfig()))
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(gold, pred) for gold in extracted_gold) else 0.0 for pred in extracted_pred)
    return 0.0

def math_task(output: dict[str, Any] | list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    Mathematical reasoning evaluation using math_verify.

    Supports: GSM8K, MATH-500, AMC12, OLYMPIAD_BENCH, MATH_ARENA, OMNI_MATH_*

    Metrics (organized by benchmark):
        - {task_name}/accuracy: correct answer rate
        - {task_name}/valid: non-empty answer rate
    """
    task_name = params.task_name

    if isinstance(output, dict):
        output_text = output["text"]
        true_answer = str(params.answer)

        metrics = {
            f"{task_name}/accuracy": 0.0,
            f"{task_name}/valid": 0.0,
        }

        predicted_answer = extract_answer(output_text)

        if predicted_answer:
            metrics[f"{task_name}/valid"] = 1.0
            try:
                metrics[f"{task_name}/accuracy"] = _math_verify_score(predicted_answer, true_answer)
            except Exception as exc:
                logger.warning("math_verify failed for %s: %s", task_name, exc)

        return metrics
    else:
        return _aggregate_metrics([math_task(o, params) for o in output])


# =============================================================================
# Fact Verification Tasks
# =============================================================================

def fact_verification_task(output: dict[str, Any] | list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    Fact verification evaluation (FEVER, FEVEROUS, AVeriTeC).

    Labels vary by dataset:
    - FEVER/FEVEROUS: SUPPORTS, REFUTES, NOT ENOUGH INFO
    - AVeriTeC: Supported, Refuted, Conflicting Evidence/Cherrypicking, Not Enough Evidence

    Metrics (organized by benchmark):
        - {task_name}/accuracy: correct label rate
        - {task_name}/valid: non-empty answer rate
    """
    task_name = params.task_name

    if isinstance(output, dict):
        output_text = output["text"]
        true_answer = str(params.answer).lower().strip()

        metrics = {
            f"{task_name}/accuracy": 0.0,
            f"{task_name}/valid": 0.0,
        }

        predicted_answer = extract_answer(output_text)

        if predicted_answer:
            metrics[f"{task_name}/valid"] = 1.0
            predicted_lower = predicted_answer.lower().strip()

            # Check for exact or partial match
            if predicted_lower == true_answer:
                metrics[f"{task_name}/accuracy"] = 1.0
            elif true_answer in predicted_lower or predicted_lower in true_answer:
                metrics[f"{task_name}/accuracy"] = 1.0

        return metrics
    else:
        return _aggregate_metrics([fact_verification_task(o, params) for o in output])


# =============================================================================
# Multiple Choice Tasks
# =============================================================================

_MCQ_ANSWER_PATTERN = re.compile(
    r"\b(?:FINAL\s+)?ANSWER(?:\s+IS)?\s*[:\-]?\s*([A-Z])\b",
    flags=re.IGNORECASE,
)


def extract_choice_letter(answer_text: str) -> str | None:
    """Extract an explicit MCQA choice without treating the first letter as the answer."""
    text = answer_text.strip().upper()

    explicit_answers = _MCQ_ANSWER_PATTERN.findall(text)
    if explicit_answers:
        return explicit_answers[-1]

    final_line = text.rsplit("\n", 1)[-1]
    bare_answer = re.fullmatch(r"\s*([A-Z])\s*[\).]?\s*", final_line)
    if bare_answer:
        return bare_answer.group(1)

    option_start = re.match(r"^\s*([A-Z])\s*[\).:\-]\s+\S", text)
    return option_start.group(1) if option_start else None

def mcq_task(output: dict[str, Any] | list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    Multiple choice question evaluation (GPQA, ACP_BENCH).

    Labels: A, B, C, D, E

    Metrics (organized by benchmark):
        - {task_name}/accuracy: correct answer rate
        - {task_name}/valid: extractable answer rate
    """
    task_name = params.task_name

    if isinstance(output, dict):
        output_text = output["text"]
        true_answer = str(params.answer).upper().strip()

        metrics = {
            f"{task_name}/accuracy": 0.0,
            f"{task_name}/valid": 0.0,
        }

        answer_text = extract_answer(output_text)
        predicted_answer = extract_choice_letter(answer_text) if answer_text else None

        if predicted_answer:
            metrics[f"{task_name}/valid"] = 1.0
            if predicted_answer == true_answer:
                metrics[f"{task_name}/accuracy"] = 1.0

        return metrics
    else:
        return _aggregate_metrics([mcq_task(o, params) for o in output])


# =============================================================================
# Code Evaluation Tasks
# =============================================================================

if hf_evaluate is not None:
    try:
        CODE_EVAL = hf_evaluate.load("code_eval")
    except Exception as exc:
        logger.warning("Could not load evaluate/code_eval: %s", exc)
        CODE_EVAL = None
else:
    CODE_EVAL = None


def _extract_python_code(text: str) -> str:
    """Extract a Python code fence when present, otherwise return plain code."""
    if "<|inner_suffix|>" in text:
        text = text.split("<|inner_suffix|>", 1)[1]
    elif "</think>" in text:
        text = text.split("</think>", 1)[1]
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def _extract_code_from_thinking(text: str) -> str:
    """Extract code from ```python blocks after skipping the thinking section."""
    return _extract_python_code(text)


def _compute_valid_ratio(output_texts: list[str]) -> float:
    """Compute ratio of outputs with proper markdown code blocks (valid format)."""
    valid_count = sum(
        1 for text in output_texts
        if "```python" in text and "```" in text.split("```python")[1]
    )
    return valid_count / len(output_texts) if output_texts else 0.0


def _build_test_reference(params: RolloutParams) -> str:
    """
    Build test reference string for code evaluation.

    Supports both HumanEval and MBPP formats:
    - HumanEval: uses 'test' + 'entry_point' with check() call
    - MBPP: uses 'test_list' (list of assertions) or 'test' directly
    """
    test = params.kwargs.get("test", "")
    entry_point = params.kwargs.get("entry_point", "")
    test_list = params.kwargs.get("test_list", [])

    # MBPP format: test_list contains assertion strings
    if test_list:
        return "\n".join(test_list)

    # HumanEval format: test contains check function, needs check() call
    if entry_point and test:
        return f"{test}\ncheck({entry_point})"

    # Fallback: use test directly
    return test


def _build_code_candidates(output_texts: list[str], params: RolloutParams) -> list[str]:
    """Build one executable candidate per sample for pass@k evaluation."""
    prompt = params.kwargs.get("prompt", "")
    entry_point = params.kwargs.get("entry_point", "")
    candidates = []
    for output_text in output_texts:
        code = _extract_python_code(output_text)
        # HumanEval commonly generates only the function body. Do not prepend
        # the prompt when the response already contains the target definition.
        if prompt and (not entry_point or f"def {entry_point}" not in code):
            # HumanEval prompts typically end in indentation spaces; appending
            # directly preserves those spaces as the generated body's indent.
            separator = "" if prompt[-1:].isspace() else "\n"
            code = prompt + separator + code
        candidates.append(code)
    return candidates


def code_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    Unified code generation evaluation for HumanEval and MBPP benchmarks.

    Requires: HuggingFace evaluate library with code_eval

    Supports both formats:
    - HumanEval: prompt + output, test with check(entry_point)
    - MBPP: output only (no prompt prefix), test_list with assertions

    Metrics:
        - {task_name}/pass@10: pass@10 metric
    """
    task_name = params.task_name

    if CODE_EVAL is None:
        return {f"{task_name}/accuracy": 0.0}

    output_texts = [output["text"] for output in outputs]
    candidates = _build_code_candidates(output_texts, params)

    test_reference = _build_test_reference(params)

    n_samples = len(output_texts)
    ks = [1]
    if n_samples >= 10:
        ks.append(10)

    pass_at_k, _ = CODE_EVAL.compute(
        references=[test_reference],
        predictions=[candidates],
        k=ks,
    )

    metrics = {}
    if "pass@1" in pass_at_k:
        metrics[f"{task_name}/accuracy"] = float(pass_at_k["pass@1"])
    if "pass@10" in pass_at_k:
        metrics[f"{task_name}/pass@10"] = float(pass_at_k["pass@10"])

    if not metrics:
        logger.warning(f"No pass@k results found in code evaluation for task {task_name}. Results: {pass_at_k}. Predictions: {candidates} ({n_samples} samples)")
        metrics[f"{task_name}/accuracy"] = 0.0

    return metrics


def code_thinking_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    Unified code evaluation with extended thinking/reasoning.

    Extracts code from ```python blocks after skipping the thinking section.
    Works for both HumanEval and MBPP benchmarks.

    Metrics:
        - {task_name}/valid: ratio of outputs with proper markdown code blocks
        - {task_name}/pass@10: pass@10 for extracted code
    """
    task_name = params.task_name

    if CODE_EVAL is None:
        return {f"{task_name}/valid": 0.0, f"{task_name}/accuracy": 0.0}

    output_texts = [output["text"] for output in outputs]

    valid_ratio = _compute_valid_ratio(output_texts)
    extracted_code = _build_code_candidates(output_texts, params)

    test_reference = _build_test_reference(params)

    n_samples = len(output_texts)
    ks = [1]
    if n_samples >= 10:
        ks.append(10)

    pass_at_k, _ = CODE_EVAL.compute(
        references=[test_reference],
        predictions=[extracted_code],
        k=ks,
    )

    metrics = {f"{task_name}/valid": valid_ratio}
    if "pass@1" in pass_at_k:
        metrics[f"{task_name}/accuracy"] = float(pass_at_k["pass@1"])
    if "pass@10" in pass_at_k:
        metrics[f"{task_name}/pass@10"] = float(pass_at_k["pass@10"])

    if f"{task_name}/accuracy" not in metrics and f"{task_name}/pass@10" not in metrics:
        logger.warning(f"No pass@k results found in code evaluation for task {task_name}. Results: {pass_at_k}. Predictions: {extracted_code} ({n_samples} samples)")
        metrics[f"{task_name}/accuracy"] = 0.0

    return metrics


# Backward compatibility aliases
def humaneval_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """HumanEval code generation evaluation. Alias for code_task."""
    return code_task(outputs, params)


def humaneval_thinking_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """HumanEval with thinking. Alias for code_thinking_task."""
    return code_thinking_task(outputs, params)


# =============================================================================
# Instruction Following Tasks
# =============================================================================

from .ifeval.instructions_registry import INSTRUCTION_DICT as IFEVAL_INSTRUCTION_DICT
from .ifbench.instructions_registry import INSTRUCTION_DICT as IFBENCH_INSTRUCTION_DICT
from .ifbench.instructions_util import _safe_nltk_download


def parse_non_reasoning_apertus(output_text: str) -> str:
    """Extract non-reasoning content from Apertus output."""
    if "<|inner_suffix|>" in output_text:
        output_text = output_text.split("<|inner_suffix|>")[1]
    return output_text.strip()


@lru_cache(maxsize=1)
def prepare_instruction_following_dependencies() -> None:
    """Prepare NLTK data needed by IFEval/IFBench before worker processes fork."""
    for resource in (
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
        "taggers/averaged_perceptron_tagger_eng",
        "stopwords",
    ):
        _safe_nltk_download(resource)


def ifeval_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    IFEval instruction following evaluation.

    Checks if outputs satisfy all specified instructions.

    Metrics:
        - ifeval/accuracy: ratio of outputs satisfying ALL instructions
    """
    output_texts = [parse_non_reasoning_apertus(output["text"]) for output in outputs]

    instruction_id_list = params.kwargs.get("instruction_id_list", [])
    instruction_kwargs_list = params.kwargs.get("instruction_kwargs", [])

    results = []
    for output_text in output_texts:
        output_results = []
        for instruction_id, instruction_kwargs in zip(
            instruction_id_list,
            instruction_kwargs_list
        ):
            instruction_cls = IFEVAL_INSTRUCTION_DICT[instruction_id]
            instruction_obj = instruction_cls(instruction_id)
            _ = instruction_obj.build_description(**{
                k: v for k, v in instruction_kwargs.items() if v is not None
            })

            try:
                output_results.append(instruction_obj.check_following(output_text))
            except Exception as e:
                print(f"Error checking instruction {instruction_id}: {e}")
                output_results.append(False)

        results.append(all(output_results))

    return {"ifeval/accuracy": float(sum(results) / len(results))}


def ifbench_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    IFBench instruction following evaluation (out-of-distribution).

    Same structure as IFEval but with different instruction set.

    Metrics:
        - ifbench/accuracy: ratio of outputs satisfying ALL instructions
    """
    output_texts = [parse_non_reasoning_apertus(output["text"]) for output in outputs]

    instruction_id_list = params.kwargs.get("instruction_id_list", [])
    instruction_kwargs_list = params.kwargs.get("instruction_kwargs", [])

    results = []
    for output_text in output_texts:
        output_results = []
        for instruction_id, instruction_kwargs in zip(
            instruction_id_list,
            instruction_kwargs_list
        ):
            instruction_cls = IFBENCH_INSTRUCTION_DICT[instruction_id]
            instruction_obj = instruction_cls(instruction_id)
            _ = instruction_obj.build_description(**{
                k: v for k, v in instruction_kwargs.items() if v is not None
            })

            try:
                output_results.append(instruction_obj.check_following(output_text))
            except Exception as e:
                print(f"Error checking instruction {instruction_id}: {e}")
                output_results.append(False)

        results.append(all(output_results))

    return {"ifbench/accuracy": float(sum(results) / len(results))}


# =============================================================================
# Task Registry
# =============================================================================

TASK_REGISTRY: dict[str, Callable[[Union[dict[str, Any], list[dict[str, Any]]], RolloutParams], dict[str, float]]] = {
    # Math reasoning (all use math_task with math_verify)
    "math": math_task,
    "gsm8k": math_task,
    "math_500": math_task,
    "olympiad_bench": math_task,
    "omni_math": math_task,
    "omni_math_easy": math_task,
    "omni_math_med": math_task,
    "omni_math_hard": math_task,

    # Fact verification
    "fact_verification": fact_verification_task,
    "averitec": fact_verification_task,

    # Multiple choice
    "mcq": mcq_task,

    # Code evaluation (unified for HumanEval and MBPP)
    "code": code_task,
    "code_thinking": code_thinking_task,
    "humaneval": code_task,
    "humaneval_thinking": code_thinking_task,
    "mbpp": code_task,
    "mbpp_thinking": code_thinking_task,

    # Instruction following
    "ifeval": ifeval_task,
    "ifbench": ifbench_task,
}


def get_task(task_id: str | None) -> Callable[[Union[dict[str, Any], list[dict[str, Any]]], RolloutParams], dict[str, float]] | None:
    """Get a task function by ID, returning None if not found."""
    if task_id is None or task_id not in TASK_REGISTRY:
        return None
    return TASK_REGISTRY[task_id]
