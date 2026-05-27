# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Math (no-format) reward function for verl.
#
# Two entry points:
#   - compute_score(...)        : per-sample (NaiveRewardManager). NO LLM judge.
#                                 Kept for backward compat / debug.
#   - compute_score_batch(...)  : batch (BatchRewardManager). MIRRORS Easy-R1's
#                                 math_no_format.compute_score 1:1, including
#                                 the rule -> sympy -> LLM judge fallback.
#
# Result dict keys consumed downstream:
#   - score          : scalar reward forwarded to advantage estimator
#   - accuracy       : per-sample correctness (0/1) — used by branch_adv (EDPO)
#   - answer_pred    : extracted boxed answer    — used by branch_adv (EDPO)
#   - format         : format score (always 1.0 here, since format check is disabled)
#   - length_penalty : soft over-long penalty
#   - overall        : alias of score for parity with Easy-R1

import re
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer

try:
    from sympy import Basic, sympify

    _HAS_SYMPY = True
except Exception:  # pragma: no cover
    _HAS_SYMPY = False

# LLM judge fallback (mirrors Easy-R1).
try:
    from .llm_judge import llm_judge_batch  # type: ignore
except ImportError:
    import importlib.util as _ilu
    import os as _os

    _spec = _ilu.spec_from_file_location(
        "llm_judge",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "llm_judge.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    llm_judge_batch = _mod.llm_judge_batch
    del _ilu, _os, _spec, _mod


TYPE_TAG_RE = re.compile(r"<type>.*?</type>", re.DOTALL)


def _normalize_ground_truth(ground_truth: Any) -> str:
    if ground_truth is None:
        return ""
    return TYPE_TAG_RE.sub("", str(ground_truth)).strip()


def _sympy_equivalent(lhs: str, rhs: str, tolerance: float = 1e-6) -> bool:
    if not _HAS_SYMPY:
        return False
    try:
        lhs_expr = sympify(lhs, strict=True)
        rhs_expr = sympify(rhs, strict=True)
    except Exception:
        return False
    if isinstance(lhs_expr, Basic) and isinstance(rhs_expr, Basic):
        if lhs_expr.is_real and rhs_expr.is_real:
            denom = abs(rhs_expr) + 1e-6
            return abs(lhs_expr - rhs_expr) / denom < tolerance
    return False


def _soft_overlong_punishment(
    response_length: int,
    max_response_length: int,
    overlong_buffer_length: int = 3072,
) -> float:
    if overlong_buffer_length <= 0:
        return 0.0
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0


def _clean_response(solution_str: str) -> str:
    return re.sub(r"\s*(<|>|/)\s*", r"\1", solution_str)


def _resolve_response_length(extra_info: Optional[dict], response: str) -> int:
    if extra_info is not None:
        for key in ("response_length", "valid_response_length", "response_len"):
            if key in extra_info:
                try:
                    return int(extra_info[key])
                except Exception:
                    pass
    # Fallback: word-count proxy for token length. With penalty_max_length=50000
    # this stays well under threshold so length_penalty=0 (Easy-R1 parity for
    # the default GRPO base run).
    return len(response.split()) if isinstance(response, str) else 0


def _rule_score(answer: str, normalized_gt: str) -> Optional[float]:
    """Returns 1.0 if rule/sympy matches, 0.0 if both miss boxed/gt, None if undecided."""
    if not answer or not normalized_gt:
        return 0.0
    ans = answer.strip()
    if grade_answer(ans, normalized_gt):
        return 1.0
    if _sympy_equivalent(ans, normalized_gt):
        return 1.0
    return None


def compute_score(
    data_source: Any = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Optional[dict] = None,
    *,
    penalty_max_length: int = 16384,
    overlong_buffer_length: int = 0,
    overlong_penalty_factor: float = 0.1,
    **kwargs,
) -> dict:
    """Per-sample reward (NaiveRewardManager). NO LLM judge — sync HTTP would
    serialize per-sample under the naive manager and stall training.

    For full Easy-R1 parity (rule -> sympy -> LLM judge), use
    `compute_score_batch` together with `reward.reward_manager.name=batch`.
    """
    response = _clean_response(solution_str)
    response_length = _resolve_response_length(extra_info, response)
    length_penalty = _soft_overlong_punishment(
        response_length=response_length,
        max_response_length=int(penalty_max_length),
        overlong_buffer_length=int(overlong_buffer_length),
    )

    normalized_gt = _normalize_ground_truth(ground_truth)
    answer = extract_boxed_content(response) or ""

    rule = _rule_score(answer, normalized_gt)
    accuracy_score = 0.0 if rule is None else float(rule)
    overall = float(accuracy_score) + float(overlong_penalty_factor) * float(length_penalty)

    return {
        "score": overall,
        "overall": overall,
        "accuracy": float(accuracy_score),
        "length_penalty": float(length_penalty),
        "answer_pred": answer.strip() if answer else "",
        "format": 1.0,
    }


def compute_score_batch(
    data_sources: Optional[list] = None,
    solution_strs: Optional[list[str]] = None,
    ground_truths: Optional[list] = None,
    extra_infos: Optional[list[dict]] = None,
    *,
    penalty_max_length: int = 16384,
    overlong_buffer_length: int = 0,
    overlong_penalty_factor: float = 0.1,
    enable_llm_judge: bool = True,
    llm_judge_host: Optional[str] = None,
    llm_judge_ports: Optional[list[int]] = None,
    llm_judge_model: Optional[str] = None,
    llm_judge_timeout: Optional[float] = None,
    llm_judge_max_retries: Optional[int] = None,
    llm_judge_max_workers: Optional[int] = None,
    **kwargs,
) -> list[dict]:
    """Batch reward (BatchRewardManager). 1:1 mirror of Easy-R1's
    `math_no_format.compute_score` flow:

        for each sample:
            rule (mathruler.grade_answer)
            -> sympy equivalence
            -> [if both fail, defer to LLM judge job queue]
        fan-out llm_judge_batch (ThreadPoolExecutor, port round-robin)
        backfill scores
        overall = accuracy + overlong_penalty_factor * length_penalty

    Defaults (host / ports / model / endpoint) come from `llm_judge.py`
    module-level env reads — same as Easy-R1, so the same training run
    talks to the same judge cluster.
    """
    solution_strs = list(solution_strs or [])
    ground_truths = list(ground_truths or [])
    extra_infos = list(extra_infos or [{} for _ in solution_strs])
    n = len(solution_strs)
    if len(ground_truths) != n:
        # Defensive: pad/truncate to align (verl already aligns these).
        ground_truths = (ground_truths + [None] * n)[:n]
    if len(extra_infos) != n:
        extra_infos = (extra_infos + [{}] * n)[:n]

    scores: list[dict] = []
    judge_jobs: list[dict] = []

    for idx in range(n):
        solution_str = solution_strs[idx] or ""
        gt = ground_truths[idx]
        ei = extra_infos[idx] or {}
        is_val = isinstance(ei, dict) and str(ei.get("split", "")).lower() == "val"

        response = _clean_response(solution_str)
        response_length = _resolve_response_length(ei, response)
        if is_val:
            # Easy-R1 val_wo_format.py: no length penalty in val.
            length_penalty = 0.0
        else:
            length_penalty = _soft_overlong_punishment(
                response_length=response_length,
                max_response_length=int(penalty_max_length),
                overlong_buffer_length=int(overlong_buffer_length),
            )

        normalized_gt = _normalize_ground_truth(gt)
        answer = extract_boxed_content(response) or ""
        ans = answer.strip() if answer else ""

        accuracy_score: Optional[float]
        if not ans or not normalized_gt:
            accuracy_score = 0.0
        elif is_val:
            # Easy-R1 val path: pure rule-based grade_answer, no sympy / no LLM judge.
            accuracy_score = 1.0 if grade_answer(ans, normalized_gt) else 0.0
        else:
            rule = _rule_score(ans, normalized_gt)
            if rule is not None:
                accuracy_score = float(rule)
            elif enable_llm_judge:
                accuracy_score = None  # placeholder, will be backfilled
                # `prompt` only used by sqa.jinja; math.jinja ignores it. We
                # still pass through extra_info["prompt"] / ["question"] when
                # present so sqa-style rewards can be plugged in later.
                judge_prompt = None
                if isinstance(ei, dict):
                    judge_prompt = ei.get("question") or ei.get("prompt")
                judge_jobs.append(
                    {
                        "index": idx,
                        "verify_type": "math",
                        "label": normalized_gt,
                        "predict": ans,
                        "prompt": judge_prompt,
                    }
                )
            else:
                accuracy_score = 0.0

        scores.append(
            {
                "accuracy": accuracy_score,
                "length_penalty": float(length_penalty),
                "answer_pred": ans,
                # Easy-R1 val_wo_format returns format=0.0; train returns 1.0.
                "format": 0.0 if is_val else 1.0,
                "_is_val": is_val,
            }
        )

    # Fan-out LLM judge (thread pool, port round-robin) — mirrors Easy-R1.
    if judge_jobs:
        results = llm_judge_batch(
            [
                {
                    "verify_type": job["verify_type"],
                    "label": job["label"],
                    "predict": job["predict"],
                    "prompt": job["prompt"],
                }
                for job in judge_jobs
            ],
            max_workers=llm_judge_max_workers,
            host=llm_judge_host,
            ports=llm_judge_ports,
            model=llm_judge_model,
            timeout=llm_judge_timeout,
            max_retries=llm_judge_max_retries,
        )
        for job, judge_ok in zip(judge_jobs, results):
            scores[job["index"]]["accuracy"] = 1.0 if judge_ok else 0.0

    # Finalize: build score / overall, fill any leftover None as 0.0.
    for s in scores:
        acc = s["accuracy"] if s["accuracy"] is not None else 0.0
        s["accuracy"] = float(acc)
        is_val = bool(s.pop("_is_val", False))
        if is_val:
            # Easy-R1 val_wo_format.py: overall == accuracy, no length-penalty mix.
            overall = float(s["accuracy"])
        else:
            overall = float(s["accuracy"]) + float(overlong_penalty_factor) * float(s["length_penalty"])
        s["score"] = overall
        s["overall"] = overall

    return scores
