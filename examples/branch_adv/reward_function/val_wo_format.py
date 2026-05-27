# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Validation reward (no-format) for verl. Per-sample API.

import re
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer


TYPE_TAG_RE = re.compile(r"<type>.*?</type>", re.DOTALL)


def _normalize_ground_truth(ground_truth: Any) -> str:
    if ground_truth is None:
        return ""
    return TYPE_TAG_RE.sub("", str(ground_truth)).strip()


def _extract_answer(response: str, allow_unboxed: bool) -> str:
    answer = extract_boxed_content(response)
    if answer:
        return answer
    if not allow_unboxed:
        return ""
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def compute_score(
    data_source: Any = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Optional[dict] = None,
    *,
    allow_unboxed: bool = False,
    format_weight: float = 0.0,
    **kwargs,
) -> dict:
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", solution_str)
    answer = _extract_answer(response, allow_unboxed)
    normalized_gt = _normalize_ground_truth(ground_truth)
    accuracy_score = 1.0 if answer and grade_answer(answer, normalized_gt) else 0.0

    return {
        "score": accuracy_score,
        "overall": accuracy_score,
        "accuracy": accuracy_score,
        "format": 0.0,
    }
