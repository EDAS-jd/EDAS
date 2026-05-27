import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
from jinja2 import Template

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "llm_judge_prompt"))

_TEMPLATE_CACHE: dict[str, Template] = {}
_TEMPLATE_LOCK = threading.Lock()

_SESSION_LOCAL = threading.local()

_DEFAULT_HOST = os.getenv("LLM_JUDGE_HOST", "10.119.97.103")
_DEFAULT_PORTS_RAW = os.getenv("LLM_JUDGE_PORTS", "9000-9007")
_DEFAULT_ENDPOINT = os.getenv("LLM_JUDGE_ENDPOINT", "/v1/chat/completions")
_DEFAULT_MODEL = os.getenv("LLM_JUDGE_MODEL", "/mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b")
_DEFAULT_TIMEOUT = float(os.getenv("LLM_JUDGE_TIMEOUT", "30"))
_DEFAULT_MAX_RETRIES = int(os.getenv("LLM_JUDGE_MAX_RETRIES", "2"))
_DEFAULT_MAX_WORKERS = int(os.getenv("LLM_JUDGE_MAX_WORKERS", "512"))
_DEFAULT_MAX_TOKENS = int(os.getenv("LLM_JUDGE_MAX_TOKENS", "1024"))
_DEBUG_ENABLED = os.getenv("LLM_JUDGE_DEBUG", "").lower() in ("1", "true", "yes")
_DEBUG_LOG_PATH = os.getenv("LLM_JUDGE_DEBUG_LOG_PATH", "")
_DEBUG_MAX_RECORDS = int(os.getenv("LLM_JUDGE_DEBUG_MAX_RECORDS", "1000"))

_PORT_INDEX = 0
_PORT_LOCK = threading.Lock()
_DEBUG_LOCK = threading.Lock()
_DEBUG_COUNTER = 0
_SCORE_PATTERN = re.compile(os.getenv("LLM_JUDGE_SCORE_PATTERN", r"\b([TF])\b(?![\s\S]*\b[TF]\b)"), re.IGNORECASE)


def _parse_ports(port_spec: str) -> list[int]:
    ports: list[int] = []
    for part in (port_spec or "").split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start <= end:
                ports.extend(range(start, end + 1))
            else:
                ports.extend(range(start, end - 1, -1))
        else:
            ports.append(int(chunk))
    seen = set()
    unique_ports = []
    for port in ports:
        if port not in seen:
            unique_ports.append(port)
            seen.add(port)
    return unique_ports


def _load_template(verify_type: str) -> Template:
    normalized = str(verify_type or "").strip().lower()
    if normalized == "llava":
        normalized = "sqa"
    if not normalized:
        raise ValueError("verify_type is required for llm judge.")
    with _TEMPLATE_LOCK:
        cached = _TEMPLATE_CACHE.get(normalized)
        if cached is not None:
            return cached
        filename = f"{normalized}.jinja"
        path = os.path.join(PROMPT_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"LLM judge prompt not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            template = Template(f.read())
        _TEMPLATE_CACHE[normalized] = template
        return template


def _next_port(ports: list[int]) -> int:
    global _PORT_INDEX
    if not ports:
        raise ValueError("No ports configured for LLM judge.")
    with _PORT_LOCK:
        port = ports[_PORT_INDEX % len(ports)]
        _PORT_INDEX += 1
    return port


def _parse_judge_content(content: str) -> Optional[bool]:
    if not content:
        return None
    text = content.strip()
    if not text:
        return None
    text_upper = text.upper()
    match = _SCORE_PATTERN.search(text_upper)
    if match:
        return match.group(1) == "T"
    return None


def _extract_chat_content(data: dict) -> str:
    try:
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if content:
            return content
        return message.get("reasoning_content", "") or ""
    except Exception:
        return ""


def _extract_completion_content(data: dict) -> str:
    try:
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("text", "") or ""
    except Exception:
        return ""


def _get_session() -> requests.Session:
    session = getattr(_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _SESSION_LOCAL.session = session
    return session


def _append_debug(record: dict) -> None:
    global _DEBUG_COUNTER
    if not _DEBUG_ENABLED or not _DEBUG_LOG_PATH:
        return
    with _DEBUG_LOCK:
        if _DEBUG_COUNTER >= _DEBUG_MAX_RECORDS:
            return
        _DEBUG_COUNTER += 1
        try:
            with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            return


def _record_debug(
    verify_type: str,
    label: str,
    predict: str,
    prompt: Optional[str],
    url: str,
    status_code: Optional[int],
    content: str,
    parsed: Optional[bool],
    error: Optional[str],
) -> None:
    if not _DEBUG_ENABLED or not _DEBUG_LOG_PATH:
        return
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verify_type": verify_type,
        "label": label,
        "predict": predict,
        "prompt_len": len(prompt) if prompt else 0,
        "url": url,
        "status_code": status_code,
        "content_head": content[:200],
        "parsed": parsed,
        "error": error,
    }
    _append_debug(record)


def _record_debug_response(url: str, status_code: Optional[int], body: Optional[dict], error: Optional[str]) -> None:
    if not _DEBUG_ENABLED or not _DEBUG_LOG_PATH:
        return
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "url": url,
        "status_code": status_code,
        "response_json": body,
        "error": error,
    }
    _append_debug(record)


def _resolve_max_workers(max_workers: Optional[int], job_count: int) -> int:
    resolved = max_workers if max_workers is not None else _DEFAULT_MAX_WORKERS
    if resolved <= 0:
        resolved = 1
    if job_count > 0:
        resolved = min(resolved, job_count)
    return max(1, resolved)


def llm_judge_is_correct(
    verify_type: str,
    label: str,
    predict: str,
    prompt: Optional[str] = None,
    host: Optional[str] = None,
    ports: Optional[list[int]] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> Optional[bool]:
    template = _load_template(verify_type)
    system_prompt = template.render(label=label, predict=predict, prompt=prompt, question=prompt)

    resolved_host = host or _DEFAULT_HOST
    resolved_ports = ports if ports is not None else _parse_ports(_DEFAULT_PORTS_RAW)
    if isinstance(resolved_ports, str):
        resolved_ports = _parse_ports(resolved_ports)
    resolved_model = model if model is not None else _DEFAULT_MODEL
    resolved_timeout = _DEFAULT_TIMEOUT if timeout is None else timeout
    resolved_retries = _DEFAULT_MAX_RETRIES if max_retries is None else max_retries
    resolved_max_tokens = _DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Respond with only T or F."},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": resolved_max_tokens,
        "stream": False,
    }
    if resolved_model:
        payload["model"] = resolved_model

    attempts = max(1, resolved_retries)
    last_error = None
    endpoint = _DEFAULT_ENDPOINT if _DEFAULT_ENDPOINT.startswith("/") else f"/{_DEFAULT_ENDPOINT}"
    for _ in range(attempts):
        port = _next_port(resolved_ports)
        url = f"http://{resolved_host}:{port}{endpoint}"
        try:
            resp = _get_session().post(url, json=payload, timeout=resolved_timeout)
        except requests.RequestException as exc:
            last_error = exc
            _record_debug(
                verify_type,
                label,
                predict,
                prompt,
                url,
                None,
                "",
                None,
                f"request_error:{exc}",
            )
            continue
        if resp.status_code != 200:
            last_error = RuntimeError(f"LLM judge status {resp.status_code}")
            _record_debug(
                verify_type,
                label,
                predict,
                prompt,
                url,
                resp.status_code,
                resp.text or "",
                None,
                f"status_error:{resp.status_code}",
            )
            continue
        try:
            data = resp.json()
        except ValueError as exc:
            last_error = exc
            _record_debug(
                verify_type,
                label,
                predict,
                prompt,
                url,
                resp.status_code,
                resp.text or "",
                None,
                f"json_error:{exc}",
            )
            continue
        _record_debug_response(url, resp.status_code, data, None)
        content = _extract_chat_content(data)
        if not content:
            content = _extract_completion_content(data)
        parsed = _parse_judge_content(content)
        _record_debug(
            verify_type,
            label,
            predict,
            prompt,
            url,
            resp.status_code,
            content,
            parsed,
            None,
        )
        if parsed is not None:
            return parsed

    _ = last_error
    return None


def llm_judge_batch(
    jobs: list[dict],
    max_workers: Optional[int] = None,
    host: Optional[str] = None,
    ports: Optional[list[int]] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> list[Optional[bool]]:
    if not jobs:
        return []
    resolved_workers = _resolve_max_workers(max_workers, len(jobs))
    if resolved_workers == 1:
        return [
            llm_judge_is_correct(
                job.get("verify_type", ""),
                label=job.get("label", ""),
                predict=job.get("predict", ""),
                prompt=job.get("prompt"),
                host=host,
                ports=ports,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
                max_tokens=max_tokens,
            )
            for job in jobs
        ]

    results: list[Optional[bool]] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        future_map = {}
        for idx, job in enumerate(jobs):
            future = executor.submit(
                llm_judge_is_correct,
                job.get("verify_type", ""),
                label=job.get("label", ""),
                predict=job.get("predict", ""),
                prompt=job.get("prompt"),
                host=host,
                ports=ports,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
                max_tokens=max_tokens,
            )
            future_map[future] = idx

        for future, idx in future_map.items():
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None

    return results
