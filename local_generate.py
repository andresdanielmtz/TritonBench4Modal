"""
Generate TritonBench-T predictions with a local OpenAI-compatible model server.

This script is intentionally separate from modal_app.py. It runs on your laptop,
talks to a local model endpoint such as Ollama at http://localhost:11434/v1,
and writes a predictions.jsonl file that modal_app.py::evaluate_only can upload
and evaluate on Modal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROMPT_HEADER = (
    "You are an expert in Triton programming, capable of writing Triton kernels "
    "and wrapper functions based on functional descriptions and function "
    "parameters. The wrapper function must fully match the provided function "
    "signature.\n\n"
    "Output a single, self-contained Python module containing: (a) the necessary "
    "imports (torch, triton, triton.language as tl), (b) the Triton kernel(s), "
    "and (c) the wrapper function that the description specifies. Wrap the "
    "entire module in one ```python ... ``` fenced code block. Do NOT include "
    "any test code or example calls - tests will be appended separately."
)


def _load_alpaca(repo_dir: Path, dataset: str) -> list[dict[str, Any]]:
    if dataset not in {"simp", "comp"}:
        raise ValueError("dataset must be 'simp' or 'comp'")

    path = repo_dir / f"data/TritonBench_T_{dataset}_alpac_v1.json"
    if not path.exists():
        raise FileNotFoundError(
            f"could not find {path}\n"
            "Clone TritonBench locally or pass --repo /path/to/TritonBench."
        )
    return json.loads(path.read_text())


def _build_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    instr = item["instruction"]
    inp = item.get("input", "") or ""
    user = instr if not inp else f"{instr}\n\n{inp}"
    return [
        {"role": "system", "content": PROMPT_HEADER},
        {"role": "user", "content": user},
    ]


def _extract_code(text: str) -> str:
    """Strip Markdown fences and return raw Python source."""
    s = text.strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip() + "\n"


def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach local model server at {url}: {e}") from e

    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def _read_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    records: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: ignoring invalid JSON on line {line_no}", file=sys.stderr)
                continue
            instruction = rec.get("instruction")
            predict = rec.get("predict")
            if isinstance(instruction, str) and isinstance(predict, str):
                records[instruction] = {"instruction": instruction, "predict": predict}
    return records


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TritonBench-T predictions using a local model server."
    )
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--dataset", choices=["simp", "comp"], default="simp")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="local_predictions.jsonl")
    parser.add_argument(
        "--repo",
        default=os.environ.get("TRITONBENCH_LOCAL_REPO", "./TritonBench"),
        help="Path to a local clone of the upstream TritonBench repository.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
        help="OpenAI-compatible API base URL for the local model server.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LOCAL_LLM_API_KEY", "ollama"),
        help="API key sent to the local OpenAI-compatible endpoint.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel requests. Keep this at 1 for most laptop models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing output file instead of resuming completed rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_dir = Path(args.repo).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    items = _load_alpaca(repo_dir, args.dataset)
    if args.limit:
        items = items[: args.limit]

    existing = {} if args.overwrite else _read_existing(output_path)
    results: list[dict[str, str] | None] = []
    pending: list[tuple[int, dict[str, Any]]] = []

    for i, item in enumerate(items):
        prior = existing.get(item["instruction"])
        if prior:
            results.append(prior)
        else:
            results.append(None)
            pending.append((i, item))

    print(f"dataset: {args.dataset}")
    print(f"items: {len(items)}")
    print(f"already complete: {len(items) - len(pending)}")
    print(f"pending: {len(pending)}")
    print(f"model: {args.model}")
    print(f"endpoint: {args.base_url.rstrip('/')}/chat/completions")
    print(f"output: {output_path}")

    if not pending:
        _write_jsonl(output_path, [r for r in results if r is not None])
        print("nothing to do")
        return 0

    started = time.monotonic()

    def do_one(idx_item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, str]]:
        i, item = idx_item
        try:
            raw = _chat_completion(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                messages=_build_messages(item),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            code = _extract_code(raw)
        except Exception as e:  # noqa: BLE001
            code = f"# generation failed: {e}\n"
        return i, {"instruction": item["instruction"], "predict": code}

    done = len(items) - len(pending)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(do_one, idx_item) for idx_item in pending]
        for future in as_completed(futures):
            i, rec = future.result()
            results[i] = rec
            done += 1
            _write_jsonl(output_path, [r for r in results if r is not None])
            elapsed = time.monotonic() - started
            print(f"{done}/{len(items)} complete ({elapsed / 60:.1f} min elapsed)")

    _write_jsonl(output_path, [r for r in results if r is not None])
    failures = sum(
        1
        for r in results
        if r is not None and r["predict"].startswith("# generation failed:")
    )
    print(f"wrote {output_path}")
    if failures:
        print(f"warning: {failures} rows contain generation failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
