"""
Pre-render Easy-R1 jinja format prompts into a verl-compatible parquet.

Easy-R1 wraps each prompt with `data.format_prompt: math_wo_format.jinja` at
load time, then feeds `[{"role": "user", "content": <rendered_jinja>}]` to
`tokenizer.apply_chat_template`. verl's RLHFDataset expects the same chat-list
shape in the parquet's prompt column but has no jinja layer, so we render the
template once here and write a new parquet alongside the original.

Usage:
    python preprocess_easyR1_parquet.py \
        --input  /path/to/train.parquet \
        --output /path/to/train.formatted.parquet \
        --template examples/branch_adv/format_prompt/math_wo_format.jinja

For the validation set, point --template at val_wo_format.jinja. Both train
and val parquets keep all other columns intact (reward_model, extra_info,
answer, ...), so the existing reward function (compute_score_batch) and
filter_groups path see identical data to the Easy-R1 run.
"""

import argparse
import os

import jinja2
import pandas as pd


def _coerce_prompt_text(value) -> str:
    """Easy-R1 only used the prompt's text. Accept both raw strings and the
    chat-list shape (when the source parquet already wrapped it)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    if hasattr(value, "tolist"):
        return _coerce_prompt_text(value.tolist())
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source parquet (Easy-R1 layout).")
    parser.add_argument("--output", required=True, help="Destination parquet (verl-compatible).")
    parser.add_argument(
        "--template",
        required=True,
        help="Path to the Easy-R1 jinja template (e.g. math_wo_format.jinja).",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Column with the raw problem text. Defaults to 'prompt'.",
    )
    parser.add_argument(
        "--answer-key",
        default=None,
        help=(
            "If the source parquet does not have a 'reward_model' column "
            "(e.g. Easy-R1 val_merged.parquet only has 'answer'), specify the "
            "column to copy into reward_model['ground_truth'] so verl's reward "
            "managers can find it. Skipped when --answer-key is omitted."
        ),
    )
    parser.add_argument(
        "--data-source",
        default=None,
        help=(
            "If neither 'data_source' nor 'reward_fn_key' is present, fill a "
            "fixed string here (verl's reward managers use this for logging "
            "buckets). Skipped when omitted."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=[None, "train", "val"],
        help=(
            "If set, inject extra_info['split'] = <value> for every row. "
            "compute_score_batch reads this to mirror Easy-R1's split-specific "
            "reward path: 'val' skips sympy + LLM judge + length_penalty and "
            "matches val_wo_format.py exactly."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    if not os.path.exists(args.template):
        raise FileNotFoundError(args.template)

    with open(args.template, encoding="utf-8") as f:
        template = jinja2.Template(f.read().strip())

    df = pd.read_parquet(args.input)
    if args.prompt_key not in df.columns:
        raise KeyError(f"--prompt-key {args.prompt_key!r} not in columns: {list(df.columns)}")

    rendered = df[args.prompt_key].map(
        lambda v: [{"role": "user", "content": template.render(content=_coerce_prompt_text(v))}]
    )
    df[args.prompt_key] = rendered

    if args.answer_key:
        if args.answer_key not in df.columns:
            raise KeyError(
                f"--answer-key {args.answer_key!r} not in columns: {list(df.columns)}"
            )
        df["reward_model"] = df[args.answer_key].map(
            lambda gt: {"ground_truth": str(gt) if gt is not None else "", "style": "rule"}
        )

    if args.data_source and "data_source" not in df.columns:
        df["data_source"] = args.data_source

    if args.split:
        if "extra_info" in df.columns:
            def _merge(existing):
                if isinstance(existing, dict):
                    base = dict(existing)
                elif hasattr(existing, "tolist"):
                    base = dict(existing.tolist()) if isinstance(existing.tolist(), dict) else {}
                else:
                    base = {}
                base["split"] = args.split
                return base
            df["extra_info"] = df["extra_info"].map(_merge)
        else:
            df["extra_info"] = [{"split": args.split} for _ in range(len(df))]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[ok] {args.input} -> {args.output}  ({len(df)} rows)")
    sample_msgs = df.iloc[0][args.prompt_key]
    print("[sample.prompt[0].content][first 400 chars]:")
    print(str(sample_msgs[0]["content"])[:400])


if __name__ == "__main__":
    main()
