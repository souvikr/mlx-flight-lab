#!/usr/bin/env python3
"""
eval_flight.py - score a model on the flight-request -> JSON task.

  python labs/eval_flight.py                                  # base model (before fine-tuning)
  python labs/eval_flight.py --adapter-path runs/flight-lora  # with your LoRA adapter
  python labs/eval_flight.py --adapter-path runs/flight-lora --data data/flight/hard.jsonl --show 20

Reports:
  valid JSON  - did the output parse as JSON at all?
  exact match - all 5 fields correct
  per-field   - accuracy of from / to / date / passengers / cabin separately

Decoding is greedy (always the most likely next token), so the score is repeatable.
"""
import argparse
import json
import os
import re
import time

from mlx_lm import batch_generate, load

FIELDS = ["from", "to", "date", "passengers", "cabin"]
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def parse(text):
    """Return the first JSON object in `text`, or None."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    p.add_argument("--adapter-path", default=None, help="folder with adapters.safetensors (from mlx_lm.lora)")
    p.add_argument("--data", default="data/flight/test.jsonl")
    p.add_argument("--limit", type=int, default=None, help="only score the first N examples")
    p.add_argument("--show", type=int, default=5, help="print this many examples (mistakes first)")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    rows = [json.loads(line) for line in open(args.data)][: args.limit]

    # Everything except the final assistant message is the prompt.
    prompts = [
        tokenizer.apply_chat_template(r["messages"][:-1], add_generation_prompt=True)
        for r in rows
    ]
    t0 = time.time()
    outputs = batch_generate(model, tokenizer, prompts, max_tokens=args.max_tokens).texts
    secs = time.time() - t0

    n = len(rows)
    valid = exact = 0
    field_ok = {f: 0 for f in FIELDS}
    results = []
    for r, out in zip(rows, outputs):
        want = json.loads(r["messages"][-1]["content"])
        got = parse(out)
        ok = {f: isinstance(got, dict) and got.get(f) == want[f] for f in FIELDS}
        valid += isinstance(got, dict)
        exact += all(ok.values())
        for f in FIELDS:
            field_ok[f] += ok[f]
        results.append((all(ok.values()), r["messages"][-2]["content"], want, out.strip(), ok))

    name = "base model" if args.model == DEFAULT_MODEL else args.model
    label = f"{name} + adapter={args.adapter_path}" if args.adapter_path else f"{name} (no adapter)"
    print(f"\n{label} | data={args.data} | {n} examples | {secs:.1f}s")
    print(f"  valid JSON : {100 * valid / n:5.1f}%")
    print(f"  exact match: {100 * exact / n:5.1f}%")
    print("  per field  : " + "  ".join(f"{f} {100 * field_ok[f] / n:.0f}%" for f in FIELDS))

    if args.show:
        print(f"\nExamples (mistakes first):")
        for good, user, want, out, ok in sorted(results, key=lambda x: x[0])[: args.show]:
            mark = "OK " if good else "BAD"
            wrong = [f for f in FIELDS if not ok[f]]
            print(f"\n[{mark}] {user}")
            print(f"   model : {out[:200]}")
            if not good:
                print(f"   wanted: {json.dumps(want)}")
                print(f"   wrong : {', '.join(wrong)}")


if __name__ == "__main__":
    main()
