#!/usr/bin/env python3
"""
01_peek_inside.py - open up the tiny language model and see how it works.

  python labs/01_peek_inside.py
  python labs/01_peek_inside.py --prompt "The capital of France is"

Shows: model size and architecture, how text becomes tokens, the exact text the
chat template builds, the probability of every candidate next token, and what
temperature does to sampling.
"""
import argparse
import os

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 64 - len(title)))


p = argparse.ArgumentParser()
p.add_argument("--model", default=os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"))
p.add_argument("--prompt", default="The best thing about flying is")
args = p.parse_args()

model, tokenizer, config = load(args.model, return_config=True)

section("1. Size and shape")
n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
dtype = model.layers[0].self_attn.q_proj.weight.dtype
print(f"parameters      : {n_params / 1e6:.1f} M  ({dtype}, ~{n_params * 2 / 1e6:.0f} MB in 16-bit)")
print(f"layers (blocks) : {config['num_hidden_layers']}")
print(f"hidden size     : {config['hidden_size']}   (each token is a vector this long)")
print(f"attention heads : {config['num_attention_heads']}  (key/value heads: {config.get('num_key_value_heads')})")
print(f"MLP size        : {config['intermediate_size']}")
print(f"vocabulary      : {config['vocab_size']} tokens")
print("one block       :", [k for k, _ in model.layers[0].named_modules() if k.count(".") == 1])

section("2. Tokenizer: text -> token ids")
ids = tokenizer.encode(args.prompt)
print(f"text   : {args.prompt!r}")
print(f"ids    : {ids}")
print(f"pieces : {[tokenizer.decode([i]) for i in ids]}")
print(f"{len(args.prompt)} characters -> {len(ids)} tokens")

section("3. Chat template: what the model actually reads")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi!"},
]
print(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
print("(fine-tuning data in 'chat' format is wrapped in exactly these markers)")

section("4. Next-token probabilities")
logits = model(mx.array(ids)[None])            # shape: (batch=1, tokens, vocab)
probs = mx.softmax(logits[0, -1].astype(mx.float32))
top = mx.argsort(-probs)[:10].tolist()
print(f"after {args.prompt!r} the model predicts:")
for i in top:
    p_i = probs[i].item()
    print(f"  {p_i * 100:5.1f}%  {'#' * int(p_i * 50):<50} {tokenizer.decode([i])!r}")

section("5. Temperature")
chat_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Describe the sky from a plane window in one sentence."}],
    add_generation_prompt=True,
)
for temp in [0.0, 0.7, 1.5]:
    mx.random.seed(42)
    out = generate(model, tokenizer, chat_prompt, max_tokens=40, sampler=make_sampler(temp=temp))
    print(f"\ntemp={temp}: {out.strip()}")
print("\n0.0 = always the top token (repeatable); higher = more random/creative.")
