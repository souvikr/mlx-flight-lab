#!/usr/bin/env python3
"""
lora_from_scratch.py - LoRA in plain MLX, with every moving part visible.

Same flight task as Lab 2, but without mlx_lm.lora doing the work:
  1. freeze the base model (134.5M weights that never change)
  2. wrap some Linear layers:   y = W x  +  scale * (x A) B
         A: (in x r) random,  B: (r x out) zeros  ->  starts as an exact no-op
  3. train ONLY A and B with cross-entropy on the answer tokens
  4. generate with the adapted model
  5. merge:  W' = W + scale * (A B)^T  and check it gives the same output

  python labs/lora_from_scratch.py
  python labs/lora_from_scratch.py --rank 1 --iters 150      # how small can r go?
  python labs/lora_from_scratch.py --targets q_proj           # adapt fewer matrices
"""
import argparse
import json
import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import generate, load


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, scale: float):
        super().__init__()
        self.base = base
        out_dims, in_dims = base.weight.shape
        self.scale = scale
        bound = 1 / math.sqrt(in_dims)
        self.lora_a = mx.random.uniform(-bound, bound, (in_dims, rank))
        self.lora_b = mx.zeros((rank, out_dims))  # zero => no change at step 0

    def __call__(self, x):
        update = (x @ self.lora_a) @ self.lora_b
        return self.base(x) + (self.scale * update).astype(x.dtype)


def add_lora(model, num_layers, targets, rank, scale):
    model.freeze()  # nothing in the base model gets gradients
    for layer in model.layers[-num_layers:]:
        for name in targets:
            parent = layer.self_attn if hasattr(layer.self_attn, name) else layer.mlp
            setattr(parent, name, LoRALinear(getattr(parent, name), rank, scale))


def encode(tokenizer, messages):
    """Token ids for the whole chat, plus how many of them are prompt."""
    full = tokenizer.apply_chat_template(messages)
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
    return full, len(prompt)


def make_batch(examples):
    """Pad to equal length. mask=1 only where the *target* token is part of the answer."""
    L = max(len(ids) for ids, _ in examples)
    tokens = np.zeros((len(examples), L), dtype=np.int32)
    mask = np.zeros((len(examples), L - 1), dtype=np.float32)
    for i, (ids, n_prompt) in enumerate(examples):
        tokens[i, : len(ids)] = ids
        mask[i, n_prompt - 1 : len(ids) - 1] = 1.0
    tokens = mx.array(tokens)
    return tokens[:, :-1], tokens[:, 1:], mx.array(mask)  # inputs, targets, mask


def loss_fn(model, inputs, targets, mask):
    logits = model(inputs).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, targets) * mask
    return ce.sum() / mask.sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"))
    p.add_argument("--data", default="data/flight")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--scale", type=float, default=20.0, help="same default as mlx_lm.lora")
    p.add_argument("--num-layers", type=int, default=16, help="adapt the last N blocks (of 30)")
    p.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="runs/scratch-lora.safetensors")
    args = p.parse_args()

    mx.random.seed(args.seed)
    random.seed(args.seed)
    model, tokenizer = load(args.model)
    total = sum(v.size for _, v in tree_flatten(model.parameters()))

    add_lora(model, args.num_layers, args.targets.split(","), args.rank, args.scale)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"base params {total / 1e6:.1f}M (frozen) | LoRA params {trainable / 1e6:.3f}M "
          f"= {100 * trainable / total:.2f}% | rank {args.rank} | last {args.num_layers} layers")

    read = lambda split: [encode(tokenizer, json.loads(l)["messages"]) for l in open(f"{args.data}/{split}.jsonl")]
    train, valid = read("train"), read("valid")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.Adam(learning_rate=args.lr)

    def val_loss():
        losses = [loss_fn(model, *make_batch(valid[i : i + 8])).item() for i in range(0, len(valid), 8)]
        return sum(losses) / len(losses)

    print(f"step    0 | val loss {val_loss():.3f}  (before training)")
    t0, order = time.time(), []
    for step in range(1, args.iters + 1):
        if len(order) < args.batch_size:
            order = random.sample(range(len(train)), len(train))  # new epoch
        batch = [train[order.pop()] for _ in range(args.batch_size)]
        loss, grads = loss_and_grad(model, *make_batch(batch))
        opt.update(model, grads)
        mx.eval(loss, model.trainable_parameters(), opt.state)
        if step % 20 == 0:
            msg = f"step {step:4d} | train loss {loss.item():.3f}"
            if step % 100 == 0:
                msg += f" | val loss {val_loss():.3f}"
            print(f"{msg} | {step / (time.time() - t0):.1f} steps/s")

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    mx.save_safetensors(args.save, dict(tree_flatten(model.trainable_parameters())))
    print(f"\nsaved LoRA weights -> {args.save} ({os.path.getsize(args.save) / 1e6:.1f} MB)")

    print("\n--- Try it on 3 test requests (cities never seen in training) ---")
    for line in open(f"{args.data}/test.jsonl").readlines()[:3]:
        msgs = json.loads(line)["messages"]
        prompt = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True)
        print(f"\nuser : {msgs[1]['content']}\nmodel: {generate(model, tokenizer, prompt, max_tokens=64).strip()}")

    print("\n--- Merge check: LoRA is just a low-rank change to W ---")
    target = args.targets.split(",")[0]
    layer = model.layers[-1]
    lora = getattr(layer.self_attn if hasattr(layer.self_attn, target) else layer.mlp, target)
    W = lora.base.weight.astype(mx.float32)                      # (out, in)
    delta = lora.scale * (lora.lora_a @ lora.lora_b).T           # (out, in)
    x = mx.random.normal((3, W.shape[1]))
    y_lora = x @ W.T + lora.scale * (x @ lora.lora_a) @ lora.lora_b
    y_merged = x @ (W + delta).T
    print(f"{target} in the last block: W is {W.shape[0]}x{W.shape[1]}, "
          f"delta has rank {np.linalg.matrix_rank(np.array(delta))}")
    print(f"max |y_lora - y_merged| = {mx.abs(y_lora - y_merged).max().item():.2e}  (~0 => identical)")
    print(f"|delta| / |W| = {(mx.linalg.norm(delta) / mx.linalg.norm(W)).item():.4f}  (a small nudge)")


if __name__ == "__main__":
    main()
