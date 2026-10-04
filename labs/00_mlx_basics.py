#!/usr/bin/env python3
"""
00_mlx_basics.py - a 10-minute tour of MLX itself (no language model yet).

  python labs/00_mlx_basics.py

Then open this file, change things, and re-run. Ideas are at the bottom.
"""
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 64 - len(title)))


section("1. Arrays")
a = mx.array([1.0, 2.0, 3.0])
b = mx.ones((3,))
print("a + b        =", a + b)
print("a.dtype      =", a.dtype, "| a.shape =", a.shape)
print("default dev  =", mx.default_device())

section("2. Lazy evaluation")
# MLX records operations and only computes when a value is needed
# (print, .item(), np.array(...), or an explicit mx.eval).
x = mx.random.normal((2048, 2048))
t = time.perf_counter()
y = x @ x
t_build = time.perf_counter() - t
t = time.perf_counter()
mx.eval(y)
t_eval = time.perf_counter() - t
print(f"y = x @ x  took {t_build * 1000:7.2f} ms  <- only builds a graph")
print(f"mx.eval(y) took {t_eval * 1000:7.2f} ms  <- the actual math")

section("3. Unified memory: pick where ops run, data never moves")
# On Apple silicon the CPU and GPU share the same RAM, so there is no .to(device).
for name, dev in [("cpu", mx.cpu), ("gpu", mx.gpu)]:
    mx.eval(mx.matmul(x, x, stream=dev))  # warm-up
    t = time.perf_counter()
    for _ in range(5):
        mx.eval(mx.matmul(x, x, stream=dev))
    sec = (time.perf_counter() - t) / 5
    print(f"{name}: 2048x2048 matmul {sec * 1000:7.1f} ms  (~{2 * 2048**3 / sec / 1e9:,.0f} GFLOP/s)")

section("4. Automatic differentiation")
def f(w):
    return (w**2).sum()          # f(w) = sum(w^2), so df/dw = 2w

w = mx.array([1.0, -2.0, 3.0])
print("f(w)     =", f(w))
print("grad f(w)=", mx.grad(f)(w), " (should be 2*w)")

section("5. Train a tiny model: learn y = 3x + 2 from noisy points")
mx.random.seed(0)
X = mx.random.uniform(-1, 1, (256, 1))
Y = 3 * X + 2 + 0.1 * mx.random.normal((256, 1))

model = nn.Linear(1, 1)


def loss_fn(model, X, Y):
    return nn.losses.mse_loss(model(X), Y)


loss_and_grad = nn.value_and_grad(model, loss_fn)
opt = optim.SGD(learning_rate=0.1)
for step in range(101):
    loss, grads = loss_and_grad(model, X, Y)   # forward + backward
    opt.update(model, grads)                   # w <- w - lr * grad
    mx.eval(model.parameters(), opt.state)     # actually run it
    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  "
              f"w={model.weight.item():.3f}  b={model.bias.item():.3f}")
print("-> w ~ 3 and b ~ 2. LoRA training (Lab 2) is this exact loop, just bigger.")

section("6. Memory")
print(f"peak memory used by MLX in this script: {mx.get_peak_memory() / 1e6:.0f} MB")

print("""
Try next (edit the file, re-run):
  * learning_rate=1.5 in section 5: the loss explodes. Why?
  * x = mx.random.normal((4096, 4096)) in section 2: how does time scale?
  * mx.random.normal((2048, 2048), dtype=mx.float16) in section 3: is half precision faster?
  * python -i labs/00_mlx_basics.py  -> drops you into a Python prompt with everything loaded
""")
