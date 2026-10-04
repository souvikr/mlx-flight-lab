#!/usr/bin/env bash
# preflight.sh - check that everything for the flight works WITHOUT internet.
#
#   ./preflight.sh
#
# The model test runs inside a macOS sandbox with networking blocked, so a pass
# here means it will also work at 35,000 ft.

cd "$(dirname "$0")"
VENV="${MLX_VENV:-$HOME/.venvs/mlx}"
PY="$VENV/bin/python"
MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
NO_NET='(version 1)(allow default)(deny network-outbound (remote ip))'
pass=0; fail=0
ok()   { printf "  \033[32m[ OK ]\033[0m %s\n" "$*"; pass=$((pass + 1)); }
bad()  { printf "  \033[31m[FAIL]\033[0m %s\n" "$*"; fail=$((fail + 1)); }
note() { printf "  \033[33m[NOTE]\033[0m %s\n" "$*"; }

echo "Preflight check (offline)"

[ "$(uname -m)" = arm64 ] && ok "Apple silicon ($(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GB RAM)" \
  || bad "not Apple silicon - MLX will not run"

if [ -x "$PY" ]; then
  ok "Python env: $VENV ($("$PY" --version 2>&1))"
  v=$("$PY" -c "import mlx.core as mx, mlx_lm; print(f'mlx {mx.__version__}, mlx-lm {mlx_lm.__version__}')" 2>/dev/null) \
    && ok "packages: $v" || bad "cannot import mlx / mlx_lm - re-run ./setup.sh while online"
  "$PY" -c "import matplotlib" 2>/dev/null && ok "matplotlib (for loss plots)" || note "matplotlib missing - plots won't work"
else
  bad "no Python env at $VENV - run ./setup.sh while online"
fi

if command -v sandbox-exec >/dev/null; then
  RUN=(sandbox-exec -p "$NO_NET"); how="network blocked"
else
  RUN=(); how="HF_HUB_OFFLINE=1"
fi
out=$("${RUN[@]}" "$VENV/bin/mlx_lm.generate" --model "$MODEL" --prompt "Say hi" --max-tokens 8 2>&1)
if echo "$out" | grep -q "Generation:"; then
  ok "model loads and generates ($how): $(echo "$out" | grep 'Generation:' | sed 's/Generation: //')"
else
  bad "model test failed ($how). Last lines:"; echo "$out" | tail -5 | sed 's/^/         /'
fi

for f in data/flight/{train,valid,test,hard}.jsonl data/wikisql/{train,valid,test}.jsonl; do
  [ -s "$f" ] && ok "$f ($(wc -l < "$f" | tr -d ' ') examples)" || bad "missing $f"
done

for f in docs/offline/mlx/index.html docs/offline/mlx-lm/LORA.md docs/offline/papers/LoRA-2106.09685.pdf README.html; do
  [ -s "$f" ] && ok "doc: $f" || note "missing doc: $f (run ./setup.sh online to fetch docs)"
done

free_gb=$(df -g . | awk 'NR==2 {print $4}')
[ "${free_gb:-0}" -ge 3 ] && ok "disk: ${free_gb} GB free" || bad "disk: only ${free_gb} GB free (want 3+)"

batt=$(pmset -g batt | grep -Eo '[0-9]+%' | head -1)
src=$(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/")
note "battery ${batt:-?} on ${src:-?} - training is a heavy load; plug in if your seat has power"
[ "$(pmset -g | awk '/lowpowermode/ {print $2}')" = 1 ] && note "Low Power Mode is ON: saves battery, but training runs slower"

echo
if [ "$fail" -eq 0 ]; then
  printf "\033[32mAll %d checks passed. You're ready to fly.\033[0m\n" "$pass"
else
  printf "\033[31m%d check(s) failed.\033[0m Fix them while you still have internet (usually: ./setup.sh).\n" "$fail"
  exit 1
fi
