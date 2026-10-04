#!/usr/bin/env bash
# setup.sh - run ONCE while you have internet. Safe to re-run: finished steps are skipped.
#
#   ./setup.sh
#
# Installs mlx-lm into ~/.venvs/mlx (override with MLX_VENV=/path), downloads the
# model, the WikiSQL data and an offline documentation library, then runs
# ./preflight.sh to prove everything works with networking blocked.
set -euo pipefail
cd "$(dirname "$0")"

VENV="${MLX_VENV:-$HOME/.venvs/mlx}"
MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
MLX_LM_VERSION="0.32.0"          # the version this repo was tested with
OFF="docs/offline"
PY="$VENV/bin/python"

step() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }
dl()   { [ -s "$2" ] && echo "   have $2" || { echo "   get  $2"; curl -fsSL "$1" -o "$2"; }; }

step "1/7 Checking for an Apple silicon Mac"
[ "$(uname -m)" = arm64 ] || { echo "MLX needs Apple silicon (M1 or newer)."; exit 1; }
echo "   $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GB RAM, macOS $(sw_vers -productVersion)"

step "2/7 Python environment at $VENV"
if [ -x "$PY" ]; then
  echo "   already exists ($("$PY" --version))"
elif command -v uv >/dev/null; then
  uv venv --seed --python 3.12 "$VENV"
else
  PYBIN=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3.10 || true)
  [ -n "$PYBIN" ] || { echo "Need Python 3.10+: brew install python@3.12  (or: brew install uv)"; exit 1; }
  "$PYBIN" -m venv "$VENV"
fi

step "3/7 Installing mlx-lm $MLX_LM_VERSION (+ matplotlib for plots, markdown for offline HTML)"
if command -v uv >/dev/null; then
  uv pip install --python "$PY" "mlx-lm==$MLX_LM_VERSION" matplotlib markdown
else
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install "mlx-lm==$MLX_LM_VERSION" matplotlib markdown
fi

step "4/7 Downloading the model ($MODEL, ~270 MB) and testing it"
"$VENV/bin/mlx_lm.generate" --model "$MODEL" --prompt "Say hello in five words." --max-tokens 20

step "5/7 Datasets"
[ -s data/flight/train.jsonl ] && echo "   have data/flight" || "$PY" labs/make_flight_data.py
mkdir -p data/wikisql
for f in train valid test; do
  dl "https://raw.githubusercontent.com/ml-explore/mlx-examples/main/lora/data/$f.jsonl" "data/wikisql/$f.jsonl"
done

step "6/7 Offline documentation -> $OFF"
mkdir -p "$OFF/mlx-lm/examples" "$OFF/cli-help" "$OFF/papers"
# mlx-lm docs shipped inside the installed package (they match your exact version)
PKG=$("$PY" -c "import mlx_lm, os; print(os.path.dirname(mlx_lm.__file__))")
cp "$PKG"/*.md "$OFF/mlx-lm/"
R="https://raw.githubusercontent.com/ml-explore/mlx-lm/v$MLX_LM_VERSION"
curl -fsSL "$R/README.md" -o "$OFF/mlx-lm/README.md"     # the full README (the packaged one is a stub)
for f in generate_response.py chat.py batch_generate_response.py tool_use.py lora_config.yaml merge_config.yaml; do
  dl "$R/examples/$f" "$OFF/mlx-lm/examples/$f"
done
dl https://raw.githubusercontent.com/ml-explore/mlx-examples/main/lora/README.md "$OFF/mlx-examples-lora-README.md"
# The MLX documentation website (~58 MB): sparse checkout of the built HTML on the gh-pages branch
if [ -s "$OFF/mlx/index.html" ]; then
  echo "   have $OFF/mlx"
else
  echo "   get  $OFF/mlx (MLX docs website, ~58 MB)"
  tmp=$(mktemp -d)
  git clone -q --depth 1 --branch gh-pages --filter=blob:none --sparse https://github.com/ml-explore/mlx.git "$tmp/mlx"
  git -C "$tmp/mlx" sparse-checkout set docs/build/html
  rm -rf "$OFF/mlx" && cp -R "$tmp/mlx/docs/build/html" "$OFF/mlx" && rm -rf "$tmp"
fi
dl https://arxiv.org/pdf/2106.09685 "$OFF/papers/LoRA-2106.09685.pdf"
dl https://arxiv.org/pdf/2305.14314 "$OFF/papers/QLoRA-2305.14314.pdf"
dl https://arxiv.org/pdf/2502.02737 "$OFF/papers/SmolLM2-2502.02737.pdf"
"$PY" - <<EOF
import shutil
from huggingface_hub import hf_hub_download
shutil.copy(hf_hub_download("$MODEL", "README.md"), "$OFF/SmolLM2-135M-Instruct-model-card.md")
EOF
for c in lora generate chat convert fuse server benchmark manage perplexity cache_prompt; do
  "$VENV/bin/mlx_lm.$c" --help > "$OFF/cli-help/mlx_lm.$c.txt" 2>&1 || true
done
"$PY" scripts/render_docs.py

step "7/7 Offline self-test"
./preflight.sh
