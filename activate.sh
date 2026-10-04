# activate.sh - run this in every new Terminal window, from the repo folder:
#
#   source activate.sh
#
# It switches on the Python environment and forces every tool to work offline.

VENV="${MLX_VENV:-$HOME/.venvs/mlx}"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "No Python environment at $VENV. Run ./setup.sh first (needs internet)."
  return 1 2>/dev/null || exit 1
fi
if [ ! -f labs/eval_flight.py ]; then
  echo "Run this from the repo folder (the one containing README.md):  cd <repo> && source activate.sh"
  return 1 2>/dev/null || exit 1
fi

source "$VENV/bin/activate"

export HF_HUB_OFFLINE=1            # never try to reach huggingface.co; use the local cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
mkdir -p runs

echo "mlx-flight-lab ready (offline mode)"
echo "  python : $(command -v python)"
echo "  model  : \$MODEL = $MODEL"
echo "  next   : ./preflight.sh   then open README.md (or README.html) at Lab 0"
