# Cheat sheet

Every command from the [README](../README.md), on one page. Run them from the repo folder after `source activate.sh`, which sets `$MODEL` = `HuggingFaceTB/SmolLM2-135M-Instruct`.

## Every session

```bash
cd mlx-flight-lab
source activate.sh        # venv + offline mode + $MODEL
./preflight.sh            # everything OK without internet?
open README.html LIBRARY.html
```

## Run the model

```bash
mlx_lm.generate --model $MODEL --prompt "Hello" --max-tokens 60
mlx_lm.generate --model $MODEL --prompt "Hello" --temp 0.8 --top-p 0.9 --seed 1
mlx_lm.generate --model $MODEL --prompt "Hi" --system-prompt "You are a pirate."
mlx_lm.generate --model $MODEL --prompt "Once upon a time" --ignore-chat-template
mlx_lm.chat     --model $MODEL                       # q = quit, r = reset, h = help
python labs/01_peek_inside.py --prompt "The pilot said"
```

## Train (LoRA)

```bash
mlx_lm.lora --model $MODEL --train --data data/flight --mask-prompt \
  --iters 300 --learning-rate 1e-4 --steps-per-eval 50 \
  --adapter-path runs/flight-lora 2>&1 | tee runs/flight-lora.log

mlx_lm.lora -c configs/flight_lora.yaml                     # same, from a config file
mlx_lm.lora -c configs/flight_lora.yaml --iters 100 --adapter-path runs/x   # flags override
```

Useful flags: `--batch-size 4` · `--num-layers 16` (or -1 = all) · `--fine-tune-type lora|dora|full` · `--optimizer adam|adamw|sgd|adafactor|muon` · `--grad-checkpoint` · `--grad-accumulation-steps N` · `--max-seq-length 512` · `--save-every 100` · `--resume-adapter-file runs/x/adapters.safetensors` · `--seed 0`. Rank, scale, layer keys and LR schedules are config-file only (see `configs/flight_lora.yaml`).

## Evaluate

```bash
python labs/eval_flight.py                                        # base model
python labs/eval_flight.py --adapter-path runs/flight-lora        # fine-tuned
python labs/eval_flight.py --adapter-path runs/flight-lora --data data/flight/hard.jsonl --show 20
mlx_lm.lora --model $MODEL --adapter-path runs/flight-lora --data data/flight --mask-prompt --test   # loss + perplexity
python labs/plot_losses.py runs/flight-lora.log && open runs/flight-lora.png
python labs/plot_losses.py runs/a.log runs/b.log -o runs/compare.png
```

## Use the adapter

```bash
mlx_lm.generate --model $MODEL --adapter-path runs/flight-lora \
  --system-prompt "Convert the flight request into JSON with keys from, to, date, passengers, cabin." \
  --prompt "2 business seats from Rome to Oslo on the 4th of May"
mlx_lm.chat --model $MODEL --adapter-path runs/flight-lora
```

## Data

```bash
python labs/make_flight_data.py                              # rebuild data/flight
python labs/make_flight_data.py --train 50 --out data/flight-50
python labs/make_flight_data.py --seed 1 --out data/flight-seed1
head -n 1 data/flight/train.jsonl | python -m json.tool
```

## WikiSQL (text format)

```bash
mlx_lm.lora --model $MODEL --train --test --data data/wikisql --iters 300 --adapter-path runs/sql
mlx_lm.generate --model $MODEL --adapter-path runs/sql --ignore-chat-template --max-tokens 40 \
  --prompt $'table: 1-10015132-16\ncolumns: Player, No., Nationality, Position, Years in Toronto, School/Club Team\nQ: What is terrence ross\' nationality\nA: '
```

## Quantize, QLoRA, fuse, serve

```bash
mlx_lm.convert --model $MODEL -q --mlx-path runs/smollm-4bit                 # 4-bit
mlx_lm.lora --model runs/smollm-4bit --train --data data/flight --mask-prompt --iters 300 \
  --learning-rate 1e-4 --adapter-path runs/qlora                            # QLoRA
mlx_lm.fuse --model $MODEL --adapter-path runs/flight-lora --save-path runs/flight-fused
mlx_lm.server --model $MODEL --adapter-path runs/flight-lora --port 8080     # Ctrl-C to stop
curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":40}'
```

## LoRA from scratch

```bash
python labs/lora_from_scratch.py
python labs/lora_from_scratch.py --rank 1 --targets q_proj,v_proj --iters 150
```

## Housekeeping

```bash
mlx_lm.manage --scan --pattern SmolLM2      # cached models
du -sh runs/*                                # what your experiments use on disk
mlx_lm.lora --help                           # or: less docs/offline/cli-help/mlx_lm.lora.txt
```
