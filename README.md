![Logo](circe.jpg)

# Circe

[ENTER DESCRIPTION HERE]

## Installation

1. Clone and download this repository

```bash
git clone https://github.com/palet-global/circe
cd circe
```

2. Set up your Python enviroment

```bash
# create virtual enviroment
python -m venv venv

# enter virtual enviroment
source venv/bin/activate
```

3. In the top-level directory run

```bash
# Install the project and its dependencies
pip install .
```

## Training Your Own Circe with our Pipeline

1. Fetch the data

```bash
python data/fetch_datasets.py --out data/processed
```

2. Supervised LoRA training

```bash
# configure Accelerate
accelerate config default

# start training
accelerate launch train/sft.py \
	--data_dir data/processed \
	--output_dir checkpoints/sft

# (optional)
# if training stops, you can resume it using the following command
accelerate launch train/sft_train.py \
	--data_dir data/processed \
	--output_dir checkpoints/sft \
	--resume_from_checkpoint checkpoints/sft/checkpoint-4000
```

3. RL using GRPO

```bash
accelerate launch train/rl_grpo.py \
    --data_dir data/processed \
    --output_dir checkpoints/grpo \
    --init_ckpt checkpoints/sft/checkpoint-13000 \
    --num_steps 3000 \
    --save_steps 500 \
    --group 4
```

4. Merge

```bash
# lets merge the weights
python train/merge_lora.py \
  --ckpt_dir checkpoints/grpo \
  --base deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# lets copy the tokenizers into merge folder
python - <<'PY'
import shutil, transformers, os
repo = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dest = "merged"
needed = [
    "tokenizer.json",
    "tokenizer_config.json",
]
for f in needed:
    try:
        src = transformers.utils.hub.cached_file(repo, f)
        shutil.copy2(src, dest)
        print("✓", f)
    except FileNotFoundError:
        print("•", f, "(not in repo, skipped)")
PY
```

5. Run Eval

```bash
# lets run the first eval
python eval/quick_squad_eval.py --model ./merged --dataset squad

# lets run the second eval
python eval/quick_squad_eval.py --model ./merged --dataset squad_es
```

6. Upload to Hugging Face

```bash
python train/upload_to_hub.py \
  --model_dir merged \
  --repo stevenr/deepseek-r1-grpo \
  --token $HF_TOKEN
```

## Troubleshooting

### Supervised LoRA training

Remove file if having problems when resuming

```bash
rm checkpoints/sft/checkpoint-8000/rng_state.pth
```

When Installing accelerate again for any reason

```bash
# Remove the stale file
rm -f ~/.cache/huggingface/accelerate/default_config.yaml

# Regenerate the config
accelerate config default
```

If you are having a problem with the version in this repo

```bash
# Try a lower version
transformers==4.38.2
```
