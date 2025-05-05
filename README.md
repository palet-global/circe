# Circe-1.5B

<!-- center-aligned, capped at 420 px wide × 240 px tall -->
<p align="center">
  <img
    src="https://cdn-uploads.huggingface.co/production/uploads/657e1ad01e3e9c41a49b732e/8IsJaxuOwuqBN0GctRUUe.png"
    alt="Circe-1.5B schematic"
    width="420"
    height="240"
  />
</p>





| ⚙️ Spec | Value |
|---------|-------|
| Base model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Trainable params | 4 M (LoRA) |
| Post-training cost | **≈ US $12** on 1×L40S |
| Training recipe | 8 h SFT → 4 h GRPO |
| Context length | up to **4 k tokens** (tested) |
| RAM @ bf16 | ~9 GB (≤ 3 GB 4-bit GPTQ) |
| Throughput | ~55 tok / s on 1×A6000 (fp16, no compile) |

**Circe-1.5B** is a single-checkpoint, 1.5 B-parameter language model that asks a simple question:  

> _“How far can you push tiny models on a tiny budget?”_

It keeps DeepSeek-R1’s strong reasoning depth but adds **fluent bilingual chat** (English & Spanish) in a checkpoint that fits on a laptop GPU.  
We intend to use it as a reproducible waypoint on the road to real-time speech-to-speech reasoning systems.

---

## 🔭 Intended Use

* **Base for new LoRAs** — domain adaptation, longer-context studies.  
* **Research** into cost-efficient RL for reasoning.  
* **Not** for high-stakes or production tasks.

See the [⚙️ Limitations](#️-limitations--bias) section before use.

---

## 💻 Hardware & Inference Tips
- **bf16 / fp16**: Needs ~9 GB VRAM.  
- **4-bit GPTQ**: < 3 GB. `bitsandbytes` works out-of-the-box.  
- Compile once (`torch.compile`) for **+10–15 %** throughput.

---

## ✍️ Current Evaluation Status
Formal **lighteval / MMLU / GSM-8K** runs are queued. Preliminary spot-checks show Circe retains DeepSeek-R1’s chain-of-thought depth on reasoning-heavy QA while adding smooth bilingual generation.

---

## ⚙️ Limitations & Bias
- No reward-model alignment. 
- Long-context (> 4 k) stability untested.  
- Training data bias from public QA pairs. Spanish coverage favors Latin American variants.  
- Minimal safety filters so **you** have to wrap with your own guardrails for production.

---

## 🔮 Roadmap
- Publish full reasoning benchmark suite & eval scripts.  
- Release code-reasoning and doc-QA adapters.  
- Attach a **24 kHz neural codec** → real-time, full-duplex voice chat without ASR → TTS hops.

---

## 🪪 License
This project is licensed under the [MIT](https://opensource.org/licenses/MIT) License. Attribution appreciated but not required.


---

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
  --repo PaletLabs/Circe \
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
