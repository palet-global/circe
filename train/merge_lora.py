"""
Merge a PEFT/LoRA checkpoint into its base model and store the
fully-baked weights under ./merged/.
"""

import argparse, os
from transformers import AutoModelForCausalLM
from peft import PeftModel

def main(cfg):
    # 1. Load base model
    print("Loading base:", cfg.base)
    base = AutoModelForCausalLM.from_pretrained(cfg.base, torch_dtype="auto")

    # 2. Attach adapters and merge
    print("Loading LoRA adapters from", cfg.ckpt_dir)
    model = PeftModel.from_pretrained(base, cfg.ckpt_dir)
    merged = model.merge_and_unload()          # returns plain nn.Module

    # 3. Save result
    out_dir = os.path.join(cfg.out_dir, "merged")
    merged.save_pretrained(out_dir, safe_serialization=True)
    print("✓ Merged model written to", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--ckpt_dir", required=True,
                    help="Directory with LoRA/PEFT checkpoint (e.g. checkpoints/grpo)")
    ap.add_argument("--out_dir", default=".",
                    help="Parent folder where ./merged will be created")
    main(ap.parse_args())
