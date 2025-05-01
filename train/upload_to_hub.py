"""
Upload a local model directory to the Hugging Face Hub.
"""

import argparse, sys
from transformers import AutoModelForCausalLM

def main(cfg):
    print("Packing model from", cfg.model_dir)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir)

    print(f"Pushing to https://huggingface.co/{cfg.repo}")
    model.push_to_hub(cfg.repo,
                      use_temp_dir=False,   # upload in-place
                      token=cfg.token)
    print("✓ Upload complete ✨")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="Folder that contains config.json + model weights")
    ap.add_argument("--repo",      required=True,
                    help="Hub repo name, e.g. myorg/deepseek-r1-grpo")
    ap.add_argument("--token",     required=True,
                    help="HF access token with write permission")
    main(ap.parse_args())
