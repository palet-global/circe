"""
GRPO-style PPO reward tuning for DeepSeek-R1-Distill-Qwen-1.5B
(stack: transformers 4.38.2 · trl 0.7.2 · peft 0.9.0).
"""

import argparse, os, random, re, unicodedata, json, torch
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GenerationConfig,                
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.trainer import ppo_trainer
from peft import PeftModel

# ---------------------------------------------------------------------------
# --------------------------- helper functions ------------------------------
# ---------------------------------------------------------------------------

def _norm(t: str) -> str:
    """Lower-cased, accent-stripped, whitespace-collapsed copy of text t."""
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", t).lower().strip()


def build_loader(arrow, steps, seed=42):
    """
    Return a DataLoader that yields *steps* dummy items,
    each turned into a dict {'prompt': ..., 'answer': ...}.
    """
    dset, rng = load_from_disk(arrow), random.Random(seed)

    def sample(_):
        row = rng.choice(dset)
        prompt = row["text"].rsplit("<|assistant|>", 1)[0] + "<|assistant|>"
        return {"prompt": prompt, "answer": row["answer_norm"]}

    return DataLoader([{}] * steps, batch_size=1, collate_fn=sample)

# ---------------------------------------------------------------------------
# ---------------------------------- main -----------------------------------
# ---------------------------------------------------------------------------
def main(cfg):
    # ---------- tokenizer ---------------------------------------------------
    tok = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=True)
    tok.padding_side = "right" # Right-pad ⇒ same as SFT.

    # ---------- base LM + value head ---------------------------------------
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.base, torch_dtype="auto" # Auto-select fp32/bf16.
    )

    # ---------- attach previously-trained LoRA adapters --------------------
    policy = PeftModel.from_pretrained(
        policy,                     # Base LM w/ value head.
        cfg.init_ckpt,              # Directory produced by SFT phase.
        is_trainable=True,          # Keep adapters trainable for PPO.
        _rename_model=False,        # Ensure param names stay compatible.
    )

    # Provide a GenerationConfig so PEFT’s .generate() inherits defaults ----
    policy.generation_config = GenerationConfig.from_pretrained(cfg.base)

    # Tell TRL that this *specific* subclass is allowed ---------------------
    ppo_trainer.SUPPORTED_ARCHITECTURES += (policy.__class__,)

    # ---------- PPO trainer -------------------------------------------------
    trainer = PPOTrainer(
        model=policy,
        tokenizer=tok,
        config=PPOConfig(
            batch_size=1,           # Single sample per PPO update.
            mini_batch_size=1,      # Same size for minibatch (no splitting).
            ppo_epochs=1,           # One optimisation pass per step.
        ),
    )
    device = trainer.accelerator.device # Shorthand for CUDA/CPU.

    # ---------- data stream -------------------------------------------------
    loader = build_loader(cfg.data_dir, cfg.num_steps) # Deterministic sampler.
    os.makedirs(cfg.output_dir, exist_ok=True) # Ensure save dir.

    gen_kwargs = dict(max_new_tokens=128, top_p=0.95, do_sample=True) # Sampling.

    # ---------- RL loop -----------------------------------------------------
    for step, batch in enumerate(loader, 1):
        prompt, gold = batch["prompt"], batch["answer"]

        # ── tokenise prompt (→ 1-D tensor) ----------------------------------
        ids = tok(prompt, return_tensors="pt").input_ids.to(device).squeeze(0)

        # ── generate continuation -------------------------------------------
        with torch.no_grad():
            out = policy.generate(
                input_ids=ids.unsqueeze(0),        # Add batch dim (1, seq).
                max_new_tokens=128,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
            )
        
        reply_ids = out[0, ids.size(0):]           # Slice off the prompt, still 1-D
        response  = tok.decode(reply_ids.tolist(), skip_special_tokens=True)

        # ── reward: 1.0 if *normalised* gold substring in response else 0.0 -
        reward_t = torch.tensor(
            [float(_norm(gold) in _norm(response))], device=device
        )

        # ── PPO update (expects lists) --------------------------------------
        trainer.step([ids], [reply_ids], [reward_t])

        # ── telemetry / checkpoints ----------------------------------------
        if step % 10 == 0:
            print(f"step {step}/{cfg.num_steps}  reward={reward_t.item():.2f}")
        if step % cfg.save_steps == 0:
            trainer.save_pretrained(Path(cfg.output_dir) / f"checkpoint-{step}")
        if step >= cfg.num_steps:
            break

    # ---------- final save --------------------------------------------------
    trainer.save_pretrained(cfg.output_dir)
    with open(Path(cfg.output_dir) / "meta.json", "w") as f:
        # Persist meta/run info.
        json.dump({"steps": cfg.num_steps}, f)
    print("✓ RL tuning finished")

# ---------------------------------------------------------------------------
# CLI entry-point: only runs when executed as a script, not imported.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   required=True)           # Arrow dataset dir.
    ap.add_argument("--output_dir", required=True)           # Where to store ckpts.
    ap.add_argument("--init_ckpt",  required=True)           # LoRA SFT checkpoint.
    ap.add_argument("--base",
                    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--num_steps",  type=int, default=3000)  # PPO iterations.
    ap.add_argument("--save_steps", type=int, default=500)   # Ckpt cadence.
    main(ap.parse_args())                                   # Kick off main()