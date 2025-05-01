"""
GRPO reward tuning • DeepSeek-R1-Distill-Qwen-1.5B
Compatible with: transformers 4.38.2 · trl 0.7.2 · peft 0.9.0
"""

import argparse, os, random, re, unicodedata, json, torch
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GenerationConfig
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead 
from peft import PeftModel
from trl.trainer import ppo_trainer

from types import MethodType

# ---------------------------------------------------------------------------
# ------------------------------ helpers ------------------------------------
# ---------------------------------------------------------------------------

# ── helpers ────────────────────────────────────────────────────────────
def _norm(t: str) -> str:
    """Return accent-free, lower-case, whitespace-collapsed copy of *t*."""
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", t).lower().strip()


def build_loader(arrow, steps, seed=42):
    """
    Deterministic DataLoader returning *steps* synthetic items.
    Each item: {'prompt': chat-prompt, 'answer': normalised answer}.
    """
    dset, rng = load_from_disk(arrow), random.Random(seed)

    def sample(_):
        row = rng.choice(dset)
        prompt = row["text"].rsplit("<|assistant|>", 1)[0] + "<|assistant|>"
        return {"prompt": prompt, "answer": row["answer_norm"]}

    return DataLoader([{}] * steps, batch_size=1, collate_fn=sample)

# ---------------------------------------------------------------------------
# --------------------------------- main ------------------------------------
# ---------------------------------------------------------------------------
def main(cfg):
    # ---------- tokenizer ---------------------------------------------------
    tok = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=True)
    tok.padding_side = "right" # Right-pad to keep left context intact.

    # ---------- 1) policy network with dormant critic -----------------------
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.base, torch_dtype="auto" # Auto-choose fp32/bf16.
    )

    # ---------- 2) disable critic head (v_head) completely ------------------
    for p in policy.v_head.parameters():                      # Iterate tensors.
        p.data.zero_()                                        # Zero-out weights.
        p.requires_grad_(False)                               # Freeze grads.
    policy.v_head.requires_grad_(False)                       # Extra safety.

    # ---------- 3) attach LoRA adapters (trainable params) -----------------
    policy = PeftModel.from_pretrained(
        policy,                          # Base + dead critic.
        cfg.init_ckpt,                   # LoRA checkpoint from SFT stage.
        is_trainable=True,               # Keep adapters trainable in PPO.
        _rename_model=False              # Preserve original parameter names.
    )

    # make value-head report an *empty* state-dict -> nothing to save, no hooks touched
    policy.v_head.state_dict = MethodType(lambda self, *a, **k: {}, policy.v_head)

    # Provide GenerationConfig so .generate() has sensible defaults ----------
    policy.generation_config = GenerationConfig.from_pretrained(cfg.base)

    # Register this PEFT subclass so PPOTrainer won’t refuse it --------------
    ppo_trainer.SUPPORTED_ARCHITECTURES += (policy.__class__,)

    # ---------- 4) PPO config (GRPO variant → vf_coef 0) --------------------
    k = cfg.group                                   # #completions per prompt.
    ppo_cfg = PPOConfig(
        batch_size=k,                               # Update uses group of k.
        mini_batch_size=k,
        ppo_epochs=1,                               # Single optimisation pass.
        vf_coef=0.0                                 # Ignore critic loss term.
    )
    trainer = PPOTrainer(model=policy, tokenizer=tok, config=ppo_cfg)
    device  = trainer.accelerator.device            # CPU / CUDA / TPU handle.

    # ---------- data & dirs -------------------------------------------------
    loader = build_loader(cfg.data_dir, cfg.num_steps)
    os.makedirs(cfg.output_dir, exist_ok=True)

    gen_kwargs = dict(max_new_tokens=128, top_p=0.95, do_sample=True) # Sampling.

    # ---------- training loop ----------------------------------------------
    for step, batch in enumerate(loader, 1):
        prompt, gold = batch["prompt"], batch["answer"]

        # ── tokenise prompt once, then replicate for k completions ----------
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)  # (1, L)
        query_ids = ids.repeat(k, 1)                                # (k, L)

        # ── sample k replies ------------------------------------------------
        with torch.no_grad():
            outs = policy.generate(
                input_ids=query_ids,
                pad_token_id=tok.eos_token_id,
                **gen_kwargs,
            )                               # (k, L+T)

        replies = [outs[i, ids.size(1):] for i in range(k)]
        texts   = [tok.decode(r, skip_special_tokens=True) for r in replies]

        # ── exact-match reward, centre by mean → GRPO advantage ------------
        rewards = torch.tensor(
            [float(_norm(gold) in _norm(t)) for t in texts], device=device
        )
        grp_rel = rewards - rewards.mean()          # Mean-centred rewards.

        # TRL expects *lists* of tensors ------------------------------------
        scores  = [r.unsqueeze(0) for r in grp_rel]          # (k,) → list[k] (1,)
        queries = [ids.squeeze(0)] * k                       # same as before

        trainer.step(queries, replies, scores) # PPO / GRPO update.

        # ── telemetry & checkpointing --------------------------------------
        if step % 10 == 0:
            print(f"step {step}/{cfg.num_steps}  reward_mean={rewards.mean():.2f}")
        if step % cfg.save_steps == 0:
            trainer.save_pretrained(Path(cfg.output_dir) / f"checkpoint-{step}")
        if step >= cfg.num_steps:
            break

    # ---------- final save --------------------------------------------------
    trainer.save_pretrained(cfg.output_dir)
    with open(Path(cfg.output_dir) / "meta.json", "w") as f:
        json.dump({"steps": cfg.num_steps, "group": k}, f)
    print("✓ GRPO tuning finished")

# ---------------------------------------------------------------------------
# CLI entry-point: only runs when executed as a script, not imported.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   required=True)        # Arrow dataset dir.
    ap.add_argument("--output_dir", required=True)        # Where to save model.
    ap.add_argument("--init_ckpt",  required=True)        # LoRA SFT checkpoint.
    ap.add_argument("--base",
                    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--num_steps",  type=int, default=3000)   # PPO iterations.
    ap.add_argument("--save_steps", type=int, default=500)    # CKPT every N steps.
    ap.add_argument("--group",      type=int, default=4,
                    help="completions per prompt (GRPO group size ≥2)")
    main(ap.parse_args())                                  # Launch main().