"""
Supervised LoRA fine-tuning on DeepSeek-R1-Distill-Qwen-1.5B.
"""

import argparse, os
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model

# ---------------------------------------------------------------------------
# --------------------------------- main ------------------------------------
# ---------------------------------------------------------------------------
def main(cfg):

    # ---------- data -------------------------------------------------------
    dset = load_from_disk(cfg.data_dir)     # Re-hydrate Arrow dataset into memory.

    # ---------- model + tokenizer -----------------------------------------
    tok = AutoTokenizer.from_pretrained(    # Download / cache tokenizer linked
        cfg.base,                           # to the base model checkpoint.
        trust_remote_code=True              # Allow custom tokenizer classes.
    )
    tok.pad_token = tok.eos_token           # Use </s> as padding token to keep
                                            # compatibility with causal LM.
    tok.padding_side = "right"              # Right-pad so left context is intact.

    model = AutoModelForCausalLM.from_pretrained(  # Grab base 1.5 B-param model
        cfg.base, torch_dtype="auto"               # and let HF auto-select FP32/
    )                                              # BF16 depending on hardware.

    # ---------- LoRA -------------------------------------------------------
    lora_cfg = LoraConfig(                 # Instantiate LoRA hyper-parameters:
        r=16,                              #  • rank of low-rank adapters
        lora_alpha=64,                     #  • scaling factor
        lora_dropout=0.05,                 #  • dropout on adapter activations
        target_modules=[                   #  • which linear layers to inject into
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        task_type=TaskType.CAUSAL_LM,      # Marks this as left-to-right LM task.
    )
    model = get_peft_model(model, lora_cfg)   # Wrap base model with LoRA adapters.
    model.print_trainable_parameters()        # Log counts: confirms only adapters
                                              # & layer-norms are trainable.

    # ---------- training args ---------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,            # Where checkpoints & logs go.
        per_device_train_batch_size=4,        # Batch size PER GPU.
        gradient_accumulation_steps=4,        # 4 × accumulation ⇒ effective 16.
        num_train_epochs=2,                   # Two full passes over dataset.
        learning_rate=2e-4,                   # Peak LR (AdamW default optimiser).
        lr_scheduler_type="cosine",           # Cosine decay to zero.
        bf16=True,                            # Use bfloat16 if hardware supports.
        fp16=False,                           # Disable fp16 (choose one or other).
        report_to=[],                         # Empty list ⇒ no W&B / TensorBoard.
        logging_steps=10,                     # Log loss every 10 optimiser steps.

        save_strategy="steps",                # Checkpoint by step count (not epoch).
        save_steps=100,                       # Save every 100 optimiser steps.
        save_total_limit=3,                   # Keep last 3 checkpoints, delete older.

        ignore_data_skip=True,                # If resuming, start at curr index
                                              # rather than skipping examples.
    )

    # ---------- trainer ----------------------------------------------------
    trainer = SFTTrainer(
        model=model,                          # LoRA-augmented model.
        tokenizer=tok,                        # Tokeniser for smart truncation.
        train_dataset=dset,                   # HF Dataset with "text" field.
        args=training_args,                   # Hyper-param bundle.
        dataset_text_field="text",            # Column to read (added during fetch).
        max_seq_length=1024,                  # Token-level truncation length.
        packing=True,                         # Pack multiple short samples into
                                              # one sequence to minimise padding.
    )

    # ^^^ Fine-tune (resume if a checkpoint path string was supplied).
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    # Write final LoRA adapters & config into PEFT-compatible directory.
    trainer.save_model(cfg.output_dir)

# ---------------------------------------------------------------------------
# CLI entry-point: only runs when executed as a script, not imported.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)          # Arrow dataset path.
    parser.add_argument("--output_dir", required=True)        # Where to write ckpts.
    parser.add_argument("--resume_from_checkpoint",           # Optional resume arg.
                        default=None)
    parser.add_argument(                                      # Base HF model hub id
        "--base",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    args = parser.parse_args()                                # Parse CLI flags.
    os.makedirs(args.output_dir, exist_ok=True)               # Ensure out dir exists.
    main(args)  