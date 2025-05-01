#!/usr/bin/env python
import argparse, json, torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# ---- global constants -----------------------------------------------------
BATCH   = 16                                  # Batch size per forward pass.
MAX_NEW = 48                                 # Max tokens generated per answer.
metric  = evaluate.load("squad")             # Exact-Match + F1 metric object.

# ---------------------------------------------------------------------------
# Collate function that **keeps each field as a ragged Python list**.
# We don’t need tensorisation here because generation length varies.
# ---------------------------------------------------------------------------
def ragged_collate(batch):
    collated = {k: [] for k in batch[0].keys()}   # Init dict with list per field.
    for ex in batch:                              # For each sample in mini-batch…
        for k, v in ex.items():                   # …append value to appropriate list.
            collated[k].append(v)
    return collated                               # Dict[str, List[Any]]

# ---------------------------------------------------------------------------
# --------------------------------- main ------------------------------------
# ---------------------------------------------------------------------------
def main(cfg):
    # Pick GPU if present.
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # ---- tokenizer --------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id # Use </s> for padding—safe for LMs.

    # ---- model ------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype="auto", device_map={"": device}
    ).eval()

    # ---- dataset + dataloader --------------------------------------------
    ds      = load_dataset(cfg.dataset, split="validation")
    loader  = DataLoader(ds, batch_size=BATCH, collate_fn=ragged_collate)

    # ---- loop over batches -----------------------------------------------
    preds, refs = [], []
    processed   = 0
    for batch in loader:
        # Build prompt per example ------------------------------------------------
        prompts = [f"Context: {c}\nQuestion: {q}\nAnswer (one brief phrase):"
           for c, q in zip(batch["context"], batch["question"])]

        enc = tok(prompts, return_tensors="pt", padding=True).to(device)

        # Generate greedy answers (one per prompt) --------------------------
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                top_p=0.9,
                temperature=0.2
            )

        # Post-process each sample within the batch ------------------------
        for j, ex_id in enumerate(batch["id"]):
            ans = tok.decode(
                out[j, enc.input_ids.shape[1]:],
                skip_special_tokens=True
            ).split("\n")[0].strip()

            preds.append({"id": ex_id, "prediction_text": ans})
            refs.append({"id": ex_id, "answers": batch["answers"][j]})

        # Progress log every 1 000 examples --------------------------------
        processed += len(batch["id"])
        if processed % 1000 == 0:
            print(f"→ processed {processed}/{len(ds)} questions")

    # ---- compute EM/F1 & dump as pretty JSON -----------------------------
    print(json.dumps(metric.compute(predictions=preds, references=refs), indent=2))

# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path or hub-id of merged checkpoint")
    ap.add_argument("--dataset", default="squad",
                    choices=["squad", "squad_es", "squad_v2"])
    main(ap.parse_args())
