"""
Download + merge English & Spanish QA data
"""

import argparse, unicodedata, re
from datasets import load_dataset, concatenate_datasets

# ---------------------------------------------------------------------------
# Helper: remove diacritics, collapse whitespace, strip ends.
# ---------------------------------------------------------------------------
def _norm(txt):
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", txt).strip()

# ---------------------------------------------------------------------------
# Helper: convert a SQuAD-style row → chat-formatted turn.
# ---------------------------------------------------------------------------
def _to_chat(s, lang):
    ans = s["answers"]["text"][0]
    return {
        "text":
            f"<|user|> ({lang}) {s['question']}\n{s['context']}\n"
            f"<|assistant|> {ans}</s>",
        "answer_norm": _norm(ans),
    }

# ---------------------------------------------------------------------------
# Helper: keep only rows that actually contain at least one answer.
# ---------------------------------------------------------------------------
def _has_answer(row):         
    return len(row["answers"]["text"]) > 0

# ---------------------------------------------------------------------------
# Fetch, transform, blend, and save datasets.
# ---------------------------------------------------------------------------
def main(out_dir):
    # --------------------------- English SQuAD v1.1 ------------------------
    en = load_dataset("rajpurkar/squad", split="train").map(_to_chat, fn_kwargs={"lang":"en"}, remove_columns=["id","title","context","question","answers"])

    # --------------------------- Spanish SQuAD v2.0 ------------------------
    es_raw = load_dataset("ccasimiro/squad_es", "v2.0.0", split="train", trust_remote_code=True)
    es = (es_raw.filter(_has_answer).map(_to_chat, fn_kwargs={"lang":"es"}, remove_columns=["id","title","context","question","answers"]))
    
    # --------------------------- XQuAD (ES slice) --------------------------
    xquad = load_dataset("xquad", "xquad.es", split="validation").map(_to_chat, fn_kwargs={"lang": "es"}, remove_columns=["context","question","id","answers"])

    # --------------------------- Combine & shuffle -------------------------
    blended = concatenate_datasets([en, es, xquad]).shuffle(42)
    
    # --------------------------- Persist to disk ---------------------------
    blended.save_to_disk(out_dir)
    print(f"Saved {len(blended):,} samples to {out_dir}")

# ---------------------------------------------------------------------------
# CLI entry-point: only runs when executed as a script, not imported.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    main(parser.parse_args().out)
