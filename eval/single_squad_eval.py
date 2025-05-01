#!/usr/bin/env python
import argparse, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

metric = evaluate.load("squad")

def answer(model, tok, q, ctx, max_new=48):
    prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    ans = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
    return ans.split("\n")[0].strip()

def main(cfg):
    tok   = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype="auto").eval()

    ds = load_dataset(cfg.dataset, split="validation")
    preds, refs = [], []
    for ex in ds:
        preds.append({"id": ex["id"], "prediction_text": answer(model, tok, ex["question"], ex["context"])})
        refs.append({"id": ex["id"], "answers": ex["answers"]})

    print(json.dumps(metric.compute(predictions=preds, references=refs), indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="squad", choices=["squad","squad_es","squad_v2"])
    main(ap.parse_args())
