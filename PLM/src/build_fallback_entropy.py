# scripts/build_fallback_entropy.py
import os, re, json, unicodedata, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from phoneme_tokenizer import PhonemeSplitter
from train_model import CustomTokenizerWrapper

def nfc(s): return unicodedata.normalize("NFC", s) if isinstance(s, str) else s

def parse_id_and_text(line: str):
    if "\t" in line:
        id_part, text_part = line.split("\t", 1)
        id_part = id_part.strip()
        return (id_part if id_part and id_part != "NA" else None, text_part.strip())
    return (None, line.strip())

def preprocess_line(text: str) -> str:
    t = nfc(text.strip())
    if t.startswith('[BOS]'): t = t[5:]
    if t.endswith('[EOS]'):   t = t[:-5]
    return nfc(t.strip())

def load_overview(overview_csv):
    df = pd.read_csv(overview_csv)
    need = {"id","speaker","conversation_id","age","sex"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"overview missing columns: {sorted(miss)}")
    # make a clean id->meta map
    m = {}
    for _, r in df.iterrows():
        oid = str(r["id"]).strip()
        m[oid] = {
            "speaker": str(r["speaker"]) if pd.notna(r["speaker"]) else "",
            "conversation": str(r["conversation_id"]) if pd.notna(r["conversation_id"]) else "",
            "age": str(int(r["age"])) if "age" in r and pd.notna(r["age"]) and float(r["age"]).is_integer() else (str(r["age"]) if "age" in r and pd.notna(r["age"]) else ""),
            "sex": str(r["sex"]) if "sex" in r and pd.notna(r["sex"]) else ""
        }
    return m

def iter_lines(paths):
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    yield line

def main(train_path, eval_path, overview_csv, model_dir, tokenizer_path,
         glottocode, csv_path=None, csv_column=None, out_csv="ent_train_eval.csv"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device).eval()

    splitter = PhonemeSplitter(
        use_csv_units=bool(csv_path),
        csv_path=csv_path,
        csv_column=csv_column
    )
    tok = CustomTokenizerWrapper(tokenizer_path, splitter)

    # special + vocab
    special = {tok.pad_token, tok.unk_token, tok.bos_token, tok.eos_token}
    vocab = tok.tokenizer.get_vocab()
    id2tok = {v:k for k,v in vocab.items()}

    id2meta = load_overview(overview_csv)

    rows = []
    sources = [p for p in [train_path, eval_path] if p and os.path.exists(p)]
    if not sources:
        raise SystemExit("No train/eval files found.")

    for line in tqdm(iter_lines(sources), desc="Entropy on train+eval"):
        oid, raw = parse_id_and_text(line)
        if oid is None or oid not in id2meta:
            # no ID or not found in overview; skip (we need speaker/conversation)
            continue
        meta = id2meta[oid]
        processed = preprocess_line(raw)
        phs = splitter.split(processed)
        joined = " ".join(phs)

        enc = tok(joined)
        pad = tok.pad([enc], return_tensors="pt")
        input_ids = pad["input_ids"].to(device)

        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits[0, :-1]
            targets = input_ids[0, 1:]

            for i, tgt in enumerate(targets):
                token_str = id2tok.get(int(tgt), "[UNK]")
                if token_str in special:  # skip BOS/EOS/etc
                    continue
                prob = torch.softmax(logits[i], dim=-1)[tgt].item()
                entropy = -np.log2(prob + 1e-12)
                rows.append({
                    "glottocode": glottocode,
                    "speaker": meta["speaker"],
                    "conversation": meta["conversation"],
                    "phoneme": token_str,
                    "entropy_rate": entropy,
                    "age": meta["age"],
                    "sex": meta["sex"]
                })

    if not rows:
        raise SystemExit("No rows computed. Are your train/eval lines ID\\tTEXT and overview aligned?")

    df = pd.DataFrame(rows)
    # average per key
    agg = (df.groupby(["glottocode","speaker","conversation","phoneme","age","sex"], dropna=False)["entropy_rate"]
             .mean().reset_index())
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    agg.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote fallback: {out_csv}  rows={len(agg)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build ent_train_eval.csv from train+eval splits.")
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--eval_path", required=True)
    ap.add_argument("--overview_csv", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--glottocode", required=True)
    ap.add_argument("--csv_path")
    ap.add_argument("--csv_column")
    ap.add_argument("--out_csv", default="ent_train_eval.csv")
    args = ap.parse_args()

    main(args.train_path, args.eval_path, args.overview_csv,
         args.model_dir, args.tokenizer_path, args.glottocode,
         csv_path=args.csv_path, csv_column=args.csv_column,
         out_csv=args.out_csv)