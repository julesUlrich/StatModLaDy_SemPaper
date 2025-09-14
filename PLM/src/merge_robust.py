# src/merge_robust.py

import re
import unicodedata
import pandas as pd
import argparse
from pathlib import Path
from functools import reduce

FILE_RE = re.compile(r'^doreco_([^_]+)_(.+)$')  # doreco_<glottocode>_<conversation>

def parse_file_into_fields(x: str):
    if not isinstance(x, str):
        return None, None
    # strip directory and extension just in case
    name = Path(x).name
    if "." in name:
        name = name.split(".", 1)[0]
    parts = name.strip().split("_", 2)  # split at most twice
    # need: <prefix>, <glottocode>, <conversation...>
    if len(parts) < 3:
        return None, None
    # parts[0] is prefix: "doreco" or "VC" (ignored)
    glottocode = parts[1]
    conversation = parts[2]
    return glottocode, conversation

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else s

def load_phoneme_map(path: str | None):
    """
    Returns a dict canonicalizer. If no file, we still map common variants.
    CSV format: from,to
    """
    base = {
        "ʤ": "dʒ",    # common affricate variant
        "e:": "eː", "o:": "oː", "i:": "iː", "u:": "uː", "a:": "aː",
    }
    if not path:
        return base
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        frm = nfc(str(row["from"]))
        to  = nfc(str(row["to"]))
        if frm:
            base[frm] = to
    return base

def canonize_phoneme(s: str, pmap: dict) -> str:
    s = nfc(str(s))
    return pmap.get(s, s)

def _standardize_entropy_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a column named 'entropy_rate'.
    Accepts 'mean_entropy' or 'entropy' and renames to 'entropy_rate'.
    """
    if "entropy_rate" in df.columns:
        return df
    if "mean_entropy" in df.columns:
        df = df.rename(columns={"mean_entropy": "entropy_rate"})
    elif "entropy" in df.columns:
        df = df.rename(columns={"entropy": "entropy_rate"})
    else:
        raise ValueError(
            "Entropy table is missing an entropy column "
            f"(expected one of: entropy_rate, mean_entropy, entropy); got {list(df.columns)}"
        )
    return df

def main(entropy_csv, speechrate_csv, out_csv,
         fallback_entropy=None, phoneme_map_csv=None, impute_sr="none"):
    # 0) Phoneme mapping
    pmap = load_phoneme_map(phoneme_map_csv)

    # 1) Load entropy (aggregated per convo) and standardize column names
    ent = pd.read_csv(entropy_csv)
    ent = _standardize_entropy_col(ent)  # -> ensures 'entropy_rate'

    # If entropy CSV carries counts, keep as ent_count (won't be clobbered later)
    if "count" in ent.columns:
        ent = ent.rename(columns={"count": "ent_count"})
        ent["ent_count"] = pd.to_numeric(ent["ent_count"], errors="coerce")

    need_ent = {"glottocode", "speaker", "conversation", "phoneme", "entropy_rate"}
    miss = need_ent - set(ent.columns)
    if miss:
        raise ValueError(f"Entropy CSV missing columns: {sorted(miss)}")

    for col in ["glottocode", "speaker", "conversation", "phoneme"]:
        ent[col] = ent[col].astype(str).map(nfc)
    ent["phoneme"] = ent["phoneme"].map(lambda x: canonize_phoneme(x, pmap))
    meta_cols = [c for c in ["age", "sex"] if c in ent.columns]

    # 2) Load speech rate
    sp = pd.read_csv(speechrate_csv)
    print(sp.head)
    sp.columns = [c.strip().lower() for c in sp.columns]

    # Accept aliases
    if "rate" not in sp.columns:
        for a in ["speech_rate", "sr", "r"]:
            if a in sp.columns:
                sp = sp.rename(columns={a: "rate"})
                break
    if "count" not in sp.columns:
        for a in ["n", "freq", "token_count", "tokens"]:
            if a in sp.columns:
                sp = sp.rename(columns={a: "count"})
                break

    needed_sp = {"file", "speaker", "phoneme", "rate"}
    missing = needed_sp - set(sp.columns)
    if missing:
        raise ValueError(f"Speech-rate CSV missing columns: {sorted(missing)}")

    if "count" not in sp.columns:
        sp["count"] = 0

    # Parse glottocode + conversation from file
    gc_list, conv_list = [], []
    for f in sp["file"]:
        gc, conv = parse_file_into_fields(f)
        gc_list.append(gc)
        conv_list.append(conv)
    sp["glottocode"] = gc_list
    sp["conversation"] = conv_list

    for col in ["glottocode", "speaker", "conversation", "phoneme"]:
        sp[col] = sp[col].astype(str).map(nfc)
    sp["phoneme"] = sp["phoneme"].map(lambda x: canonize_phoneme(x, pmap))

    sp["rate"] = pd.to_numeric(sp["rate"], errors="coerce")
    sp["count"] = pd.to_numeric(sp["count"], errors="coerce").fillna(0).astype(int)

    key = ["glottocode", "speaker", "conversation", "phoneme"]
    sp_small = (
        sp.groupby(key, as_index=False)
          .agg(speech_rate=("rate", "mean"),   # change to "sum" if that's what you intend
               sr_count=("count", "sum"))
    )

    # 3) Initial inner merge (entropy × speech-rate)
    merged = ent.merge(sp_small, on=key, how="inner")

    # 4) Diagnostics before backfills
    ent_keys = ent[key].drop_duplicates()
    sp_keys = sp_small[key].drop_duplicates()

    unmatched_ent = ent_keys.merge(sp_keys, on=key, how="left", indicator=True)
    unmatched_ent = unmatched_ent[unmatched_ent["_merge"] == "left_only"].drop(columns="_merge")

    unmatched_sp = sp_keys.merge(ent_keys, on=key, how="left", indicator=True)
    unmatched_sp = unmatched_sp[unmatched_sp["_merge"] == "left_only"].drop(columns="_merge")

    print(f"Merged rows: {len(merged)}")
    print(f"Distinct key matches possible: {len(ent_keys.merge(sp_keys, on=key, how='inner'))}")
    if not unmatched_ent.empty:
        print(f"⚠️  {len(unmatched_ent)} entropy key rows had no speech-rate match. Example:")
        print(unmatched_ent.head(5).to_string(index=False))
    if not unmatched_sp.empty:
        print(f"⚠️  {len(unmatched_sp)} speech-rate key rows had no entropy match. Example:")
        print(unmatched_sp.head(5).to_string(index=False))

    # 5) Backfill ENTROPY for SR-only rows from fallback (ent_train_eval.csv)
    fb = None
    if fallback_entropy:
        fb = pd.read_csv(fallback_entropy)
        fb = _standardize_entropy_col(fb)  # ensures 'entropy_rate'

        # carry fallback count, if present
        if "count" in fb.columns:
            fb = fb.rename(columns={"count": "fb_count"})
            fb["fb_count"] = pd.to_numeric(fb["fb_count"], errors="coerce")

        need_fb = {"glottocode", "speaker", "conversation", "phoneme", "entropy_rate"}
        miss = need_fb - set(fb.columns)
        if miss:
            raise ValueError(f"Fallback entropy CSV missing columns: {sorted(miss)}")

        for col in key:
            fb[col] = fb[col].astype(str).map(nfc)
        fb["phoneme"] = fb["phoneme"].map(lambda x: canonize_phoneme(x, pmap))

        # average entropy per key (fallback)
        fb_mean = fb.groupby(key, as_index=False)["entropy_rate"].mean()

        # rows that are in SR but not in ENT
        need_entropy = unmatched_sp.merge(fb_mean, on=key, how="left")
        have_entropy = need_entropy[~need_entropy["entropy_rate"] .isna()].copy()
        if not have_entropy.empty:
            print(f"Backfilling entropy for {len(have_entropy)} SR-only rows using fallback.")
            # attach speech_rate + sr_count from sp_small
            have_entropy = have_entropy.merge(sp_small, on=key, how="left")

            # attach fb_count if available
            if "fb_count" in fb.columns:
                fb_counts = fb[key + ["fb_count"]].dropna().drop_duplicates()
                have_entropy = have_entropy.merge(fb_counts, on=key, how="left")

            # attach age/sex from entropy side if present
            if meta_cols:
                meta = ent[key + meta_cols].drop_duplicates()
                have_entropy = have_entropy.merge(meta, on=key, how="left")

            merged = pd.concat([merged, have_entropy], ignore_index=True)

    # 6) Impute speech-rate for ENT-only if requested
    if impute_sr != "none" and not unmatched_ent.empty:
        to_impute = ent.merge(unmatched_ent, on=key, how="inner")

        if impute_sr in {"speaker_mean", "backoff"}:
            A = (
                sp_small.groupby(["glottocode", "speaker", "phoneme"], as_index=False)["speech_rate"]
                        .mean().rename(columns={"speech_rate": "sr_A"})
            )
            B = (
                sp_small.groupby(["glottocode", "phoneme"], as_index=False)["speech_rate"]
                        .mean().rename(columns={"speech_rate": "sr_B"})
            )
            C = (
                sp_small.groupby(["glottocode", "speaker"], as_index=False)["speech_rate"]
                        .mean().rename(columns={"speech_rate": "sr_C"})
            )
            D = (
                sp_small.groupby(["glottocode"], as_index=False)["speech_rate"]
                        .mean().rename(columns={"speech_rate": "sr_D"})
            )

            tmp = to_impute.copy()
            tmp = tmp.merge(A, on=["glottocode", "speaker", "phoneme"], how="left")
            tmp = tmp.merge(B, on=["glottocode", "phoneme"], how="left")
            tmp = tmp.merge(C, on=["glottocode", "speaker"], how="left")
            tmp = tmp.merge(D, on=["glottocode"], how="left")

            if impute_sr == "speaker_mean":
                tmp["speech_rate"] = tmp["sr_A"]
            else:
                tmp["speech_rate"] = (
                    tmp["sr_A"].combine_first(tmp["sr_B"])
                               .combine_first(tmp["sr_C"])
                               .combine_first(tmp["sr_D"])
                )

            have_sr = tmp[~tmp["speech_rate"].isna()].copy()
            if not have_sr.empty:
                # attach sr_count if known
                have_sr = have_sr.merge(
                    sp_small[key + ["sr_count"]],
                    on=key, how="left"
                )
                print(f"Imputing speech-rate for {len(have_sr)} ENT-only rows (strategy={impute_sr}).")
                merged = pd.concat([merged, have_sr], ignore_index=True)

        elif impute_sr == "zero":
            to_impute = to_impute.copy()
            to_impute["speech_rate"] = 0.0
            to_impute = to_impute.merge(sp_small[key + ["sr_count"]], on=key, how="left")
            print(f"Imputing speech-rate for {len(to_impute)} ENT-only rows (strategy=zero).")
            merged = pd.concat([merged, to_impute], ignore_index=True)
        else:
            raise ValueError("--impute_sr must be one of: none | speaker_mean | backoff | zero")

    # 7) Build unified 'count' with priority: SR → fallback → entropy
    # Attach ent_count reference (if entropy carried one)
    if "ent_count" in ent.columns:
        ent_counts = ent[key + ["ent_count"]].drop_duplicates()
        merged = merged.merge(ent_counts, on=key, how="left")

    # If we had fallback with fb_count, attach it for rows added earlier
    if isinstance(fb, pd.DataFrame) and "fb_count" in fb.columns:
        fb_counts_ref = fb[key + ["fb_count"]].drop_duplicates()
        merged = merged.merge(fb_counts_ref, on=key, how="left")

    # Coalesce
    merged["count"] = (
        pd.to_numeric(merged.get("sr_count"), errors="coerce")
          .fillna(pd.to_numeric(merged.get("fb_count"), errors="coerce"))
          .fillna(pd.to_numeric(merged.get("ent_count"), errors="coerce"))
          .fillna(0)
          .astype(int)
    )

    # quick diagnostic for zero-count keys
    zero_keys = (merged.groupby(key, as_index=False)["count"]
                      .sum().query("count == 0"))
    if not zero_keys.empty:
        print("⚠️  Keys with total count = 0 after merge (no SR & no fallback counts):")
        print(zero_keys.head(10).to_string(index=False))

    # 8) Final numerics + info rate
    merged["entropy_rate"] = pd.to_numeric(merged["entropy_rate"], errors="coerce")
    merged["speech_rate"] = pd.to_numeric(merged["speech_rate"], errors="coerce")
    merged["information_rate"] = merged["entropy_rate"] * merged["speech_rate"]

    # 9) Reorder columns
    out_cols = ["glottocode", "speaker", "conversation", "phoneme",
                "entropy_rate", "speech_rate", "information_rate", "count"]
    for c in ["age", "sex"]:
        if c in meta_cols and c not in out_cols:
            out_cols.insert(2, c)

    # Create any missing (with appropriate default types)
    numeric_cols = {"entropy_rate", "speech_rate", "information_rate", "count"}
    for c in out_cols:
        if c not in merged.columns:
            merged[c] = 0 if c in numeric_cols else ""

    merged = merged[out_cols].copy()

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote merged table: {out_csv}")
    print(f"Final rows: {len(merged)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robustly merge entropy aggregates with speech-rate table.")
    ap.add_argument("--entropy_csv", required=True,
                    help="CSV from create_result_summary.py (row per glottocode/speaker/conversation/phoneme).")
    ap.add_argument("--speechrate_csv", required=True,
                    help="CSV with columns: file,speaker,phoneme,total_duration,count,average_duration,rate")
    ap.add_argument("--out_csv", required=True, help="Path to write merged CSV.")
    ap.add_argument("--fallback_entropy", help="Optional: ent_train_eval.csv to backfill entropy for SR-only rows.")
    ap.add_argument("--phoneme_map", dest="phoneme_map_csv",
                    help="Optional CSV 'from,to' to canonicalize phoneme labels.")
    ap.add_argument("--impute_sr", default="none",
                    choices=["none", "speaker_mean", "backoff", "zero"],
                    help="Impute speech-rate for ENT-only rows.")
    args = ap.parse_args()

    main(args.entropy_csv, args.speechrate_csv, args.out_csv,
         fallback_entropy=args.fallback_entropy,
         phoneme_map_csv=args.phoneme_map_csv,
         impute_sr=args.impute_sr)