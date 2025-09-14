# src/split_corpus.py

import os
import glob
import argparse
import pandas as pd
from collections import defaultdict
import re
import unicodedata
from typing import Dict, Optional, Tuple

# -----------------------------
# Filename parsing
# -----------------------------

def extract_speaker_and_conversation_id(filename: str) -> Tuple[str, str]:
    """
    Extract speaker ID and conversation ID from filenames like:
      doreco_bora1263_llijchu_ine_II1_00-16_JUM_utterances.txt -> ('JUM','llijchu_ine_II1_00-16')
      doreco_bora1263_ovehe_1_JUM_utterances.txt               -> ('JUM','ovehe_1')
    """
    basename = os.path.basename(filename)
    m = re.match(r'doreco_[^_]+_(.+)_([^_]+)_utterances\.txt', basename)
    if not m:
        raise ValueError(f"Cannot extract speaker/conversation from: {basename}")
    conversation_id = m.group(1)
    speaker_id = m.group(2)
    return speaker_id, conversation_id

# -----------------------------
# I/O helpers
# -----------------------------

def load_utterances_with_metadata(file_path):
    """Return list of (utterance_text, speaker_id, conversation_id)."""
    speaker_id, conversation_id = extract_speaker_and_conversation_id(file_path)
    with open(file_path, encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f if line.strip()]
        # DO NOT strip BOS/EOS here; we’ll match on a cleaned copy but preserve original in output
        lines = [unicodedata.normalize('NFC', line) for line in lines]
    return [(line, speaker_id, conversation_id) for line in lines]

def clean_for_matching(text: str) -> str:
    """Remove [BOS]/[EOS], trim, normalize NFC."""
    t = text.strip()
    if t.startswith('[BOS]'):
        t = t[5:]
    if t.endswith('[EOS]'):
        t = t[:-5]
    return unicodedata.normalize('NFC', t.strip())

# -----------------------------
# Overview handling
# -----------------------------

def load_overview_file(overview_path: str) -> pd.DataFrame:
    if not os.path.exists(overview_path):
        raise FileNotFoundError(f"Overview file not found: {overview_path}")
    df = pd.read_csv(overview_path)
    required = ['id', 'ref', 'speaker', 'ipa', 'orthography']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in overview")
    return df

def create_text_to_id_mapping(overview_df: pd.DataFrame) -> Dict[str, str]:
    """
    Map IPA/orthography (NFC, stripped) -> string(id).
    We add both raw and NFC variants to be robust.
    """
    mapping: Dict[str, str] = {}
    for _, row in overview_df.iterrows():
        id_str = str(row['id']).strip()
        if pd.notna(row['ipa']):
            ipa = str(row['ipa']).strip()
            if ipa:
                mapping[ipa] = id_str
                mapping[unicodedata.normalize('NFC', ipa)] = id_str
        if pd.notna(row['orthography']):
            orth = str(row['orthography']).strip()
            if orth:
                mapping[orth] = id_str
                mapping[unicodedata.normalize('NFC', orth)] = id_str
    return mapping

def create_id_to_index(overview_df: pd.DataFrame) -> Dict[str, int]:
    """Map string(id) -> pandas row index (to update split/conversation in overview)."""
    return {str(row['id']).strip(): i for i, row in overview_df.iterrows()}

# -----------------------------
# Split logic
# -----------------------------

def split_by_character_count_three_way(utterances_with_meta, train_ratio, eval_ratio):
    total_chars = sum(len(u[0]) for u in utterances_with_meta)
    train_chars_target = int(total_chars * train_ratio)
    eval_chars_target = int(total_chars * eval_ratio)

    train_utts, eval_utts, test_utts = [], [], []
    char_count = 0
    for utt in utterances_with_meta:
        if char_count < train_chars_target:
            train_utts.append(utt)
        elif char_count < train_chars_target + eval_chars_target:
            eval_utts.append(utt)
        else:
            test_utts.append(utt)
        char_count += len(utt[0])
    return train_utts, eval_utts, test_utts

def split_by_character_count_strict(utterances_with_meta, split_ratio):
    joined = "".join([u[0] for u in utterances_with_meta])
    split_point = int(len(joined) * split_ratio)
    train_text = joined[:split_point]
    test_text = joined[split_point:]
    # Keep some metadata (speaker/conv of the first group)
    return [(train_text, utterances_with_meta[0][1], utterances_with_meta[0][2])], \
           [(test_text,  utterances_with_meta[0][1], utterances_with_meta[0][2])]

# -----------------------------
# Matching
# -----------------------------

def match_to_overview_id(utterance_text: str, text2id: Dict[str, str]) -> Optional[str]:
    """
    Return overview 'id' (as string) if we can match by IPA/orthography,
    after removing BOS/EOS and NFC-normalizing; else None.
    """
    cleaned = clean_for_matching(utterance_text)
    # direct
    if cleaned in text2id:
        return text2id[cleaned]
    # try again with NFC (already NFC, but harmless)
    nfc = unicodedata.normalize('NFC', cleaned)
    return text2id.get(nfc, None)

# -----------------------------
# Overview enrichment
# -----------------------------

def update_overview_with_splits(overview_df: pd.DataFrame,
                                id_to_split: Dict[str, str],
                                id_to_conversation: Dict[str, str]) -> pd.DataFrame:
    df = overview_df.copy()
    if 'split' not in df.columns:
        df['split'] = 'unknown'
    else:
        df['split'] = df['split'].fillna('unknown')

    if 'conversation_id' not in df.columns:
        df['conversation_id'] = 'unknown'
    else:
        df['conversation_id'] = df['conversation_id'].fillna('unknown')

    for i, row in df.iterrows():
        sid = str(row['id']).strip()
        if sid in id_to_split:
            df.at[i, 'split'] = id_to_split[sid]
        if sid in id_to_conversation:
            df.at[i, 'conversation_id'] = id_to_conversation[sid]
    return df

# -----------------------------
# Main split
# -----------------------------

def split_and_write_corpus(
        input_dir: str,
        output_dir: str,
        overview_file: str,
        train_filename: str = "train.txt",
        eval_filename: str = "eval.txt",
        enhanced_overview_filename: str = "enhanced_overview.csv",
        train_ratio: float = 0.8,
        eval_ratio: float = 0.1,
        create_eval_split: bool = True
):
    os.makedirs(output_dir, exist_ok=True)

    # Load overview & mappings
    print(f"Loading overview: {overview_file}")
    overview_df = load_overview_file(overview_file)
    text2id = create_text_to_id_mapping(overview_df)
    id2idx = create_id_to_index(overview_df)

    # Load utterances grouped by (speaker, conversation)
    speaker_conversation_utterances = defaultdict(lambda: defaultdict(list))
    files = glob.glob(os.path.join(input_dir,"utterances", "*_utterances.txt"))
    print(f"Found {len(files)} utterance files")

    for fp in files:
        items = load_utterances_with_metadata(fp)  # (text, speaker, conv)
        if items:
            spk = items[0][1]
            conv = items[0][2]
            speaker_conversation_utterances[spk][conv] = items
            print(f"  {spk}/{conv}: {len(items)} lines")

    # Prepare outputs
    train_path = os.path.join(output_dir, train_filename)
    eval_path  = os.path.join(output_dir, eval_filename) if create_eval_split else None
    manifest_rows = []  # for splits_manifest.csv

    counters = {'total_train_chars': 0, 'total_eval_chars': 0, 'total_test_chars': 0}

    if create_eval_split:
        test_ratio = 1.0 - train_ratio - eval_ratio
        if test_ratio < 0:
            raise ValueError("train_ratio + eval_ratio must be < 1.0")
        print(f"3-way split: train {train_ratio:.0%}, eval {eval_ratio:.0%}, test {test_ratio:.0%}")
    else:
        test_ratio = 1.0 - train_ratio
        print(f"2-way split: train {train_ratio:.0%}, test {test_ratio:.0%}")

    id_to_split = {}         # overview id -> split
    id_to_conversation = {}  # overview id -> conversation_id

    with open(train_path, "w", encoding="utf-8-sig") as f_train:
        if create_eval_split:
            with open(eval_path, "w", encoding="utf-8-sig") as f_eval:
                # iterate speakers/conversations
                for spk, convs in speaker_conversation_utterances.items():
                    for conv, utts in convs.items():
                        tr, ev, te = split_by_character_count_three_way(utts, train_ratio, eval_ratio)

                        # write helper
                        def write_block(block, split_name, handle, write_test_to=None):
                            nonlocal counters
                            for (text, s, c) in block:
                                overview_id = match_to_overview_id(text, text2id)
                                overview_id_out = overview_id if overview_id else "NA"

                                # write line with ID prefix
                                line = f"{overview_id_out}\t{text}\n"
                                handle.write(line) if handle else write_test_to.write(line)

                                # update overview split/conv if known
                                if overview_id:
                                    id_to_split[overview_id] = split_name
                                    id_to_conversation[overview_id] = c

                                # manifest row
                                manifest_rows.append({
                                    "overview_id": overview_id_out,
                                    "split": split_name,
                                    "speaker": s,
                                    "conversation_id": c,
                                    "text_preview": clean_for_matching(text)[:60]
                                })

                            # counters
                            chars = sum(len(t[0]) for t in block)
                            if split_name == "train": counters['total_train_chars'] += chars
                            elif split_name == "eval": counters['total_eval_chars'] += chars
                            else: counters['total_test_chars'] += chars

                        # train
                        write_block(tr, "train", f_train)
                        # eval
                        write_block(ev, "eval", f_eval)
                        # test -> file per speaker/conv
                        test_path = os.path.join(output_dir, f"test_{spk}_{conv}.txt")
                        with open(test_path, "w", encoding="utf-8-sig") as f_test:
                            write_block(te, "test", None, write_test_to=f_test)
        else:
            # legacy two-way
            for spk, convs in speaker_conversation_utterances.items():
                for conv, utts in convs.items():
                    tr, te = split_by_character_count_strict(utts, train_ratio)

                    def write_block(block, split_name, handle, write_test_to=None):
                        nonlocal counters
                        for (text, s, c) in block:
                            overview_id = match_to_overview_id(text, text2id)
                            overview_id_out = overview_id if overview_id else "NA"
                            line = f"{overview_id_out}\t{text}\n"
                            handle.write(line) if handle else write_test_to.write(line)
                            if overview_id:
                                id_to_split[overview_id] = split_name
                                id_to_conversation[overview_id] = c
                            manifest_rows.append({
                                "overview_id": overview_id_out,
                                "split": split_name,
                                "speaker": s,
                                "conversation_id": c,
                                "text_preview": clean_for_matching(text)[:60]
                            })
                        chars = sum(len(t[0]) for t in block)
                        if split_name == "train": counters['total_train_chars'] += chars
                        else: counters['total_test_chars'] += chars

                    write_block(tr, "train", f_train)
                    test_path = os.path.join(output_dir, f"test_{spk}_{conv}.txt")
                    with open(test_path, "w", encoding="utf-8-sig") as f_test:
                        write_block(te, "test", None, write_test_to=f_test)

    # Save enhanced overview
    enhanced = update_overview_with_splits(overview_df, id_to_split, id_to_conversation)
    enhanced_overview_path = os.path.join(output_dir, enhanced_overview_filename)
    enhanced.to_csv(enhanced_overview_path, index=False, encoding="utf-8-sig")

    # Save manifest
    manifest_path = os.path.join(output_dir, "splits_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")

    # Report unmatched
    unmatched = [r for r in manifest_rows if r["overview_id"] == "NA"]
    if unmatched:
        print(f"⚠️  {len(unmatched)} utterances could not be matched to overview (id='NA').")
        print("   Example previews:")
        for row in unmatched[:5]:
            print(f"   - [{row['split']}] {row['speaker']}/{row['conversation_id']}: '{row['text_preview']}'")

    # Summary
    print("\n" + "=" * 60)
    print("CORPUS SPLIT SUMMARY")
    print("=" * 60)
    print(f"Train chars: {counters['total_train_chars']:,}")
    if create_eval_split:
        print(f"Eval  chars: {counters['total_eval_chars']:,}")
    print(f"Test  chars: {counters['total_test_chars']:,}")
    total = counters['total_train_chars'] + counters.get('total_eval_chars', 0) + counters['total_test_chars']
    print(f"Total chars: {total:,}")
    print("\nOutputs:")
    print(f"  Train: {train_path}")
    if create_eval_split:
        print(f"  Eval:  {eval_path}")
    print(f"  Test:  {output_dir}/test_*_*.txt")
    print(f"  Enhanced overview: {enhanced_overview_path}")
    print(f"  Manifest: {manifest_path}")

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split corpus and write ID-prefixed lines for deterministic matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Dir with doreco_*_utterances.txt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir for splits")
    parser.add_argument("--overview_file", type=str, required=True, help="Path to overview CSV")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--train_filename", type=str, default="train.txt")
    parser.add_argument("--eval_filename", type=str, default="eval.txt")
    parser.add_argument("--enhanced_overview_filename", type=str, default="enhanced_overview.csv")
    parser.add_argument("--eval_split", action="store_true", help="3-way split (train/eval/test)")
    parser.add_argument("--legacy_mode", action="store_true", help="2-way split (train/test)")
    args = parser.parse_args()

    if args.legacy_mode:
        create_eval_split = False
        tr = args.train_ratio
        ev = 0.0
        print("Legacy mode (2-way split)")
    else:
        create_eval_split = args.eval_split
        tr = args.train_ratio
        ev = args.eval_ratio if create_eval_split else 0.0
        if create_eval_split and (tr + ev >= 1.0):
            raise SystemExit(f"train_ratio ({tr}) + eval_ratio ({ev}) must be < 1.0")

    split_and_write_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overview_file=args.overview_file,
        train_filename=args.train_filename,
        eval_filename=args.eval_filename,
        enhanced_overview_filename=args.enhanced_overview_filename,
        train_ratio=tr,
        eval_ratio=ev,
        create_eval_split=create_eval_split
    )