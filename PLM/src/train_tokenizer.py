# src/train_tokenizer.py

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.processors import TemplateProcessing
from phoneme_tokenizer import PhonemeSplitter
import os
import argparse
import tempfile
import unicodedata
from collections import Counter
import pandas as pd


def preprocess_line(line: str) -> str:
    """
    Remove leading ID (if present) and strip BOS/EOS tokens before splitting.
    """
    line = line.strip()

    # Remove leading ID if corpus is in "<id>\t<text>" format
    if "\t" in line:
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].isdigit():
            line = parts[1]

    # Normalize Unicode
    line = unicodedata.normalize("NFC", line)

    # Remove special tokens from raw text
    for token in ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]:
        line = line.replace(token, "")

    return line.strip()


def create_fixed_vocabulary_tokenizer(
            corpus_files,
            output_path,
            use_csv_units=False,
            csv_path=None,
            csv_column=None,
            add_frequent_combinations=False,
            min_freq_for_combinations=10,
            handle_missing="warn"
    ):
    splitter = PhonemeSplitter(
        use_csv_units=use_csv_units,
        csv_path=csv_path,
        csv_column=csv_column
    )

    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    SPECIAL_TOKENS = set(special_tokens)

    print("Building vocabulary from phonemes...")
    phoneme_counter = Counter()

    # Count phonemes in corpus
    for corpus_file in corpus_files:
        with open(corpus_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                clean_line = preprocess_line(line)
                if clean_line:
                    phonemes = splitter.split(clean_line)
                    phonemes = [p for p in phonemes if p not in SPECIAL_TOKENS]
                    phoneme_counter.update(phonemes)

    print(f"Found {len(phoneme_counter)} unique phonemes in corpus")

    # Base vocabulary: special tokens + phonemes
    base_vocab = special_tokens.copy()

    if use_csv_units and csv_path:
        df = pd.read_csv(csv_path)
        csv_phonemes = set(df[csv_column].dropna().astype(str).tolist())
        csv_phonemes = set([unicodedata.normalize("NFC", ph) for ph in csv_phonemes])
        base_vocab.extend(sorted(csv_phonemes))
        print(f"Added {len(csv_phonemes)} phonemes from CSV")

        corpus_phonemes = set(phoneme_counter.keys()) - SPECIAL_TOKENS
        missing_in_csv = corpus_phonemes - csv_phonemes

        if missing_in_csv:
            if handle_missing == "warn":
                print(f"Warning: {len(missing_in_csv)} phonemes in corpus not found in CSV: {missing_in_csv}")
            elif handle_missing == "add":
                print(f"Adding {len(missing_in_csv)} phonemes missing from CSV to vocabulary.")
                base_vocab.extend(sorted(missing_in_csv))
            elif handle_missing == "ignore":
                print(f"Ignoring {len(missing_in_csv)} phonemes missing from CSV.")
                # Drop them from the phoneme counter so they’re not used in the tokenizer
                for ph in missing_in_csv:
                    phoneme_counter.pop(ph, None)
            else:
                raise ValueError(f"Unknown handle_missing value: {handle_missing}")
    else:
        base_vocab.extend(sorted(phoneme_counter.keys()))
        print(f"Added {len(phoneme_counter)} phonemes from corpus")

    if add_frequent_combinations:
        print("Finding frequent phoneme combinations...")
        combinations = find_frequent_combinations(corpus_files, splitter, min_freq_for_combinations)
        base_vocab.extend(combinations)
        print(f"Added {len(combinations)} frequent combinations")

    vocab = {token: idx for idx, token in enumerate(base_vocab)}
    print(f"Final vocabulary size: {len(vocab)}")

    # Create temporary preprocessed corpus
    temp_files = []
    try:
        for corpus_file in corpus_files:
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8-sig')
            temp_files.append(temp_file.name)
            temp_file.close()

            with open(corpus_file, 'r', encoding='utf-8-sig') as infile, \
                 open(temp_file.name, 'w', encoding='utf-8-sig') as outfile:
                for line in infile:
                    clean_line = preprocess_line(line)
                    if clean_line:
                        phonemes = splitter.split(clean_line)
                        phonemes = [unicodedata.normalize("NFC", ph) for ph in phonemes]

                        if use_csv_units and handle_missing == "ignore":
                            phonemes = [ph for ph in phonemes if ph in csv_phonemes or ph in SPECIAL_TOKENS]

                        outfile.write(' '.join(phonemes) + '\n')

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[("[BOS]", vocab["[BOS]"]),
                            ("[EOS]", vocab["[EOS]"])]
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tokenizer.save(output_path)
        print(f"Tokenizer saved to {output_path}")

        vocab_info_path = output_path.replace('.json', '_vocab_info.txt')
        with open(vocab_info_path, 'w', encoding='utf-8-sig') as f:
            f.write(f"Vocabulary size: {len(vocab)}\n")
            f.write(f"Special tokens: {special_tokens}\n")
            f.write(f"Phonemes: {len(base_vocab) - len(special_tokens)}\n\n")
            f.write("Vocabulary:\n")
            for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{idx:4d}: {token}\n")
        print(f"Vocabulary info saved to {vocab_info_path}")

    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass


def find_frequent_combinations(corpus_files, splitter, min_freq):
    from collections import Counter
    bigram_counter = Counter()
    trigram_counter = Counter()

    for corpus_file in corpus_files:
        with open(corpus_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                clean_line = preprocess_line(line)
                if clean_line:
                    phonemes = splitter.split(clean_line)
                    for i in range(len(phonemes) - 1):
                        bigram = phonemes[i] + phonemes[i + 1]
                        bigram_counter[bigram] += 1
                    for i in range(len(phonemes) - 2):
                        trigram = phonemes[i] + phonemes[i + 1] + phonemes[i + 2]
                        trigram_counter[trigram] += 1

    frequent_combinations = [combo for combo, freq in bigram_counter.items() if freq >= min_freq]
    frequent_combinations += [combo for combo, freq in trigram_counter.items() if freq >= min_freq]
    return sorted(frequent_combinations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--phoneme_csv", type=str)
    parser.add_argument("--column", type=str)
    parser.add_argument("--fixed_vocab", action="store_true")
    parser.add_argument("--add_combinations", action="store_true")
    parser.add_argument("--min_combo_freq", type=int, default=10)
    parser.add_argument(
        "--handle_missing",
        type=str,
        choices=["ignore", "add"],
        default="warn",
        help="What to do with phonemes not in the CSV: 'ignore' (drop), 'add' (include them), or 'warn' (default)."
    )
    args = parser.parse_args()

    if args.fixed_vocab:
        create_fixed_vocabulary_tokenizer(
            corpus_files=[args.train_path],
            output_path=args.output_path,
            use_csv_units=bool(args.phoneme_csv),
            csv_path=args.phoneme_csv,
            csv_column=args.column,
            add_frequent_combinations=args.add_combinations,
            min_freq_for_combinations=args.min_combo_freq,
            handle_missing=args.handle_missing
        )
    else:
        print("BPE tokenizer training not updated for ID-stripping. Use --fixed_vocab for phoneme-level tokenizer.")