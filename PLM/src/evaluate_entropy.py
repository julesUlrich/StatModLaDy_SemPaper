# evaluate_entropy.py

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from transformers import GPT2LMHeadModel
from phoneme_tokenizer import PhonemeSplitter
import glob
import re
import unicodedata
from train_model import CustomTokenizerWrapper, preprocess_line

def nfc(x):
    import unicodedata
    return unicodedata.normalize("NFC", x) if isinstance(x, str) else x

class EnhancedEntropyAnalyzer:
    def __init__(self, model_dir, tokenizer_path, overview_csv_path,
                 csv_path=None, csv_column=None, glottocode=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(self.device).eval()
        self.glottocode = glottocode or ""

        # tokenizer + splitter
        self.splitter = PhonemeSplitter(
            use_csv_units=bool(csv_path),
            csv_path=csv_path,
            csv_column=csv_column
        )
        self.tokenizer = CustomTokenizerWrapper(tokenizer_path, self.splitter)

        # load overview + build both maps
        self.overview_df = pd.read_csv(overview_csv_path)
        self.create_id_to_overview_mapping()
        self.create_text_to_overview_mapping()

        self.special_tokens = {
            self.tokenizer.pad_token,
            self.tokenizer.unk_token,
            self.tokenizer.bos_token,
            self.tokenizer.eos_token
        }
        vocab = self.tokenizer.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in vocab.items()}

    def create_text_to_overview_mapping(self):
        """Map cleaned IPA/orthography -> overview id (string)."""
        self.text_to_overview = {}
        for _, row in self.overview_df.iterrows():
            oid = str(row["id"]).strip()
            # IPA
            if pd.notna(row.get("ipa")):
                ipa = str(row["ipa"]).strip()
                if ipa:
                    self.text_to_overview[ipa] = oid
                    self.text_to_overview[unicodedata.normalize("NFC", ipa)] = oid
            # Orthography
            if pd.notna(row.get("orthography")):
                orth = str(row["orthography"]).strip()
                if orth:
                    self.text_to_overview[orth] = oid
                    self.text_to_overview[unicodedata.normalize("NFC", orth)] = oid

    @staticmethod
    def parse_id_and_text(line: str):
        """Return (overview_id or None, raw_text). Lines may be 'ID\\tTEXT' or just TEXT."""
        if "\t" in line:
            id_part, text_part = line.split("\t", 1)
            id_part = id_part.strip()
            return (id_part if id_part and id_part != "NA" else None, text_part.strip())
        return (None, line.strip())

    def create_id_to_overview_mapping(self):
        """Create mapping from original overview 'id' to metadata (incl. age/sex)."""
        self.id_to_overview = {}
        for idx, row in self.overview_df.iterrows():
            overview_id = str(row['id']).strip()
            age = row['age'] if 'age' in row and pd.notna(row['age']) else ""
            sex = row['sex'] if 'sex' in row and pd.notna(row['sex']) else ""
            # Cast safely to strings (avoid NaN)
            age_str = str(int(age)) if isinstance(age, (int, float)) and not pd.isna(age) and float(
                age).is_integer() else str(age)
            sex_str = str(sex)

            self.id_to_overview[overview_id] = {
                'orthography': str(row['orthography']) if pd.notna(row['orthography']) else "",
                'ref': str(row['ref']) if pd.notna(row['ref']) else "",
                'speaker': str(row['speaker']) if pd.notna(row['speaker']) else "",
                'conversation_id': str(row.get('conversation_id', '')) if pd.notna(row.get('conversation_id')) else "",
                'split': str(row.get('split', '')) if pd.notna(row.get('split')) else "",
                'age': age_str,
                'sex': sex_str,
                'original_overview_id': overview_id,
                'overview_row_index': idx
            }

    def clean_utterance(self, text):
        """Remove BOS/EOS tokens from utterance text"""
        text = text.strip()
        if text.startswith('[BOS]'):
            text = text[5:]
        if text.endswith('[EOS]'):
            text = text[:-5]
        return text.strip()

    def get_orthographic_info_by_id(self, utterance_id):
        return self.id_to_overview.get(str(utterance_id), {
            'orthography': '',
            'ref': '',
            'speaker': '',
            'conversation_id': '',
            'split': '',
            'age': '',
            'sex': '',
            'original_overview_id': utterance_id,
            'overview_row_index': -1
        })

    def parse_filename(self, filename):
        """Extract speaker and conversation from test filename"""
        # Expected format: test_{speaker}_{conversation}.txt
        basename = os.path.basename(filename)
        match = re.match(r'test_([^_]+)_(.+)\.txt', basename)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def analyze_test_files(self, test_files_dir, max_context_length, output_dir,
                           no_plots=False, exclude_special_from_csv=False):
        # Find all test files
        test_files = glob.glob(os.path.join(test_files_dir, "test_*.txt"))
        if not test_files:
            raise ValueError(f"No test files found in {test_files_dir}")
        print(f"Found {len(test_files)} test files to analyze")

        # Organize by speaker
        speaker_files = defaultdict(list)
        for file_path in test_files:
            speaker, conversation = self.parse_filename(file_path)
            if speaker and conversation:
                speaker_files[speaker].append((file_path, conversation))
            else:
                print(f"Warning: Could not parse filename {file_path}")

        # Global statistics
        global_entropy_data = []
        global_phonemes = []
        global_entropies = []
        global_context_entropy = defaultdict(lambda: defaultdict(list))
        global_sentence_counter = 0

        # Process each speaker
        for speaker_id, files_and_convs in speaker_files.items():
            print(f"\nProcessing speaker: {speaker_id}")
            speaker_output_dir = os.path.join(output_dir, speaker_id)
            os.makedirs(speaker_output_dir, exist_ok=True)

            speaker_entropy_data = []
            speaker_phonemes = []
            speaker_entropies = []
            speaker_context_entropy = defaultdict(lambda: defaultdict(list))

            # Process each conversation
            for file_path, conversation_id in files_and_convs:
                print(f"  Analyzing conversation: {conversation_id}")

                conv_entropy_data = []
                conv_phonemes = []
                conv_entropies = []
                conv_context_entropy = defaultdict(lambda: defaultdict(list))

                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    lines = [line.strip() for line in f if line.strip()]

                for sentence_idx, line in enumerate(tqdm(lines, desc=f"Analyzing {conversation_id}", leave=False)):
                    line_id, raw_line = self.parse_id_and_text(line)

                    if line_id is not None:
                        ortho_info = self.get_orthographic_info_by_id(line_id)
                        global_id = ortho_info["original_overview_id"]
                    else:
                        # Fall back: strip BOS/EOS and NFC-normalize, try text lookup
                        cleaned = self.clean_utterance(raw_line)
                        nfc = unicodedata.normalize("NFC", cleaned)
                        maybe_id = self.text_to_overview.get(nfc)
                        if maybe_id is None:
                            maybe_id = self.text_to_overview.get(cleaned)
                        if maybe_id is not None:
                            ortho_info = self.get_orthographic_info_by_id(maybe_id)
                            global_id = ortho_info["original_overview_id"]
                        else:
                            ortho_info = self.get_orthographic_info_by_id("")  # fallback with blanks
                            global_id = ""

                    processed_line = preprocess_line(raw_line)  # same preprocessing as training
                    phonemes = self.splitter.split(processed_line)
                    joined = " ".join(phonemes)

                    encoded = self.tokenizer(joined)
                    padded = self.tokenizer.pad([encoded], return_tensors="pt")
                    input_ids = padded["input_ids"].to(self.device)

                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        logits = outputs.logits[0, :-1]
                        targets = input_ids[0, 1:]

                        sentence_entropies = []
                        sentence_phonemes_clean = []

                        for i, target_id in enumerate(targets):
                            token_str = self.id_to_token.get(int(target_id), "[UNK]")
                            probs = torch.nn.functional.softmax(logits[i], dim=-1)
                            p = probs.clamp_min(1e-12)  # numerical safety (avoid log(0))
                            entropy_bits = float((-p * p.log2()).sum().item())
                            is_special = token_str in self.special_tokens

                            entropy_entry = {
                                "glottocode": self.glottocode,
                                "speaker": speaker_id,
                                "conversation": conversation_id,
                                "conversation_sentence_id": sentence_idx,
                                "language_sentence_id": global_sentence_counter,
                                "global_id": global_id,
                                "position": i,
                                "phoneme": token_str,
                                "entropy": entropy_bits,
                                "orthography": ortho_info.get("orthography", ""),
                                "ref": ortho_info.get("ref", ""),
                                "age": ortho_info.get("age", ""),
                                "sex": ortho_info.get("sex", ""),
                            }

                            if not (exclude_special_from_csv and is_special):
                                conv_entropy_data.append(entropy_entry)
                                speaker_entropy_data.append(entropy_entry)
                                global_entropy_data.append(entropy_entry)

                            if not is_special:
                                conv_phonemes.append(token_str)
                                speaker_phonemes.append(token_str)
                                global_phonemes.append(token_str)

                                conv_entropies.append(entropy_bits)
                                speaker_entropies.append(entropy_bits)
                                global_entropies.append(entropy_bits)

                                sentence_entropies.append(entropy_bits)
                                sentence_phonemes_clean.append(token_str)

                                for context_len in range(0, min(i, max_context_length) + 1):
                                    conv_context_entropy[context_len][token_str].append(entropy_bits)
                                    speaker_context_entropy[context_len][token_str].append(entropy_bits)
                                    global_context_entropy[context_len][token_str].append(entropy_bits)

                    global_sentence_counter += 1

                    if not no_plots and sentence_entropies:
                        plot_filename = f"entropy_sentence_{speaker_id}_{conversation_id}_{sentence_idx:04d}.png"
                        plot_path = os.path.join(speaker_output_dir, plot_filename)
                        self.plot_entropy_progression(
                            entropies=sentence_entropies,
                            phonemes=sentence_phonemes_clean,
                            ipa_text=processed_line,
                            orthographic_text=ortho_info['orthography'],
                            speaker_id=speaker_id,
                            conversation_id=conversation_id,
                            sentence_id=sentence_idx,
                            path=plot_path
                        )

                # Conversation-level outputs
                if conv_entropies:
                    conv_stats = self.compute_statistics(conv_entropies, conv_phonemes, conv_context_entropy)
                    # Pull age/sex for this conversation from the rows we just wrote
                    ages = [r["age"] for r in conv_entropy_data if r.get("age")]
                    sexes = [r["sex"] for r in conv_entropy_data if r.get("sex")]
                    conv_meta = {
                        "glottocode": self.glottocode,
                        "speaker": speaker_id,
                        "conversation": conversation_id,
                        "age": ages[0] if ages else "",
                        "sex": sexes[0] if sexes else ""
                    }
                    conv_stats_out = {"metadata": conv_meta, **conv_stats}
                    with open(os.path.join(speaker_output_dir,
                                           f"conversation_{conversation_id}_entropy_stats.json"), 'w') as f:
                        json.dump(conv_stats_out, f, indent=2)

                if conv_entropy_data:
                    pd.DataFrame(conv_entropy_data).to_csv(
                        os.path.join(speaker_output_dir, f"conversation_{conversation_id}_phoneme_entropies.csv"),
                        index=False
                    )

            # Speaker-level outputs
            if speaker_entropies:
                spk_ages = [r["age"] for r in speaker_entropy_data if r.get("age")]
                spk_sexes = [r["sex"] for r in speaker_entropy_data if r.get("sex")]
                speaker_stats = self.compute_statistics(speaker_entropies, speaker_phonemes, speaker_context_entropy)
                speaker_stats_out = {
                    "metadata": {
                        "glottocode": self.glottocode,
                        "speaker": speaker_id,
                        "age": spk_ages[0] if spk_ages else "",
                        "sex": spk_sexes[0] if spk_sexes else ""
                    },
                    **speaker_stats
                }
                with open(os.path.join(speaker_output_dir, f"speaker_{speaker_id}_entropy_stats.json"), 'w') as f:
                    json.dump(speaker_stats_out, f, indent=2)

            if speaker_entropy_data:
                pd.DataFrame(speaker_entropy_data).to_csv(
                    os.path.join(speaker_output_dir, f"speaker_{speaker_id}_phoneme_entropies.csv"),
                    index=False
                )

        # Global outputs
        os.makedirs(output_dir, exist_ok=True)
        if global_entropies:
            global_stats = self.compute_statistics(global_entropies, global_phonemes, global_context_entropy)
            global_stats["glottocode"] = self.glottocode
            with open(os.path.join(output_dir, "language_entropy_stats.json"), 'w') as f:
                json.dump(global_stats, f, indent=2)

        if global_entropy_data:
            pd.DataFrame(global_entropy_data).to_csv(
                os.path.join(output_dir, "all_phoneme_entropies.csv"),
                index=False
            )

        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"Total speakers analyzed: {len(speaker_files)}")
        print(f"Total rows in all_phoneme_entropies.csv: {len(global_entropy_data)}")

    def analyze_train_eval_fallback(self, data_dir: str, output_dir: str,
                                    exclude_special_from_csv: bool = True,
                                    max_context_length: int = 8):
        """
        Build a fallback table from train + eval splits.
        Output: ent_train_eval.csv with columns:
          glottocode, speaker, conversation, phoneme, entropy_rate, count
        """
        train_path = os.path.join(data_dir, "train.txt")
        eval_path = os.path.join(data_dir, "eval.txt")
        files = [p for p in [train_path, eval_path] if os.path.exists(p)]
        if not files:
            print("No train/eval files found for fallback (skipping ent_train_eval.csv).")
            return

        print(f"\nBuilding fallback ENT from {len(files)} file(s): {[os.path.basename(p) for p in files]}")

        # Aggregators keyed by (speaker, conversation, phoneme)
        from collections import defaultdict
        buckets = defaultdict(list)

        def process_line(line: str):
            # Use the same parsing as in test analysis
            line_id, raw_text = self.parse_id_and_text(line)

            # We *require* an ID here, because we need speaker/conversation from the overview mapping
            if line_id is None:
                return  # cannot place this line into a (speaker, conversation) bucket

            meta = self.get_orthographic_info_by_id(line_id)
            spk = meta.get("speaker") or ""
            conv = meta.get("conversation_id") or ""
            if not spk or not conv:
                # If conversation is missing in overview, we skip (can't aggregate properly)
                return

            # Same preprocessing as training
            processed = preprocess_line(raw_text)
            phonemes = self.splitter.split(processed)
            joined = " ".join(phonemes)

            encoded = self.tokenizer(joined)
            padded = self.tokenizer.pad([encoded], return_tensors="pt")
            input_ids = padded["input_ids"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, :-1]
                targets = input_ids[0, 1:]

                for i, target_id in enumerate(targets):
                    token_str = self.id_to_token.get(int(target_id), "[UNK]")
                    is_special = token_str in self.special_tokens
                    if exclude_special_from_csv and is_special:
                        continue

                    # Only count non-special tokens
                    if not is_special:
                        probs = torch.nn.functional.softmax(logits[i], dim=-1)
                        p = probs.clamp_min(1e-12)  # numerical safety (avoid log(0))
                        entropy_bits = float((-p * p.log2()).sum().item())
                        buckets[(spk, conv, token_str)].append(entropy_bits)

        # Stream both files
        for path in files:
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    process_line(line)

        # Aggregate mean ENT per (speaker, conversation, phoneme)
        rows = []
        for (spk, conv, ph), ent_list in buckets.items():
            if not ent_list:
                continue

            # Try to pull age/sex from any row in this conversation (via overview id lookup).
            # We don’t have an id here, but age/sex are constant per speaker in your overview,
            # so we can grab via any row for that (speaker, conv). We'll do a cheap lookup:
            o = self.overview_df
            spk_n = nfc(spk)
            conv_n = nfc(conv)

            col_speaker = o["speaker"].astype(str).map(nfc)
            col_conv = o.get("conversation_id", pd.Series([""] * len(o))).astype(str).map(nfc)

            cand = o[(col_speaker == spk_n) & (col_conv == conv_n)]

            age = ""
            sex = ""
            if not cand.empty:
                a = cand.iloc[0].get("age", "")
                s = cand.iloc[0].get("sex", "")
                try:
                    age = str(int(a)) if pd.notna(a) and float(a) == int(a) else str(a)
                except Exception:
                    age = "" if pd.isna(a) else str(a)
                sex = "" if pd.isna(s) else str(s)

            rows.append({
                "glottocode": self.glottocode,
                "speaker": spk,
                "conversation": conv,
                "phoneme": ph,
                "entropy_rate": float(np.mean(ent_list)),
                "count": int(len(ent_list)),
                "age": age,
                "sex": sex,
            })

        out_path = os.path.join(output_dir, "ent_train_eval.csv")
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Fallback ENT table written to: {out_path}  ({len(rows)} rows)")

    def compute_statistics(self, entropies, phonemes, context_entropy):
        """Compute comprehensive statistics for a set of entropies"""
        # Convert phoneme counts to regular Python integers for JSON serialization
        phoneme_counts = pd.Series(phonemes).value_counts()
        phoneme_counts_dict = {str(k): int(v) for k, v in phoneme_counts.items()}

        stats = {
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "median_entropy": float(np.median(entropies)),
            "min_entropy": float(np.min(entropies)),
            "max_entropy": float(np.max(entropies)),
            "total_phonemes": int(len(phonemes)),
            "unique_phonemes": int(len(set(phonemes))),
            "phoneme_counts": phoneme_counts_dict
        }

        # Context-specific statistics
        context_stats = {}
        for context_len, entries in context_entropy.items():
            context_stats[f"context_{context_len}"] = {
                phoneme: {
                    "mean_entropy": float(np.mean(entropies)),
                    "std_entropy": float(np.std(entropies)),
                    "count": int(len(entropies))
                }
                for phoneme, entropies in entries.items()
            }

        stats["context_entropy_stats"] = context_stats
        return stats

    def plot_entropy_progression(self, entropies, phonemes, ipa_text, orthographic_text,
                                 speaker_id, conversation_id, sentence_id, path):
        """Enhanced plot with both IPA and orthographic text"""
        plt.figure(figsize=(12, 6), dpi=600)

        line_colour = "#0D47A1"
        mean_colour = "#FF9A6B"

        # Main entropy plot with phonemes as x-axis labels
        plt.plot(range(len(entropies)), entropies, marker="o", linewidth=1.5, color=line_colour, zorder=3)

        plt.title(f"Entropy Progression - Speaker: {speaker_id}, Conv: {conversation_id}, Sentence: {sentence_id}")
        plt.xlabel("Phoneme")
        plt.ylabel("Entropy (bits)")
        plt.gca().set_facecolor('#F0F0F0')
        plt.grid(True, alpha=1, linestyle='-', linewidth=0.8, color='white', zorder=1)

        # Set phonemes as x-axis labels
        plt.xticks(range(len(phonemes)), phonemes, rotation=45, ha='right')

        # Add mean entropy line
        mean_entropy = sum(entropies)/len(entropies)
        plt.axhline(y=mean_entropy, color=mean_colour, linestyle='--', alpha=0.7,
                    label=f'Mean: {mean_entropy:.2f} bits')
        plt.legend()

        # Add text information at the bottom
        plt.figtext(0.02, 0.02, f"IPA: {ipa_text[:100]}{'...' if len(ipa_text) > 100 else ''}",
                    fontsize=8, ha='left')
        plt.figtext(0.02, 0.05,
                    f"Orthography: {orthographic_text[:100]}{'...' if len(orthographic_text) > 100 else ''}",
                    fontsize=8, ha='left')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for text
        plt.savefig(path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced entropy analysis with speaker/conversation breakdown")
    parser.add_argument("--test_files_dir", type=str, required=True,
                        help="Directory containing test_*.txt files")
    parser.add_argument("--overview_csv", type=str, required=True,
                        help="Path to enhanced overview CSV file with split information")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained model")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer file")
    parser.add_argument("--csv_path", type=str,
                        help="Optional CSV file with phoneme inventory")
    parser.add_argument("--csv_column", type=str,
                        help="Column in CSV to use for phonemes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save entropy analysis results")
    parser.add_argument("--max_context_length", type=int, default=8,
                        help="Maximum context length for analysis")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating individual sentence plots")
    parser.add_argument("--exclude_special_from_csv", action="store_true",
                        help="Exclude special tokens ([PAD], [UNK], etc.) from CSV output")
    parser.add_argument("--glottocode", type=str, required=True,
                        help="Language code (e.g., bora1263) to store in outputs")
    parser.add_argument("--build_fallback", action="store_true",
                        help="Also compute ent_train_eval.csv from train+eval.")

    args = parser.parse_args()

    analyzer = EnhancedEntropyAnalyzer(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        overview_csv_path=args.overview_csv,
        csv_path=args.csv_path,
        csv_column=args.csv_column,
        glottocode=args.glottocode
    )

    analyzer.analyze_test_files(
        test_files_dir=args.test_files_dir,
        max_context_length=args.max_context_length,
        output_dir=args.output_dir,
        no_plots=args.no_plots,
        exclude_special_from_csv=args.exclude_special_from_csv
    )

    if args.build_fallback:
        analyzer.analyze_train_eval_fallback(
            data_dir=args.test_files_dir,
            output_dir=args.output_dir,
            exclude_special_from_csv=args.exclude_special_from_csv,
            max_context_length=args.max_context_length
        )