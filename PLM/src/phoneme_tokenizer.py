# src/phoneme_tokenizer.py

import unicodedata
import pandas as pd
from typing import List, Set, Optional


class PhonemeSplitter:
    def __init__(self,
                 use_csv_units: bool = False,
                 csv_path: Optional[str] = None,
                 csv_column: Optional[str] = None):
        """
        Phoneme splitter that supports custom rules and optional unit loading from CSV.

        Args:
            use_csv_units: Whether to use units loaded from CSV.
            csv_path: Path to the CSV file.
            csv_column: Name of the column with phoneme units.
        """
        self.use_csv_units = use_csv_units
        self.csv_units = []

        if use_csv_units:
            assert csv_path and csv_column, "CSV path and column must be specified if use_csv_units is True"
            self.load_units_from_csv(csv_path, csv_column)

        self.suffix_symbols = {'ː', 'ʲ', 'ʰ'}  # ː ʲ ʰ
        self.prefix_symbols = {'ˈ', 'ˌ'}  # ˈ ˌ
        self.diphthong_patterns = {
            'aːɪ', 'eːɪ', 'oːɪ', 'aːʊ',
            'eːʊ', 'ɪə', 'ʊə', 'eə'
        }

    def load_units_from_csv(self, path: str, column: str) -> None:
        df = pd.read_csv(path)
        units = df[column].dropna().astype(str).tolist()

        # 🔧 Normalize units to NFC here
        units = [unicodedata.normalize("NFC", u) for u in units if u.strip()]

        # Sort by length for greedy matching
        self.csv_units = sorted(units, key=len, reverse=True)

    def _is_modifier_or_diacritic(self, char: str) -> bool:
        code = ord(char)
        return unicodedata.combining(char) > 0 or (0x02B0 <= code <= 0x02FF)

    def split(self, text: str) -> List[str]:
        if not text:
            return []

        text = unicodedata.normalize("NFC", text)

        special_tokens = {"[BOS]", "[EOS]", "[PAD]", "[UNK]"}
        out: List[str] = []
        i = 0

        while i < len(text):
            # 1) If a special token starts at i, take it whole.
            matched_special = None
            for tok in special_tokens:
                if text.startswith(tok, i):
                    matched_special = tok
                    break

            if matched_special:
                out.append(matched_special)
                i += len(matched_special)
                continue

            # 2) Otherwise, gather the longest span until the next special token (or end),
            #    and split *that chunk* using CSV/rule logic.
            j = i
            while j < len(text) and not any(text.startswith(tok, j) for tok in special_tokens):
                j += 1

            chunk = text[i:j]  # non-special span
            if chunk:
                if self.use_csv_units:
                    out.extend(self._csv_based_split(chunk))
                else:
                    out.extend(self._rule_based_split(chunk))

            i = j

        return out

    def _csv_based_split(self, text: str) -> List[str]:
        text = unicodedata.normalize("NFC", text)  # Normalize whole string first
        result = []
        i = 0
        while i < len(text):
            matched = False
            for unit in self.csv_units:
                slice_candidate = text[i:i + len(unit)]
                # Normalize slice to NFC before comparing
                if unicodedata.normalize("NFC", slice_candidate) == unit:
                    result.append(unit)
                    i += len(slice_candidate)
                    matched = True
                    break
            if not matched:
                char = text[i]
                # Attach combining marks to previous phoneme
                if unicodedata.combining(char) and result:
                    result[-1] += char
                else:
                    result.append(char)
                i += 1
        return result
    def _rule_based_split(self, text: str) -> List[str]:
        result = []
        i = 0

        while i < len(text):
            char = text[i]

            # Check for diphthong patterns first (greedy matching)
            diphthong_found = False
            for pattern in sorted(self.diphthong_patterns, key=len, reverse=True):
                if text[i:i + len(pattern)] == pattern:
                    result.append(pattern)
                    i += len(pattern)
                    diphthong_found = True
                    break
            if diphthong_found:
                continue

            # Diacritic or modifier letter - attach to previous character
            if self._is_modifier_or_diacritic(char):
                if result:
                    result[-1] += char
                else:
                    result.append(char)
                i += 1
                continue

            # Prefix symbols attach to next character
            if char in self.prefix_symbols:
                if i + 1 < len(text):
                    # Look ahead to get the next non-diacritic character
                    next_char = text[i + 1]
                    combined = char + next_char
                    result.append(combined)
                    i += 2
                    # Check if there are diacritics following the next character
                    while i < len(text) and self._is_modifier_or_diacritic(text[i]):
                        result[-1] += text[i]
                        i += 1
                else:
                    result.append(char)
                    i += 1
                continue

            # Suffix symbols attach to previous character
            if char in self.suffix_symbols:
                if result:
                    result[-1] += char
                else:
                    result.append(char)
                i += 1
                continue

            # Regular character - stands alone
            result.append(char)
            i += 1

        return [token for token in result if token]  # Filter out empty tokens