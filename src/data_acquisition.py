# Project Nur — Quran Data Acquisition and Parsing
# Bismillah
#
# Downloads the Quran text from Tanzil.net in clean UTF-8 format
# and parses it into a structured dataset with verse-level metadata.

import os
import re
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from surah_metadata import SURAH_METADATA, RevelationPeriod, get_period_distribution


@dataclass
class Verse:
    """A single Quranic verse with full metadata."""
    surah_number: int
    verse_number: int
    text_uthmani: str              # Uthmani script (standard Mushaf)
    text_simple: str               # Simplified Arabic (no diacritics)
    surah_name_arabic: str
    surah_name_english: str
    surah_name_transliterated: str
    revelation_period: str         # early_meccan, middle_meccan, late_meccan, medinan
    meccan_or_medinan: str         # Meccan or Medinan (binary)
    chronological_order: int       # Surah's position in revelation chronology
    verse_id: str                  # Unique ID: "surah:verse" e.g. "2:255"


class QuranDataAcquisition:
    """
    Handles downloading, parsing, and structuring the Quranic text.
    
    Data source: Tanzil.net — the most widely used digital Quran resource,
    used by Quran.com and other major platforms.
    """

    # Primary source: risan/quran-json on GitHub (reliable, structured JSON)
    QURAN_JSON_URL = "https://raw.githubusercontent.com/risan/quran-json/master/data/quran.json"

    def __init__(self, data_dir: str = "data/quran"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.verses: List[Verse] = []

    def download_quran_text(self, force: bool = False) -> Path:
        """
        Download the Quran text from GitHub (risan/quran-json).
        
        Returns path to the JSON file.
        """
        json_path = self.data_dir / "quran.json"

        if json_path.exists() and not force:
            print(f"  [OK] Quran JSON already exists: {json_path}")
            return json_path

        print(f"  [DL] Downloading Quran text from GitHub...")
        try:
            response = requests.get(self.QURAN_JSON_URL, timeout=30)
            response.raise_for_status()
            json_path.write_text(response.text, encoding="utf-8")
            print(f"  [OK] Saved to {json_path} ({len(response.text)} chars)")
        except requests.RequestException as e:
            print(f"  [ERR] Download failed: {e}")
            print(f"  -> Manual download: {self.QURAN_JSON_URL}")
            print(f"  -> Save to: {json_path}")
            raise

        return json_path

    def parse_quran_json(self, filepath: Path) -> Dict[str, str]:
        """
        Parse the risan/quran-json format.
        
        Format: {"1": [{"chapter":1, "verse":1, "text":"..."},...], "2": [...]}
        
        Returns: dict mapping "surah:verse" -> text
        """
        raw = json.loads(filepath.read_text(encoding="utf-8"))
        verses = {}
        for surah_num_str, verse_list in raw.items():
            for verse_obj in verse_list:
                verse_id = f"{verse_obj['chapter']}:{verse_obj['verse']}"
                verses[verse_id] = verse_obj["text"].strip()
        return verses

    def build_dataset(self) -> pd.DataFrame:
        """
        Build the complete structured dataset.
        
        Downloads text if needed, parses it, enriches with metadata,
        and returns a pandas DataFrame.
        """
        print("Building Quranic dataset...")
        print("=" * 60)

        # Step 1: Download
        print("\n[1/4] Acquiring text data...")
        json_path = self.download_quran_text()

        # Step 2: Parse
        print("\n[2/4] Parsing JSON...")
        all_verses = self.parse_quran_json(json_path)
        print(f"  Parsed: {len(all_verses)} verses")

        # Step 3: Merge with metadata
        print("\n[3/4] Enriching with surah metadata...")
        self.verses = []

        for surah_num, surah_meta in SURAH_METADATA.items():
            for verse_num in range(1, surah_meta.verse_count + 1):
                verse_id = f"{surah_num}:{verse_num}"

                verse_text = all_verses.get(verse_id, "")

                if not verse_text:
                    print(f"  [WARN] Missing verse: {verse_id}")
                    continue

                verse = Verse(
                    surah_number=surah_num,
                    verse_number=verse_num,
                    text_uthmani=verse_text,
                    text_simple=verse_text,  # Same source; both columns populated
                    surah_name_arabic=surah_meta.name_arabic,
                    surah_name_english=surah_meta.name_english,
                    surah_name_transliterated=surah_meta.name_transliterated,
                    revelation_period=surah_meta.revelation_period.value,
                    meccan_or_medinan=surah_meta.meccan_or_medinan,
                    chronological_order=surah_meta.chronological_order,
                    verse_id=verse_id,
                )
                self.verses.append(verse)

        print(f"  Total verses assembled: {len(self.verses)}")

        # Step 4: Create DataFrame
        print("\n[4/4] Creating structured DataFrame...")
        df = pd.DataFrame([asdict(v) for v in self.verses])

        # Add computed columns
        df["text_length_chars"] = df["text_uthmani"].str.len()
        df["text_length_words"] = df["text_uthmani"].str.split().str.len()

        # Validate
        self._validate_dataset(df)

        # Save
        output_path = self.data_dir / "quran_dataset.parquet"
        df.to_parquet(output_path, index=False)
        print(f"\n  ✓ Dataset saved to: {output_path}")

        # Also save as CSV for human inspection
        csv_path = self.data_dir / "quran_dataset.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  ✓ CSV copy saved to: {csv_path}")

        self._print_summary(df)
        return df

    def _validate_dataset(self, df: pd.DataFrame):
        """Validate the assembled dataset for completeness."""
        print("\n  Validation:")

        # Check total verse count
        expected_total = sum(s.verse_count for s in SURAH_METADATA.values())
        actual_total = len(df)
        status = "✓" if actual_total == expected_total else "✗"
        print(f"    {status} Total verses: {actual_total} (expected {expected_total})")

        # Check no empty texts
        empty_uthmani = df["text_uthmani"].isna().sum() + (df["text_uthmani"] == "").sum()
        status = "✓" if empty_uthmani == 0 else "✗"
        print(f"    {status} Empty Uthmani texts: {empty_uthmani}")

        # Check all surahs present
        surahs_present = df["surah_number"].nunique()
        status = "✓" if surahs_present == 114 else "✗"
        print(f"    {status} Surahs present: {surahs_present} (expected 114)")

        # Check revelation period distribution
        period_counts = df.groupby("revelation_period").size()
        print(f"    Verses by revelation period:")
        for period in ["early_meccan", "middle_meccan", "late_meccan", "medinan"]:
            count = period_counts.get(period, 0)
            pct = count / len(df) * 100
            print(f"      {period}: {count} verses ({pct:.1f}%)")

    def _print_summary(self, df: pd.DataFrame):
        """Print a summary of the dataset."""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"  Total verses:        {len(df)}")
        print(f"  Total surahs:        {df['surah_number'].nunique()}")
        print(f"  Total words:         {df['text_length_words'].sum():,}")
        print(f"  Total characters:    {df['text_length_chars'].sum():,}")
        print(f"  Avg words/verse:     {df['text_length_words'].mean():.1f}")
        print(f"  Avg chars/verse:     {df['text_length_chars'].mean():.1f}")
        print(f"  Longest verse:       {df.loc[df['text_length_words'].idxmax(), 'verse_id']} "
              f"({df['text_length_words'].max()} words)")
        print(f"  Shortest verse:      {df.loc[df['text_length_words'].idxmin(), 'verse_id']} "
              f"({df['text_length_words'].min()} words)")
        print("=" * 60)


class ShuffledQuranGenerator:
    """
    Generate shuffled versions of the Quran for null hypothesis testing.
    
    Three levels of shuffling:
    1. Word-level: shuffle words within each verse (destroys syntax, preserves vocabulary per verse)
    2. Verse-level: shuffle verses within each surah (destroys verse order, preserves surah grouping)
    3. Surah-level: shuffle surahs (destroys surah order, preserves internal structure)
    """

    def __init__(self, quran_df: pd.DataFrame, seed: int = 42):
        self.quran_df = quran_df.copy()
        self.rng = np.random.default_rng(seed)

    def shuffle_words(self) -> pd.DataFrame:
        """Shuffle words within each verse."""
        df = self.quran_df.copy()
        df["text_uthmani"] = df["text_uthmani"].apply(self._shuffle_words_in_text)
        df["text_simple"] = df["text_simple"].apply(self._shuffle_words_in_text)
        return df

    def shuffle_verses(self) -> pd.DataFrame:
        """Shuffle verses within each surah."""
        df = self.quran_df.copy()
        shuffled_dfs = []
        for surah_num in range(1, 115):
            surah_df = df[df["surah_number"] == surah_num].copy()
            # Shuffle the text columns while keeping metadata fixed
            texts_uthmani = surah_df["text_uthmani"].values.copy()
            texts_simple = surah_df["text_simple"].values.copy()
            self.rng.shuffle(texts_uthmani)
            self.rng.shuffle(texts_simple)
            surah_df["text_uthmani"] = texts_uthmani
            surah_df["text_simple"] = texts_simple
            shuffled_dfs.append(surah_df)
        return pd.concat(shuffled_dfs, ignore_index=True)

    def shuffle_surahs(self) -> pd.DataFrame:
        """Shuffle surah order (keeping internal verse order intact)."""
        df = self.quran_df.copy()
        surah_groups = [group for _, group in df.groupby("surah_number")]
        self.rng.shuffle(surah_groups)
        return pd.concat(surah_groups, ignore_index=True)

    def _shuffle_words_in_text(self, text: str) -> str:
        """Shuffle the words in a single text string."""
        words = text.split()
        self.rng.shuffle(words)
        return " ".join(words)

    def generate_all_controls(self, output_dir: str = "data/controls/shuffled"):
        """Generate and save all shuffled controls."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Generating shuffled Quran controls...")

        # Word-level shuffle
        word_shuffled = self.shuffle_words()
        word_shuffled.to_parquet(output_path / "quran_shuffled_words.parquet", index=False)
        print(f"  ✓ Word-shuffled: {len(word_shuffled)} verses")

        # Verse-level shuffle
        verse_shuffled = self.shuffle_verses()
        verse_shuffled.to_parquet(output_path / "quran_shuffled_verses.parquet", index=False)
        print(f"  ✓ Verse-shuffled: {len(verse_shuffled)} verses")

        # Surah-level shuffle
        surah_shuffled = self.shuffle_surahs()
        surah_shuffled.to_parquet(output_path / "quran_shuffled_surahs.parquet", index=False)
        print(f"  ✓ Surah-shuffled: {len(surah_shuffled)} verses")

        print("  All shuffled controls generated.")


if __name__ == "__main__":
    # Build the Quran dataset
    acquisition = QuranDataAcquisition(data_dir="data/quran")
    df = acquisition.build_dataset()

    # Generate shuffled controls
    shuffler = ShuffledQuranGenerator(df)
    shuffler.generate_all_controls()
