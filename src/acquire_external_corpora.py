# Project Nur — External Control Corpora Acquisition
# Bismillah
#
# Downloads and normalizes the four external control corpora
# for comparison against the Quran's mathematical signatures.

import json
import csv
import io
import re
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics for normalization."""
    if not text:
        return ""
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]')
    text = text.replace('\u0640', '')
    text = text.replace('\ufdfa', '')
    return diacritics.sub('', text).strip()


def download_file(url: str, save_path: Path, label: str = "") -> bool:
    """Download a file with progress indication."""
    if save_path.exists():
        print(f"  [OK] {label} already exists: {save_path}")
        return True
    
    print(f"  [DL] Downloading {label}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Project-Nur/1.0'})
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(data)
            print(f"  [OK] Saved {len(data):,} bytes to {save_path}")
            return True
    except Exception as e:
        print(f"  [ERR] Download failed: {e}")
        return False


def acquire_hadith(output_dir: Path) -> pd.DataFrame:
    """Download and parse Sahih al-Bukhari from HuggingFace."""
    print("\n" + "=" * 60)
    print("CORPUS: SAHIH AL-BUKHARI (Hadith)")
    print("=" * 60)
    
    url = "https://huggingface.co/datasets/meeAtif/hadith_datasets/resolve/main/Sahih%20al-Bukhari.json"
    json_path = output_dir / "bukhari_raw.json"
    
    if not download_file(url, json_path, "Sahih al-Bukhari"):
        return pd.DataFrame()
    
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    
    # Parse — it's a flat list of dicts with 'Arabic_Text' key
    records = []
    hadith_list = raw if isinstance(raw, list) else raw.get("hadiths", raw.get("data", []))
    
    for item in hadith_list:
        arabic = item.get("Arabic_Text", item.get("arabic", item.get("hadith_arabic", "")))
        if not arabic or not isinstance(arabic, str):
            continue
            
        arabic = strip_diacritics(arabic)
        if len(arabic.split()) < 3:
            continue
            
        records.append({
            "text_simple": arabic,
            "text_uthmani": arabic,
            "source": "bukhari",
            "id": item.get("Reference", len(records)),
        })
    
    df = pd.DataFrame(records)
    print(f"  Parsed: {len(df)} hadiths")
    print(f"  Total words: {df['text_simple'].str.split().str.len().sum():,}")
    
    # Save
    out_path = output_dir / "hadith_bukhari.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved to: {out_path}")
    return df


def acquire_poetry(output_dir: Path) -> pd.DataFrame:
    """Download and parse Pre-Islamic poetry from Tarab dataset."""
    print("\n" + "=" * 60)
    print("CORPUS: PRE-ISLAMIC ARABIC POETRY")
    print("=" * 60)
    
    # Try multiple poetry sources
    urls = [
        "https://huggingface.co/datasets/drelhaj/Tarab/resolve/main/tarab_full.csv",
        "https://huggingface.co/datasets/drelhaj/Tarab/resolve/main/Tarab_poetry.csv",
    ]
    csv_path = output_dir / "tarab_poetry.csv"
    
    downloaded = False
    for url in urls:
        if download_file(url, csv_path, f"Poetry from {url.split('/')[-1]}"):
            downloaded = True
            break
    
    if not downloaded:
        print("  [ERR] All poetry download URLs failed.")
        return pd.DataFrame()
    
    
    # Parse CSV — look for pre-Islamic verses
    records = []
    try:
        df_raw = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        print(f"  Raw CSV: {len(df_raw)} rows, columns: {list(df_raw.columns)}")
        
        # Find the text and era columns
        text_col = None
        era_col = None
        for col in df_raw.columns:
            col_lower = col.lower().strip()
            if "verse" in col_lower or "lyric" in col_lower or "text" in col_lower or "poem" in col_lower:
                text_col = col
            if "era" in col_lower or "origin" in col_lower or "age" in col_lower or "period" in col_lower:
                era_col = col
        
        if text_col is None:
            # Just use the first column that has Arabic text
            for col in df_raw.columns:
                sample = str(df_raw[col].iloc[0]) if len(df_raw) > 0 else ""
                if any('\u0600' <= c <= '\u06FF' for c in sample):
                    text_col = col
                    break
        
        print(f"  Text column: {text_col}")
        print(f"  Era column: {era_col}")
        
        if text_col is None:
            print("  [ERR] Could not find text column")
            return pd.DataFrame()
        
        # If we have era info, filter for pre-Islamic; otherwise take all
        if era_col:
            pre_islamic_keywords = ["جاهلي", "pre-islamic", "jahili", "Pre-Islamic", "Jahiliyyah"]
            mask = df_raw[era_col].astype(str).str.contains("|".join(pre_islamic_keywords), case=False, na=False)
            filtered = df_raw[mask]
            if len(filtered) < 100:
                print(f"  [WARN] Only {len(filtered)} pre-Islamic verses found. Using all poetry.")
                filtered = df_raw
        else:
            filtered = df_raw
        
        for _, row in filtered.iterrows():
            text = str(row[text_col])
            text = strip_diacritics(text)
            if len(text.split()) < 2:
                continue
            records.append({
                "text_simple": text,
                "text_uthmani": text,
                "source": "pre_islamic_poetry",
            })
            if len(records) >= 10000:
                break
                
    except Exception as e:
        print(f"  [ERR] Parse error: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    print(f"  Parsed: {len(df)} poetic verses")
    print(f"  Total words: {df['text_simple'].str.split().str.len().sum():,}")
    
    out_path = output_dir / "poetry_pre_islamic.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved to: {out_path}")
    return df


def acquire_bible(output_dir: Path) -> pd.DataFrame:
    """Download and parse Arabic Van Dyke Bible."""
    print("\n" + "=" * 60)
    print("CORPUS: ARABIC BIBLE (Van Dyke)")
    print("=" * 60)
    
    url = "https://downloads.biblesupersearch.com/bibles/json/bibles_json_6.0.zip"
    zip_path = output_dir / "bibles_json.zip"
    
    if not download_file(url, zip_path, "Bible SuperSearch JSON"):
        return pd.DataFrame()
    
    # Extract SVD (Smith Van Dyke) Arabic Bible from ZIP
    records = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            svd_files = [n for n in zf.namelist() if "svd" in n.lower() and n.endswith(".json")]
            print(f"  Found SVD files: {svd_files}")
            
            if not svd_files:
                print("  [ERR] No Arabic Bible found in archive")
                return pd.DataFrame()
            
            with zf.open(svd_files[0]) as f:
                bible_data = json.loads(f.read().decode("utf-8"))
            
            # Structure: {metadata: {...}, verses: [{book_name, book, chapter, verse, text}, ...]}
            verse_list = bible_data.get("verses", [])
            for item in verse_list:
                text = item.get("text", "")
                if not text:
                    continue
                text = strip_diacritics(re.sub(r'<[^>]+>', '', str(text)))
                if len(text.split()) >= 2:
                    records.append({
                        "text_simple": text,
                        "text_uthmani": text,
                        "source": "bible_svd",
                        "book": item.get("book_name", ""),
                        "chapter": item.get("chapter", 0),
                        "verse": item.get("verse", 0),
                    })
                            
    except Exception as e:
        print(f"  [ERR] Parse error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    print(f"  Parsed: {len(df)} verses")
    if len(df) > 0:
        print(f"  Total words: {df['text_simple'].str.split().str.len().sum():,}")
    
    out_path = output_dir / "bible_vandyke.parquet"
    if len(df) > 0:
        df.to_parquet(out_path, index=False)
        print(f"  Saved to: {out_path}")
    return df


def main():
    output_dir = Path("data/controls/external")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PROJECT NUR — EXTERNAL CONTROL CORPORA ACQUISITION")
    print("=" * 60)
    
    results = {}
    
    # 1. Hadith
    hadith_df = acquire_hadith(output_dir)
    results["hadith"] = len(hadith_df)
    
    # 2. Poetry
    poetry_df = acquire_poetry(output_dir)
    results["poetry"] = len(poetry_df)
    
    # 3. Bible
    bible_df = acquire_bible(output_dir)
    results["bible"] = len(bible_df)
    
    print("\n\n" + "=" * 60)
    print("ACQUISITION SUMMARY")
    print("=" * 60)
    for name, count in results.items():
        status = "OK" if count > 0 else "FAILED"
        print(f"  [{status}] {name}: {count} records")
    
    print("\nAcquisition complete.")


if __name__ == "__main__":
    main()
