# Project Nur — Cross-Corpus Comparison: Phase 3 Metrics on External Corpora
# Bismillah
#
# Runs the SAME Phase 3 analysis (perplexity, attention, compression)
# on the external corpora and compares against the Quran.

import sys
sys.path.insert(0, "src")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from phase3_information_theory import TransformerInternalsAnalyzer, TextIntrinsicAnalyzer


def run_cross_corpus_comparison():
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)

    # Load all corpora
    corpora = {}
    
    # Quran
    quran_df = pd.read_parquet("data/quran/quran_dataset.parquet")
    corpora["QURAN"] = quran_df["text_uthmani"].tolist()
    
    # External
    for name, path in [
        ("HADITH_BUKHARI", "data/controls/external/hadith_bukhari.parquet"),
        ("POETRY_JAHILI", "data/controls/external/poetry_pre_islamic.parquet"),
        ("BIBLE_VANDYKE", "data/controls/external/bible_vandyke.parquet"),
    ]:
        p = Path(path)
        if p.exists():
            df = pd.read_parquet(p)
            corpora[name] = df["text_simple"].tolist()
            print(f"  Loaded {name}: {len(corpora[name])} texts")
    
    # Shuffled control
    shuf_path = Path("data/controls/shuffled/quran_shuffled_words.parquet")
    if shuf_path.exists():
        df = pd.read_parquet(shuf_path)
        corpora["QURAN_SHUFFLED"] = df["text_uthmani"].tolist()

    results = {}

    # ── Part A: Text-Intrinsic ──
    print("\n\n### PART A: TEXT-INTRINSIC METRICS ###")
    intrinsic = TextIntrinsicAnalyzer()

    for name, texts in corpora.items():
        full_text = " ".join(texts)
        char_ent = intrinsic.character_entropy(full_text)
        word_ent = intrinsic.word_entropy(full_text)
        compress = intrinsic.compressibility(full_text)
        zipf = intrinsic.zipf_analysis(full_text)
        mi_50 = intrinsic.mutual_information_distance(full_text, window=50)
        mi_200 = intrinsic.mutual_information_distance(full_text, window=200)

        results[f"{name}_intrinsic"] = {
            "char_entropy": char_ent,
            "word_entropy": word_ent,
            "gzip_ratio": compress["gzip_ratio"],
            "bz2_ratio": compress["bz2_ratio"],
            "lzma_ratio": compress["lzma_ratio"],
            "zipf_exponent": zipf["zipf_exponent"],
            "zipf_r_squared": zipf["zipf_r_squared"],
            "vocabulary_size": zipf["vocabulary_size"],
            "type_token_ratio": zipf["type_token_ratio"],
            "mutual_info_w50": mi_50,
            "mutual_info_w200": mi_200,
        }
        print(f"\n  [{name}]")
        print(f"    Char entropy:   {char_ent:.4f}")
        print(f"    gzip ratio:     {compress['gzip_ratio']:.4f}")
        print(f"    bz2 ratio:      {compress['bz2_ratio']:.4f}")
        print(f"    Zipf exp:       {zipf['zipf_exponent']:.4f} (R²={zipf['zipf_r_squared']:.4f})")
        print(f"    Vocab size:     {zipf['vocabulary_size']}")
        print(f"    TTR:            {zipf['type_token_ratio']:.4f}")
        print(f"    MI(w=50):       {mi_50:.4f}")

    # ── Part B: Transformer Internals ──
    print("\n\n### PART B: TRANSFORMER INTERNALS ###")
    transformer = TransformerInternalsAnalyzer()

    for name, texts in corpora.items():
        print(f"\n{'=' * 50}")
        print(f"  CORPUS: {name}")
        print(f"{'=' * 50}")

        ppl = transformer.compute_pseudo_perplexity(texts, name)
        results[f"{name}_perplexity"] = ppl
        print(f"  PSEUDO-PERPLEXITY: {ppl['pseudo_perplexity']:.2f}")

        att = transformer.analyze_attention_patterns(texts, name)
        results[f"{name}_attention"] = att
        print(f"  ATTENTION ENTROPY: {att['mean_attention_entropy']:.4f}")
        print(f"  LONG-RANGE ATT:    {att['long_range_attention_ratio']:.4f}")

    # ── Part C: Comparative Table ──
    print(f"\n\n{'=' * 90}")
    print("CROSS-CORPUS COMPARISON — FULL TABLE")
    print(f"{'=' * 90}")

    corpus_names = list(corpora.keys())
    header = f"{'Metric':<25}" + "".join(f"{n:>15}" for n in corpus_names)
    print(f"\n{header}")
    print("-" * (25 + 15 * len(corpus_names)))

    metrics = [
        ("Char Entropy", "intrinsic", "char_entropy"),
        ("gzip Ratio", "intrinsic", "gzip_ratio"),
        ("bz2 Ratio", "intrinsic", "bz2_ratio"),
        ("Zipf Exponent", "intrinsic", "zipf_exponent"),
        ("Zipf R²", "intrinsic", "zipf_r_squared"),
        ("Vocab Size", "intrinsic", "vocabulary_size"),
        ("TTR", "intrinsic", "type_token_ratio"),
        ("MI (w=50)", "intrinsic", "mutual_info_w50"),
        ("Pseudo-Perplexity", "perplexity", "pseudo_perplexity"),
        ("Attention Entropy", "attention", "mean_attention_entropy"),
        ("Long-Range Att.", "attention", "long_range_attention_ratio"),
    ]

    for label, suffix, key in metrics:
        vals = []
        for c in corpus_names:
            r = results.get(f"{c}_{suffix}", {})
            v = r.get(key, 0)
            vals.append(v)
        
        if any(v > 100 for v in vals):
            row = f"{label:<25}" + "".join(f"{v:>15.1f}" for v in vals)
        else:
            row = f"{label:<25}" + "".join(f"{v:>15.4f}" for v in vals)
        print(row)

    # Save
    def clean(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    out = results_dir / "cross_corpus_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(clean(results), f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved to {out}")


if __name__ == "__main__":
    run_cross_corpus_comparison()
