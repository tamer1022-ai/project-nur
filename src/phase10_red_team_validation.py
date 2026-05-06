# Project Nur — Phase 10: Red Team Validation
# Bismillah
#
# Addresses the 4 CRITICAL gaps identified in the epistemological red-team:
# 1. Uthmani vs Simple script confound
# 2. Tokens-per-word ratio confound
# 3. Length vs Theme regression
# 4. Verse-aligned vs window-aligned PPL

import numpy as np
import pandas as pd
import json
import math
import re
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.stats import spearmanr, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")


def strip_diacritics(text):
    return re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]').sub(
        '', text.replace('\u0640', '')).strip()


MAKKI = {1,6,7,10,11,12,13,14,15,16,17,18,19,20,21,23,25,26,27,28,29,30,
    31,32,34,35,36,37,38,39,40,41,42,43,44,45,46,50,51,52,53,54,55,56,67,68,69,
    70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,
    95,96,97,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114}

THEMES = {
    "aqeedah": [1,112,113,114,109,111,108,105,106,107,102,103,104,97,95,93,94,110,101,100,99],
    "eschatology": [81,82,84,88,69,56,78,79,75,83,99,101,44,54],
    "narrative": [12,18,28,20,7,11,26,27,21,37,71],
    "legislation": [2,4,5,24,33,65,66,49,58,60],
    "exhortation": [31,39,35,16,6,10,13,14,45,46,29,30,34,42,43],
}

SURAH_THEME = {}
for theme, surahs in THEMES.items():
    for s in surahs:
        SURAH_THEME[s] = theme


def compute_ppl_for_text(text, tokenizer, model, device, mask_id, max_masks=8):
    """Compute PPL for a single text string."""
    text = strip_diacritics(text)
    words = text.split()
    if len(words) < 3:
        return float('nan'), 0

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device)
    seq = ids.shape[1]
    n_tokens = seq - 2  # exclude [CLS] and [SEP]

    if seq <= 2:
        return float('nan'), n_tokens

    positions = list(range(1, seq - 1))
    if len(positions) > max_masks:
        positions = np.random.choice(positions, max_masks, replace=False).tolist()

    losses = []
    with torch.no_grad():
        for pos in positions:
            masked = ids.clone()
            true_tok = ids[0, pos].item()
            masked[0, pos] = mask_id
            out = model(masked, attention_mask=att)
            probs = torch.softmax(out.logits[0, pos], dim=-1)
            p = probs[true_tok].item()
            if p > 0:
                losses.append(-math.log(p))

    if not losses:
        return float('nan'), n_tokens
    return float(np.exp(np.mean(losses))), n_tokens


def extract_windows(text, window_size=15):
    """Extract fixed-size windows from text."""
    words = strip_diacritics(text).split()
    windows = []
    for i in range(0, len(words) - window_size + 1, window_size):
        w = " ".join(words[i:i+window_size])
        windows.append(w)
    return windows


def main():
    print("=" * 70)
    print("PHASE 10: RED TEAM VALIDATION")
    print("Addressing critical methodological gaps")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)

    # Load Quran with BOTH text variants
    quran = pd.read_parquet("data/quran/quran_dataset.parquet")
    print(f"  Loaded {len(quran)} verses")
    print(f"  Columns: {quran.columns.tolist()}")

    has_simple = "text_simple" in quran.columns
    print(f"  Has text_simple column: {has_simple}")

    # =========================================================================
    # TEST 1: TOKENS-PER-WORD RATIO
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 1: TOKENS-PER-WORD RATIO")
    print("Are tokenizer fragmentation differences driving PPL differences?")
    print(f"{'='*70}")

    models_to_test = {
        "CAMeLBERT": "CAMeL-Lab/bert-base-arabic-camelbert-ca",
        "AraBERT": "aubmindlab/bert-base-arabertv02",
    }

    # Prepare corpus texts
    quran_text_uthmani = " ".join(strip_diacritics(t) for t in quran["text_uthmani"].tolist())
    quran_words_uthmani = quran_text_uthmani.split()

    if has_simple:
        quran_text_simple = " ".join(strip_diacritics(t) for t in quran["text_simple"].tolist())
        quran_words_simple = quran_text_simple.split()

    # Load Hadith for comparison
    hadith_df = pd.read_parquet("data/controls/external/hadith_bukhari.parquet")
    hadith_col = [c for c in hadith_df.columns if 'text' in c.lower() or 'arabic' in c.lower() or 'content' in c.lower()]
    if hadith_col:
        hadith_texts = [strip_diacritics(str(t)) for t in hadith_df[hadith_col[0]].dropna().tolist()[:500]]
    else:
        hadith_texts = [strip_diacritics(str(t)) for t in hadith_df.iloc[:, 0].dropna().tolist()[:500]]
    hadith_all = " ".join(hadith_texts)
    hadith_words = hadith_all.split()
    print(f"  Loaded {len(hadith_texts)} hadith texts, {len(hadith_words)} words")

    tpw_results = {}

    for model_name, model_id in models_to_test.items():
        print(f"\n  Loading tokenizer: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Quran Uthmani
        sample_uthmani = quran_words_uthmani[:3000]
        n_words_u = len(sample_uthmani)
        tokens_u = tokenizer(" ".join(sample_uthmani), add_special_tokens=False)
        n_tokens_u = len(tokens_u["input_ids"])
        tpw_u = n_tokens_u / n_words_u

        # Quran Simple (if available)
        if has_simple:
            sample_simple = quran_words_simple[:3000]
            n_words_s = len(sample_simple)
            tokens_s = tokenizer(" ".join(sample_simple), add_special_tokens=False)
            n_tokens_s = len(tokens_s["input_ids"])
            tpw_s = n_tokens_s / n_words_s

        # Hadith
        sample_hadith = hadith_words[:3000]
        n_words_h = len(sample_hadith)
        tokens_h = tokenizer(" ".join(sample_hadith), add_special_tokens=False)
        n_tokens_h = len(tokens_h["input_ids"])
        tpw_h = n_tokens_h / n_words_h

        print(f"\n  {model_name} Tokens-Per-Word:")
        print(f"    Quran (Uthmani):  {tpw_u:.3f} ({n_tokens_u} tokens / {n_words_u} words)")
        if has_simple:
            print(f"    Quran (Simple):   {tpw_s:.3f} ({n_tokens_s} tokens / {n_words_s} words)")
        print(f"    Hadith:           {tpw_h:.3f} ({n_tokens_h} tokens / {n_words_h} words)")

        ratio = tpw_u / tpw_h
        print(f"    Quran/Hadith ratio: {ratio:.3f}x")
        if ratio > 1.5:
            print(f"    ⚠️  WARNING: Quran tokenizes into {ratio:.1f}x more tokens — potential confound!")
        elif ratio > 1.2:
            print(f"    🟡 CAUTION: Moderate tokenization difference")
        else:
            print(f"    ✅ Tokenization difference is small — unlikely to explain 154x PPL gap")

        tpw_results[model_name] = {
            "quran_uthmani_tpw": tpw_u,
            "quran_simple_tpw": tpw_s if has_simple else None,
            "hadith_tpw": tpw_h,
            "ratio": ratio,
        }

    # =========================================================================
    # TEST 2: UTHMANI vs SIMPLE SCRIPT
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("TEST 2: UTHMANI vs SIMPLE SCRIPT PPL COMPARISON")
    print("Is the Uthmani orthography inflating PPL?")
    print(f"{'='*70}")

    if not has_simple:
        print("  ⚠️  text_simple column not available. Using diacritics-stripped Uthmani as proxy.")
        # Create a "simple" version by additional normalization
        def extra_normalize(text):
            text = strip_diacritics(text)
            # Normalize hamza variations
            text = text.replace('إ', 'ا').replace('أ', 'ا').replace('آ', 'ا').replace('ٱ', 'ا')
            # Normalize ta marbuta
            text = text.replace('ة', 'ه')
            # Normalize alif maqsura
            text = text.replace('ى', 'ي')
            return text

        quran["text_normalized"] = quran["text_uthmani"].apply(extra_normalize)
        simple_col = "text_normalized"
    else:
        simple_col = "text_simple"

    # Load CAMeLBERT for PPL comparison
    model_id = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
    print(f"\n  Loading {model_id} for PPL comparison...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device).eval()
    mask_id = tokenizer.mask_token_id

    # Sample 200 15-word windows from each version
    n_windows = 200

    uthmani_full = " ".join(strip_diacritics(t) for t in quran["text_uthmani"])
    simple_full = " ".join(strip_diacritics(t) for t in quran[simple_col])

    windows_uthmani = extract_windows(uthmani_full, 15)[:n_windows]
    windows_simple = extract_windows(simple_full, 15)[:n_windows]

    print(f"  Extracted {len(windows_uthmani)} Uthmani windows, {len(windows_simple)} Simple windows")

    # Compute PPL on 50 random windows from each
    n_eval = 50
    idx = np.random.choice(min(len(windows_uthmani), len(windows_simple)), n_eval, replace=False)

    ppl_uthmani = []
    ppl_simple = []
    for i in tqdm(idx, desc="  Comparing scripts"):
        p_u, _ = compute_ppl_for_text(windows_uthmani[i], tokenizer, model, device, mask_id, max_masks=12)
        p_s, _ = compute_ppl_for_text(windows_simple[i], tokenizer, model, device, mask_id, max_masks=12)
        if not math.isnan(p_u): ppl_uthmani.append(p_u)
        if not math.isnan(p_s): ppl_simple.append(p_s)

    mean_u = np.mean(ppl_uthmani)
    mean_s = np.mean(ppl_simple)
    ratio = mean_u / mean_s if mean_s > 0 else float('inf')

    print(f"\n  RESULTS:")
    print(f"    Uthmani PPL: {mean_u:.1f} (median: {np.median(ppl_uthmani):.1f}, n={len(ppl_uthmani)})")
    print(f"    Simple PPL:  {mean_s:.1f} (median: {np.median(ppl_simple):.1f}, n={len(ppl_simple)})")
    print(f"    Ratio (Uthmani/Simple): {ratio:.3f}x")

    if ratio > 2.0:
        print(f"    🔴 CRITICAL: Uthmani inflates PPL by {ratio:.1f}x — orthography IS a major confound!")
    elif ratio > 1.3:
        print(f"    🟡 CAUTION: Moderate orthographic effect ({ratio:.1f}x)")
    else:
        print(f"    ✅ Orthographic effect is small ({ratio:.2f}x) — NOT a primary confound")

    # Also compute Hadith PPL for comparison
    print(f"\n  Computing Hadith PPL baseline for comparison...")
    hadith_windows = extract_windows(hadith_all, 15)[:n_windows]
    hadith_ppl = []
    eval_idx = np.random.choice(len(hadith_windows), min(n_eval, len(hadith_windows)), replace=False)
    for i in tqdm(eval_idx, desc="  Hadith baseline"):
        p, _ = compute_ppl_for_text(hadith_windows[i], tokenizer, model, device, mask_id, max_masks=12)
        if not math.isnan(p): hadith_ppl.append(p)

    mean_h = np.mean(hadith_ppl) if hadith_ppl else 0
    print(f"    Hadith PPL: {mean_h:.1f} (median: {np.median(hadith_ppl):.1f}, n={len(hadith_ppl)})")
    print(f"\n  ADJUSTED GAP (using Simple text):")
    print(f"    Quran(Simple)/Hadith ratio: {mean_s/mean_h:.1f}x")
    print(f"    Original Quran(Uthmani)/Hadith ratio: {mean_u/mean_h:.1f}x")

    # =========================================================================
    # TEST 3: LENGTH vs THEME REGRESSION
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("TEST 3: LENGTH vs THEME — PARTIAL CORRELATION")
    print("Does theme predict PPL AFTER controlling for length?")
    print(f"{'='*70}")

    # Load Phase 9 results
    with open("data/results/verse_anomaly_map.json", "r", encoding="utf-8") as f:
        p9 = json.load(f)

    df = pd.DataFrame(p9["surah_stats"])
    df["theme"] = df["surah"].map(SURAH_THEME).fillna("other")
    df["log_ppl"] = np.log(df["ppl"])
    df["log_words"] = np.log(df["total_words"])

    # Spearman: length vs log_ppl
    rho_len, p_len = spearmanr(df["total_words"], df["log_ppl"])
    print(f"\n  Length vs log(PPL): rho={rho_len:.3f}, p={p_len:.6f}")

    # For each theme pair, compare PPL WITHIN similar-length surahs
    print(f"\n  LENGTH-MATCHED THEME COMPARISON:")
    print(f"  (Comparing only surahs with 20-100 total words to control for length)")

    df_mid = df[(df["total_words"] >= 20) & (df["total_words"] <= 100)]
    print(f"  Surahs in 20-100 word range: {len(df_mid)}")

    for theme in ["aqeedah", "eschatology", "narrative", "legislation", "exhortation"]:
        theme_data = df_mid[df_mid["theme"] == theme]["ppl"]
        other_data = df_mid[df_mid["theme"] != theme]["ppl"]
        if len(theme_data) >= 3 and len(other_data) >= 3:
            u_stat, p_val = mannwhitneyu(theme_data, other_data, alternative='two-sided')
            print(f"    {theme:<15}: median PPL={theme_data.median():>8.0f} (n={len(theme_data)}) vs others={other_data.median():>8.0f} (n={len(other_data)}) | p={p_val:.4f} {'*' if p_val < 0.05 else 'ns'}")
        else:
            print(f"    {theme:<15}: insufficient data in this length range (n={len(theme_data)})")

    # Also: simple OLS with length as covariate
    from numpy.linalg import lstsq
    # Create dummy variables for themes
    for theme in THEMES:
        df[f"is_{theme}"] = (df["theme"] == theme).astype(int)

    X = df[["log_words", "is_aqeedah", "is_eschatology", "is_narrative", "is_legislation"]].values
    X = np.column_stack([np.ones(len(X)), X])  # intercept
    y = df["log_ppl"].values
    coeffs, residuals, rank, sv = lstsq(X, y, rcond=None)

    print(f"\n  OLS REGRESSION: log(PPL) ~ log(words) + theme dummies")
    labels = ["intercept", "log(words)", "aqeedah", "eschatology", "narrative", "legislation"]
    for label, coeff in zip(labels, coeffs):
        direction = "↑ PPL" if coeff > 0 else "↓ PPL"
        print(f"    {label:<15}: β = {coeff:>7.3f}  ({direction})")

    # R² calculation
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"    R² = {r2:.3f}")

    # R² with length only
    X_len = np.column_stack([np.ones(len(df)), df["log_words"].values])
    coeffs_len, _, _, _ = lstsq(X_len, y, rcond=None)
    y_pred_len = X_len @ coeffs_len
    ss_res_len = np.sum((y - y_pred_len) ** 2)
    r2_len = 1 - ss_res_len / ss_tot
    print(f"    R² (length only) = {r2_len:.3f}")
    print(f"    R² improvement from adding theme = {r2 - r2_len:.3f}")

    if r2 - r2_len < 0.05:
        print(f"    🔴 Theme adds <5% explanatory power — thematic finding is likely a LENGTH ARTIFACT")
    elif r2 - r2_len < 0.10:
        print(f"    🟡 Theme adds moderate explanatory power — signal exists but weak")
    else:
        print(f"    ✅ Theme adds significant explanatory power — thematic finding SURVIVES length control")

    # =========================================================================
    # TEST 4: VERSE-ALIGNED vs WINDOW-ALIGNED PPL
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("TEST 4: VERSE-ALIGNED vs WINDOW-ALIGNED PPL")
    print("Does cutting across verse boundaries inflate PPL?")
    print(f"{'='*70}")

    # Sample 50 individual verses (10+ words each)
    long_verses = quran[quran["text_length_words"] >= 10].sample(50, random_state=42)
    verse_ppls = []
    for _, row in tqdm(long_verses.iterrows(), total=50, desc="  Verse-aligned"):
        p, _ = compute_ppl_for_text(row["text_uthmani"], tokenizer, model, device, mask_id, max_masks=12)
        if not math.isnan(p):
            verse_ppls.append(p)

    # 50 random 15-word windows (cross-verse)
    window_ppls = []
    w_idx = np.random.choice(len(windows_uthmani), 50, replace=False)
    for i in tqdm(w_idx, desc="  Window-aligned"):
        p, _ = compute_ppl_for_text(windows_uthmani[i], tokenizer, model, device, mask_id, max_masks=12)
        if not math.isnan(p):
            window_ppls.append(p)

    mean_verse = np.mean(verse_ppls)
    mean_window = np.mean(window_ppls)
    ratio_vw = mean_window / mean_verse if mean_verse > 0 else float('inf')

    print(f"\n  RESULTS:")
    print(f"    Verse-aligned PPL:  {mean_verse:.1f} (median: {np.median(verse_ppls):.1f}, n={len(verse_ppls)})")
    print(f"    Window-aligned PPL: {mean_window:.1f} (median: {np.median(window_ppls):.1f}, n={len(window_ppls)})")
    print(f"    Window/Verse ratio: {ratio_vw:.3f}x")

    if ratio_vw > 2.0:
        print(f"    🔴 CRITICAL: Windowing inflates PPL by {ratio_vw:.1f}x — cross-verse boundaries ARE a confound!")
    elif ratio_vw > 1.3:
        print(f"    🟡 CAUTION: Moderate windowing effect ({ratio_vw:.1f}x)")
    else:
        print(f"    ✅ Windowing effect is small ({ratio_vw:.2f}x) — NOT a primary confound")

    print(f"\n  VERSE-ALIGNED vs HADITH:")
    print(f"    Quran (verse-aligned): {mean_verse:.1f}")
    print(f"    Hadith (windowed):     {mean_h:.1f}")
    print(f"    Ratio: {mean_verse/mean_h:.1f}x")

    # =========================================================================
    # CONSOLIDATED RESULTS
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("CONSOLIDATED RED-TEAM VALIDATION RESULTS")
    print(f"{'='*70}")

    results = {
        "test1_tokenization": tpw_results,
        "test2_orthography": {
            "uthmani_mean_ppl": float(mean_u),
            "simple_mean_ppl": float(mean_s),
            "ratio": float(ratio),
            "hadith_mean_ppl": float(mean_h),
            "adjusted_quran_hadith_gap": float(mean_s / mean_h) if mean_h > 0 else None,
        },
        "test3_length_theme": {
            "r2_full_model": float(r2),
            "r2_length_only": float(r2_len),
            "r2_improvement_from_theme": float(r2 - r2_len),
            "coefficients": {l: float(c) for l, c in zip(labels, coeffs)},
        },
        "test4_verse_vs_window": {
            "verse_aligned_ppl": float(mean_verse),
            "window_aligned_ppl": float(mean_window),
            "ratio": float(ratio_vw),
            "verse_vs_hadith_ratio": float(mean_verse / mean_h) if mean_h > 0 else None,
        },
    }

    out = Path("data/results/red_team_validation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved to {out}")

    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
