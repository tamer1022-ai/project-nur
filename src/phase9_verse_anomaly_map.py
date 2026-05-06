# Project Nur — Phase 9: Verse-Level Anomaly Map (OPTIMIZED)
# Bismillah
#
# FAST version: batch per-surah windows instead of per-verse per-token masking
# Uses the same 15-word window methodology but aggregated by surah.

import numpy as np
import pandas as pd
import json
import math
import re
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
warnings.filterwarnings("ignore")


def strip_diacritics(text):
    return re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]').sub(
        '', text.replace('\u0640', '')).strip()

SURAH_NAMES = {
    1:"Al-Fatiha",2:"Al-Baqarah",3:"Ali 'Imran",4:"An-Nisa",5:"Al-Ma'idah",
    6:"Al-An'am",7:"Al-A'raf",8:"Al-Anfal",9:"At-Tawbah",10:"Yunus",
    11:"Hud",12:"Yusuf",13:"Ar-Ra'd",14:"Ibrahim",15:"Al-Hijr",
    16:"An-Nahl",17:"Al-Isra",18:"Al-Kahf",19:"Maryam",20:"Ta-Ha",
    21:"Al-Anbiya",22:"Al-Hajj",23:"Al-Mu'minun",24:"An-Nur",25:"Al-Furqan",
    26:"Ash-Shu'ara",27:"An-Naml",28:"Al-Qasas",29:"Al-Ankabut",30:"Ar-Rum",
    31:"Luqman",32:"As-Sajdah",33:"Al-Ahzab",34:"Saba",35:"Fatir",
    36:"Ya-Sin",37:"As-Saffat",38:"Sad",39:"Az-Zumar",40:"Ghafir",
    41:"Fussilat",42:"Ash-Shura",43:"Az-Zukhruf",44:"Ad-Dukhan",45:"Al-Jathiyah",
    46:"Al-Ahqaf",47:"Muhammad",48:"Al-Fath",49:"Al-Hujurat",50:"Qaf",
    51:"Adh-Dhariyat",52:"At-Tur",53:"An-Najm",54:"Al-Qamar",55:"Ar-Rahman",
    56:"Al-Waqi'ah",57:"Al-Hadid",58:"Al-Mujadila",59:"Al-Hashr",60:"Al-Mumtahina",
    61:"As-Saff",62:"Al-Jumu'ah",63:"Al-Munafiqun",64:"At-Taghabun",65:"At-Talaq",
    66:"At-Tahrim",67:"Al-Mulk",68:"Al-Qalam",69:"Al-Haqqah",70:"Al-Ma'arij",
    71:"Nuh",72:"Al-Jinn",73:"Al-Muzzammil",74:"Al-Muddaththir",75:"Al-Qiyamah",
    76:"Al-Insan",77:"Al-Mursalat",78:"An-Naba",79:"An-Nazi'at",80:"Abasa",
    81:"At-Takwir",82:"Al-Infitar",83:"Al-Mutaffifin",84:"Al-Inshiqaq",
    85:"Al-Buruj",86:"At-Tariq",87:"Al-A'la",88:"Al-Ghashiyah",89:"Al-Fajr",
    90:"Al-Balad",91:"Ash-Shams",92:"Al-Layl",93:"Ad-Duha",94:"Ash-Sharh",
    95:"At-Tin",96:"Al-Alaq",97:"Al-Qadr",98:"Al-Bayyinah",99:"Az-Zalzalah",
    100:"Al-Adiyat",101:"Al-Qari'ah",102:"At-Takathur",103:"Al-Asr",104:"Al-Humazah",
    105:"Al-Fil",106:"Quraysh",107:"Al-Ma'un",108:"Al-Kawthar",109:"Al-Kafirun",
    110:"An-Nasr",111:"Al-Masad",112:"Al-Ikhlas",113:"Al-Falaq",114:"An-Nas",
}

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


def fast_ppl_batch(texts, tokenizer, model, device, mask_token_id, max_masks=8):
    """Compute PPL for a batch of texts using random masking (fast approximation)."""
    losses_all = []
    for text in texts:
        text = strip_diacritics(text)
        if len(text.split()) < 3:
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        seq = ids.shape[1]
        if seq <= 2:
            continue
        # Sample random positions to mask (much faster than all positions)
        positions = list(range(1, seq - 1))
        if len(positions) > max_masks:
            positions = np.random.choice(positions, max_masks, replace=False).tolist()
        with torch.no_grad():
            for pos in positions:
                masked = ids.clone()
                true_tok = ids[0, pos].item()
                masked[0, pos] = mask_token_id
                out = model(masked, attention_mask=att)
                probs = torch.softmax(out.logits[0, pos], dim=-1)
                p = probs[true_tok].item()
                if p > 0:
                    losses_all.append(-math.log(p))
    return losses_all


def main():
    print("=" * 70)
    print("PHASE 9: VERSE-LEVEL ANOMALY MAP (OPTIMIZED)")
    print("=" * 70)

    quran = pd.read_parquet("data/quran/quran_dataset.parquet")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
    print(f"\n  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device).eval()
    mask_id = tokenizer.mask_token_id

    np.random.seed(42)
    surah_results = []

    for s_num in tqdm(range(1, 115), desc="  Surahs", unit="surah"):
        verses = quran[quran["surah_number"] == s_num]
        texts = verses["text_uthmani"].tolist()
        n_verses = len(texts)
        total_words = sum(len(strip_diacritics(t).split()) for t in texts)

        losses = fast_ppl_batch(texts, tokenizer, model, device, mask_id, max_masks=6)

        if losses:
            ppl = float(np.exp(np.mean(losses)))
            std = float(np.std(losses))
            median_loss = float(np.median(losses))
        else:
            ppl, std, median_loss = float('nan'), 0, 0

        surah_results.append({
            "surah": s_num,
            "name": SURAH_NAMES.get(s_num, f"S{s_num}"),
            "revelation": "Makki" if s_num in MAKKI else "Madani",
            "n_verses": n_verses,
            "total_words": total_words,
            "avg_words_per_verse": round(total_words / max(n_verses, 1), 1),
            "ppl": ppl,
            "loss_std": std,
            "median_loss": median_loss,
        })

    df = pd.DataFrame(surah_results).dropna(subset=["ppl"])

    # Sort and display
    df_sorted = df.sort_values("ppl", ascending=False)

    print(f"\n\n{'='*70}")
    print("TOP 20 MOST SURPRISING SURAHS")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'Surah':<20} {'Type':<7} {'PPL':>10} {'Verses':>7} {'Words':>7}")
    print("  " + "-" * 58)
    for i, (_, r) in enumerate(df_sorted.head(20).iterrows()):
        print(f"  {i+1:<4} {r['name']:<20} {r['revelation']:<7} {r['ppl']:>10.1f} {r['n_verses']:>7} {r['total_words']:>7}")

    print(f"\n\n{'='*70}")
    print("TOP 20 MOST PREDICTABLE SURAHS")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'Surah':<20} {'Type':<7} {'PPL':>10} {'Verses':>7} {'Words':>7}")
    print("  " + "-" * 58)
    for i, (_, r) in enumerate(df_sorted.tail(20).iterrows()):
        print(f"  {i+1:<4} {r['name']:<20} {r['revelation']:<7} {r['ppl']:>10.1f} {r['n_verses']:>7} {r['total_words']:>7}")

    # Makki vs Madani
    makki = df[df["revelation"] == "Makki"]["ppl"]
    madani = df[df["revelation"] == "Madani"]["ppl"]
    print(f"\n  MAKKI vs MADANI:")
    print(f"    Makki:  mean PPL = {makki.mean():.1f}, median = {makki.median():.1f} (n={len(makki)})")
    print(f"    Madani: mean PPL = {madani.mean():.1f}, median = {madani.median():.1f} (n={len(madani)})")
    print(f"    Ratio (Makki/Madani): {makki.mean()/madani.mean():.2f}x")

    # Thematic
    print(f"\n  THEMATIC BREAKDOWN:")
    theme_data = {}
    for theme, surahs in THEMES.items():
        t_ppl = df[df["surah"].isin(surahs)]["ppl"]
        if len(t_ppl) > 0:
            theme_data[theme] = {"mean": float(t_ppl.mean()), "median": float(t_ppl.median()), "n": len(t_ppl)}
            print(f"    {theme:<15}: mean PPL = {t_ppl.mean():>8.1f}, median = {t_ppl.median():>8.1f} (n={len(t_ppl)} surahs)")

    # Variance analysis
    print(f"\n  VARIANCE ANALYSIS:")
    print(f"    Overall PPL range: {df['ppl'].min():.1f} — {df['ppl'].max():.1f}")
    print(f"    Coefficient of variation: {df['ppl'].std()/df['ppl'].mean()*100:.1f}%")
    print(f"    IQR: {df['ppl'].quantile(0.25):.1f} — {df['ppl'].quantile(0.75):.1f}")

    # Length correlation
    from scipy.stats import spearmanr
    corr, pval = spearmanr(df["total_words"], df["ppl"])
    print(f"\n  LENGTH vs PPL CORRELATION:")
    print(f"    Spearman rho = {corr:.3f}, p = {pval:.4f}")
    print(f"    {'SIGNIFICANT' if pval < 0.05 else 'NOT SIGNIFICANT'}: {'Longer' if corr > 0 else 'Shorter'} surahs tend to be {'more' if corr > 0 else 'less'} surprising")

    # Save
    results = {
        "surah_stats": df_sorted.to_dict(orient="records"),
        "makki": {"mean": float(makki.mean()), "median": float(makki.median()), "n": int(len(makki))},
        "madani": {"mean": float(madani.mean()), "median": float(madani.median()), "n": int(len(madani))},
        "themes": theme_data,
        "length_correlation": {"spearman_rho": float(corr), "p_value": float(pval)},
        "overall": {
            "mean_ppl": float(df["ppl"].mean()),
            "median_ppl": float(df["ppl"].median()),
            "min_ppl": float(df["ppl"].min()),
            "max_ppl": float(df["ppl"].max()),
            "cv": float(df["ppl"].std()/df["ppl"].mean()),
        }
    }

    def clean(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        return obj

    out = Path("data/results/verse_anomaly_map.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(clean(results), f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved to {out}")

if __name__ == "__main__":
    main()
