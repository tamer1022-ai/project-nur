# Project Nur — Phase 8: Synthetic Arabic Control
# Bismillah
#
# CAN any generative model produce text with the Quran's
# information-theoretic signature?
#
# We generate Arabic text using ArGPT2 (autoregressive Arabic model)
# with multiple seed strategies, then run the SAME validation pipeline.

import numpy as np
import pandas as pd
import json
import math
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM,
    pipeline
)
import warnings
warnings.filterwarnings("ignore")


def strip_diacritics(text: str) -> str:
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]')
    return diacritics.sub('', text.replace('\u0640', '')).strip()


class SyntheticArabicGenerator:
    """Generate Arabic text using ArGPT2 with different seeding strategies."""

    def __init__(self, n_texts: int = 500, max_length: int = 40, seed: int = 42):
        self.n_texts = n_texts
        self.max_length = max_length
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_corpus(self, model_id: str, seeds: List[str], 
                       label: str, temperature: float = 0.9,
                       top_k: int = 50, top_p: float = 0.95) -> List[str]:
        """Generate texts from an autoregressive model with given seeds."""
        print(f"\n  Generating {label} ({self.n_texts} texts)...")
        print(f"  Model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.to(self.device)
        model.eval()

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts = []
        torch.manual_seed(self.seed)

        for i in tqdm(range(self.n_texts), desc=f"  {label}", unit="text"):
            # Cycle through seeds
            seed_text = seeds[i % len(seeds)]

            inputs = tokenizer(seed_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.2,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = strip_diacritics(generated)

            # Take only the generated part (after seed)
            if len(generated) > len(seed_text):
                texts.append(generated)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Generated {len(texts)} texts, avg words: {np.mean([len(t.split()) for t in texts]):.1f}")
        return texts

    def generate_all(self) -> Dict[str, List[str]]:
        """Generate synthetic corpora with different strategies."""
        print("=" * 60)
        print("PHASE 8: SYNTHETIC ARABIC TEXT GENERATION")
        print("=" * 60)

        model_id = "aubmindlab/aragpt2-base"

        # Strategy 1: Quran-style seeds (Quranic openings)
        quran_seeds = [
            "بسم الله الرحمن الرحيم",
            "الحمد لله رب العالمين",
            "قل هو الله أحد",
            "والعصر ان الانسان",
            "يا ايها الذين امنوا",
            "ان الله يامر بالعدل",
            "وما ارسلناك الا رحمة",
            "الم ذلك الكتاب لا ريب",
            "قل اعوذ برب الفلق",
            "تبارك الذي نزل الفرقان",
        ]

        # Strategy 2: Generic Arabic seeds
        generic_seeds = [
            "كان في قديم الزمان",
            "ومن المعروف ان",
            "قال الحكيم في ذلك",
            "واذا نظرنا الى",
            "ولا شك ان هذا",
            "في يوم من الايام",
            "اما بعد فان",
            "ولقد كان الناس",
            "والذي يظهر لنا",
            "فمن اراد ان يعرف",
        ]

        # Strategy 3: Religious prose seeds (non-Quranic)
        religious_seeds = [
            "اعلم رحمك الله ان",
            "والايمان بالله واجب",
            "ومن صفات المؤمنين",
            "ان التقوى اساس كل",
            "والصلاة عماد الدين",
            "ومن اراد الاخرة فعليه",
            "والزكاة ركن من اركان",
            "ان الصبر مفتاح الفرج",
            "والتوبة واجبة على كل",
            "ومن يتوكل على الله",
        ]

        corpora = {}

        # Generate from each strategy
        corpora["SYNTH_QURAN_STYLE"] = self.generate_corpus(
            model_id, quran_seeds, "Quran-Style Synthetic", temperature=0.8
        )

        corpora["SYNTH_GENERIC"] = self.generate_corpus(
            model_id, generic_seeds, "Generic Arabic Synthetic", temperature=0.9
        )

        corpora["SYNTH_RELIGIOUS"] = self.generate_corpus(
            model_id, religious_seeds, "Religious Prose Synthetic", temperature=0.85
        )

        # Save
        out_dir = Path("data/controls/synthetic")
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, texts in corpora.items():
            df = pd.DataFrame({"text_simple": texts, "source": name})
            df.to_parquet(out_dir / f"{name.lower()}.parquet", index=False)
            print(f"  Saved {name}: {len(texts)} texts -> {out_dir / f'{name.lower()}.parquet'}")

        return corpora


class SyntheticValidator:
    """Run the same perplexity pipeline on synthetic corpora."""

    EVAL_MODELS = {
        "CAMeLBERT-CA": "CAMeL-Lab/bert-base-arabic-camelbert-ca",
        "AraBERT-v2": "aubmindlab/bert-base-arabertv02",
    }

    def __init__(self, window_words: int = 15, n_samples: int = 200, seed: int = 42):
        self.window_words = window_words
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_windows(self, texts: List[str]) -> List[str]:
        all_words = []
        for t in texts:
            all_words.extend(strip_diacritics(t).split())
        windows = []
        for i in range(0, len(all_words) - self.window_words, self.window_words):
            windows.append(" ".join(all_words[i:i + self.window_words]))
        if len(windows) > self.n_samples:
            idx = self.rng.choice(len(windows), self.n_samples, replace=False)
            windows = [windows[i] for i in idx]
        return windows

    def compute_ppl(self, model_name, model_id, windows: List[str]) -> Dict:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        model.to(self.device)
        model.eval()
        mask_token_id = tokenizer.mask_token_id
        all_losses = []

        for window in tqdm(windows[:50], desc=f"  PPL-{model_name}", unit="win"):
            encoded = tokenizer(window, return_tensors="pt", truncation=True, max_length=64)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            seq_len = input_ids.shape[1]
            if seq_len <= 2:
                continue
            with torch.no_grad():
                for pos in range(1, seq_len - 1):
                    masked = input_ids.clone()
                    true_token = input_ids[0, pos].item()
                    masked[0, pos] = mask_token_id
                    outputs = model(masked, attention_mask=attention_mask)
                    logits = outputs.logits[0, pos]
                    probs = torch.softmax(logits, dim=-1)
                    token_prob = probs[true_token].item()
                    if token_prob > 0:
                        all_losses.append(-math.log(token_prob))

        losses = np.array(all_losses)
        ppl = float(np.exp(np.mean(losses))) if len(losses) > 0 else float('inf')

        # Bootstrap CI
        boot_ppls = []
        for _ in range(1000):
            sample = self.rng.choice(losses, len(losses), replace=True)
            boot_ppls.append(float(np.exp(np.mean(sample))))
        ci_low = float(np.percentile(boot_ppls, 2.5))
        ci_high = float(np.percentile(boot_ppls, 97.5))

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"ppl": ppl, "ci_low": ci_low, "ci_high": ci_high, "n_tokens": len(losses)}

    def run(self, synthetic_corpora: Dict[str, List[str]]):
        print(f"\n\n{'='*60}")
        print("SYNTHETIC VALIDATION — Same pipeline as Phase 6")
        print(f"{'='*60}")

        # Also load the Quran for direct comparison
        quran_df = pd.read_parquet("data/quran/quran_dataset.parquet")
        all_corpora = {"QURAN": quran_df["text_uthmani"].tolist()}
        all_corpora.update(synthetic_corpora)

        # Extract windows
        corpus_windows = {}
        for name, texts in all_corpora.items():
            w = self.extract_windows(texts)
            corpus_windows[name] = w
            print(f"  {name}: {len(w)} windows")

        # Run each eval model
        results = {}
        for model_name, model_id in self.EVAL_MODELS.items():
            print(f"\n  === {model_name} ===")
            results[model_name] = {}
            for corpus_name, windows in corpus_windows.items():
                print(f"\n  Corpus: {corpus_name}")
                r = self.compute_ppl(model_name, model_id, windows)
                results[model_name][corpus_name] = r
                print(f"    PPL: {r['ppl']:.2f} (CI: [{r['ci_low']:.2f}, {r['ci_high']:.2f}])")

        # Print comparison table
        print(f"\n\n{'='*80}")
        print("SYNTHETIC vs QURAN — CAN AI REPLICATE THE SIGNATURE?")
        print(f"{'='*80}")

        for model_name in results:
            print(f"\n  {model_name}:")
            print(f"    {'Corpus':<25} {'PPL':>10} {'CI_Low':>10} {'CI_High':>10} {'vs Quran':>12}")
            print("    " + "-" * 70)
            q_ppl = results[model_name].get("QURAN", {}).get("ppl", 0)
            for corpus_name in corpus_windows:
                r = results[model_name][corpus_name]
                ratio = r["ppl"] / q_ppl if q_ppl > 0 else 0
                marker = " ◀ QURAN" if corpus_name == "QURAN" else ""
                print(f"    {corpus_name:<25} {r['ppl']:>10.2f} {r['ci_low']:>10.2f} {r['ci_high']:>10.2f} {ratio:>10.2f}x{marker}")

        # Save
        def clean(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            return obj

        out = Path("data/results/synthetic_validation.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(clean(results), f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Results saved to: {out}")

        return results


if __name__ == "__main__":
    # Step 1: Generate synthetic Arabic text
    generator = SyntheticArabicGenerator(n_texts=500, max_length=40)
    synthetic = generator.generate_all()

    # Step 2: Run the same validation pipeline
    validator = SyntheticValidator(window_words=15, n_samples=200)
    validator.run(synthetic)
