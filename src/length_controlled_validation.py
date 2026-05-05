# Project Nur — Length-Controlled Multi-Model Validation
# Bismillah
#
# This eliminates the text-length confound and validates across
# multiple transformer architectures. If the anomaly persists
# across models AND controlled text lengths, it's structural.

import numpy as np
import pandas as pd
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
warnings.filterwarnings("ignore")


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics."""
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]')
    return diacritics.sub('', text.replace('\u0640', '')).strip()


class LengthControlledValidator:
    """
    Controls for text length by using FIXED TOKEN WINDOWS across all corpora.
    
    Method: For each corpus, we extract windows of exactly N tokens,
    then compute pseudo-perplexity on these equal-length windows.
    This eliminates text length as a confounding variable.
    """

    MODELS = {
        "CAMeLBERT-CA": "CAMeL-Lab/bert-base-arabic-camelbert-ca",
        "AraBERT-v2": "aubmindlab/bert-base-arabertv02",
        "XLM-RoBERTa": "xlm-roberta-base",
    }

    def __init__(self, window_tokens: int = 15, n_samples: int = 200, seed: int = 42):
        """
        Args:
            window_tokens: Fixed number of WORDS per sample (controls length)
            n_samples: Number of equal-length samples per corpus
            seed: Random seed for reproducibility
        """
        self.window_tokens = window_tokens
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_fixed_windows(self, texts: List[str], corpus_name: str) -> List[str]:
        """
        Extract fixed-length word windows from a corpus.
        Ensures all corpora have EXACTLY the same text length distribution.
        """
        # Concatenate all texts with verse boundary markers
        all_words = []
        for text in texts:
            words = strip_diacritics(text).split()
            all_words.extend(words)

        # Extract non-overlapping windows of fixed length
        windows = []
        for i in range(0, len(all_words) - self.window_tokens, self.window_tokens):
            window = " ".join(all_words[i:i + self.window_tokens])
            windows.append(window)

        # Randomly sample n_samples windows
        if len(windows) > self.n_samples:
            indices = self.rng.choice(len(windows), self.n_samples, replace=False)
            windows = [windows[i] for i in indices]
        
        print(f"  [{corpus_name}] Extracted {len(windows)} windows of {self.window_tokens} words each")
        return windows

    def compute_perplexity_for_model(
        self,
        model_name: str,
        model_id: str,
        corpus_windows: Dict[str, List[str]],
    ) -> Dict[str, Dict]:
        """Compute pseudo-perplexity for one model across all corpora."""
        print(f"\n{'=' * 60}")
        print(f"  MODEL: {model_name} ({model_id})")
        print(f"{'=' * 60}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        model.to(self.device)
        model.eval()

        mask_token_id = tokenizer.mask_token_id
        results = {}

        for corpus_name, windows in corpus_windows.items():
            print(f"\n  Processing {corpus_name} ({len(windows)} windows)...")
            all_losses = []

            for window in tqdm(windows, desc=f"  {corpus_name}", unit="win"):
                encoded = tokenizer(
                    window, return_tensors="pt", 
                    truncation=True, max_length=64,
                    padding=False,
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                seq_len = input_ids.shape[1]

                if seq_len <= 2:
                    continue

                # Mask each non-special token and measure prediction loss
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

            # Bootstrap 95% CI
            ci_low, ci_high = self._bootstrap_ci(losses)

            results[corpus_name] = {
                "pseudo_perplexity": ppl,
                "mean_loss": float(np.mean(losses)),
                "std_loss": float(np.std(losses)),
                "median_loss": float(np.median(losses)),
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "n_tokens": len(losses),
                "n_windows": len(windows),
            }

            print(f"    PPL: {ppl:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
            print(f"    Tokens analyzed: {len(losses)}")

        # Cleanup GPU memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _bootstrap_ci(self, losses: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        """Bootstrap 95% CI for perplexity."""
        if len(losses) < 10:
            return 0.0, 0.0
        
        boot_ppls = []
        for _ in range(n_boot):
            sample = self.rng.choice(losses, len(losses), replace=True)
            boot_ppls.append(float(np.exp(np.mean(sample))))
        
        low = float(np.percentile(boot_ppls, 100 * alpha / 2))
        high = float(np.percentile(boot_ppls, 100 * (1 - alpha / 2)))
        return low, high

    def run(self) -> Dict:
        """Execute the full length-controlled multi-model validation."""
        print("=" * 70)
        print("PROJECT NUR — LENGTH-CONTROLLED MULTI-MODEL VALIDATION")
        print(f"Window size: {self.window_tokens} words | Samples: {self.n_samples}")
        print("=" * 70)

        # Load corpora
        corpora_raw = {}
        
        quran = pd.read_parquet("data/quran/quran_dataset.parquet")
        corpora_raw["QURAN"] = quran["text_uthmani"].tolist()

        for name, path in [
            ("HADITH", "data/controls/external/hadith_bukhari.parquet"),
            ("POETRY", "data/controls/external/poetry_pre_islamic.parquet"),
            ("BIBLE", "data/controls/external/bible_vandyke.parquet"),
        ]:
            p = Path(path)
            if p.exists():
                df = pd.read_parquet(p)
                corpora_raw[name] = df["text_simple"].tolist()

        shuf = Path("data/controls/shuffled/quran_shuffled_words.parquet")
        if shuf.exists():
            df = pd.read_parquet(shuf)
            corpora_raw["QURAN_SHUF"] = df["text_uthmani"].tolist()

        # Extract fixed-length windows
        print("\n### STEP 1: Extracting equal-length windows ###")
        corpus_windows = {}
        for name, texts in corpora_raw.items():
            corpus_windows[name] = self.extract_fixed_windows(texts, name)

        # Run each model
        all_results = {}
        print("\n### STEP 2: Multi-model perplexity analysis ###")
        
        for model_name, model_id in self.MODELS.items():
            try:
                model_results = self.compute_perplexity_for_model(
                    model_name, model_id, corpus_windows
                )
                all_results[model_name] = model_results
            except Exception as e:
                print(f"\n  [ERR] Model {model_name} failed: {e}")
                continue

        # Statistical tests
        print("\n\n### STEP 3: Statistical significance ###")
        self._statistical_tests(all_results)

        # Comparative table
        self._print_comparative_table(all_results)

        # Save
        self._save_results(all_results)

        return all_results

    def _statistical_tests(self, all_results: Dict):
        """Run pairwise statistical tests between Quran and each control."""
        for model_name, model_results in all_results.items():
            print(f"\n  [{model_name}]")
            if "QURAN" not in model_results:
                continue
            
            quran_ppl = model_results["QURAN"]["pseudo_perplexity"]
            
            for corpus_name, corpus_data in model_results.items():
                if corpus_name == "QURAN":
                    continue
                
                control_ppl = corpus_data["pseudo_perplexity"]
                ratio = quran_ppl / control_ppl if control_ppl > 0 else float('inf')
                
                # Check CI overlap
                q_low = model_results["QURAN"]["ci_95_low"]
                q_high = model_results["QURAN"]["ci_95_high"]
                c_low = corpus_data["ci_95_low"]
                c_high = corpus_data["ci_95_high"]
                
                overlaps = q_low <= c_high and c_low <= q_high
                sig = "NOT SIGNIFICANT" if overlaps else "SIGNIFICANT"
                
                print(f"    QURAN vs {corpus_name}: ratio={ratio:.2f}x, "
                      f"Q=[{q_low:.1f},{q_high:.1f}] vs C=[{c_low:.1f},{c_high:.1f}] — {sig}")

    def _print_comparative_table(self, all_results: Dict):
        """Print the master comparison table."""
        print(f"\n\n{'=' * 90}")
        print("MASTER COMPARISON — LENGTH-CONTROLLED PSEUDO-PERPLEXITY")
        print(f"Window: {self.window_tokens} words | Samples: {self.n_samples} per corpus")
        print(f"{'=' * 90}")

        corpora = ["QURAN", "HADITH", "POETRY", "BIBLE", "QURAN_SHUF"]
        
        for model_name, model_results in all_results.items():
            print(f"\n  {model_name}:")
            header = f"    {'Corpus':<15} {'PPL':>10} {'CI_Low':>10} {'CI_High':>10} {'Tokens':>8}"
            print(header)
            print("    " + "-" * 55)
            
            for corpus in corpora:
                if corpus in model_results:
                    r = model_results[corpus]
                    print(f"    {corpus:<15} {r['pseudo_perplexity']:>10.2f} "
                          f"{r['ci_95_low']:>10.2f} {r['ci_95_high']:>10.2f} "
                          f"{r['n_tokens']:>8}")

    def _save_results(self, all_results: Dict):
        """Save results."""
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

        out = Path("data/results/length_controlled_validation.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(clean(all_results), f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Results saved to: {out}")


if __name__ == "__main__":
    validator = LengthControlledValidator(
        window_tokens=15,   # Fixed at 15 words per sample
        n_samples=200,      # 200 samples per corpus
    )
    validator.run()
