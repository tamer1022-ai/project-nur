# Project Nur — Phase 3: Information-Theoretic Fingerprinting
# Bismillah
#
# THE REAL QUESTION (from Tamer's insight):
# What does the Quran DO to the transformer's internal math?
# How does it affect the "equation" when the machine reads it?
#
# We measure THREE things:
# 1. PERPLEXITY — How surprised is the model by each token?
# 2. ATTENTION ENTROPY — How do attention patterns differ?
# 3. TEXT-INTRINSIC — Character entropy, compressibility, Zipf's law

import numpy as np
import pandas as pd
import json
import gzip
import bz2
import lzma
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from scipy import stats
from scipy.stats import entropy as scipy_entropy
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class TransformerInternalsAnalyzer:
    """
    Analyzes how the transformer's internal math behaves
    when processing different texts. This is Tamer's core insight:
    not what the output looks like, but what the TEXT DOES to the MACHINE.
    """

    MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-ca"

    def __init__(self, device: Optional[str] = None, batch_size: int = 16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"Loading model for internal analysis... (device: {self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.MODEL_NAME, output_attentions=True, output_hidden_states=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        print(f"  Model: {self.n_layers} layers, {self.n_heads} heads")

    def compute_pseudo_perplexity(self, texts: List[str], label: str = "") -> Dict:
        """
        Compute pseudo-perplexity for masked language model.
        
        For each token, mask it, predict it, measure how confident the model is.
        Lower perplexity = the text is more "natural" / coherent to the model.
        
        If the Quran has lower perplexity than comparable texts despite being
        composed over 23 years across different contexts, that's the signal.
        """
        print(f"\n  Computing pseudo-perplexity for {label} ({len(texts)} texts)...")
        all_token_losses = []
        
        for text_i, text in enumerate(tqdm(texts[:50], desc=f"  PPL-{label}", unit="verse")):
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = encoded["input_ids"].to(self.device)
            seq_len = input_ids.shape[1]
            
            if seq_len <= 2:  # Only [CLS] and [SEP]
                continue

            # Mask each token one at a time and measure prediction loss
            with torch.no_grad():
                for mask_pos in range(1, seq_len - 1):  # Skip CLS and SEP
                    masked_input = input_ids.clone()
                    true_token = input_ids[0, mask_pos].item()
                    masked_input[0, mask_pos] = self.tokenizer.mask_token_id
                    
                    outputs = self.model(masked_input, attention_mask=encoded["attention_mask"].to(self.device))
                    logits = outputs.logits[0, mask_pos]
                    probs = torch.softmax(logits, dim=-1)
                    token_prob = probs[true_token].item()
                    
                    if token_prob > 0:
                        all_token_losses.append(-math.log(token_prob))

        losses = np.array(all_token_losses)
        ppl = float(np.exp(np.mean(losses))) if len(losses) > 0 else float('inf')
        
        return {
            "pseudo_perplexity": ppl,
            "mean_loss": float(np.mean(losses)) if len(losses) > 0 else 0,
            "std_loss": float(np.std(losses)) if len(losses) > 0 else 0,
            "median_loss": float(np.median(losses)) if len(losses) > 0 else 0,
            "n_tokens": len(losses),
        }

    def analyze_attention_patterns(self, texts: List[str], label: str = "") -> Dict:
        """
        Analyze how the transformer's attention behaves on this text.
        
        Measures:
        - Attention entropy per head (high = distributed, low = focused)
        - Long-range attention ratio (how much attention goes to distant tokens)
        - Attention pattern consistency across layers
        """
        print(f"\n  Analyzing attention patterns for {label}...")
        
        head_entropies_by_layer = {l: [] for l in range(self.n_layers)}
        long_range_ratios = []
        
        sample = texts[:100]  # Sample for speed
        
        for text in tqdm(sample, desc=f"  ATT-{label}", unit="verse"):
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            seq_len = input_ids.shape[1]
            
            if seq_len < 4:
                continue
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
            
            for layer_idx, layer_att in enumerate(attentions):
                att = layer_att[0].cpu().numpy()  # (heads, seq, seq)
                
                for head_idx in range(self.n_heads):
                    head_att = att[head_idx]  # (seq, seq)
                    # Entropy of attention distribution for each query position
                    for q in range(1, seq_len - 1):  # Skip CLS/SEP
                        dist = head_att[q, :seq_len]
                        dist = dist / (dist.sum() + 1e-12)
                        ent = float(scipy_entropy(dist + 1e-12))
                        head_entropies_by_layer[layer_idx].append(ent)
                    
                    # Long-range attention: ratio of attention to tokens > 5 positions away
                    if seq_len > 10:
                        total_att = 0
                        long_att = 0
                        for q in range(1, seq_len - 1):
                            for k in range(seq_len):
                                total_att += head_att[q, k]
                                if abs(q - k) > 5:
                                    long_att += head_att[q, k]
                        if total_att > 0:
                            long_range_ratios.append(long_att / total_att)

        # Aggregate
        layer_entropy_means = {}
        for l in range(self.n_layers):
            vals = head_entropies_by_layer[l]
            if vals:
                layer_entropy_means[f"layer_{l}"] = float(np.mean(vals))

        return {
            "mean_attention_entropy": float(np.mean([v for vals in head_entropies_by_layer.values() for v in vals])),
            "entropy_by_layer": layer_entropy_means,
            "long_range_attention_ratio": float(np.mean(long_range_ratios)) if long_range_ratios else 0,
            "long_range_std": float(np.std(long_range_ratios)) if long_range_ratios else 0,
        }

    def analyze_hidden_state_trajectory(self, texts: List[str], label: str = "") -> Dict:
        """
        How do token representations change as they pass through layers?
        
        Measures the "journey" of meaning through the network.
        If the Quran causes different trajectories, that's visible here.
        """
        print(f"\n  Analyzing hidden state trajectories for {label}...")
        
        layer_similarities = {l: [] for l in range(self.n_layers - 1)}
        layer_norms = {l: [] for l in range(self.n_layers)}
        
        sample = texts[:100]
        
        for text in tqdm(sample, desc=f"  HID-{label}", unit="verse"):
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encoded["input_ids"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
            
            # Measure layer-to-layer change
            for l in range(len(hidden_states) - 1):
                h_curr = hidden_states[l][0].cpu().numpy()    # (seq, hidden)
                h_next = hidden_states[l+1][0].cpu().numpy()  # (seq, hidden)
                
                # Cosine similarity between consecutive layers (how much does meaning change?)
                for t in range(1, h_curr.shape[0] - 1):
                    cos_sim = np.dot(h_curr[t], h_next[t]) / (
                        np.linalg.norm(h_curr[t]) * np.linalg.norm(h_next[t]) + 1e-12
                    )
                    if l < self.n_layers - 1:
                        layer_similarities[l].append(float(cos_sim))
                
                # Norm of hidden states (how "activated" is the network?)
                if l < self.n_layers:
                    mean_norm = float(np.mean(np.linalg.norm(h_curr[1:-1], axis=1)))
                    layer_norms[l].append(mean_norm)
        
        trajectory = {}
        for l in range(self.n_layers - 1):
            if layer_similarities[l]:
                trajectory[f"layer_{l}_to_{l+1}_similarity"] = float(np.mean(layer_similarities[l]))
        
        norms = {}
        for l in range(self.n_layers):
            if layer_norms[l]:
                norms[f"layer_{l}_norm"] = float(np.mean(layer_norms[l]))
        
        return {
            "layer_transition_similarities": trajectory,
            "layer_norms": norms,
            "mean_transition_similarity": float(np.mean(
                [np.mean(v) for v in layer_similarities.values() if v]
            )),
        }


class TextIntrinsicAnalyzer:
    """
    Text-intrinsic information theory metrics.
    These don't use a model — they measure properties of the text itself.
    """

    @staticmethod
    def character_entropy(text: str) -> float:
        """Shannon entropy at character level."""
        counts = Counter(text)
        total = len(text)
        probs = [c / total for c in counts.values()]
        return float(scipy_entropy(probs, base=2))

    @staticmethod
    def word_entropy(text: str) -> float:
        """Shannon entropy at word level."""
        words = text.split()
        counts = Counter(words)
        total = len(words)
        probs = [c / total for c in counts.values()]
        return float(scipy_entropy(probs, base=2))

    @staticmethod
    def compressibility(text: str) -> Dict[str, float]:
        """
        Approximate Kolmogorov complexity via compression ratio.
        Lower ratio = more compressible = more internal structure.
        """
        text_bytes = text.encode("utf-8")
        original_size = len(text_bytes)
        
        results = {}
        for name, compress_fn in [("gzip", gzip.compress), ("bz2", bz2.compress), ("lzma", lzma.compress)]:
            compressed = compress_fn(text_bytes)
            ratio = len(compressed) / original_size
            results[f"{name}_ratio"] = float(ratio)
            results[f"{name}_compressed_bytes"] = len(compressed)
        
        results["original_bytes"] = original_size
        return results

    @staticmethod
    def zipf_analysis(text: str) -> Dict[str, float]:
        """
        How well does word frequency follow Zipf's law?
        Zipf's law: frequency ∝ 1/rank
        Tighter fit = more "natural" language structure.
        """
        words = text.split()
        counts = Counter(words)
        frequencies = sorted(counts.values(), reverse=True)
        ranks = np.arange(1, len(frequencies) + 1)
        
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)
        
        # Linear regression on log-log scale
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
        
        return {
            "zipf_exponent": float(-slope),  # Should be ~1.0 for perfect Zipf
            "zipf_r_squared": float(r_value ** 2),  # Goodness of fit
            "zipf_p_value": float(p_value),
            "vocabulary_size": len(counts),
            "total_words": len(words),
            "type_token_ratio": len(counts) / len(words) if words else 0,
        }

    @staticmethod  
    def mutual_information_distance(text: str, window: int = 50) -> float:
        """
        Mutual information between words at distance `window` apart.
        High MI at long distances = long-range coherence in the text.
        """
        words = text.split()
        if len(words) < window + 100:
            return 0.0
        
        # Build co-occurrence counts
        pair_counts = Counter()
        word_counts_a = Counter()
        word_counts_b = Counter()
        n_pairs = 0
        
        for i in range(len(words) - window):
            a = words[i]
            b = words[i + window]
            pair_counts[(a, b)] += 1
            word_counts_a[a] += 1
            word_counts_b[b] += 1
            n_pairs += 1
        
        if n_pairs == 0:
            return 0.0
        
        mi = 0.0
        for (a, b), count in pair_counts.items():
            p_ab = count / n_pairs
            p_a = word_counts_a[a] / n_pairs
            p_b = word_counts_b[b] / n_pairs
            if p_ab > 0 and p_a > 0 and p_b > 0:
                mi += p_ab * math.log2(p_ab / (p_a * p_b))
        
        return float(mi)


class Phase3Runner:
    """Orchestrates Phase 3: what the Quran does to the transformer's math."""

    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        print("=" * 70)
        print("PROJECT NUR — PHASE 3: INFORMATION-THEORETIC FINGERPRINTING")
        print("What does the Quran DO to the transformer's internal math?")
        print("=" * 70)

        # Load texts
        quran_df = pd.read_parquet("data/quran/quran_dataset.parquet")
        quran_texts = quran_df["text_uthmani"].tolist()
        quran_full_text = " ".join(quran_texts)

        # Load shuffled controls
        controls = {}
        for name in ["quran_shuffled_words", "quran_shuffled_verses"]:
            path = Path(f"data/controls/shuffled/{name}.parquet")
            if path.exists():
                df = pd.read_parquet(path)
                controls[name] = df["text_uthmani"].tolist()

        results = {}

        # ─── PART A: Text-Intrinsic Metrics ───
        print("\n\n### PART A: TEXT-INTRINSIC INFORMATION THEORY ###")
        intrinsic = TextIntrinsicAnalyzer()

        for corpus_name, texts in [("QURAN", quran_texts)] + [(k, v) for k, v in controls.items()]:
            full_text = " ".join(texts)
            print(f"\n  [{corpus_name}]")

            char_ent = intrinsic.character_entropy(full_text)
            word_ent = intrinsic.word_entropy(full_text)
            compress = intrinsic.compressibility(full_text)
            zipf = intrinsic.zipf_analysis(full_text)
            mi_10 = intrinsic.mutual_information_distance(full_text, window=10)
            mi_50 = intrinsic.mutual_information_distance(full_text, window=50)
            mi_200 = intrinsic.mutual_information_distance(full_text, window=200)

            results[f"{corpus_name}_intrinsic"] = {
                "char_entropy": char_ent,
                "word_entropy": word_ent,
                "compression": compress,
                "zipf": zipf,
                "mutual_info_w10": mi_10,
                "mutual_info_w50": mi_50,
                "mutual_info_w200": mi_200,
            }

            print(f"    Char entropy:   {char_ent:.4f} bits")
            print(f"    Word entropy:   {word_ent:.4f} bits")
            print(f"    Zipf exponent:  {zipf['zipf_exponent']:.4f} (ideal=1.0)")
            print(f"    Zipf R-squared: {zipf['zipf_r_squared']:.4f}")
            print(f"    gzip ratio:     {compress['gzip_ratio']:.4f}")
            print(f"    bz2 ratio:      {compress['bz2_ratio']:.4f}")
            print(f"    lzma ratio:     {compress['lzma_ratio']:.4f}")
            print(f"    MI (w=10):      {mi_10:.6f}")
            print(f"    MI (w=50):      {mi_50:.6f}")
            print(f"    MI (w=200):     {mi_200:.6f}")
            print(f"    Vocab size:     {zipf['vocabulary_size']}")
            print(f"    TTR:            {zipf['type_token_ratio']:.4f}")

        # ─── PART B: Transformer Internals ───
        print("\n\n### PART B: TRANSFORMER INTERNAL ANALYSIS ###")
        print("### What does the text DO to the machine? ###")
        
        transformer = TransformerInternalsAnalyzer()

        for corpus_name, texts in [("QURAN", quran_texts)] + [(k, v) for k, v in controls.items()]:
            print(f"\n\n{'=' * 50}")
            print(f"  CORPUS: {corpus_name}")
            print(f"{'=' * 50}")

            # Pseudo-perplexity
            ppl = transformer.compute_pseudo_perplexity(texts, corpus_name)
            results[f"{corpus_name}_perplexity"] = ppl
            print(f"\n  PSEUDO-PERPLEXITY: {ppl['pseudo_perplexity']:.2f}")
            print(f"  Mean token loss:   {ppl['mean_loss']:.4f}")
            print(f"  Tokens analyzed:   {ppl['n_tokens']}")

            # Attention patterns
            att = transformer.analyze_attention_patterns(texts, corpus_name)
            results[f"{corpus_name}_attention"] = att
            print(f"\n  ATTENTION ENTROPY: {att['mean_attention_entropy']:.4f}")
            print(f"  Long-range ratio:  {att['long_range_attention_ratio']:.4f}")

            # Hidden state trajectory
            hid = transformer.analyze_hidden_state_trajectory(texts, corpus_name)
            results[f"{corpus_name}_trajectory"] = hid
            print(f"\n  LAYER TRANSITION:  {hid['mean_transition_similarity']:.4f}")

        # ─── PART C: Comparative Report ───
        self._comparative_report(results)
        self._save_results(results)

        return results

    def _comparative_report(self, results: Dict):
        print(f"\n\n{'=' * 70}")
        print("PHASE 3 COMPARATIVE REPORT")
        print(f"{'=' * 70}")

        corpora = ["QURAN", "quran_shuffled_words", "quran_shuffled_verses"]
        
        print(f"\n{'Metric':<30} {'QURAN':>12} {'SHUF_WORDS':>12} {'SHUF_VERSES':>12}")
        print("-" * 70)

        metrics = [
            ("Char Entropy (bits)", "intrinsic", "char_entropy"),
            ("Word Entropy (bits)", "intrinsic", "word_entropy"),
            ("Zipf Exponent", "intrinsic", lambda r: r.get("zipf", {}).get("zipf_exponent", 0)),
            ("Zipf R-squared", "intrinsic", lambda r: r.get("zipf", {}).get("zipf_r_squared", 0)),
            ("gzip Ratio", "intrinsic", lambda r: r.get("compression", {}).get("gzip_ratio", 0)),
            ("MI (window=50)", "intrinsic", "mutual_info_w50"),
            ("MI (window=200)", "intrinsic", "mutual_info_w200"),
            ("Pseudo-Perplexity", "perplexity", "pseudo_perplexity"),
            ("Attention Entropy", "attention", "mean_attention_entropy"),
            ("Long-Range Att.", "attention", "long_range_attention_ratio"),
            ("Layer Transition", "trajectory", "mean_transition_similarity"),
        ]

        for label, suffix, key in metrics:
            vals = []
            for c in corpora:
                r = results.get(f"{c}_{suffix}", {})
                if callable(key):
                    v = key(r)
                else:
                    v = r.get(key, 0)
                vals.append(v)
            
            print(f"{label:<30} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}")

    def _save_results(self, results: Dict):
        # Convert any non-serializable values
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

        output_path = self.results_dir / "phase3_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean(results), f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Results saved to: {output_path}")


if __name__ == "__main__":
    runner = Phase3Runner()
    runner.run()
