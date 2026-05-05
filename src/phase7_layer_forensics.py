# Project Nur — Phase 7: Layer-by-Layer Attention Forensics
# Bismillah
#
# WHY does the perplexity inversion happen?
# We dissect each model layer-by-layer to find WHERE
# the Quran causes anomalous internal behavior.
#
# For each model × each corpus, we extract:
# 1. Per-layer attention entropy (how diffuse/focused is attention)
# 2. Per-layer long-range attention ratio (how far back does the model look)
# 3. Per-layer hidden state norm (how much "energy" flows through)
# 4. Per-layer hidden state transition (how much does representation change)

import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def strip_diacritics(text: str) -> str:
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]')
    return diacritics.sub('', text.replace('\u0640', '')).strip()


class LayerForensics:
    """
    Dissects transformer behavior layer-by-layer across corpora.
    Answers: WHERE in the network does the Quran diverge?
    """

    MODELS = {
        "CAMeLBERT-CA": "CAMeL-Lab/bert-base-arabic-camelbert-ca",
        "AraBERT-v2": "aubmindlab/bert-base-arabertv02",
        "XLM-RoBERTa": "xlm-roberta-base",
    }

    def __init__(self, n_samples: int = 100, window_words: int = 15, seed: int = 42):
        self.n_samples = n_samples
        self.window_words = window_words
        self.rng = np.random.default_rng(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_windows(self, texts: List[str], name: str) -> List[str]:
        """Extract fixed-length word windows."""
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

    def analyze_model(self, model_name: str, model_id: str, 
                      corpus_windows: Dict[str, List[str]]) -> Dict:
        """Full layer forensics for one model."""
        print(f"\n{'=' * 60}")
        print(f"  MODEL: {model_name}")
        print(f"{'=' * 60}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, output_attentions=True,
                                          output_hidden_states=True)
        model.to(self.device)
        model.eval()

        n_layers = model.config.num_hidden_layers
        results = {}

        for corpus_name, windows in corpus_windows.items():
            print(f"\n  Analyzing {corpus_name} ({len(windows)} windows)...")

            # Accumulators per layer
            layer_entropies = [[] for _ in range(n_layers)]
            layer_long_range = [[] for _ in range(n_layers)]
            layer_norms = [[] for _ in range(n_layers)]
            layer_transitions = [[] for _ in range(n_layers)]

            for window in tqdm(windows, desc=f"  {corpus_name}", unit="win"):
                encoded = tokenizer(window, return_tensors="pt", 
                                    truncation=True, max_length=64)
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                seq_len = input_ids.shape[1]

                if seq_len < 3:
                    continue

                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)

                attentions = outputs.attentions      # tuple of (1, heads, seq, seq)
                hidden_states = outputs.hidden_states # tuple of (1, seq, dim), len = layers+1

                for layer_i in range(n_layers):
                    # --- Attention analysis ---
                    att = attentions[layer_i][0]  # (heads, seq, seq)
                    # Average across heads
                    att_avg = att.mean(dim=0)  # (seq, seq)

                    # Entropy of attention distribution per token, then average
                    # Only for non-padding tokens
                    valid_len = attention_mask.sum().item()
                    entropies = []
                    long_range_scores = []

                    for pos in range(1, min(int(valid_len), seq_len) - 1):
                        dist = att_avg[pos, :int(valid_len)]
                        dist = dist.clamp(min=1e-12)
                        ent = -(dist * dist.log()).sum().item()
                        entropies.append(ent)

                        # Long-range: fraction of attention to tokens > 5 positions away
                        if pos > 5:
                            long_range = dist[:pos - 5].sum().item()
                        else:
                            long_range = 0.0
                        long_range_scores.append(long_range)

                    if entropies:
                        layer_entropies[layer_i].append(np.mean(entropies))
                        layer_long_range[layer_i].append(np.mean(long_range_scores))

                    # --- Hidden state analysis ---
                    hs = hidden_states[layer_i + 1][0]  # (seq, dim) — layer output
                    hs_prev = hidden_states[layer_i][0]  # (seq, dim) — previous layer

                    # Norm of hidden states (average across valid tokens)
                    norms = torch.norm(hs[:int(valid_len)], dim=-1).mean().item()
                    layer_norms[layer_i].append(norms)

                    # Transition: cosine similarity between consecutive layers
                    cos_sim = torch.nn.functional.cosine_similarity(
                        hs[:int(valid_len)], hs_prev[:int(valid_len)], dim=-1
                    ).mean().item()
                    layer_transitions[layer_i].append(cos_sim)

            # Aggregate per layer
            corpus_result = {
                "n_layers": n_layers,
                "layers": {}
            }

            for layer_i in range(n_layers):
                corpus_result["layers"][f"layer_{layer_i}"] = {
                    "attention_entropy": float(np.mean(layer_entropies[layer_i])) if layer_entropies[layer_i] else 0,
                    "attention_entropy_std": float(np.std(layer_entropies[layer_i])) if layer_entropies[layer_i] else 0,
                    "long_range_ratio": float(np.mean(layer_long_range[layer_i])) if layer_long_range[layer_i] else 0,
                    "hidden_norm": float(np.mean(layer_norms[layer_i])) if layer_norms[layer_i] else 0,
                    "transition_similarity": float(np.mean(layer_transitions[layer_i])) if layer_transitions[layer_i] else 0,
                }

            results[corpus_name] = corpus_result

        # Cleanup
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def run(self) -> Dict:
        print("=" * 70)
        print("PROJECT NUR — PHASE 7: LAYER-BY-LAYER ATTENTION FORENSICS")
        print("WHERE in the transformer does the Quran diverge?")
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

        # Extract equal-length windows
        print("\n### Extracting fixed-length windows ###")
        corpus_windows = {}
        for name, texts in corpora_raw.items():
            corpus_windows[name] = self.extract_windows(texts, name)
            print(f"  {name}: {len(corpus_windows[name])} windows")

        # Analyze per model
        all_results = {}
        for model_name, model_id in self.MODELS.items():
            try:
                model_results = self.analyze_model(model_name, model_id, corpus_windows)
                all_results[model_name] = model_results
            except Exception as e:
                print(f"\n  [ERR] {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Print divergence report
        self._divergence_report(all_results)

        # Save
        self._save(all_results)
        return all_results

    def _divergence_report(self, all_results: Dict):
        """Find WHERE the Quran diverges most from controls."""
        print(f"\n\n{'=' * 90}")
        print("DIVERGENCE REPORT — Where does the Quran break from other texts?")
        print(f"{'=' * 90}")

        for model_name, model_results in all_results.items():
            print(f"\n\n  ══ {model_name} ══")
            
            if "QURAN" not in model_results:
                continue

            n_layers = model_results["QURAN"]["n_layers"]
            controls = [c for c in model_results if c != "QURAN"]

            # Per-layer comparison
            print(f"\n  {'Layer':<8} {'Metric':<22} {'QURAN':>8} {'Avg_Ctrl':>10} {'Delta%':>10} {'Verdict':>12}")
            print("  " + "-" * 72)

            for layer_i in range(n_layers):
                layer_key = f"layer_{layer_i}"
                q = model_results["QURAN"]["layers"][layer_key]

                for metric in ["attention_entropy", "long_range_ratio", "hidden_norm", "transition_similarity"]:
                    q_val = q[metric]
                    ctrl_vals = [model_results[c]["layers"][layer_key][metric] 
                                for c in controls if layer_key in model_results[c]["layers"]]
                    
                    if not ctrl_vals:
                        continue
                    
                    ctrl_avg = np.mean(ctrl_vals)
                    if ctrl_avg != 0:
                        delta_pct = ((q_val - ctrl_avg) / abs(ctrl_avg)) * 100
                    else:
                        delta_pct = 0

                    # Flag large divergences
                    if abs(delta_pct) > 15:
                        verdict = "▐█ ANOMALY"
                    elif abs(delta_pct) > 8:
                        verdict = "▐▌ Notable"
                    else:
                        verdict = ""

                    short_metric = metric.replace("attention_", "att_").replace("transition_", "trans_").replace("hidden_", "h_")
                    
                    if verdict:
                        print(f"  L{layer_i:<6} {short_metric:<22} {q_val:>8.4f} {ctrl_avg:>10.4f} {delta_pct:>+9.1f}% {verdict}")

            # Summary: which layers diverge most?
            print(f"\n  LAYER DIVERGENCE RANKING (by total absolute delta):")
            layer_total_delta = []
            for layer_i in range(n_layers):
                layer_key = f"layer_{layer_i}"
                q = model_results["QURAN"]["layers"][layer_key]
                total = 0
                for metric in ["attention_entropy", "long_range_ratio", "hidden_norm", "transition_similarity"]:
                    q_val = q[metric]
                    ctrl_vals = [model_results[c]["layers"][layer_key][metric] 
                                for c in controls if layer_key in model_results[c]["layers"]]
                    if ctrl_vals:
                        ctrl_avg = np.mean(ctrl_vals)
                        if ctrl_avg != 0:
                            total += abs((q_val - ctrl_avg) / abs(ctrl_avg)) * 100
                layer_total_delta.append((layer_i, total))

            layer_total_delta.sort(key=lambda x: x[1], reverse=True)
            for rank, (layer_i, delta) in enumerate(layer_total_delta[:5]):
                bar = "█" * int(delta / 5)
                print(f"    #{rank+1}  Layer {layer_i:>2}: {delta:>6.1f}% total divergence  {bar}")

    def _save(self, all_results: Dict):
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

        out = Path("data/results/layer_forensics.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(clean(all_results), f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Results saved to: {out}")


if __name__ == "__main__":
    forensics = LayerForensics(n_samples=100, window_words=15)
    forensics.run()
