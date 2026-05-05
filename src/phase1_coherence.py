# Project Nur — Phase 1: Cross-Temporal Semantic Coherence Analysis
# Bismillah
#
# THE CORE QUESTION:
# Verses revealed across 23 years, in different cities, addressing different
# circumstances — do they maintain mathematical coherence in the embedding space
# at a level that exceeds what any human-authored text achieves across
# comparable time spans?
#
# This is the proof-of-concept experiment. If the signal exists here,
# we expand to Phases 2-6. If not, we recalibrate.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class CoherenceMetrics:
    """Results of a coherence analysis for one corpus."""
    corpus_name: str
    
    # Intra-period coherence: how similar are verses WITHIN each period?
    intra_coherence_by_period: Dict[str, float]  # period -> mean cosine similarity
    intra_coherence_overall: float                 # weighted average across periods
    
    # Inter-period coherence: how similar are verses ACROSS periods?
    inter_coherence_by_pair: Dict[str, float]     # "periodA_vs_periodB" -> mean cosine sim
    inter_coherence_overall: float                 # average across all period pairs
    
    # THE KEY METRIC: coherence ratio = inter / intra
    # High ratio = text maintains coherence across time periods
    # A ratio close to 1.0 means cross-period coherence ≈ within-period coherence
    # (the text is as coherent across 23 years as within a single period)
    coherence_ratio: float
    
    # Variance metrics
    intra_variance: float    # How much does within-period coherence vary?
    inter_variance: float    # How much does cross-period coherence vary?
    
    # Additional metrics
    global_coherence: float  # Mean cosine similarity across ALL verse pairs
    n_verses: int


class CrossTemporalCoherenceAnalyzer:
    """
    Analyzes whether a text maintains semantic coherence across its
    temporal composition periods, and compares this against controls.
    """

    REVELATION_PERIODS = ["early_meccan", "middle_meccan", "late_meccan", "medinan"]

    def __init__(self, sample_size: int = 500, seed: int = 42):
        """
        Args:
            sample_size: Max verses to sample per period for pairwise computation.
                         Computing all 6236×6236 pairs is ~39M comparisons.
                         We sample to make computation tractable while maintaining
                         statistical validity.
            seed: Random seed for reproducibility.
        """
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)

    def compute_coherence(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        corpus_name: str = "unnamed",
        period_column: str = "revelation_period",
    ) -> CoherenceMetrics:
        """
        Compute cross-temporal coherence metrics for a corpus.
        
        Args:
            embeddings: (n_verses, embedding_dim) array
            metadata: DataFrame with period labels for each verse
            corpus_name: Name for reporting
            period_column: Column name containing period labels
            
        Returns:
            CoherenceMetrics with all computed values
        """
        print(f"\n{'─' * 50}")
        print(f"Computing coherence: {corpus_name}")
        print(f"{'─' * 50}")

        periods = self.REVELATION_PERIODS
        period_indices = {}
        for period in periods:
            idx = metadata[metadata[period_column] == period].index.tolist()
            if len(idx) > self.sample_size:
                idx = self.rng.choice(idx, self.sample_size, replace=False).tolist()
            period_indices[period] = idx
            print(f"  {period}: {len(idx)} verses (sampled from {(metadata[period_column] == period).sum()})")

        # ─── Intra-period coherence ───
        # Mean cosine similarity between verse pairs WITHIN each period
        intra_coherence = {}
        for period in periods:
            idx = period_indices[period]
            if len(idx) < 2:
                intra_coherence[period] = 0.0
                continue
            period_emb = embeddings[idx]
            sim_matrix = cosine_similarity(period_emb)
            # Extract upper triangle (exclude diagonal = self-similarity = 1.0)
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            intra_coherence[period] = float(np.mean(upper_tri))
            print(f"  Intra-{period}: {intra_coherence[period]:.6f} "
                  f"(σ={np.std(upper_tri):.6f}, n_pairs={len(upper_tri)})")

        # Weighted average intra-coherence (weighted by number of verses)
        weights = [len(period_indices[p]) for p in periods]
        values = [intra_coherence[p] for p in periods]
        intra_overall = float(np.average(values, weights=weights))

        # ─── Inter-period coherence ───
        # Mean cosine similarity between verse pairs ACROSS different periods
        inter_coherence = {}
        for i, p1 in enumerate(periods):
            for j, p2 in enumerate(periods):
                if j <= i:
                    continue
                idx1 = period_indices[p1]
                idx2 = period_indices[p2]
                if not idx1 or not idx2:
                    inter_coherence[f"{p1}_vs_{p2}"] = 0.0
                    continue

                emb1 = embeddings[idx1]
                emb2 = embeddings[idx2]
                cross_sim = cosine_similarity(emb1, emb2)
                mean_sim = float(np.mean(cross_sim))
                inter_coherence[f"{p1}_vs_{p2}"] = mean_sim
                print(f"  Inter-{p1[:5]}-{p2[:5]}: {mean_sim:.6f} "
                      f"(σ={np.std(cross_sim):.6f}, n_pairs={cross_sim.size})")

        inter_overall = float(np.mean(list(inter_coherence.values())))

        # ─── Coherence Ratio ───
        coherence_ratio = inter_overall / intra_overall if intra_overall > 0 else 0.0
        print(f"\n  ► Intra-period (overall):  {intra_overall:.6f}")
        print(f"  ► Inter-period (overall):  {inter_overall:.6f}")
        print(f"  ► COHERENCE RATIO:         {coherence_ratio:.6f}")

        # ─── Global coherence ───
        # Sample global pairs for overall coherence
        n_global = min(2000, len(embeddings))
        global_idx = self.rng.choice(len(embeddings), n_global, replace=False)
        global_emb = embeddings[global_idx]
        global_sim = cosine_similarity(global_emb)
        global_upper = global_sim[np.triu_indices_from(global_sim, k=1)]
        global_coherence = float(np.mean(global_upper))

        return CoherenceMetrics(
            corpus_name=corpus_name,
            intra_coherence_by_period=intra_coherence,
            intra_coherence_overall=intra_overall,
            inter_coherence_by_pair=inter_coherence,
            inter_coherence_overall=inter_overall,
            coherence_ratio=coherence_ratio,
            intra_variance=float(np.var(list(intra_coherence.values()))),
            inter_variance=float(np.var(list(inter_coherence.values()))),
            global_coherence=global_coherence,
            n_verses=len(embeddings),
        )

    def permutation_test(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        observed_ratio: float,
        n_permutations: int = 10000,
        period_column: str = "revelation_period",
    ) -> Tuple[float, np.ndarray]:
        """
        Permutation test for the coherence ratio.
        
        Null hypothesis: The period labels are independent of the embeddings.
        We shuffle the period labels and recompute the coherence ratio.
        If the observed ratio is extreme relative to the null distribution,
        the coherence is genuine — not an artifact of the embedding space.
        
        Args:
            embeddings: The verse embeddings
            metadata: DataFrame with period labels
            observed_ratio: The observed coherence ratio to test against
            n_permutations: Number of permutation iterations
            period_column: Column with period labels
            
        Returns:
            (p_value, null_distribution)
        """
        print(f"\n  Running permutation test ({n_permutations} iterations)...")

        null_ratios = []
        periods_array = metadata[period_column].values.copy()

        for perm_i in range(n_permutations):
            # Shuffle period labels
            shuffled_periods = self.rng.permutation(periods_array)
            shuffled_meta = metadata.copy()
            shuffled_meta[period_column] = shuffled_periods

            # Compute coherence ratio with shuffled labels
            # Use smaller sample for speed
            small_analyzer = CrossTemporalCoherenceAnalyzer(
                sample_size=min(100, self.sample_size),
                seed=self.rng.integers(0, 2**31),
            )

            # Suppress output during permutations
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                metrics = small_analyzer.compute_coherence(
                    embeddings, shuffled_meta, f"perm_{perm_i}", period_column
                )
                null_ratios.append(metrics.coherence_ratio)
            finally:
                sys.stdout = old_stdout

            if (perm_i + 1) % 1000 == 0:
                print(f"    Completed {perm_i + 1}/{n_permutations} permutations")

        null_ratios = np.array(null_ratios)

        # P-value: proportion of null ratios >= observed ratio
        p_value = float(np.mean(null_ratios >= observed_ratio))

        print(f"  Permutation test results:")
        print(f"    Observed ratio:   {observed_ratio:.6f}")
        print(f"    Null mean:        {np.mean(null_ratios):.6f}")
        print(f"    Null std:         {np.std(null_ratios):.6f}")
        print(f"    P-value:          {p_value:.6f}")
        print(f"    Significant (p<0.001): {'YES ✓' if p_value < 0.001 else 'NO ✗'}")

        return p_value, null_ratios

    def compute_effect_size(
        self,
        quran_ratio: float,
        control_ratios: List[float],
    ) -> Dict[str, float]:
        """
        Compute effect size (Cohen's d) of Quran coherence ratio vs. controls.
        
        A large effect size (|d| > 0.8) indicates a meaningful difference.
        """
        if len(control_ratios) < 2:
            return {"cohens_d": float("nan"), "interpretation": "insufficient controls"}

        control_mean = np.mean(control_ratios)
        control_std = np.std(control_ratios, ddof=1)

        if control_std == 0:
            return {"cohens_d": float("inf") if quran_ratio > control_mean else 0.0,
                    "interpretation": "zero variance in controls"}

        d = (quran_ratio - control_mean) / control_std

        # Interpretation thresholds (Cohen, 1988)
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            "cohens_d": float(d),
            "quran_ratio": quran_ratio,
            "control_mean": float(control_mean),
            "control_std": float(control_std),
            "interpretation": interpretation,
        }


class Phase1Runner:
    """
    Orchestrates the complete Phase 1 analysis:
    1. Load Quran embeddings + metadata
    2. Load control embeddings + metadata
    3. Compute coherence for each corpus
    4. Run statistical tests
    5. Generate report
    """

    def __init__(
        self,
        quran_embeddings_path: str = "data/embeddings/quran_embeddings.npy",
        quran_metadata_path: str = "data/embeddings/quran_embeddings.meta.parquet",
        controls_dir: str = "data/embeddings",
        results_dir: str = "data/results",
        sample_size: int = 500,
        n_permutations: int = 10000,
    ):
        self.quran_emb_path = Path(quran_embeddings_path)
        self.quran_meta_path = Path(quran_metadata_path)
        self.controls_dir = Path(controls_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = CrossTemporalCoherenceAnalyzer(sample_size=sample_size)
        self.n_permutations = n_permutations

    def run(self) -> Dict:
        """Execute the complete Phase 1 analysis."""
        print("=" * 70)
        print("PROJECT NUR — PHASE 1: CROSS-TEMPORAL SEMANTIC COHERENCE")
        print("=" * 70)

        results = {}

        # ─── 1. Quran Analysis ───
        print("\n\n█ STEP 1: Analyzing the Quran")
        print("█" * 50)
        quran_emb = np.load(self.quran_emb_path)
        quran_meta = pd.read_parquet(self.quran_meta_path)
        quran_metrics = self.analyzer.compute_coherence(
            quran_emb, quran_meta, "QURAN"
        )
        results["quran"] = quran_metrics

        # ─── 2. Control Analyses ───
        control_results = {}
        control_files = {
            "shuffled_words": "quran_shuffled_words",
            "shuffled_verses": "quran_shuffled_verses",
            "shuffled_surahs": "quran_shuffled_surahs",
        }

        for control_name, file_prefix in control_files.items():
            emb_path = self.controls_dir / f"{file_prefix}_embeddings.npy"
            meta_path = self.controls_dir / f"{file_prefix}_embeddings.meta.parquet"

            if not emb_path.exists():
                print(f"\n  ⚠ Control not found: {emb_path} — skipping")
                continue

            print(f"\n\n█ CONTROL: {control_name}")
            print("█" * 50)
            control_emb = np.load(emb_path)
            control_meta = pd.read_parquet(meta_path)
            control_metrics = self.analyzer.compute_coherence(
                control_emb, control_meta, control_name
            )
            control_results[control_name] = control_metrics

        results["controls"] = control_results

        # ─── 3. Permutation Test ───
        print("\n\n█ STEP 3: Permutation Test")
        print("█" * 50)
        p_value, null_dist = self.analyzer.permutation_test(
            quran_emb, quran_meta,
            quran_metrics.coherence_ratio,
            n_permutations=self.n_permutations,
        )
        results["permutation_test"] = {
            "p_value": p_value,
            "null_mean": float(np.mean(null_dist)),
            "null_std": float(np.std(null_dist)),
            "observed_ratio": quran_metrics.coherence_ratio,
            "significant": p_value < 0.001,
        }

        # Save null distribution for visualization
        np.save(self.results_dir / "null_distribution.npy", null_dist)

        # ─── 4. Effect Size ───
        if control_results:
            print("\n\n█ STEP 4: Effect Size Analysis")
            print("█" * 50)
            control_ratios = [m.coherence_ratio for m in control_results.values()]
            effect = self.analyzer.compute_effect_size(
                quran_metrics.coherence_ratio, control_ratios
            )
            results["effect_size"] = effect
            print(f"  Cohen's d: {effect['cohens_d']:.4f} ({effect['interpretation']})")

        # ─── 5. Generate Report ───
        self._generate_report(results)

        # ─── 6. Save Results ───
        self._save_results(results)

        return results

    def _generate_report(self, results: Dict):
        """Generate a human-readable Phase 1 report."""
        print("\n\n" + "=" * 70)
        print("PHASE 1 REPORT — CROSS-TEMPORAL SEMANTIC COHERENCE")
        print("=" * 70)

        quran = results["quran"]
        print(f"\n┌─ QURAN ─────────────────────────────────────────┐")
        print(f"│ Intra-period coherence:  {quran.intra_coherence_overall:.6f}          │")
        print(f"│ Inter-period coherence:  {quran.inter_coherence_overall:.6f}          │")
        print(f"│ COHERENCE RATIO:         {quran.coherence_ratio:.6f}          │")
        print(f"│ Global coherence:        {quran.global_coherence:.6f}          │")
        print(f"│ Verses analyzed:         {quran.n_verses}                  │")
        print(f"└─────────────────────────────────────────────────┘")

        if results.get("controls"):
            print(f"\n┌─ CONTROLS ──────────────────────────────────────┐")
            for name, metrics in results["controls"].items():
                print(f"│ {name:25s} ratio: {metrics.coherence_ratio:.6f}    │")
            print(f"└─────────────────────────────────────────────────┘")

        if results.get("permutation_test"):
            perm = results["permutation_test"]
            print(f"\n┌─ PERMUTATION TEST ──────────────────────────────┐")
            print(f"│ Observed ratio:   {perm['observed_ratio']:.6f}                   │")
            print(f"│ Null mean:        {perm['null_mean']:.6f}                   │")
            print(f"│ Null std:         {perm['null_std']:.6f}                   │")
            print(f"│ P-value:          {perm['p_value']:.6f}                   │")
            sig = "YES ✓" if perm["significant"] else "NO ✗"
            print(f"│ Significant:      {sig:30s}│")
            print(f"└─────────────────────────────────────────────────┘")

        if results.get("effect_size"):
            eff = results["effect_size"]
            print(f"\n┌─ EFFECT SIZE ───────────────────────────────────┐")
            print(f"│ Cohen's d:        {eff['cohens_d']:.4f}                      │")
            print(f"│ Interpretation:   {eff['interpretation']:30s}│")
            print(f"└─────────────────────────────────────────────────┘")

        # Verdict
        print(f"\n{'=' * 70}")
        if results.get("permutation_test", {}).get("significant"):
            print("VERDICT: The Quran's cross-temporal semantic coherence is")
            print("statistically significant — it exceeds what random period")
            print("assignment would produce (p < 0.001).")
            if results.get("effect_size", {}).get("interpretation") in ("large", "medium"):
                print("\nThe effect size is " + results["effect_size"]["interpretation"] +
                      ", indicating a meaningful difference from control texts.")
            print("\n→ PROCEED TO PHASE 2.")
        else:
            print("VERDICT: Phase 1 results do not show statistically significant")
            print("cross-temporal coherence anomaly at the p < 0.001 level.")
            print("\n→ RECALIBRATE methodology before proceeding.")
        print("=" * 70)

    def _save_results(self, results: Dict):
        """Save all results to disk."""
        # Convert dataclass objects to dicts for JSON serialization
        serializable = {}

        if "quran" in results:
            q = results["quran"]
            serializable["quran"] = {
                "corpus_name": q.corpus_name,
                "intra_coherence_by_period": q.intra_coherence_by_period,
                "intra_coherence_overall": q.intra_coherence_overall,
                "inter_coherence_by_pair": q.inter_coherence_by_pair,
                "inter_coherence_overall": q.inter_coherence_overall,
                "coherence_ratio": q.coherence_ratio,
                "global_coherence": q.global_coherence,
                "n_verses": q.n_verses,
            }

        if "controls" in results:
            serializable["controls"] = {}
            for name, m in results["controls"].items():
                serializable["controls"][name] = {
                    "corpus_name": m.corpus_name,
                    "coherence_ratio": m.coherence_ratio,
                    "intra_coherence_overall": m.intra_coherence_overall,
                    "inter_coherence_overall": m.inter_coherence_overall,
                    "global_coherence": m.global_coherence,
                }

        if "permutation_test" in results:
            serializable["permutation_test"] = results["permutation_test"]

        if "effect_size" in results:
            serializable["effect_size"] = results["effect_size"]

        output_path = self.results_dir / "phase1_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    runner = Phase1Runner(
        sample_size=500,
        n_permutations=10000,
    )
    results = runner.run()
