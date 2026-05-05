# Project Nur — Phase 2: Ring Composition Detection in Vector Space
# Bismillah
#
# THE CORE QUESTION:
# Michel Cuypers identified ring compositions (A-B-C-X-C'-B'-A') manually
# in individual surahs. Can we detect these computationally across the
# ENTIRE text, including structures too large or subtle for human detection?
#
# Ring composition signature in vector space:
#   If a surah has ring structure, its verse-to-verse cosine similarity
#   matrix will show ANTI-DIAGONAL SYMMETRY — verse 1 mirrors verse N,
#   verse 2 mirrors verse N-1, etc.
#
# We quantify this with a "mirror score" and compare against controls.

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import correlate2d
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SurahRingAnalysis:
    """Ring composition analysis results for a single surah."""
    surah_number: int
    surah_name: str
    n_verses: int
    mirror_score: float          # Correlation between upper-left and flipped lower-right
    anti_diagonal_mean: float    # Mean similarity along the anti-diagonal
    diagonal_mean: float         # Mean similarity along the main diagonal (adjacent verses)
    anti_diag_to_diag_ratio: float  # anti_diagonal / diagonal — high = ring structure
    center_prominence: float     # How much the center verse(s) stand out semantically
    symmetry_p_value: float      # P-value from permutation test of mirror score
    is_significant: bool         # Whether mirror_score is significant at p < 0.05


@dataclass
class RingCompositionResults:
    """Complete Phase 2 results."""
    corpus_name: str
    surah_analyses: List[SurahRingAnalysis]
    significant_surahs: List[int]  # Surah numbers with significant ring structure
    mean_mirror_score: float
    median_mirror_score: float
    top_10_surahs: List[Tuple[int, str, float]]  # (number, name, score)


class RingCompositionDetector:
    """
    Detects ring (chiastic) composition in texts using embedding similarity.
    
    Ring composition: A-B-C-X-C'-B'-A'
    In the similarity matrix, this manifests as:
    - High values along the anti-diagonal (verse i mirrors verse n-i)
    - A prominent center (the 'X' element — the thematic pivot)
    - Symmetry between the upper-left and lower-right triangles
    """

    def __init__(self, min_verses: int = 5, n_permutations: int = 1000, seed: int = 42):
        """
        Args:
            min_verses: Minimum verses in a surah to analyze (too few = meaningless)
            n_permutations: Number of permutations for significance testing
            seed: Random seed
        """
        self.min_verses = min_verses
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(seed)

    def compute_mirror_score(self, sim_matrix: np.ndarray) -> float:
        """
        Compute the mirror score: correlation between the upper triangle
        and the horizontally+vertically flipped upper triangle.
        
        A perfect ring composition would have mirror_score = 1.0.
        Random text would have mirror_score ≈ 0.0.
        """
        n = sim_matrix.shape[0]
        if n < self.min_verses:
            return 0.0

        # Get upper triangle (excluding diagonal)
        upper_tri_indices = np.triu_indices(n, k=1)
        upper_tri = sim_matrix[upper_tri_indices]

        # Flip the matrix (reverse both axes) and get its upper triangle
        flipped = sim_matrix[::-1, ::-1]
        flipped_upper_tri = flipped[upper_tri_indices]

        # Correlation between original and flipped upper triangles
        if np.std(upper_tri) == 0 or np.std(flipped_upper_tri) == 0:
            return 0.0

        corr, _ = stats.pearsonr(upper_tri, flipped_upper_tri)
        return float(corr)

    def compute_anti_diagonal_strength(self, sim_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute the strength of the anti-diagonal relative to the main diagonal.
        
        Anti-diagonal: sim(verse_i, verse_{n-1-i}) — the "mirror" pairs
        Main diagonal (k=1): sim(verse_i, verse_{i+1}) — adjacent verses
        
        Returns: (anti_diagonal_mean, diagonal_mean, ratio)
        """
        n = sim_matrix.shape[0]
        if n < self.min_verses:
            return 0.0, 0.0, 0.0

        # Anti-diagonal values: sim(i, n-1-i)
        anti_diag = [sim_matrix[i, n - 1 - i] for i in range(n) if i != n - 1 - i]
        anti_diag_mean = float(np.mean(anti_diag)) if anti_diag else 0.0

        # Main diagonal (k=1): adjacent verse similarities
        diag_k1 = [sim_matrix[i, i + 1] for i in range(n - 1)]
        diag_mean = float(np.mean(diag_k1)) if diag_k1 else 0.0

        ratio = anti_diag_mean / diag_mean if diag_mean > 0 else 0.0

        return anti_diag_mean, diag_mean, ratio

    def compute_center_prominence(self, sim_matrix: np.ndarray) -> float:
        """
        Compute how prominent the center of the surah is semantically.
        
        In ring composition, the center verse(s) are the thematic pivot.
        They should have higher-than-average similarity to ALL other verses
        (since the entire surah revolves around this center).
        
        Returns: z-score of center verse's mean similarity vs. all verses' mean similarity
        """
        n = sim_matrix.shape[0]
        if n < self.min_verses:
            return 0.0

        # Mean similarity of each verse to all others
        verse_mean_sims = np.mean(sim_matrix, axis=1)  # excludes self? No — includes diagonal

        # For true comparison, zero out diagonal
        sim_no_diag = sim_matrix.copy()
        np.fill_diagonal(sim_no_diag, 0)
        verse_mean_sims = np.sum(sim_no_diag, axis=1) / (n - 1)

        # Center verse(s)
        center_idx = n // 2
        if n % 2 == 0:
            # Even: average the two center verses
            center_sim = (verse_mean_sims[center_idx - 1] + verse_mean_sims[center_idx]) / 2
        else:
            center_sim = verse_mean_sims[center_idx]

        # Z-score
        mean_all = np.mean(verse_mean_sims)
        std_all = np.std(verse_mean_sims)
        if std_all == 0:
            return 0.0

        return float((center_sim - mean_all) / std_all)

    def permutation_test_mirror(self, embeddings: np.ndarray, observed_score: float) -> float:
        """
        Permutation test: shuffle verse order and recompute mirror score.
        
        If the observed mirror score is higher than most shuffled versions,
        the ring structure is genuine — not an artifact.
        """
        n = embeddings.shape[0]
        null_scores = []

        for _ in range(self.n_permutations):
            perm_idx = self.rng.permutation(n)
            perm_emb = embeddings[perm_idx]
            perm_sim = cosine_similarity(perm_emb)
            null_scores.append(self.compute_mirror_score(perm_sim))

        null_scores = np.array(null_scores)
        p_value = float(np.mean(null_scores >= observed_score))
        return p_value

    def analyze_surah(
        self,
        embeddings: np.ndarray,
        surah_number: int,
        surah_name: str,
        run_permutation: bool = True,
    ) -> SurahRingAnalysis:
        """Analyze a single surah for ring composition."""
        n = embeddings.shape[0]

        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)

        # Mirror score
        mirror_score = self.compute_mirror_score(sim_matrix)

        # Anti-diagonal strength
        anti_diag_mean, diag_mean, ratio = self.compute_anti_diagonal_strength(sim_matrix)

        # Center prominence
        center_prom = self.compute_center_prominence(sim_matrix)

        # Permutation test (only for surahs with notable mirror scores)
        p_value = 1.0
        if run_permutation and mirror_score > 0.3 and n >= self.min_verses:
            p_value = self.permutation_test_mirror(embeddings, mirror_score)

        return SurahRingAnalysis(
            surah_number=surah_number,
            surah_name=surah_name,
            n_verses=n,
            mirror_score=mirror_score,
            anti_diagonal_mean=anti_diag_mean,
            diagonal_mean=diag_mean,
            anti_diag_to_diag_ratio=ratio,
            center_prominence=center_prom,
            symmetry_p_value=p_value,
            is_significant=(p_value < 0.05 and mirror_score > 0.3),
        )

    def analyze_corpus(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        corpus_name: str = "QURAN",
    ) -> RingCompositionResults:
        """
        Analyze all surahs in a corpus for ring composition.
        """
        print(f"\n{'=' * 70}")
        print(f"PHASE 2: RING COMPOSITION DETECTION — {corpus_name}")
        print(f"{'=' * 70}")

        surah_analyses = []
        surah_numbers = sorted(metadata["surah_number"].unique())

        for surah_num in surah_numbers:
            mask = metadata["surah_number"] == surah_num
            surah_idx = metadata[mask].index.tolist()
            surah_emb = embeddings[surah_idx]
            n_verses = len(surah_idx)

            # Get surah name from metadata
            surah_name = metadata[mask].iloc[0].get("surah_name_english", f"Surah {surah_num}")
            if pd.isna(surah_name):
                surah_name = f"Surah {surah_num}"

            if n_verses < self.min_verses:
                continue

            analysis = self.analyze_surah(
                surah_emb, surah_num, surah_name,
                run_permutation=True,
            )
            surah_analyses.append(analysis)

            # Progress reporting
            marker = ""
            if analysis.mirror_score > 0.5:
                marker = " <<<< STRONG"
            elif analysis.mirror_score > 0.3:
                marker = " << NOTABLE"

            sig = " [SIG]" if analysis.is_significant else ""
            print(f"  Surah {surah_num:3d} ({surah_name:25s}) | "
                  f"verses={n_verses:3d} | "
                  f"mirror={analysis.mirror_score:.4f} | "
                  f"anti/diag={analysis.anti_diag_to_diag_ratio:.4f} | "
                  f"center_z={analysis.center_prominence:+.2f}"
                  f"{marker}{sig}")

        # Aggregate results
        significant = [a.surah_number for a in surah_analyses if a.is_significant]
        scores = [a.mirror_score for a in surah_analyses]
        sorted_analyses = sorted(surah_analyses, key=lambda a: a.mirror_score, reverse=True)
        top_10 = [(a.surah_number, a.surah_name, a.mirror_score) for a in sorted_analyses[:10]]

        results = RingCompositionResults(
            corpus_name=corpus_name,
            surah_analyses=surah_analyses,
            significant_surahs=significant,
            mean_mirror_score=float(np.mean(scores)) if scores else 0.0,
            median_mirror_score=float(np.median(scores)) if scores else 0.0,
            top_10_surahs=top_10,
        )

        self._print_summary(results)
        return results

    def _print_summary(self, results: RingCompositionResults):
        """Print Phase 2 summary."""
        print(f"\n{'─' * 70}")
        print(f"PHASE 2 SUMMARY — {results.corpus_name}")
        print(f"{'─' * 70}")
        print(f"  Surahs analyzed:     {len(results.surah_analyses)}")
        print(f"  Mean mirror score:   {results.mean_mirror_score:.4f}")
        print(f"  Median mirror score: {results.median_mirror_score:.4f}")
        print(f"  Significant surahs:  {len(results.significant_surahs)}")

        if results.significant_surahs:
            print(f"\n  SIGNIFICANT RING STRUCTURES DETECTED:")
            for a in results.surah_analyses:
                if a.is_significant:
                    print(f"    Surah {a.surah_number} ({a.surah_name}) — "
                          f"mirror={a.mirror_score:.4f}, p={a.symmetry_p_value:.4f}, "
                          f"center_z={a.center_prominence:+.2f}")

        print(f"\n  TOP 10 BY MIRROR SCORE:")
        for num, name, score in results.top_10_surahs:
            print(f"    {num:3d}. {name:25s} — {score:.4f}")


class Phase2Runner:
    """Orchestrates the complete Phase 2 analysis."""

    def __init__(
        self,
        quran_embeddings_path: str = "data/embeddings/quran_embeddings.npy",
        quran_metadata_path: str = "data/embeddings/quran_embeddings.meta.parquet",
        controls_dir: str = "data/embeddings",
        results_dir: str = "data/results",
        min_verses: int = 5,
        n_permutations: int = 1000,
    ):
        self.quran_emb_path = Path(quran_embeddings_path)
        self.quran_meta_path = Path(quran_metadata_path)
        self.controls_dir = Path(controls_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.detector = RingCompositionDetector(
            min_verses=min_verses,
            n_permutations=n_permutations,
        )

    def run(self) -> Dict:
        """Execute Phase 2."""
        print("=" * 70)
        print("PROJECT NUR — PHASE 2: RING COMPOSITION DETECTION")
        print("=" * 70)

        results = {}

        # 1. Analyze Quran
        print("\n\n### STEP 1: Analyzing QURAN ###")
        quran_emb = np.load(self.quran_emb_path)
        quran_meta = pd.read_parquet(self.quran_meta_path)

        # Need surah names — load full dataset
        quran_full_path = Path("data/quran/quran_dataset.parquet")
        if quran_full_path.exists():
            quran_full = pd.read_parquet(quran_full_path)
            # Add surah name to metadata
            name_map = quran_full.groupby("surah_number")["surah_name_english"].first().to_dict()
            quran_meta["surah_name_english"] = quran_meta["surah_number"].map(name_map)
        else:
            quran_meta["surah_name_english"] = quran_meta["surah_number"].apply(lambda x: f"Surah {x}")

        quran_results = self.detector.analyze_corpus(quran_emb, quran_meta, "QURAN")
        results["quran"] = quran_results

        # 2. Analyze shuffled controls
        control_files = {
            "shuffled_words": "quran_shuffled_words",
            "shuffled_verses": "quran_shuffled_verses",
        }

        for control_name, file_prefix in control_files.items():
            emb_path = self.controls_dir / f"{file_prefix}_embeddings.npy"
            meta_path = self.controls_dir / f"{file_prefix}_embeddings.meta.parquet"

            if not emb_path.exists():
                print(f"\n  [SKIP] Control not found: {emb_path}")
                continue

            print(f"\n\n### CONTROL: {control_name} ###")
            control_emb = np.load(emb_path)
            control_meta = pd.read_parquet(meta_path)
            control_meta["surah_name_english"] = control_meta["surah_number"].apply(
                lambda x: f"Surah {x}")

            control_results = self.detector.analyze_corpus(
                control_emb, control_meta, control_name
            )
            results[control_name] = control_results

        # 3. Comparative analysis
        self._comparative_analysis(results)

        # 4. Save results
        self._save_results(results)

        return results

    def _comparative_analysis(self, results: Dict):
        """Compare Quran ring structure against controls."""
        if "quran" not in results:
            return

        print(f"\n\n{'=' * 70}")
        print("COMPARATIVE ANALYSIS — RING COMPOSITION")
        print(f"{'=' * 70}")

        quran = results["quran"]
        print(f"\n  QURAN:")
        print(f"    Mean mirror score:  {quran.mean_mirror_score:.4f}")
        print(f"    Significant surahs: {len(quran.significant_surahs)} / {len(quran.surah_analyses)}")

        for name, control in results.items():
            if name == "quran":
                continue
            print(f"\n  {name.upper()}:")
            print(f"    Mean mirror score:  {control.mean_mirror_score:.4f}")
            print(f"    Significant surahs: {len(control.significant_surahs)} / {len(control.surah_analyses)}")

            # Effect size
            quran_scores = [a.mirror_score for a in quran.surah_analyses]
            control_scores = [a.mirror_score for a in control.surah_analyses]

            if len(quran_scores) > 1 and len(control_scores) > 1:
                t_stat, p_val = stats.ttest_ind(quran_scores, control_scores)
                pooled_std = np.sqrt(
                    (np.var(quran_scores, ddof=1) + np.var(control_scores, ddof=1)) / 2
                )
                cohens_d = (np.mean(quran_scores) - np.mean(control_scores)) / pooled_std if pooled_std > 0 else 0
                print(f"    t-test p-value:     {p_val:.6f}")
                print(f"    Cohen's d:          {cohens_d:.4f}")
                sig = "YES" if p_val < 0.001 else "NO"
                print(f"    Significant:        {sig}")

        # Cross-reference with Cuypers' known findings
        print(f"\n\n  CROSS-REFERENCE WITH KNOWN RING STRUCTURES:")
        print(f"  (Cuypers identified ring composition in these surahs)")
        known_ring_surahs = [1, 2, 5, 12, 24, 33, 47, 48, 58, 71, 74, 96, 108, 112]
        for s_num in known_ring_surahs:
            for a in quran.surah_analyses:
                if a.surah_number == s_num:
                    marker = " *** CONFIRMED" if a.mirror_score > 0.3 else ""
                    print(f"    Surah {s_num:3d} ({a.surah_name:25s}) — "
                          f"mirror={a.mirror_score:.4f}{marker}")
                    break

    def _save_results(self, results: Dict):
        """Save Phase 2 results."""
        serializable = {}

        for corpus_name, res in results.items():
            corpus_data = {
                "corpus_name": res.corpus_name,
                "mean_mirror_score": res.mean_mirror_score,
                "median_mirror_score": res.median_mirror_score,
                "significant_surahs": res.significant_surahs,
                "n_significant": len(res.significant_surahs),
                "n_analyzed": len(res.surah_analyses),
                "top_10": [
                    {"surah": num, "name": name, "score": score}
                    for num, name, score in res.top_10_surahs
                ],
                "all_surahs": [
                    {
                        "surah_number": a.surah_number,
                        "surah_name": a.surah_name,
                        "n_verses": a.n_verses,
                        "mirror_score": a.mirror_score,
                        "anti_diagonal_mean": a.anti_diagonal_mean,
                        "diagonal_mean": a.diagonal_mean,
                        "anti_diag_to_diag_ratio": a.anti_diag_to_diag_ratio,
                        "center_prominence": a.center_prominence,
                        "p_value": a.symmetry_p_value,
                        "is_significant": a.is_significant,
                    }
                    for a in res.surah_analyses
                ],
            }
            serializable[corpus_name] = corpus_data

        output_path = self.results_dir / "phase2_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Results saved to: {output_path}")


if __name__ == "__main__":
    runner = Phase2Runner(
        min_verses=5,
        n_permutations=1000,
    )
    results = runner.run()
