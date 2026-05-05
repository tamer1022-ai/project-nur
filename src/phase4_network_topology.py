# Project Nur — Phase 4: Semantic Network Topology
# Bismillah
#
# THE QUESTION:
# When we map the Quran's vocabulary as a network (words connected when
# they co-occur in the same verse), does the resulting graph have
# anomalous topological properties compared to controls?
#
# Arabic is a root-based language. Words sharing the same trilateral root
# are semantically related. The Quran's network of word co-occurrences
# reveals its deep structural organization.
#
# We measure:
# 1. Small-world coefficient (clustering vs path length)
# 2. Scale-free properties (degree distribution power law)
# 3. Network resilience (robustness to node removal)
# 4. Community structure (modularity)
# 5. Semantic hub analysis (which words are most central?)

import numpy as np
import pandas as pd
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# We use networkx for graph analysis
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[WARN] networkx not installed. Run: pip install networkx")


def strip_arabic_diacritics(text: str) -> str:
    """Remove Arabic diacritics/tashkeel for root-level comparison."""
    # Arabic diacritics Unicode range
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
                            r'\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u08D3-\u08E1'
                            r'\u08E3-\u08FF\uFE70-\uFE7F]')
    # Also remove tatweel and special marks
    text = text.replace('\u0640', '')  # tatweel
    text = re.sub(r'[\u06D7-\u06ED]', '', text)  # Quranic signs
    return diacritics.sub('', text)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for consistent comparison."""
    text = strip_arabic_diacritics(text)
    # Normalize alef variants
    text = re.sub(r'[إأآٱا]', 'ا', text)
    # Normalize teh marbuta
    text = text.replace('ة', 'ه')
    # Normalize alef maksura
    text = text.replace('ى', 'ي')
    # Remove non-Arabic characters except spaces
    text = re.sub(r'[^\u0621-\u064A\s]', '', text)
    return text.strip()


class SemanticNetworkBuilder:
    """
    Builds a word co-occurrence network from verse-level data.
    Two words are connected if they appear in the same verse.
    Edge weight = number of verses they co-occur in.
    """

    def __init__(self, min_word_freq: int = 3, min_word_len: int = 2):
        """
        Args:
            min_word_freq: Minimum frequency for a word to be included
            min_word_len: Minimum character length (filters particles)
        """
        self.min_word_freq = min_word_freq
        self.min_word_len = min_word_len

    def build_network(self, verses: List[str], label: str = "") -> 'nx.Graph':
        """Build co-occurrence network from verse texts."""
        if not HAS_NX:
            raise ImportError("networkx required")

        print(f"\n  Building network for {label}...")

        # Tokenize and normalize
        verse_words = []
        word_freq = Counter()

        for verse in verses:
            normalized = normalize_arabic(verse)
            words = [w for w in normalized.split() if len(w) >= self.min_word_len]
            verse_words.append(words)
            word_freq.update(words)

        # Filter vocabulary
        vocab = {w for w, c in word_freq.items() if c >= self.min_word_freq}
        print(f"    Total unique words: {len(word_freq)}")
        print(f"    After freq filter:  {len(vocab)}")

        # Build graph
        G = nx.Graph()

        # Add nodes with frequency attribute
        for word in vocab:
            G.add_node(word, freq=word_freq[word])

        # Add edges from co-occurrence
        edge_weights = Counter()
        for words in verse_words:
            # Filter to vocabulary
            filtered = [w for w in words if w in vocab]
            # All pairs in this verse
            for w1, w2 in combinations(set(filtered), 2):
                pair = tuple(sorted([w1, w2]))
                edge_weights[pair] += 1

        for (w1, w2), weight in edge_weights.items():
            G.add_edge(w1, w2, weight=weight)

        print(f"    Nodes: {G.number_of_nodes()}")
        print(f"    Edges: {G.number_of_edges()}")
        print(f"    Density: {nx.density(G):.6f}")

        return G


class NetworkTopologyAnalyzer:
    """Analyzes topological properties of the semantic network."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def analyze(self, G: 'nx.Graph', label: str = "") -> Dict:
        """Complete topological analysis of a network."""
        if not HAS_NX:
            raise ImportError("networkx required")

        print(f"\n  Analyzing topology: {label}")
        results = {}

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        results["n_nodes"] = n_nodes
        results["n_edges"] = n_edges
        results["density"] = float(nx.density(G))

        # ─── 1. Degree Distribution ───
        degrees = [d for _, d in G.degree()]
        results["degree_mean"] = float(np.mean(degrees))
        results["degree_median"] = float(np.median(degrees))
        results["degree_std"] = float(np.std(degrees))
        results["degree_max"] = max(degrees)

        # Power law fit (scale-free test)
        degree_counts = Counter(degrees)
        deg_vals = sorted(degree_counts.keys())
        deg_freqs = [degree_counts[d] for d in deg_vals]

        if len(deg_vals) > 5 and min(deg_vals) > 0:
            log_degs = np.log(deg_vals)
            log_freqs = np.log(deg_freqs)
            slope, intercept, r_value, p_val, _ = stats.linregress(log_degs, log_freqs)
            results["power_law_exponent"] = float(-slope)
            results["power_law_r_squared"] = float(r_value ** 2)
            print(f"    Power law exponent: {-slope:.3f} (R²={r_value**2:.3f})")
        else:
            results["power_law_exponent"] = 0
            results["power_law_r_squared"] = 0

        # ─── 2. Clustering Coefficient ───
        avg_clustering = nx.average_clustering(G)
        results["clustering_coefficient"] = float(avg_clustering)
        print(f"    Clustering coeff:   {avg_clustering:.4f}")

        # ─── 3. Connected Components ───
        components = list(nx.connected_components(G))
        largest_cc = max(components, key=len)
        results["n_components"] = len(components)
        results["largest_component_size"] = len(largest_cc)
        results["largest_component_fraction"] = len(largest_cc) / n_nodes

        # Work on largest connected component for path-based metrics
        G_cc = G.subgraph(largest_cc).copy()
        print(f"    Components: {len(components)}, largest: {len(largest_cc)} ({len(largest_cc)/n_nodes*100:.1f}%)")

        # ─── 4. Average Path Length (sampled for speed) ───
        if len(G_cc) > 1:
            sample_size = min(200, len(G_cc))
            sample_nodes = list(self.rng.choice(list(G_cc.nodes()), sample_size, replace=False))
            path_lengths = []
            for source in sample_nodes[:50]:
                lengths = nx.single_source_shortest_path_length(G_cc, source)
                path_lengths.extend(lengths.values())
            avg_path = float(np.mean(path_lengths)) if path_lengths else 0
            results["avg_path_length"] = avg_path
            print(f"    Avg path length:    {avg_path:.3f}")
        else:
            results["avg_path_length"] = 0

        # ─── 5. Small-World Coefficient ───
        # σ = (C/C_rand) / (L/L_rand)
        # σ > 1 indicates small-world properties
        if len(G_cc) > 10 and n_edges > 0:
            # Generate random graph with same n, m
            n_cc = len(G_cc)
            m_cc = G_cc.number_of_edges()
            
            rand_clusterings = []
            rand_paths = []
            for _ in range(5):
                G_rand = nx.gnm_random_graph(n_cc, m_cc, seed=int(self.rng.integers(0, 2**31)))
                rand_clusterings.append(nx.average_clustering(G_rand))
                if nx.is_connected(G_rand):
                    # Sample path lengths
                    rand_sample = list(self.rng.choice(list(G_rand.nodes()), min(50, n_cc), replace=False))
                    rp = []
                    for s in rand_sample[:20]:
                        rp.extend(nx.single_source_shortest_path_length(G_rand, s).values())
                    rand_paths.append(float(np.mean(rp)) if rp else avg_path)

            C_rand = float(np.mean(rand_clusterings)) if rand_clusterings else 1e-6
            L_rand = float(np.mean(rand_paths)) if rand_paths else avg_path

            if C_rand > 0 and L_rand > 0 and avg_path > 0:
                sigma = (avg_clustering / max(C_rand, 1e-6)) / (avg_path / max(L_rand, 1e-6))
                results["small_world_sigma"] = float(sigma)
                results["random_clustering"] = C_rand
                results["random_path_length"] = L_rand
                print(f"    Small-world σ:      {sigma:.3f} (>1 = small-world)")
            else:
                results["small_world_sigma"] = 0
        else:
            results["small_world_sigma"] = 0

        # ─── 6. Modularity (Community Detection) ───
        try:
            communities = nx.community.greedy_modularity_communities(G_cc)
            modularity = nx.community.modularity(G_cc, communities)
            results["modularity"] = float(modularity)
            results["n_communities"] = len(communities)
            community_sizes = sorted([len(c) for c in communities], reverse=True)
            results["community_sizes_top5"] = community_sizes[:5]
            print(f"    Modularity:         {modularity:.4f}")
            print(f"    Communities:        {len(communities)}")
        except Exception:
            results["modularity"] = 0
            results["n_communities"] = 0

        # ─── 7. Network Resilience ───
        # Remove top hubs and measure how quickly the network fragments
        if len(G_cc) > 50:
            hub_removal_curve = self._compute_resilience(G_cc)
            results["resilience_auc"] = float(np.trapezoid(hub_removal_curve))
            results["resilience_curve"] = [float(x) for x in hub_removal_curve]
            print(f"    Resilience AUC:     {results['resilience_auc']:.4f}")

        # ─── 8. Top Hubs ───
        degree_centrality = nx.degree_centrality(G)
        top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        results["top_hubs"] = [{"word": w, "centrality": float(c)} for w, c in top_hubs]

        betweenness = nx.betweenness_centrality(G_cc, k=min(100, len(G_cc)))
        top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
        results["top_betweenness"] = [{"word": w, "centrality": float(c)} for w, c in top_between]

        return results

    def _compute_resilience(self, G: 'nx.Graph', steps: int = 20) -> List[float]:
        """Remove top-degree nodes progressively, measure largest component fraction."""
        G_copy = G.copy()
        n_original = len(G_copy)
        curve = [1.0]  # Start at 100%

        nodes_per_step = max(1, len(G_copy) // steps)

        for _ in range(steps - 1):
            if len(G_copy) < 2:
                curve.append(0.0)
                continue
            # Remove highest-degree nodes
            degrees = dict(G_copy.degree())
            to_remove = sorted(degrees, key=degrees.get, reverse=True)[:nodes_per_step]
            G_copy.remove_nodes_from(to_remove)

            if len(G_copy) == 0:
                curve.append(0.0)
            else:
                largest = max(nx.connected_components(G_copy), key=len)
                curve.append(len(largest) / n_original)

        return curve


class Phase4Runner:
    """Orchestrates Phase 4: Semantic Network Topology."""

    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        print("=" * 70)
        print("PROJECT NUR — PHASE 4: SEMANTIC NETWORK TOPOLOGY")
        print("How does the Quran's word-network differ structurally?")
        print("=" * 70)

        # Load data
        quran_df = pd.read_parquet("data/quran/quran_dataset.parquet")
        quran_verses = quran_df["text_uthmani"].tolist()

        controls = {}
        for name in ["quran_shuffled_words", "quran_shuffled_verses"]:
            path = Path(f"data/controls/shuffled/{name}.parquet")
            if path.exists():
                df = pd.read_parquet(path)
                controls[name] = df["text_uthmani"].tolist()

        builder = SemanticNetworkBuilder(min_word_freq=3, min_word_len=2)
        analyzer = NetworkTopologyAnalyzer()

        results = {}

        # Analyze each corpus
        for corpus_name, verses in [("QURAN", quran_verses)] + list(controls.items()):
            print(f"\n\n{'=' * 60}")
            print(f"  CORPUS: {corpus_name}")
            print(f"{'=' * 60}")

            G = builder.build_network(verses, corpus_name)
            topo = analyzer.analyze(G, corpus_name)
            results[corpus_name] = topo

        # Comparative report
        self._comparative_report(results)
        self._save_results(results)

        return results

    def _comparative_report(self, results: Dict):
        print(f"\n\n{'=' * 70}")
        print("PHASE 4 COMPARATIVE REPORT — NETWORK TOPOLOGY")
        print(f"{'=' * 70}")

        metrics = [
            ("Nodes", "n_nodes"),
            ("Edges", "n_edges"),
            ("Density", "density"),
            ("Mean Degree", "degree_mean"),
            ("Clustering Coeff", "clustering_coefficient"),
            ("Avg Path Length", "avg_path_length"),
            ("Small-World σ", "small_world_sigma"),
            ("Power Law Exp", "power_law_exponent"),
            ("Power Law R²", "power_law_r_squared"),
            ("Modularity", "modularity"),
            ("Communities", "n_communities"),
            ("Resilience AUC", "resilience_auc"),
            ("Largest Component %", "largest_component_fraction"),
        ]

        corpora = list(results.keys())
        header = f"{'Metric':<25}" + "".join(f"{c:>18}" for c in corpora)
        print(f"\n{header}")
        print("-" * (25 + 18 * len(corpora)))

        for label, key in metrics:
            vals = []
            for c in corpora:
                v = results[c].get(key, 0)
                vals.append(v)
            row = f"{label:<25}" + "".join(f"{v:>18.4f}" for v in vals)
            print(row)

        # Print top hubs for Quran
        if "QURAN" in results and "top_hubs" in results["QURAN"]:
            print(f"\n  QURAN — TOP 15 SEMANTIC HUBS (by degree centrality):")
            for hub in results["QURAN"]["top_hubs"][:15]:
                print(f"    {hub['word']:>20s}  centrality={hub['centrality']:.4f}")

    def _save_results(self, results: Dict):
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

        output_path = self.results_dir / "phase4_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean(results), f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Results saved to: {output_path}")


if __name__ == "__main__":
    runner = Phase4Runner()
    runner.run()
