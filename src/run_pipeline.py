# Project Nur — Master Pipeline
# Bismillah
#
# End-to-end orchestrator: data acquisition → embedding → analysis → report
# Run this to execute the complete Phase 1 proof-of-concept.

import sys
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_phase0_data():
    """Phase 0: Acquire and prepare all data."""
    from data_acquisition import QuranDataAcquisition, ShuffledQuranGenerator

    print("\n" + "=" * 70)
    print("PHASE 0: DATA ACQUISITION")
    print("=" * 70)

    # Build Quran dataset
    acquisition = QuranDataAcquisition(data_dir="data/quran")
    df = acquisition.build_dataset()

    # Generate shuffled controls
    shuffler = ShuffledQuranGenerator(df)
    shuffler.generate_all_controls(output_dir="data/controls/shuffled")

    return df


def run_phase0_embeddings(quran_df=None):
    """Phase 0.5: Generate embeddings for Quran and controls."""
    import pandas as pd
    import numpy as np
    from embedding_pipeline import QuranEmbeddingPipeline

    print("\n" + "=" * 70)
    print("PHASE 0.5: EMBEDDING GENERATION")
    print("=" * 70)

    # Load Quran data if not provided
    if quran_df is None:
        quran_path = Path("data/quran/quran_dataset.parquet")
        if not quran_path.exists():
            print("ERROR: Quran dataset not found. Run Phase 0 first.")
            return
        quran_df = pd.read_parquet(quran_path)

    # Initialize embedding pipeline
    pipeline = QuranEmbeddingPipeline(
        batch_size=32,
        pooling="mean",  # Mean pooling captures distributed meaning
    )

    # Embed Quran
    print("\n\n█ Embedding: QURAN")
    pipeline.embed_quran_dataset(
        quran_df,
        text_column="text_simple",
        output_path="data/embeddings/quran_embeddings.npy",
    )

    # Embed shuffled controls
    shuffled_controls = {
        "quran_shuffled_words": "data/controls/shuffled/quran_shuffled_words.parquet",
        "quran_shuffled_verses": "data/controls/shuffled/quran_shuffled_verses.parquet",
        "quran_shuffled_surahs": "data/controls/shuffled/quran_shuffled_surahs.parquet",
    }

    for control_name, control_path in shuffled_controls.items():
        path = Path(control_path)
        if not path.exists():
            print(f"\n  ⚠ Control not found: {path} — skipping")
            continue

        print(f"\n\n█ Embedding: {control_name}")
        control_df = pd.read_parquet(path)
        pipeline.embed_control_corpus(
            control_df,
            corpus_name=control_name,
            text_column="text_simple",
            output_dir="data/embeddings",
        )


def run_phase1():
    """Phase 1: Cross-Temporal Semantic Coherence Analysis."""
    from phase1_coherence import Phase1Runner

    print("\n" + "=" * 70)
    print("PHASE 1: CROSS-TEMPORAL SEMANTIC COHERENCE ANALYSIS")
    print("=" * 70)

    runner = Phase1Runner(
        quran_embeddings_path="data/embeddings/quran_embeddings.npy",
        quran_metadata_path="data/embeddings/quran_embeddings.meta.parquet",
        controls_dir="data/embeddings",
        results_dir="data/results",
        sample_size=500,
        n_permutations=10000,
    )

    results = runner.run()
    return results


def validate_metadata():
    """Quick validation of surah metadata."""
    from surah_metadata import validate_metadata as _validate
    print("\n" + "=" * 70)
    print("METADATA VALIDATION")
    print("=" * 70)
    _validate()


def main():
    parser = argparse.ArgumentParser(
        description="Project Nur — Computational Analysis of Quranic Structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all              # Run everything (recommended first time)
  python run_pipeline.py --validate         # Just validate metadata
  python run_pipeline.py --data             # Just acquire data
  python run_pipeline.py --embed            # Just generate embeddings
  python run_pipeline.py --analyze          # Just run Phase 1 analysis
        """,
    )

    parser.add_argument("--all", action="store_true",
                        help="Run the complete pipeline (data → embed → analyze)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate surah metadata only")
    parser.add_argument("--data", action="store_true",
                        help="Run data acquisition only")
    parser.add_argument("--embed", action="store_true",
                        help="Run embedding generation only")
    parser.add_argument("--analyze", action="store_true",
                        help="Run Phase 1 analysis only")

    args = parser.parse_args()

    # Default to --all if no flags provided
    if not any([args.all, args.validate, args.data, args.embed, args.analyze]):
        args.all = True

    start_time = time.time()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              PROJECT NUR (نور) — PHASE 1 PIPELINE              ║")
    print("║     Computational Discovery of Latent Mathematical Structure    ║")
    print("║                       in the Quran                              ║")
    print("║                                                                 ║")
    print("║                       Bismillah                                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    try:
        if args.all or args.validate:
            validate_metadata()

        quran_df = None
        if args.all or args.data:
            quran_df = run_phase0_data()

        if args.all or args.embed:
            run_phase0_embeddings(quran_df)

        if args.all or args.analyze:
            results = run_phase1()

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline error: {e}")
        raise

    elapsed = time.time() - start_time
    print(f"\n\nTotal pipeline time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print("Pipeline complete. Alhamdulillah.")


if __name__ == "__main__":
    main()
