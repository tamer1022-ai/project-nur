# Project Nur (نور) — Computational Structural Quran Analysis

**بسم الله الرحمن الرحيم**

A rigorous, data-driven computational analysis of the Quran's latent mathematical structure using modern transformer architectures and information-theoretic methods.

## Live Dashboard
**[View the Interactive Dashboard →](https://tamer1022-ai.github.io/project-nur/)**

## Key Findings

### 1. The Perplexity Inversion
The Classical Arabic model (CAMeLBERT-CA) finds the Quran **154x harder to predict** than Hadith. The Modern Arabic model (AraBERT-v2) finds it the **easiest** text. The mathematical signature flips across architectures — no other text exhibits this.

### 2. The Ordering Signal
Shuffling the Quran's words increases model surprise by **9-40x** depending on architecture. The word sequence carries more mathematical structure than any other Arabic text tested.

### 3. The Compression Paradox
The Quran is the **most compressible** Arabic text (gzip ratio 0.198) yet the **most surprising** to its specialist model. Structured but unpredictable — simultaneously.

### 4. AI Cannot Replicate the Signature
We generated 1,500 synthetic Arabic texts using ArGPT2. Even when explicitly mimicking Quranic style, AI produces text **19-28x more predictable** than the actual Quran on the Classical Arabic model.

## Methodology

- **3 Transformer Models**: CAMeLBERT-CA, AraBERT-v2, XLM-RoBERTa
- **5 Corpora**: Quran, Sahih al-Bukhari, Pre-Islamic Poetry, Arabic Bible (Van Dyke), Shuffled Quran
- **Length-Controlled**: Fixed 15-word windows (200 samples per corpus)
- **Statistical Rigor**: Bootstrap 95% confidence intervals, zero CI overlap on all comparisons
- **Layer Forensics**: 12-layer attention entropy and long-range attention analysis

## Results Summary

| Corpus | CAMeLBERT PPL | AraBERT PPL | XLM-R PPL |
|---|---|---|---|
| **Quran** | **888.86** | **11.22** | **23.46** |
| Hadith | 5.74 | 8.52 | 12.74 |
| Poetry | 205.27 | 387.07 | 134.12 |
| Bible | 152.50 | 46.84 | 37.64 |
| Quran (Shuffled) | 7,974.77 | 457.21 | 60.38 |

## Author

**Tamer Al-Zawahreh** — May 2026

Built with amanah. Data-driven. No apologetics. Let the numbers speak.

## License

This research is shared for educational and academic purposes.
