# Project Nur — Embedding Pipeline
# Bismillah
#
# Generates verse-level embeddings using CAMeLBERT-CA (Classical Arabic BERT).
# Handles batching, GPU acceleration, and multiple pooling strategies.

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Literal
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class QuranEmbeddingPipeline:
    """
    Generates high-dimensional embeddings for Quranic verses
    using CAMeLBERT-CA (Classical Arabic BERT from NYU Abu Dhabi).
    
    This is the core mathematical transformation: Arabic text → vectors
    in 768-dimensional space. Every subsequent analysis operates on these vectors.
    """

    # CAMeLBERT-CA: Pre-trained on Classical Arabic texts
    # This is the best available model for Classical/Quranic Arabic
    MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-ca"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 32,
        pooling: Literal["cls", "mean", "max"] = "mean",
        max_length: int = 512,
    ):
        """
        Initialize the embedding pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: "cuda", "cpu", or None (auto-detect)
            batch_size: Number of verses to embed at once
            pooling: How to aggregate token embeddings into verse embedding
                     "cls"  — use the [CLS] token (captures overall sentence meaning)
                     "mean" — average all token embeddings (captures distributed meaning)
                     "max"  — max-pool across tokens (captures strongest features)
            max_length: Maximum token sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.pooling = pooling
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Embedding Pipeline Configuration:")
        print(f"  Model:    {model_name}")
        print(f"  Device:   {self.device}")
        print(f"  Pooling:  {pooling}")
        print(f"  Batch:    {batch_size}")
        print(f"  Max Len:  {max_length}")

        # Load model and tokenizer
        print(f"\nLoading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"  ✓ Model loaded. Embedding dim: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of text strings into vectors.
        
        Args:
            texts: List of Arabic text strings
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + self.batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Forward pass
                outputs = self.model(**encoded)
                hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
                attention_mask = encoded["attention_mask"]   # (batch, seq_len)

                # Pool token embeddings into verse embeddings
                if self.pooling == "cls":
                    # Use [CLS] token embedding
                    embeddings = hidden_states[:, 0, :]
                elif self.pooling == "mean":
                    # Mean pooling with attention mask
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                elif self.pooling == "max":
                    # Max pooling with attention mask
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    hidden_states[mask_expanded == 0] = -1e9
                    embeddings = torch.max(hidden_states, dim=1)[0]
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling}")

                # Normalize embeddings to unit length (for cosine similarity)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def embed_quran_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = "text_simple",
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Embed all verses in a Quran dataset DataFrame.
        
        Args:
            df: DataFrame with verse text and metadata
            text_column: Column containing the Arabic text to embed
            output_path: If provided, save embeddings to this path
            
        Returns:
            numpy array of shape (n_verses, embedding_dim)
        """
        texts = df[text_column].tolist()
        print(f"\nEmbedding {len(texts)} verses using {text_column}...")

        embeddings = self.embed_texts(texts)

        print(f"  ✓ Generated embeddings: shape {embeddings.shape}")
        print(f"  ✓ Embedding norm range: [{np.linalg.norm(embeddings, axis=1).min():.4f}, "
              f"{np.linalg.norm(embeddings, axis=1).max():.4f}]")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, embeddings)
            print(f"  ✓ Saved to: {output_path}")

            # Also save metadata mapping
            meta_path = output_path.with_suffix(".meta.parquet")
            meta_df = df[["verse_id", "surah_number", "verse_number",
                          "revelation_period", "meccan_or_medinan",
                          "chronological_order"]].copy()
            meta_df.to_parquet(meta_path, index=False)
            print(f"  ✓ Metadata saved to: {meta_path}")

        return embeddings

    def embed_control_corpus(
        self,
        df: pd.DataFrame,
        corpus_name: str,
        text_column: str = "text_simple",
        output_dir: str = "data/embeddings",
    ) -> np.ndarray:
        """
        Embed a control corpus using the same pipeline.
        
        Ensures identical processing for fair comparison.
        """
        print(f"\n{'=' * 60}")
        print(f"Embedding control corpus: {corpus_name}")
        print(f"{'=' * 60}")

        output_path = Path(output_dir) / f"{corpus_name}_embeddings.npy"
        return self.embed_quran_dataset(df, text_column, str(output_path))


if __name__ == "__main__":
    # Quick test with a few verses
    print("Testing embedding pipeline...")
    pipeline = QuranEmbeddingPipeline(batch_size=4)

    test_texts = [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "الرحمن الرحيم",
        "مالك يوم الدين",
    ]

    embeddings = pipeline.embed_texts(test_texts, show_progress=False)
    print(f"\nTest embeddings shape: {embeddings.shape}")

    # Compute pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    print(f"\nPairwise cosine similarities:")
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            print(f"  '{test_texts[i][:30]}...' ↔ '{test_texts[j][:30]}...': {sim_matrix[i, j]:.4f}")
