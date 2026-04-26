"""
BERT token-level encoder for the constraint network.

Key difference from MiniLM sentence encoder:
- MiniLM: paragraph → split into sentences → one vector per sentence
  Problem: "he went" and "she went" map to nearly identical vectors
  
- BERT tokens: paragraph → tokenize whole text → keep per-token hidden states → pool into windows
  Advantage: word order, tense, pronouns, entity names all preserved in the representation
  
The constraint network receives the same shape: (batch, num_windows, dim)
But now each window contains token-level information, not compressed sentence meaning.

Usage:
  encoder = BERTWindowEncoder("bert-base-uncased")
  embeddings = encoder.encode_paragraph("Marie Curie was born in Warsaw...")
  # shape: (num_windows, 768)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BERTWindowEncoder:
    """
    Encodes text into windowed token-level representations using BERT.
    
    Process:
    1. Tokenize the full paragraph
    2. Run through frozen BERT
    3. Take hidden states from a chosen layer
    4. Pool tokens into fixed-size windows (mean pooling)
    
    Output: (num_windows, hidden_dim) — same interface as SentenceEncoderWrapper
    """

    def __init__(self, model_name="bert-base-uncased", device="cpu",
                 layer=-2, window_size=16, max_tokens=512):
        self.device = device
        self.layer = layer
        self.window_size = window_size
        self.max_tokens = max_tokens

        print(f"Loading BERT encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name,
                                                output_hidden_states=True).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.dim = self.model.config.hidden_size
        print(f"  Hidden dim: {self.dim}, Layer: {layer}, Window: {window_size}")

    def encode_paragraph(self, text, max_windows=16):
        """
        Encode a paragraph into windowed representations.
        Returns: (num_windows, dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_tokens, padding=False
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get chosen layer's hidden states
            hidden = outputs.hidden_states[self.layer]  # (1, seq_len, dim)
            hidden = hidden.squeeze(0)  # (seq_len, dim)

        # Remove [CLS] and [SEP]
        hidden = hidden[1:-1]  # (seq_len-2, dim)

        if hidden.shape[0] == 0:
            return torch.zeros(1, self.dim, device=self.device)

        # Pool into windows
        seq_len = hidden.shape[0]
        n_windows = min(seq_len // self.window_size, max_windows)
        if n_windows == 0:
            # Sequence shorter than window — just mean pool everything
            return hidden.mean(dim=0, keepdim=True)

        # Trim to exact multiple of window_size
        trimmed = hidden[:n_windows * self.window_size]
        # Reshape and mean pool
        windowed = trimmed.reshape(n_windows, self.window_size, self.dim)
        pooled = windowed.mean(dim=1)  # (n_windows, dim)

        return pooled

    def encode_sentences(self, sentences):
        """
        Compatibility method: encode a list of sentences as one paragraph.
        Joins them and encodes as a single sequence.
        """
        text = " ".join(sentences)
        return self.encode_paragraph(text)

    @staticmethod
    def split_sentences(text):
        """Reuse the same splitter for compatibility."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 5]


def precompute_bert_embeddings(paragraphs, encoder, max_windows, device,
                                corruptions_per_para=5, cache_path=None):
    """
    Pre-compute all BERT window embeddings.
    Same interface as the MiniLM version.
    """
    import os, time, random
    from nl_adaptation import (
        corrupt_shuffle, corrupt_negate, corrupt_swap_entities,
        corrupt_tense_shift, corrupt_coreference_break, corrupt_topic_splice,
    )

    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        return data["pos"], data["neg"], data["types"]

    N = len(paragraphs)
    dim = encoder.dim
    corruption_types = ["shuffle", "negate", "entity_swap",
                        "tense_shift", "coref_break", "topic_splice"]

    pos_embs = torch.zeros(N, max_windows, dim)
    neg_embs = torch.zeros(N * corruptions_per_para, max_windows, dim)
    neg_types = []

    print(f"  Encoding {N} paragraphs + {N * corruptions_per_para} corruptions (BERT)...")
    t0 = time.time()

    for idx, sents in enumerate(paragraphs):
        text = " ".join(sents)

        # Encode coherent
        with torch.no_grad():
            emb = encoder.encode_paragraph(text, max_windows=max_windows).cpu()
        n = min(emb.shape[0], max_windows)
        pos_embs[idx, :n] = emb[:n]

        # Corruptions
        for k in range(corruptions_per_para):
            random.seed(idx * 1000 + k)
            ctype = corruption_types[k % len(corruption_types)]

            donor_idx = (idx + k + 1) % N
            donor_sents = paragraphs[donor_idx]

            if ctype == "shuffle":
                csents = corrupt_shuffle(sents).sentences
            elif ctype == "negate":
                csents = corrupt_negate(sents).sentences
            elif ctype == "entity_swap":
                csents = corrupt_swap_entities(sents).sentences
            elif ctype == "tense_shift":
                csents = corrupt_tense_shift(sents).sentences
            elif ctype == "coref_break":
                csents = corrupt_coreference_break(sents).sentences
            elif ctype == "topic_splice":
                csents = corrupt_topic_splice(sents, donor_sents).sentences
            else:
                csents = corrupt_shuffle(sents).sentences

            ctext = " ".join(csents)
            with torch.no_grad():
                c_emb = encoder.encode_paragraph(ctext, max_windows=max_windows).cpu()
            cn = min(c_emb.shape[0], max_windows)
            neg_idx = idx * corruptions_per_para + k
            neg_embs[neg_idx, :cn] = c_emb[:cn]
            neg_types.append(ctype)

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (N - idx - 1) / rate
            print(f"    {idx+1}/{N} ({rate:.0f}/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({N/elapsed:.0f} paragraphs/s)")

    if cache_path:
        print(f"  Caching to {cache_path}")
        torch.save({"pos": pos_embs, "neg": neg_embs, "types": neg_types}, cache_path)

    return pos_embs, neg_embs, neg_types


if __name__ == "__main__":
    # Quick test
    encoder = BERTWindowEncoder("bert-base-uncased", "cuda" if torch.cuda.is_available() else "cpu")

    text = ("Marie Curie was born in Warsaw in 1867. "
            "She moved to Paris to study physics at the Sorbonne. "
            "Her research on radioactivity was groundbreaking. "
            "She became the first woman to win a Nobel Prize.")

    emb = encoder.encode_paragraph(text)
    print(f"Input: {len(text)} chars")
    print(f"Output: {emb.shape}")  # (num_windows, 768)

    # Test that word order matters
    shuffled = ("She became the first woman to win a Nobel Prize. "
                "Her research on radioactivity was groundbreaking. "
                "Marie Curie was born in Warsaw in 1867. "
                "She moved to Paris to study physics at the Sorbonne.")

    emb_orig = encoder.encode_paragraph(text)
    emb_shuf = encoder.encode_paragraph(shuffled)

    diff = (emb_orig - emb_shuf).norm().item()
    print(f"Embedding distance (original vs shuffled): {diff:.4f}")
    print(f"  (MiniLM would give ~0 here, BERT gives meaningful distance)")

    # Test pronoun sensitivity
    text_he = "The scientist published his findings. He was awarded a prize."
    text_she = "The scientist published her findings. She was awarded a prize."
    emb_he = encoder.encode_paragraph(text_he)
    emb_she = encoder.encode_paragraph(text_she)
    diff_pron = (emb_he - emb_she).norm().item()
    print(f"Embedding distance (he/his vs she/her): {diff_pron:.4f}")
    print(f"  (MiniLM would give ~0, BERT preserves pronoun differences)")
