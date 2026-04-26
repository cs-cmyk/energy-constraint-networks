"""
Corruption clustering analysis.

Tests whether corruptions targeting the same structural property
produce similar perturbations in the constraint network's
internal representation space.

For each corruption type:
1. Encode coherent paragraph through BERT → constraint network
2. Encode corrupted paragraph through BERT → constraint network
3. Compute displacement = representation(corrupted) - representation(coherent)
4. Average displacement vectors across paragraphs

Then compute 15×15 cosine similarity matrix between mean displacement vectors.
If entity_swap and role_title_swap (both referential consistency) cluster together
while being distant from shuffle (discourse ordering), the clustering claim
is empirical, not intuitive.

Run: python analyze_clustering.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import numpy as np
from model import ConstraintNetwork
from bert_encoder import BERTWindowEncoder
from nl_adaptation import (
    SentenceEncoderWrapper,
    corrupt_shuffle, corrupt_negate, corrupt_swap_entities,
    corrupt_tense_shift, corrupt_coreference_break, corrupt_topic_splice,
)
from corruptions_extended import (
    corrupt_name_substitution, corrupt_definite_description_swap,
    corrupt_demonstrative_confusion, corrupt_singular_plural_mismatch,
    corrupt_role_title_swap, corrupt_temporal_reference_break,
    corrupt_repetition_injection, corrupt_number_contradiction,
    corrupt_bridging_reference_break,
)


class ConstraintNetworkWithRepresentations(nn.Module):
    """Wrapper that extracts internal representations before the energy head."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Returns (energy, pre_energy_representation)."""
        if self.model.embedding is not None and x.dtype == torch.long:
            x = self.model.embedding(x)

        x = self.model.input_proj(x)

        for block in self.model.blocks:
            x = block(x)

        # x is now the representation BEFORE the energy head
        # This is what we want — the learned structural representation
        representation = x.clone()

        # Also compute energy for reference
        h = self.model.energy_norm(x)
        per_pos_energy = self.model.energy_mlp(h).squeeze(-1)
        mean_energy = per_pos_energy.mean(dim=1)
        max_energy = per_pos_energy.max(dim=1).values
        energy = mean_energy + self.model.alpha * max_energy

        return energy, representation


def load_val_paragraphs(num=200, min_sents=5):
    from datasets import load_dataset
    print("Loading Wikipedia validation paragraphs...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    paragraphs = []
    for item in ds:
        text = item["text"].strip()
        if len(text) < 150:
            continue
        sents = SentenceEncoderWrapper.split_sentences(text)
        if len(sents) >= min_sents:
            paragraphs.append(sents[:8])
            if len(paragraphs) >= num:
                break
    print(f"  Found {len(paragraphs)} paragraphs")
    return paragraphs


def encode_text(text, encoder, max_windows=32):
    embs = encoder.encode_paragraph(text, max_windows=max_windows)
    if embs.shape[0] < max_windows:
        pad = torch.zeros(max_windows - embs.shape[0], embs.shape[1],
                          device=embs.device)
        embs = torch.cat([embs, pad], dim=0)
    else:
        embs = embs[:max_windows]
    return embs.unsqueeze(0)


def verify_corruption(original, corrupted):
    return any(a != b for a, b in zip(original, corrupted)) or len(original) != len(corrupted)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    encoder = BERTWindowEncoder("bert-base-uncased", str(device),
                                 layer=-2, window_size=8, max_tokens=512)

    # Load model
    model = ConstraintNetwork(
        d_model=384, d_state=64, vocab_size=None,
        max_seq_len=32, dropout=0.15, alpha=0.3
    ).to(device)
    model.input_proj = nn.Linear(768, 384).to(device)
    model.load_state_dict(
        torch.load("nl_bert_constraint_best.pt", weights_only=True, map_location=device))
    model.eval()

    # Wrap to extract representations
    wrapped = ConstraintNetworkWithRepresentations(model).to(device)
    wrapped.eval()
    print("Model loaded.\n")

    # Add this to the top of analyze_clustering.py after model loads
    text = "Marie Curie was born in Warsaw in 1867. She moved to Paris to study physics."
    embs = encoder.encode_paragraph(text, max_windows=32).to(device)
    if embs.shape[0] < 32:
        pad = torch.zeros(32 - embs.shape[0], embs.shape[1], device=device)
        embs = torch.cat([embs, pad], dim=0)
    energy = model(embs.unsqueeze(0))
    print(f"Test energy: {energy.item():+.4f}")
    # Should be around +2.0 to +3.0 for coherent text
    # If it's wildly different (e.g., -50 or +500), the fallback isn't compatible


    paragraphs = load_val_paragraphs(200, min_sents=5)
    eval_paras = paragraphs[:100]
    donor_paras = paragraphs[100:]
    donor_sents = random.choice(donor_paras) if donor_paras else None

    # All 15 corruption types with category labels
    corruption_info = {
        # Trained types
        "shuffle":        {"fn": lambda s: corrupt_shuffle(s).sentences,        "category": "Ordering",     "trained": True},
        "negate":         {"fn": lambda s: corrupt_negate(s).sentences,         "category": "Semantic",     "trained": True},
        "entity_swap":    {"fn": lambda s: corrupt_swap_entities(s).sentences,  "category": "Referential",  "trained": True},
        "tense_shift":    {"fn": lambda s: corrupt_tense_shift(s).sentences,    "category": "Surface",      "trained": True},
        "coref_pronoun":  {"fn": lambda s: corrupt_coreference_break(s).sentences, "category": "Referential", "trained": True},
        "topic_splice":   {"fn": lambda s: corrupt_topic_splice(s, donor_sents).sentences if donor_sents else corrupt_shuffle(s).sentences,
                                                                                 "category": "Semantic",     "trained": True},
        # Unseen types
        "name_sub":       {"fn": lambda s: corrupt_name_substitution(s).sentences,       "category": "Referential",  "trained": False},
        "desc_swap":      {"fn": lambda s: corrupt_definite_description_swap(s).sentences,"category": "Referential",  "trained": False},
        "demonstrative":  {"fn": lambda s: corrupt_demonstrative_confusion(s).sentences, "category": "Referential",  "trained": False},
        "singular_plural":{"fn": lambda s: corrupt_singular_plural_mismatch(s).sentences,"category": "Referential",  "trained": False},
        "role_title":     {"fn": lambda s: corrupt_role_title_swap(s).sentences,         "category": "Surface",      "trained": False},
        "temporal_break": {"fn": lambda s: corrupt_temporal_reference_break(s).sentences,"category": "Ordering",     "trained": False},
        "repetition":     {"fn": lambda s: corrupt_repetition_injection(s).sentences,    "category": "Ordering",     "trained": False},
        "number_change":  {"fn": lambda s: corrupt_number_contradiction(s).sentences,    "category": "Semantic",     "trained": False},
        "bridging_break": {"fn": lambda s: corrupt_bridging_reference_break(s).sentences,"category": "Referential",  "trained": False},
    }

    # Compute mean displacement vectors
    print("Computing displacement vectors...")
    displacement_vectors = {k: [] for k in corruption_info}
    energy_gaps = {k: [] for k in corruption_info}

    for idx, sents in enumerate(eval_paras):
        text_coh = " ".join(sents)

        with torch.no_grad():
            embs_coh = encode_text(text_coh, encoder, max_windows=32).to(device)
            e_coh, rep_coh = wrapped(embs_coh)
            # Pool representation to a single vector: mean across positions
            rep_coh_pooled = rep_coh.mean(dim=1)  # (1, d_model)

        for ctype, info in corruption_info.items():
            random.seed(idx + hash(ctype) % 100000)
            try:
                csents = info["fn"](sents)
            except Exception:
                continue

            if not verify_corruption(sents, csents):
                continue

            text_cor = " ".join(csents)
            with torch.no_grad():
                embs_cor = encode_text(text_cor, encoder, max_windows=32).to(device)
                e_cor, rep_cor = wrapped(embs_cor)
                rep_cor_pooled = rep_cor.mean(dim=1)  # (1, d_model)

            # Displacement vector
            disp = (rep_cor_pooled - rep_coh_pooled).squeeze(0).cpu()
            displacement_vectors[ctype].append(disp)
            energy_gaps[ctype].append((e_cor - e_coh).item())

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(eval_paras)}")

    # Compute mean displacement per corruption type
    print("\nComputing mean displacements and similarity matrix...")
    mean_displacements = {}
    for ctype, disps in displacement_vectors.items():
        if len(disps) > 0:
            stacked = torch.stack(disps)
            mean_displacements[ctype] = stacked.mean(dim=0)

    # Compute cosine similarity matrix
    ctypes_ordered = [k for k in corruption_info if k in mean_displacements]
    n = len(ctypes_ordered)
    sim_matrix = np.zeros((n, n))

    for i, c1 in enumerate(ctypes_ordered):
        for j, c2 in enumerate(ctypes_ordered):
            v1 = mean_displacements[c1]
            v2 = mean_displacements[c2]
            cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            sim_matrix[i, j] = cos

    # Print results
    print(f"\n{'='*70}")
    print("CORRUPTION DISPLACEMENT SIMILARITY MATRIX")
    print(f"{'='*70}")

    # Print header
    short = [c[:8] for c in ctypes_ordered]
    header = f"{'':>16s}" + "".join(f"{s:>9s}" for s in short)
    print(header)

    for i, c1 in enumerate(ctypes_ordered):
        cat = corruption_info[c1]["category"]
        trained = "T" if corruption_info[c1]["trained"] else "U"
        label = f"{c1[:12]:>12s}[{trained}]"
        row = "".join(f"{sim_matrix[i,j]:>9.3f}" for j in range(n))
        print(f"{label} {row}")

    # Compute within-category vs between-category similarity
    print(f"\n{'='*70}")
    print("WITHIN-CATEGORY vs BETWEEN-CATEGORY SIMILARITY")
    print(f"{'='*70}")

    categories = {}
    for i, c in enumerate(ctypes_ordered):
        cat = corruption_info[c]["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)

    within_sims = []
    between_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            cat_i = corruption_info[ctypes_ordered[i]]["category"]
            cat_j = corruption_info[ctypes_ordered[j]]["category"]
            if cat_i == cat_j:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])

    within_avg = np.mean(within_sims) if within_sims else 0
    between_avg = np.mean(between_sims) if between_sims else 0

    print(f"  Within-category avg similarity:  {within_avg:.4f} ({len(within_sims)} pairs)")
    print(f"  Between-category avg similarity: {between_avg:.4f} ({len(between_sims)} pairs)")
    print(f"  Ratio: {within_avg / max(between_avg, 1e-8):.2f}x")

    # Per-category breakdown
    print(f"\n  Per-category within-similarity:")
    for cat, indices in categories.items():
        if len(indices) < 2:
            print(f"    {cat:<15s}: N/A (only 1 member)")
            continue
        cat_sims = []
        for i in indices:
            for j in indices:
                if i < j:
                    cat_sims.append(sim_matrix[i, j])
        if cat_sims:
            print(f"    {cat:<15s}: {np.mean(cat_sims):.4f} ({len(cat_sims)} pairs)")

    # Energy gap per type for reference
    print(f"\n{'='*70}")
    print("ENERGY GAPS (for reference)")
    print(f"{'='*70}")
    for c in ctypes_ordered:
        gaps = energy_gaps[c]
        cat = corruption_info[c]["category"]
        trained = "T" if corruption_info[c]["trained"] else "U"
        if gaps:
            print(f"  {c:<18s} [{trained}] {cat:<12s} gap={np.mean(gaps):+.4f} (n={len(gaps)})")

    # Save results
    results = {
        "similarity_matrix": sim_matrix.tolist(),
        "corruption_order": ctypes_ordered,
        "categories": {c: corruption_info[c]["category"] for c in ctypes_ordered},
        "trained": {c: corruption_info[c]["trained"] for c in ctypes_ordered},
        "within_category_sim": float(within_avg),
        "between_category_sim": float(between_avg),
        "ratio": float(within_avg / max(between_avg, 1e-8)),
        "per_category_within": {},
        "energy_gaps": {c: float(np.mean(energy_gaps[c])) for c in ctypes_ordered if energy_gaps[c]},
    }
    for cat, indices in categories.items():
        cat_sims = [sim_matrix[i, j] for i in indices for j in indices if i < j]
        if cat_sims:
            results["per_category_within"][cat] = float(np.mean(cat_sims))

    with open("clustering_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to clustering_analysis.json")


if __name__ == "__main__":
    main()
