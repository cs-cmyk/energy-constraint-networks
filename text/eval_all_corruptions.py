"""
Evaluate all corruption types (original 6 + extended 9) through
the BERT-trained constraint network on Wikipedia validation text.

Run: python eval_all_corruptions.py
"""

import torch
import torch.nn as nn
import json
import random
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


def evaluate(text, encoder, model, device, max_windows=32):
    embs = encoder.encode_paragraph(text, max_windows=max_windows).to(device)
    n_real = embs.shape[0]
    if embs.shape[0] < max_windows:
        pad = torch.zeros(max_windows - embs.shape[0], embs.shape[1], device=device)
        embs = torch.cat([embs, pad], dim=0)
    embs = embs.unsqueeze(0)
    with torch.no_grad():
        energy, per_pos = model(embs, return_per_position=True)
    return energy.item(), [per_pos[0, i].item() for i in range(n_real)]


def verify_corruption(original, corrupted):
    return any(a != b for a, b in zip(original, corrupted)) or len(original) != len(corrupted)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    encoder = BERTWindowEncoder("bert-base-uncased", str(device),
                                 layer=-2, window_size=8, max_tokens=512)

    model = ConstraintNetwork(
        d_model=384, d_state=64, vocab_size=None,
        max_seq_len=32, dropout=0.15, alpha=0.3
    ).to(device)
    model.input_proj = nn.Linear(768, 384).to(device)
    model.load_state_dict(
        torch.load("nl_bert_constraint_best.pt", weights_only=True, map_location=device))
    model.eval()
    print("BERT constraint network loaded.\n")

    paragraphs = load_val_paragraphs(200, min_sents=5)
    eval_paras = paragraphs[:100]
    donor_paras = paragraphs[100:]
    donor_sents = random.choice(donor_paras) if donor_paras else None

    # All 15 corruption types
    corruption_fns = {
        # Original 6
        "shuffle": lambda s: corrupt_shuffle(s).sentences,
        "negate": lambda s: corrupt_negate(s).sentences,
        "entity_swap": lambda s: corrupt_swap_entities(s).sentences,
        "tense_shift": lambda s: corrupt_tense_shift(s).sentences,
        "coref_pronoun": lambda s: corrupt_coreference_break(s).sentences,
        "topic_splice": lambda s: corrupt_topic_splice(s, donor_sents).sentences if donor_sents else corrupt_shuffle(s).sentences,
        # Extended 9
        "name_substitution": lambda s: corrupt_name_substitution(s).sentences,
        "description_swap": lambda s: corrupt_definite_description_swap(s).sentences,
        "demonstrative": lambda s: corrupt_demonstrative_confusion(s).sentences,
        "singular_plural": lambda s: corrupt_singular_plural_mismatch(s).sentences,
        "role_title_swap": lambda s: corrupt_role_title_swap(s).sentences,
        "temporal_break": lambda s: corrupt_temporal_reference_break(s).sentences,
        "repetition": lambda s: corrupt_repetition_injection(s).sentences,
        "number_change": lambda s: corrupt_number_contradiction(s).sentences,
        "bridging_break": lambda s: corrupt_bridging_reference_break(s).sentences,
    }

    print("=" * 70)
    print("EVALUATION: 15 corruption types on Wikipedia (BERT constraint network)")
    print("=" * 70)

    coherent_energies = []
    results_by_type = {k: {"energies": [], "detected": 0, "total": 0, "skipped": 0}
                       for k in corruption_fns}
    all_detailed = []

    for idx, sents in enumerate(eval_paras):
        random.seed(idx)
        text_coh = " ".join(sents)
        e_coh, pp_coh = evaluate(text_coh, encoder, model, device)
        coherent_energies.append(e_coh)

        for ctype, fn in corruption_fns.items():
            random.seed(idx + hash(ctype) % 100000)
            try:
                csents = fn(sents)
            except Exception:
                results_by_type[ctype]["skipped"] += 1
                continue

            if not verify_corruption(sents, csents):
                results_by_type[ctype]["skipped"] += 1
                continue

            text_cor = " ".join(csents)
            e_cor, pp_cor = evaluate(text_cor, encoder, model, device)
            detected = e_coh < e_cor
            results_by_type[ctype]["energies"].append(e_cor)
            results_by_type[ctype]["detected"] += (1 if detected else 0)
            results_by_type[ctype]["total"] += 1

            # Save detailed info for best examples
            all_detailed.append({
                "idx": idx, "type": ctype,
                "coherent_energy": e_coh, "corrupted_energy": e_cor,
                "gap": e_cor - e_coh, "detected": detected,
                "coherent_sents": sents, "corrupted_sents": csents,
                "coherent_pp": pp_coh, "corrupted_pp": pp_cor,
            })

    # ==========================================
    # Summary table
    # ==========================================
    coh_avg = sum(coherent_energies) / len(coherent_energies)

    print(f"\n  Coherent avg energy: {coh_avg:+.4f} ({len(coherent_energies)} paragraphs)")
    print(f"\n  {'Corruption':<22s} | {'Accuracy':>8s} | {'Avg Gap':>8s} | {'Valid':>5s} | {'Skip':>4s}")
    print(f"  {'-'*22} | {'-'*8} | {'-'*8} | {'-'*5} | {'-'*4}")

    sorted_types = sorted(results_by_type.keys(),
                           key=lambda k: (results_by_type[k]["detected"] /
                                          max(results_by_type[k]["total"], 1)),
                           reverse=True)

    for ctype in sorted_types:
        r = results_by_type[ctype]
        if r["total"] == 0:
            print(f"  {ctype:<22s} | {'N/A':>8s} | {'N/A':>8s} | {0:>5d} | {r['skipped']:>4d}")
            continue
        acc = r["detected"] / r["total"]
        avg_e = sum(r["energies"]) / len(r["energies"])
        gap = avg_e - coh_avg
        print(f"  {ctype:<22s} | {acc:>7.1%} | {gap:>+8.4f} | {r['total']:>5d} | {r['skipped']:>4d}")

    total_detected = sum(r["detected"] for r in results_by_type.values())
    total_pairs = sum(r["total"] for r in results_by_type.values())
    print(f"\n  Overall: {total_detected}/{total_pairs} = {total_detected/max(total_pairs,1):.1%}")

    # ==========================================
    # Group by category
    # ==========================================
    print(f"\n{'='*70}")
    print("BY CATEGORY")
    print(f"{'='*70}")

    categories = {
        "Coreference (broad)": ["coref_pronoun", "name_substitution", "singular_plural",
                                 "description_swap", "demonstrative", "bridging_break"],
        "Structural ordering": ["shuffle", "temporal_break", "repetition"],
        "Semantic consistency": ["negate", "entity_swap", "topic_splice", "number_change"],
        "Surface features": ["tense_shift", "role_title_swap"],
    }

    for cat_name, ctypes in categories.items():
        total_d = sum(results_by_type[c]["detected"] for c in ctypes if c in results_by_type)
        total_t = sum(results_by_type[c]["total"] for c in ctypes if c in results_by_type)
        if total_t:
            print(f"  {cat_name:<30s}: {total_d}/{total_t} = {total_d/total_t:.1%}")

    # ==========================================
    # Best detection examples per type
    # ==========================================
    print(f"\n{'='*70}")
    print("BEST DETECTION EXAMPLES")
    print(f"{'='*70}")

    shown_types = set()
    best_by_type = {}
    for det in sorted(all_detailed, key=lambda d: d["gap"], reverse=True):
        if det["type"] not in best_by_type and det["detected"]:
            best_by_type[det["type"]] = det

    for ctype in sorted_types[:10]:
        if ctype not in best_by_type:
            continue
        det = best_by_type[ctype]

        print(f"\n--- {ctype} (gap={det['gap']:+.4f}) ---")
        print(f"  Coherent (E={det['coherent_energy']:+.4f}):")
        for i, s in enumerate(det["coherent_sents"][:6]):
            e = det["coherent_pp"][i] if i < len(det["coherent_pp"]) else 0
            print(f"    {i+1}. [{e:+.4f}] {s}")

        print(f"  Corrupted (E={det['corrupted_energy']:+.4f}):")
        for i, s in enumerate(det["corrupted_sents"][:6]):
            e = det["corrupted_pp"][i] if i < len(det["corrupted_pp"]) else 0
            changed = " <<<" if (i < len(det["coherent_sents"]) and
                                  s != det["coherent_sents"][i]) else ""
            print(f"    {i+1}. [{e:+.4f}]{changed} {s}")

    # Save
    output = {
        "coherent_avg": coh_avg,
        "overall_accuracy": total_detected / max(total_pairs, 1),
        "per_type": {},
        "categories": {},
    }
    for ctype in sorted_types:
        r = results_by_type[ctype]
        if r["total"]:
            output["per_type"][ctype] = {
                "accuracy": r["detected"] / r["total"],
                "avg_gap": (sum(r["energies"]) / len(r["energies"])) - coh_avg,
                "valid": r["total"],
                "skipped": r["skipped"],
            }
    for cat_name, ctypes in categories.items():
        total_d = sum(results_by_type[c]["detected"] for c in ctypes if c in results_by_type)
        total_t = sum(results_by_type[c]["total"] for c in ctypes if c in results_by_type)
        if total_t:
            output["categories"][cat_name] = {"accuracy": total_d / total_t, "total": total_t}

    with open("all_corruptions_eval.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to all_corruptions_eval.json")


if __name__ == "__main__":
    main()
