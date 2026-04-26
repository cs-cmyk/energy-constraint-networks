"""
Evaluate BERT-trained constraint network on in-distribution Wikipedia text.
Same as eval_in_distribution.py but uses BERTWindowEncoder.

Run: python eval_bert.py
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


def evaluate_paragraph(sentences, encoder, constraint_net, device, max_windows=32):
    text = " ".join(sentences)
    embs = encoder.encode_paragraph(text, max_windows=max_windows).to(device)
    if embs.shape[0] < max_windows:
        pad = torch.zeros(max_windows - embs.shape[0], embs.shape[1], device=device)
        embs = torch.cat([embs, pad], dim=0)
    else:
        embs = embs[:max_windows]
    embs = embs.unsqueeze(0)
    with torch.no_grad():
        energy, per_pos = constraint_net(embs, return_per_position=True)
    n_real = min(len(sentences), max_windows)
    return energy.item(), [per_pos[0, i].item() for i in range(embs.shape[1])]


def verify_corruption(original, corrupted):
    """Check that corruption actually changed something."""
    return any(a != b for a, b in zip(original, corrupted)) or len(original) != len(corrupted)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    encoder = BERTWindowEncoder("bert-base-uncased", str(device),
                                 layer=-2, window_size=8, max_tokens=512)

    # Load BERT-trained constraint network
    constraint_net = ConstraintNetwork(
        d_model=384, d_state=64, vocab_size=None,
        max_seq_len=32, dropout=0.15, alpha=0.3
    ).to(device)
    constraint_net.input_proj = nn.Linear(768, 384).to(device)
    constraint_net.load_state_dict(
        torch.load("nl_bert_constraint_best.pt", weights_only=True, map_location=device))
    constraint_net.eval()
    print("BERT constraint network loaded.\n")

    paragraphs = load_val_paragraphs(150, min_sents=5)
    eval_paras = paragraphs[:100]
    donor_paras = paragraphs[100:]

    corruption_types = {
        "shuffle": lambda s: corrupt_shuffle(s).sentences,
        "negate": lambda s: corrupt_negate(s).sentences,
        "entity_swap": lambda s: corrupt_swap_entities(s).sentences,
        "tense_shift": lambda s: corrupt_tense_shift(s).sentences,
        "coref_break": lambda s: corrupt_coreference_break(s).sentences,
    }
    if donor_paras:
        donor_sents = random.choice(donor_paras)
        corruption_types["topic_splice"] = lambda s: corrupt_topic_splice(s, donor_sents).sentences

    print("=" * 70)
    print("EVALUATION: BERT constraint network (in-distribution)")
    print("=" * 70)

    coherent_energies = []
    corrupted_energies = {k: [] for k in corruption_types}
    all_results = []
    skipped = {k: 0 for k in corruption_types}

    for idx, sents in enumerate(eval_paras):
        random.seed(idx)
        e_coh, _ = evaluate_paragraph(sents, encoder, constraint_net, device)
        coherent_energies.append(e_coh)

        for ctype, corrupt_fn in corruption_types.items():
            random.seed(idx + 10000)
            corrupted_sents = corrupt_fn(sents)

            # Check corruption actually worked
            if not verify_corruption(sents, corrupted_sents):
                skipped[ctype] += 1
                continue

            e_cor, _ = evaluate_paragraph(corrupted_sents, encoder, constraint_net, device)
            corrupted_energies[ctype].append(e_cor)
            all_results.append({
                "idx": idx, "type": ctype,
                "coherent_energy": e_coh, "corrupted_energy": e_cor,
                "detected": e_coh < e_cor,
            })

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    coh_avg = sum(coherent_energies) / len(coherent_energies)
    print(f"\n  Coherent avg energy: {coh_avg:+.4f} ({len(coherent_energies)} paragraphs)")

    print(f"\n  {'Corruption':<20s} | {'Avg Energy':>10s} | {'Gap':>8s} | {'Accuracy':>8s} | {'Skipped':>7s}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*7}")

    for ctype in corruption_types:
        energies = corrupted_energies[ctype]
        if not energies:
            print(f"  {ctype:<20s} | {'N/A':>10s} | {'N/A':>8s} | {'N/A':>8s} | {skipped[ctype]:>7d}")
            continue
        avg = sum(energies) / len(energies)
        gap = avg - coh_avg

        pairs = [(r["coherent_energy"], r["corrupted_energy"])
                 for r in all_results if r["type"] == ctype]
        correct = sum(1 for c, x in pairs if c < x)
        acc = correct / len(pairs) if pairs else 0

        print(f"  {ctype:<20s} | {avg:>+10.4f} | {gap:>+8.4f} | {acc:>7.1%} | {skipped[ctype]:>7d}")

    valid_results = [r for r in all_results]
    overall_correct = sum(1 for r in valid_results if r["detected"])
    overall_total = len(valid_results)
    if overall_total:
        print(f"\n  Overall paired accuracy: {overall_correct}/{overall_total} "
              f"= {overall_correct/overall_total:.1%}")

    # Detailed examples
    print(f"\n{'='*70}")
    print("DETAILED EXAMPLES (best detections)")
    print(f"{'='*70}")

    # Find paragraphs with largest gaps
    best_detections = sorted(all_results, key=lambda r: r["corrupted_energy"] - r["coherent_energy"], reverse=True)

    shown_types = set()
    for det in best_detections:
        if det["type"] in shown_types:
            continue
        shown_types.add(det["type"])

        idx = det["idx"]
        sents = eval_paras[idx]
        ctype = det["type"]
        gap = det["corrupted_energy"] - det["coherent_energy"]

        e_coh, pp_coh = evaluate_paragraph(sents, encoder, constraint_net, device)

        random.seed(idx + 10000)
        csents = corruption_types[ctype](sents)
        e_cor, pp_cor = evaluate_paragraph(csents, encoder, constraint_net, device)

        print(f"\n--- {ctype} (gap={gap:+.4f}) ---")
        print(f"  Coherent (E={e_coh:+.4f}):")
        for i, s in enumerate(sents):
            print(f"    {i+1}. [{pp_coh[i]:+.4f}] {s}")
        print(f"  Corrupted (E={e_cor:+.4f}):")
        for i, s in enumerate(csents):
            changed = " <<<" if (i < len(sents) and s != sents[i]) else ""
            print(f"    {i+1}. [{pp_cor[i]:+.4f}]{changed} {s}")

        if len(shown_types) >= 5:
            break

    output = {
        "coherent_avg": coh_avg,
        "overall_accuracy": overall_correct / overall_total if overall_total else 0,
        "per_type": {},
        "skipped": skipped,
    }
    for ctype in corruption_types:
        pairs = [(r["coherent_energy"], r["corrupted_energy"])
                 for r in all_results if r["type"] == ctype]
        if pairs:
            correct = sum(1 for c, x in pairs if c < x)
            output["per_type"][ctype] = {
                "accuracy": correct / len(pairs),
                "n_valid": len(pairs),
                "n_skipped": skipped[ctype],
            }

    with open("bert_eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to bert_eval_results.json")


if __name__ == "__main__":
    main()
