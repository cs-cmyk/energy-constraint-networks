"""
Run real sentence examples through BERT-trained constraint network.
These are the examples we want to show people — with real energy scores.

Run: python eval_showcase.py
"""

import torch
import torch.nn as nn
import json
from model import ConstraintNetwork
from bert_encoder import BERTWindowEncoder


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
    print("Model loaded.\n")

    # ==========================================
    # Examples grouped by category
    # ==========================================

    examples = [
        # --- COHERENT PARAGRAPHS ---
        {
            "name": "Coherent: Marie Curie biography",
            "category": "coherent",
            "sentences": [
                "Marie Curie was born in Warsaw in 1867.",
                "She moved to Paris to study physics at the Sorbonne.",
                "Her research on radioactivity was groundbreaking.",
                "She became the first woman to win a Nobel Prize.",
                "Her discoveries transformed the field of nuclear physics.",
            ]
        },
        {
            "name": "Coherent: photosynthesis",
            "category": "coherent",
            "sentences": [
                "Photosynthesis converts sunlight into chemical energy.",
                "The process takes place primarily in the chloroplasts of plant cells.",
                "Light reactions occur in the thylakoid membranes.",
                "The Calvin cycle then fixes carbon dioxide into glucose.",
                "This process is essential for nearly all life on Earth.",
            ]
        },
        {
            "name": "Coherent: lighthouse narrative",
            "category": "coherent",
            "sentences": [
                "The old lighthouse had stood on the cliff for over a century.",
                "Its beam could be seen from twenty miles out to sea.",
                "Generations of keepers had maintained its rotating lens.",
                "The last keeper retired when the light was automated in 1985.",
                "Today, the lighthouse serves as a museum and visitor center.",
            ]
        },

        # --- ENTITY / TOPIC VIOLATIONS ---
        {
            "name": "Entity swap: basketball in physics bio",
            "category": "entity_swap",
            "sentences": [
                "Marie Curie was born in Warsaw in 1867.",
                "She moved to Paris to study physics at the Sorbonne.",
                "Her research on basketball was groundbreaking.",
                "He became the first woman to win a Nobel Prize.",
                "Her discoveries transformed the field of nuclear physics.",
            ]
        },
        {
            "name": "Topic splice: birds in engineering text",
            "category": "topic_splice",
            "sentences": [
                "The bridge was designed by a team of structural engineers.",
                "Construction began in March and continued through the summer.",
                "The migration patterns of Arctic terns span over 40,000 miles.",
                "Their feathers provide insulation during long flights.",
                "The project was completed under budget by November.",
            ]
        },

        # --- CONTRADICTIONS ---
        {
            "name": "Contradiction: vaccine efficacy",
            "category": "contradiction",
            "sentences": [
                "The vaccine was tested in a large clinical trial.",
                "Results showed a 95% efficacy rate against infection.",
                "The study did not find any evidence that it worked.",
                "Regulators approved it for emergency use shortly after.",
            ]
        },
        {
            "name": "Self-contradiction: company profits",
            "category": "self_contradiction",
            "sentences": [
                "The company reported record profits in the third quarter.",
                "Revenue grew 23% compared to the same period last year.",
                "However, the company has been losing money every quarter this year.",
                "The CEO attributed the strong performance to new product launches.",
                "Investors responded positively, pushing the stock up 8%.",
            ]
        },

        # --- HALLUCINATION PATTERNS ---
        {
            "name": "Hallucination: Fleming invents telephone",
            "category": "hallucination",
            "sentences": [
                "Alexander Fleming discovered penicillin in 1928.",
                "The discovery happened when mold contaminated a petri dish.",
                "He noticed that bacteria near the mold had been destroyed.",
                "Fleming then invented the telephone in his laboratory.",
                "The telephone revolutionized global shipping routes.",
            ]
        },
        {
            "name": "Gradual drift: Renaissance to fiber optics",
            "category": "gradual_drift",
            "sentences": [
                "The Renaissance began in Italy in the 14th century.",
                "It was characterized by a renewed interest in classical art and learning.",
                "Florence became a major center for artistic innovation.",
                "Many of these innovations were influenced by advances in optics.",
                "Modern optical fiber networks now transmit data at the speed of light.",
                "Submarine cables carry 99% of intercontinental internet traffic.",
            ]
        },
        {
            "name": "Subtle hallucination: JWST false claims",
            "category": "subtle_hallucination",
            "sentences": [
                "The James Webb Space Telescope launched in December 2021.",
                "It orbits the Sun at the second Lagrange point, about 1.5 million km from Earth.",
                "Its primary mirror is 6.5 meters in diameter.",
                "The telescope has already discovered several new planets with breathable atmospheres.",
                "NASA confirmed that one of these planets has liquid oceans visible from orbit.",
            ]
        },

        # --- STRUCTURAL VIOLATIONS ---
        {
            "name": "Shuffled: Curie sentences reordered",
            "category": "shuffle",
            "sentences": [
                "She became the first woman to win a Nobel Prize.",
                "Her discoveries transformed the field of nuclear physics.",
                "Marie Curie was born in Warsaw in 1867.",
                "Her research on radioactivity was groundbreaking.",
                "She moved to Paris to study physics at the Sorbonne.",
            ]
        },
        {
            "name": "Coreference break: pronoun swap",
            "category": "coref_break",
            "sentences": [
                "The CEO announced the company's new strategy last week.",
                "He emphasized the importance of sustainable growth.",
                "She then outlined three key initiatives for the quarter.",
                "They expected significant revenue increases by year end.",
                "It was considered a bold move by industry analysts.",
            ]
        },
    ]

    # ==========================================
    # Evaluate
    # ==========================================

    results = []
    print("=" * 70)
    print("REAL ENERGY SCORES FROM BERT CONSTRAINT NETWORK")
    print("=" * 70)

    for ex in examples:
        text = " ".join(ex["sentences"])
        energy, per_pos = evaluate(text, encoder, model, device)

        print(f"\n--- {ex['name']} ---")
        print(f"  Overall energy: {energy:+.4f}  [{ex['category']}]")

        # Map windows back to approximate sentence positions
        # Each window = 8 tokens ≈ half a sentence
        for i, (sent, e) in enumerate(zip(ex["sentences"], per_pos[:len(ex["sentences"])])):
            marker = ""
            print(f"  {i+1}. [{e:+.4f}] {sent}")

        results.append({
            "name": ex["name"],
            "category": ex["category"],
            "energy": energy,
            "per_position": per_pos[:len(ex["sentences"])],
            "sentences": ex["sentences"],
        })

    # ==========================================
    # Summary by category
    # ==========================================

    print(f"\n{'='*70}")
    print("SUMMARY BY CATEGORY")
    print(f"{'='*70}")

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["energy"])

    sorted_cats = sorted(categories.items(), key=lambda x: sum(x[1])/len(x[1]))

    for cat, energies in sorted_cats:
        avg = sum(energies) / len(energies)
        label = "COHERENT" if cat == "coherent" else "VIOLATION"
        print(f"  {cat:<25s} | E={avg:+.4f} | {label}")

    coherent_avg = sum(categories["coherent"]) / len(categories["coherent"])
    violation_energies = [e for cat, es in categories.items() if cat != "coherent" for e in es]
    violation_avg = sum(violation_energies) / len(violation_energies)

    print(f"\n  Coherent average:  {coherent_avg:+.4f}")
    print(f"  Violation average: {violation_avg:+.4f}")
    print(f"  Gap: {violation_avg - coherent_avg:+.4f}")
    print(f"  All violations higher than all coherent: "
          f"{all(v > max(categories['coherent']) for v in violation_energies)}")

    with open("showcase_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to showcase_results.json")


if __name__ == "__main__":
    main()
