"""
Natural Language Adaptation of the Constraint Network.

Key changes from synthetic:
  1. Frozen sentence encoder (e.g. all-MiniLM-L6-v2) replaces token embeddings
  2. Operates on sentence-level representations, not token-level
  3. Corruption strategies work on real paragraphs
  4. Constraint network architecture unchanged — takes continuous vectors

Pipeline:
  Raw text → sentence split → sentence encoder → sequence of vectors
  → constraint network → energy score

Install:
  pip install sentence-transformers datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# 1. Sentence Encoder Wrapper
# ============================================================

class SentenceEncoderWrapper:
    """
    Wraps a frozen sentence-transformers model.
    Converts paragraphs into sequences of sentence embeddings.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim = self.model.get_sentence_embedding_dimension()
        self.device = device
        print(f"Sentence encoder: {model_name}, dim={self.dim}")

    def encode_paragraph(self, text: str, max_sentences: int = 32) -> torch.Tensor:
        """Split text into sentences, encode each, return (num_sentences, dim)."""
        sentences = self.split_sentences(text)[:max_sentences]
        if not sentences:
            return torch.zeros(1, self.dim)
        with torch.no_grad():
            embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings

    def encode_sentences(self, sentences: List[str]) -> torch.Tensor:
        """Encode a list of sentences directly."""
        if not sentences:
            return torch.zeros(1, self.dim)
        with torch.no_grad():
            return self.model.encode(sentences, convert_to_tensor=True)

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Simple sentence splitter. For production, use spaCy or nltk."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 5]


# ============================================================
# 2. Natural Language Corruption Strategies
# ============================================================

@dataclass
class CorruptionResult:
    sentences: List[str]
    corruption_type: str
    corrupted_indices: List[int] = field(default_factory=list)


def corrupt_shuffle(sentences: List[str]) -> CorruptionResult:
    """Shuffle sentence order. Breaks discourse flow and causal chains."""
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    return CorruptionResult(shuffled, "shuffle", list(range(len(shuffled))))


def corrupt_swap_entities(sentences: List[str]) -> CorruptionResult:
    """
    Swap named entities between sentences.
    "Alice went to Paris. She loved the Eiffel Tower."
    → "Bob went to Paris. She loved the Eiffel Tower."
    Creates coreference and consistency violations.
    """
    # Simple heuristic: swap capitalized words between sentences
    result = sentences.copy()
    corrupted = []

    # Find capitalized words (proxy for entities) per sentence
    entity_pattern = re.compile(r'\b[A-Z][a-z]+\b')
    sentence_entities = []
    for s in result:
        entities = entity_pattern.findall(s)
        # Filter common sentence starters
        entities = [e for e in entities if e.lower() not in
                    {"the", "this", "that", "these", "those", "there", "then",
                     "they", "their", "what", "when", "where", "which", "while",
                     "however", "moreover", "furthermore", "although", "because",
                     "after", "before", "during", "since", "until", "into"}]
        sentence_entities.append(entities)

    # Find two sentences with entities and swap one
    entity_sents = [(i, ents) for i, ents in enumerate(sentence_entities) if ents]
    if len(entity_sents) >= 2:
        idx1, ents1 = random.choice(entity_sents)
        idx2, ents2 = random.choice([x for x in entity_sents if x[0] != idx1])
        e1 = random.choice(ents1)
        e2 = random.choice(ents2)
        result[idx1] = result[idx1].replace(e1, e2, 1)
        corrupted = [idx1]

    return CorruptionResult(result, "entity_swap", corrupted)


def corrupt_negate(sentences: List[str]) -> CorruptionResult:
    """
    Insert negation to create contradictions.
    "The system processes requests quickly."
    → "The system does not process requests quickly."
    """
    result = sentences.copy()
    corrupted = []

    # Simple patterns for negation insertion
    negation_targets = [
        (r'\b(is|are|was|were)\b', r'\1 not'),
        (r'\b(can|could|will|would|should|must|may|might)\b', r'\1 not'),
        (r'\b(has|have|had)\b', r'\1 not'),
        (r'\b(do|does|did)\b', r'\1 not'),
    ]

    # Pick a random sentence to negate
    indices = list(range(len(result)))
    random.shuffle(indices)

    for idx in indices:
        for pattern, replacement in negation_targets:
            new_sent = re.sub(pattern, replacement, result[idx], count=1)
            if new_sent != result[idx]:
                result[idx] = new_sent
                corrupted.append(idx)
                break
        if corrupted:
            break

    return CorruptionResult(result, "negate", corrupted)


def corrupt_tense_shift(sentences: List[str]) -> CorruptionResult:
    """
    Shift tense mid-paragraph.
    Past → present or present → past for a subset of sentences.
    Creates temporal inconsistency.
    """
    result = sentences.copy()
    n = len(result)
    if n < 3:
        return CorruptionResult(result, "tense_shift", [])

    # Shift the middle third to opposite tense (crude approximation)
    start = n // 3
    end = 2 * n // 3
    corrupted = list(range(start, end))

    simple_past_to_present = [
        (r'\bwas\b', 'is'), (r'\bwere\b', 'are'),
        (r'\bhad\b', 'has'), (r'\bdid\b', 'does'),
        (r'\bwent\b', 'goes'), (r'\bcame\b', 'comes'),
        (r'\bmade\b', 'makes'), (r'\btook\b', 'takes'),
        (r'\bgave\b', 'gives'), (r'\bfound\b', 'finds'),
    ]

    for idx in corrupted:
        for pattern, replacement in simple_past_to_present:
            result[idx] = re.sub(pattern, replacement, result[idx])

    return CorruptionResult(result, "tense_shift", corrupted)


def corrupt_topic_splice(sentences: List[str], donor_sentences: List[str]) -> CorruptionResult:
    """
    Splice in sentences from a completely different paragraph.
    Breaks topical coherence while maintaining local grammaticality.
    This is the hardest corruption to detect — requires understanding topic flow.
    """
    result = sentences.copy()
    if len(result) < 3 or len(donor_sentences) < 1:
        return CorruptionResult(result, "topic_splice", [])

    # Replace 1-2 sentences in the middle with donor sentences
    n_replace = min(2, len(donor_sentences), len(result) - 2)
    start = random.randint(1, len(result) - n_replace - 1)
    corrupted = list(range(start, start + n_replace))

    donor_picks = random.sample(donor_sentences, n_replace)
    for i, idx in enumerate(corrupted):
        result[idx] = donor_picks[i]

    return CorruptionResult(result, "topic_splice", corrupted)


def corrupt_coreference_break(sentences: List[str]) -> CorruptionResult:
    """
    Break coreference chains by replacing pronouns with wrong referents.
    "Alice went to the store. She bought milk."
    → "Alice went to the store. He bought milk."
    """
    result = sentences.copy()
    corrupted = []

    pronoun_swaps = {
        'he': 'she', 'she': 'he', 'his': 'her', 'her': 'his',
        'him': 'her', 'He': 'She', 'She': 'He', 'His': 'Her',
        'Her': 'His', 'Him': 'Her',
        'it': 'they', 'It': 'They', 'its': 'their', 'Its': 'Their',
    }

    for idx in range(1, len(result)):  # skip first sentence
        words = result[idx].split()
        swapped = False
        for i, w in enumerate(words):
            clean_w = w.rstrip('.,!?;:')
            if clean_w in pronoun_swaps:
                suffix = w[len(clean_w):]
                words[i] = pronoun_swaps[clean_w] + suffix
                swapped = True
                break
        if swapped:
            result[idx] = ' '.join(words)
            corrupted.append(idx)
            break  # one swap is enough

    return CorruptionResult(result, "coref_break", corrupted)


CORRUPTION_STRATEGIES = [
    corrupt_shuffle,
    corrupt_swap_entities,
    corrupt_negate,
    corrupt_tense_shift,
    corrupt_coreference_break,
    # topic_splice needs donor sentences, handled separately
]


def apply_random_corruption(sentences: List[str],
                            donor_sentences: Optional[List[str]] = None
                            ) -> CorruptionResult:
    """Apply a random corruption strategy."""
    strategies = CORRUPTION_STRATEGIES.copy()
    if donor_sentences:
        strategies.append(lambda s: corrupt_topic_splice(s, donor_sentences))

    strategy = random.choice(strategies)
    return strategy(sentences)


# ============================================================
# 3. Natural Language Dataset
# ============================================================

class NaturalLanguageConstraintDataset(torch.utils.data.Dataset):
    """
    Dataset for training constraint network on natural language.

    Sources for coherent text (positive examples):
    - Wikipedia paragraphs
    - Book excerpts
    - News articles
    - Any well-edited multi-sentence text

    Each item returns:
    - positive: sentence embeddings of coherent paragraph
    - negative: sentence embeddings of corrupted version
    - corruption_type: which strategy was applied
    """

    def __init__(self, texts: List[str], encoder: SentenceEncoderWrapper,
                 max_sentences: int = 16, cache_embeddings: bool = True):
        self.encoder = encoder
        self.max_sentences = max_sentences
        self.dim = encoder.dim

        # Process texts into sentence lists
        print(f"Processing {len(texts)} texts...")
        self.paragraphs = []
        for text in texts:
            sents = encoder.split_sentences(text)
            if len(sents) >= 4:  # need enough sentences for meaningful corruption
                self.paragraphs.append(sents[:max_sentences])

        print(f"  Valid paragraphs: {len(self.paragraphs)}")

        # Optionally cache embeddings
        self.cached = {}
        if cache_embeddings:
            print("  Caching embeddings...")
            for i, sents in enumerate(self.paragraphs):
                self.cached[i] = encoder.encode_sentences(sents).cpu()
            print(f"  Cached {len(self.cached)} paragraphs")

    def __len__(self):
        return len(self.paragraphs)

    def _pad_embeddings(self, emb: torch.Tensor) -> torch.Tensor:
        """Pad to max_sentences."""
        n = emb.shape[0]
        if n >= self.max_sentences:
            return emb[:self.max_sentences]
        pad = torch.zeros(self.max_sentences - n, self.dim)
        return torch.cat([emb, pad], dim=0)

    def __getitem__(self, idx):
        sentences = self.paragraphs[idx]

        # Get positive embeddings
        if idx in self.cached:
            pos_emb = self.cached[idx]
        else:
            pos_emb = self.encoder.encode_sentences(sentences).cpu()

        # Create corruption
        donor_idx = random.randint(0, len(self.paragraphs) - 1)
        while donor_idx == idx:
            donor_idx = random.randint(0, len(self.paragraphs) - 1)
        donor_sents = self.paragraphs[donor_idx]

        corruption = apply_random_corruption(sentences, donor_sents)

        # Encode corrupted version
        neg_emb = self.encoder.encode_sentences(corruption.sentences).cpu()

        return (
            self._pad_embeddings(pos_emb),
            self._pad_embeddings(neg_emb),
            corruption.corruption_type,
        )


# ============================================================
# 4. Adapted Constraint Network
# ============================================================

def create_nl_constraint_network(encoder_dim=384, d_model=256, d_state=64):
    """
    Create constraint network for natural language.

    Key difference: no embedding layer, input projection maps
    sentence encoder dim → constraint network dim.

    Architecture is identical to synthetic version:
    Input projection → SSM blocks + 2-head attention → energy head
    """
    from model import ConstraintNetwork

    net = ConstraintNetwork(
        d_model=d_model,
        d_state=d_state,
        vocab_size=None,  # no embedding — takes continuous vectors
        max_seq_len=32,   # max sentences per paragraph
        dropout=0.1,
        alpha=0.3,
    )

    # Replace input projection to match encoder dim
    net.input_proj = nn.Linear(encoder_dim, d_model)

    return net


# ============================================================
# 5. Data Loading Utilities
# ============================================================

def load_wikipedia_paragraphs(num_paragraphs=5000, min_sentences=4):
    """Load paragraphs from Wikipedia via HuggingFace datasets."""
    from datasets import load_dataset

    print(f"Loading Wikipedia paragraphs...")
    ds = load_dataset("wikipedia", "20220301.en", split="train",
                      streaming=True, trust_remote_code=True)

    paragraphs = []
    for article in ds:
        text = article["text"]
        # Split into paragraphs
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) < 100:
                continue
            sents = SentenceEncoderWrapper.split_sentences(para)
            if len(sents) >= min_sentences:
                paragraphs.append(para)
                if len(paragraphs) >= num_paragraphs:
                    break
        if len(paragraphs) >= num_paragraphs:
            break

    print(f"  Loaded {len(paragraphs)} paragraphs")
    return paragraphs


def load_simple_paragraphs(num_paragraphs=2000):
    """
    Fallback: generate simple multi-sentence texts.
    Use this if you can't download Wikipedia.
    """
    from datasets import load_dataset

    print("Loading text data...")
    # Try bookcorpus or similar
    try:
        ds = load_dataset("bookcorpus", split="train", streaming=True)
        texts = []
        buffer = []
        for item in ds:
            sent = item["text"].strip()
            if len(sent) > 20:
                buffer.append(sent)
            if len(buffer) >= 8:
                texts.append(buffer.copy())
                buffer = buffer[4:]  # overlap
            if len(texts) >= num_paragraphs:
                break
        return [" ".join(t) for t in texts]
    except Exception:
        pass

    # Final fallback: use the texts already in memory
    print("  Using synthetic multi-sentence texts as fallback")
    templates = [
        "The {adj} {noun} {verb} the {noun2}. It was {adj2} and {adj3}. "
        "Many people noticed the change. The effect spread {adv}. "
        "By the end, everything had transformed. No one could deny the result.",

        "{Name} worked at the {place}. Every morning, {pronoun} arrived early. "
        "The routine was predictable but comfortable. Colleagues respected {poss} dedication. "
        "One day, something unexpected happened. {Name} discovered a {adj} opportunity.",
    ]
    nouns = ["system", "process", "structure", "pattern", "network", "signal"]
    adjs = ["complex", "elegant", "robust", "fragile", "ancient", "modern"]
    verbs = ["transformed", "disrupted", "enhanced", "revealed", "connected"]
    names = ["Alice", "Marcus", "Elena", "James", "Priya", "Chen"]
    places = ["institute", "laboratory", "university", "company", "observatory"]

    paragraphs = []
    for i in range(num_paragraphs):
        random.seed(i)
        t = random.choice(templates)
        text = t.format(
            adj=random.choice(adjs), noun=random.choice(nouns),
            verb=random.choice(verbs), noun2=random.choice(nouns),
            adj2=random.choice(adjs), adj3=random.choice(adjs),
            adv=random.choice(["quickly", "slowly", "gradually", "suddenly"]),
            Name=random.choice(names), place=random.choice(places),
            pronoun=random.choice(["he", "she"]),
            poss=random.choice(["his", "her"]),
        )
        paragraphs.append(text)

    return paragraphs


# ============================================================
# 6. Training Script
# ============================================================

def train_nl_constraint(
    paragraphs: List[str] = None,
    encoder_name: str = "all-MiniLM-L6-v2",
    d_model: int = 256,
    d_state: int = 64,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    margin: float = 5.0,
    device: str = None,
):
    """Full training pipeline for natural language constraint network."""
    import time, json
    from model import ConstraintLoss

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    if paragraphs is None:
        try:
            paragraphs = load_wikipedia_paragraphs(3000)
        except Exception as e:
            print(f"Wikipedia load failed ({e}), using fallback")
            paragraphs = load_simple_paragraphs(2000)

    # Split train/val
    random.shuffle(paragraphs)
    split = int(len(paragraphs) * 0.85)
    train_texts = paragraphs[:split]
    val_texts = paragraphs[split:]

    # Encoder
    encoder = SentenceEncoderWrapper(encoder_name, device)

    # Datasets
    print("\nBuilding training dataset...")
    train_ds = NaturalLanguageConstraintDataset(train_texts, encoder, max_sentences=16)
    print("Building validation dataset...")
    val_ds = NaturalLanguageConstraintDataset(val_texts, encoder, max_sentences=16)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = create_nl_constraint_network(encoder.dim, d_model, d_state).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nConstraint network params: {params:,}")
    print(f"  encoder dim: {encoder.dim} → model dim: {d_model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = ConstraintLoss(margin)

    history = []
    best_acc = 0

    print(f"\nTraining for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(epochs):
        t0 = time.time()

        # Train
        model.train()
        tl = 0; n = 0
        for pos_emb, neg_emb, _ in train_dl:
            pos_emb = pos_emb.to(device)
            neg_emb = neg_emb.to(device)

            ep = model(pos_emb)
            en = model(neg_emb)
            loss = criterion(ep, en)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); n += 1

        scheduler.step()

        # Validate
        model.eval()
        vl = 0; correct = 0; total = 0; vn = 0
        type_correct = {}; type_total = {}

        with torch.no_grad():
            for pos_emb, neg_emb, ctypes in val_dl:
                pos_emb = pos_emb.to(device)
                neg_emb = neg_emb.to(device)

                ep = model(pos_emb)
                en = model(neg_emb)
                vl += criterion(ep, en).item(); vn += 1

                preds = (ep < en)
                correct += preds.sum().item()
                total += preds.shape[0]

                for j, ct in enumerate(ctypes):
                    if ct not in type_correct:
                        type_correct[ct] = 0
                        type_total[ct] = 0
                    type_total[ct] += 1
                    if preds[j]:
                        type_correct[ct] += 1

        acc = correct / max(total, 1)
        dt = time.time() - t0

        rec = {"epoch": epoch+1, "train_loss": tl/n, "val_loss": vl/vn,
               "val_acc": acc, "time": dt}
        history.append(rec)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "nl_constraint_best.pt")
            tag = " *"
        else:
            tag = ""

        if (epoch+1) % 5 == 0 or epoch == 0 or tag:
            print(f"E{epoch+1:3d} | loss={tl/n:.4f} val={vl/vn:.4f} "
                  f"acc={acc:.3f} | {dt:.1f}s{tag}")

    print(f"\nBest accuracy: {best_acc:.3f}")

    # Per-type breakdown
    print("\nPer-corruption accuracy:")
    for ct in sorted(type_total.keys()):
        ct_acc = type_correct[ct] / type_total[ct]
        print(f"  {ct:20s}: {ct_acc:.2f} ({type_total[ct]} samples)")

    results = {
        "history": history,
        "best_acc": best_acc,
        "per_type": {ct: type_correct[ct]/type_total[ct] for ct in type_total},
        "params": params,
        "encoder": encoder_name,
        "d_model": d_model,
    }

    with open("nl_constraint_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to nl_constraint_results.json")

    return model, encoder, results


# ============================================================
# 7. Integration with Generative LM
# ============================================================

class NLConstraintIntegration:
    """
    Sketch of how to integrate the NL constraint network
    with a real language model (e.g. GPT-2, Pythia, etc.).

    The key insight: we don't need the constraint network to process
    tokens. We need it to process the LM's hidden states, which are
    already continuous vectors in a learned representation space.

    Architecture:
      LM hidden states (at chosen layer)
      → projection to constraint space
      → constraint network evaluates structural energy
      → energy added to LM training loss

    This is identical to what we did with the synthetic model,
    just at larger scale.
    """

    def __init__(self, constraint_net, lm_hidden_dim, constraint_dim,
                 lambda_max=0.1, device="cpu"):
        self.constraint_net = constraint_net.to(device)
        self.constraint_net.eval()
        for p in self.constraint_net.parameters():
            p.requires_grad = False

        # Projection from LM hidden space to constraint space
        self.projection = nn.Linear(lm_hidden_dim, constraint_dim).to(device)
        self.lambda_max = lambda_max
        self.device = device

    def compute_constraint_loss(self, hidden_states, step, warmup=1000, tau=200):
        """
        Given LM hidden states (batch, seq_len, lm_hidden_dim),
        compute constraint energy for regularization.

        Returns scalar energy to add to LM loss.
        """
        projected = self.projection(hidden_states)
        energy = self.constraint_net(projected).mean()

        # Sigmoid schedule
        lam = self.lambda_max * torch.sigmoid(
            torch.tensor((step - warmup) / tau, dtype=torch.float32)
        ).item()

        return lam * energy, energy.item(), lam

    def get_trainable_params(self):
        """Only the projection layer is trainable."""
        return self.projection.parameters()


def integration_example():
    """
    Example of how you'd wire this into a real LM training loop.
    This is pseudocode — adapt to your specific LM framework.
    """
    print("""
    # === Integration with a real LM (pseudocode) ===

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Load your LM
    lm = AutoModelForCausalLM.from_pretrained("pythia-70m")
    tokenizer = AutoTokenizer.from_pretrained("pythia-70m")

    # 2. Load trained constraint network
    constraint_net = create_nl_constraint_network(encoder_dim=512, d_model=256)
    constraint_net.load_state_dict(torch.load("nl_constraint_best.pt"))

    # 3. Create integration
    integration = NLConstraintIntegration(
        constraint_net,
        lm_hidden_dim=512,     # pythia-70m hidden dim
        constraint_dim=256,     # constraint network dim
        lambda_max=0.2,
    )

    # 4. Training loop
    optimizer = AdamW(
        list(lm.parameters()) + list(integration.get_trainable_params()),
        lr=5e-5
    )

    for step, batch in enumerate(dataloader):
        # Forward pass with hidden state extraction
        outputs = lm(batch["input_ids"], output_hidden_states=True)
        logits = outputs.logits

        # LM loss
        lm_loss = cross_entropy(logits, batch["labels"])

        # Constraint loss (from middle layer hidden states)
        middle_layer = len(outputs.hidden_states) // 2
        hidden = outputs.hidden_states[middle_layer]
        constraint_loss, energy, lam = integration.compute_constraint_loss(
            hidden, step, warmup=2000, tau=500
        )

        # Combined loss
        total_loss = lm_loss + constraint_loss

        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: lm={lm_loss:.4f} energy={energy:.3f} λ={lam:.4f}")
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("Natural Language Constraint Network")
    print("=" * 60)

    # Quick demo with fallback data
    print("\n--- Corruption Strategy Demo ---\n")

    text = (
        "Marie Curie was born in Warsaw in 1867. She moved to Paris to study physics. "
        "Her research on radioactivity was groundbreaking. She became the first woman "
        "to win a Nobel Prize. Her work changed the field of nuclear physics forever. "
        "Many scientists built upon her discoveries in the following decades."
    )

    sentences = SentenceEncoderWrapper.split_sentences(text)
    print(f"Original ({len(sentences)} sentences):")
    for i, s in enumerate(sentences):
        print(f"  {i}: {s}")

    print()
    for strategy in [corrupt_shuffle, corrupt_swap_entities, corrupt_negate,
                     corrupt_tense_shift, corrupt_coreference_break]:
        random.seed(42)
        result = strategy(sentences)
        print(f"{result.corruption_type} (corrupted: {result.corrupted_indices}):")
        for i, s in enumerate(result.sentences):
            marker = " <<<" if i in result.corrupted_indices else ""
            print(f"  {i}: {s}{marker}")
        print()

    print("\n--- Integration Pseudocode ---")
    integration_example()

    print("\n--- To train on real data ---")
    print("  python nl_adaptation.py --train")
    print("  (requires: pip install sentence-transformers datasets)")

    import sys
    if "--train" in sys.argv:
        train_nl_constraint()
