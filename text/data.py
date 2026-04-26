"""
Synthetic data generator for constraint network training.
Creates sequences that follow structural rules (positive) and
corrupted versions that violate them (negative).

Rules are compositional and verifiable:
1. Entity consistency: introduced entities must be referenced consistently
2. Causal ordering: cause must precede effect
3. Attribute consistency: entity attributes don't contradict
4. Scope rules: local modifiers apply to adjacent elements
"""

import torch
import random
from dataclasses import dataclass
from typing import List, Tuple


# Simple vocabulary for synthetic data
ENTITIES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
ATTRIBUTES = ["red", "blue", "green", "large", "small", "bright", "dark", "warm"]
ACTIONS = ["moves", "grows", "splits", "merges", "rotates", "shifts", "pulses", "fades"]
LOCATIONS = ["north", "south", "east", "west", "center", "edge", "top", "bottom"]
CONNECTORS = ["then", "next", "after", "so", "thus", "hence", "therefore", "meanwhile"]
REFERENCES = ["it", "this", "that", "the_entity"]

# Token vocabulary
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>"]
ALL_TOKENS = SPECIAL_TOKENS + ENTITIES + ATTRIBUTES + ACTIONS + LOCATIONS + CONNECTORS + REFERENCES

TOKEN_TO_ID = {t: i for i, t in enumerate(ALL_TOKENS)}
ID_TO_TOKEN = {i: t for t, i in TOKEN_TO_ID.items()}
VOCAB_SIZE = len(ALL_TOKENS)


@dataclass
class StructuredSequence:
    """A sequence with known structural properties."""
    tokens: List[str]
    token_ids: List[int]
    rules_followed: List[str]
    rules_violated: List[str]
    is_coherent: bool


def generate_coherent_sequence(num_statements: int = 8, seed: int = None) -> StructuredSequence:
    """Generate a structurally coherent sequence following all rules."""
    if seed is not None:
        random.seed(seed)

    tokens = ["<bos>"]
    rules_followed = []

    # Pick entities for this sequence
    active_entities = random.sample(ENTITIES, k=random.randint(2, 4))
    entity_attributes = {}
    entity_locations = {}

    for i in range(num_statements):
        statement = []

        if i == 0:
            # First statement: introduce an entity with attribute and location
            entity = active_entities[0]
            attr = random.choice(ATTRIBUTES)
            loc = random.choice(LOCATIONS)
            statement = [entity, attr, loc]
            entity_attributes[entity] = attr
            entity_locations[entity] = loc
            rules_followed.append(f"introduce:{entity}={attr},{loc}")

        elif i < len(active_entities):
            # Introduce remaining entities
            entity = active_entities[i]
            attr = random.choice(ATTRIBUTES)
            loc = random.choice(LOCATIONS)
            statement = [entity, attr, loc]
            entity_attributes[entity] = attr
            entity_locations[entity] = loc
            rules_followed.append(f"introduce:{entity}={attr},{loc}")

        else:
            # Reference existing entity consistently
            entity = random.choice(list(entity_attributes.keys()))
            action = random.choice(ACTIONS)

            # Use consistent attribute (rule: attribute consistency)
            attr = entity_attributes[entity]

            if random.random() < 0.4:
                # Use pronoun reference (rule: entity consistency)
                statement = ["the_entity", attr, action]
                rules_followed.append(f"consistent_ref:{entity}={attr}")
            else:
                statement = [entity, action]
                rules_followed.append(f"direct_ref:{entity}")

            # Maybe update location with causal connector
            if random.random() < 0.3:
                new_loc = random.choice(LOCATIONS)
                connector = random.choice(CONNECTORS)
                statement.extend([connector, new_loc])
                entity_locations[entity] = new_loc
                rules_followed.append(f"causal_move:{entity}->{new_loc}")

        tokens.extend(statement)
        if i < num_statements - 1:
            tokens.append("<sep>")

    tokens.append("<eos>")

    token_ids = [TOKEN_TO_ID.get(t, TOKEN_TO_ID["<pad>"]) for t in tokens]

    return StructuredSequence(
        tokens=tokens,
        token_ids=token_ids,
        rules_followed=rules_followed,
        rules_violated=[],
        is_coherent=True,
    )


def corrupt_sequence(seq: StructuredSequence, corruption_type: str = "random") -> StructuredSequence:
    """Create a corrupted version that violates specific structural rules."""
    tokens = seq.tokens.copy()
    rules_violated = []

    if corruption_type == "random" :
        corruption_type = random.choice([
            "shuffle", "swap_attributes", "break_reference",
            "reverse_causal", "swap_entities"
        ])

    if corruption_type == "shuffle":
        # Shuffle statement order (breaks causal ordering)
        statements = []
        current = []
        for t in tokens[1:-1]:  # skip bos/eos
            if t == "<sep>":
                if current:
                    statements.append(current)
                current = []
            else:
                current.append(t)
        if current:
            statements.append(current)

        random.shuffle(statements)
        tokens = ["<bos>"]
        for i, stmt in enumerate(statements):
            tokens.extend(stmt)
            if i < len(statements) - 1:
                tokens.append("<sep>")
        tokens.append("<eos>")
        rules_violated.append("causal_order_violated")

    elif corruption_type == "swap_attributes":
        # Replace attributes with inconsistent ones
        for i, t in enumerate(tokens):
            if t in ATTRIBUTES and random.random() < 0.5:
                new_attr = random.choice([a for a in ATTRIBUTES if a != t])
                tokens[i] = new_attr
                rules_violated.append(f"attribute_inconsistency:{t}->{new_attr}")

    elif corruption_type == "break_reference":
        # Replace entity references with wrong entities
        for i, t in enumerate(tokens):
            if t in ENTITIES and random.random() < 0.4:
                new_entity = random.choice([e for e in ENTITIES if e != t])
                tokens[i] = new_entity
                rules_violated.append(f"broken_reference:{t}->{new_entity}")

    elif corruption_type == "reverse_causal":
        # Reverse connector meaning (put effects before causes)
        statements = []
        current = []
        for t in tokens[1:-1]:
            if t == "<sep>":
                if current:
                    statements.append(current)
                current = []
            else:
                current.append(t)
        if current:
            statements.append(current)

        if len(statements) >= 2:
            # Reverse pairs
            for i in range(0, len(statements) - 1, 2):
                statements[i], statements[i + 1] = statements[i + 1], statements[i]

        tokens = ["<bos>"]
        for i, stmt in enumerate(statements):
            tokens.extend(stmt)
            if i < len(statements) - 1:
                tokens.append("<sep>")
        tokens.append("<eos>")
        rules_violated.append("causal_reversal")

    elif corruption_type == "swap_entities":
        # Swap all occurrences of two entities
        entities_in_seq = [t for t in tokens if t in ENTITIES]
        if len(set(entities_in_seq)) >= 2:
            e1, e2 = random.sample(list(set(entities_in_seq)), 2)
            tokens = [e2 if t == e1 else (e1 if t == e2 else t) for t in tokens]
            rules_violated.append(f"entity_swap:{e1}<->{e2}")

    token_ids = [TOKEN_TO_ID.get(t, TOKEN_TO_ID["<pad>"]) for t in tokens]

    return StructuredSequence(
        tokens=tokens,
        token_ids=token_ids,
        rules_followed=[],
        rules_violated=rules_violated,
        is_coherent=False,
    )


class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset of positive/negative sequence pairs."""

    def __init__(self, num_samples=10000, num_statements=8, max_seq_len=128):
        self.pairs = []
        self.max_seq_len = max_seq_len

        for i in range(num_samples):
            pos = generate_coherent_sequence(num_statements, seed=i * 7 + 42)
            neg = corrupt_sequence(pos)
            self.pairs.append((pos, neg))

    def __len__(self):
        return len(self.pairs)

    def _pad(self, ids):
        if len(ids) >= self.max_seq_len:
            return ids[: self.max_seq_len]
        return ids + [TOKEN_TO_ID["<pad>"]] * (self.max_seq_len - len(ids))

    def __getitem__(self, idx):
        pos, neg = self.pairs[idx]
        pos_ids = torch.tensor(self._pad(pos.token_ids), dtype=torch.long)
        neg_ids = torch.tensor(self._pad(neg.token_ids), dtype=torch.long)
        return pos_ids, neg_ids


if __name__ == "__main__":
    # Demo
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Tokens: {ALL_TOKENS}\n")

    seq = generate_coherent_sequence(6, seed=42)
    print("=== Coherent sequence ===")
    print(" ".join(seq.tokens))
    print(f"Rules: {seq.rules_followed}\n")

    corrupted = corrupt_sequence(seq, "swap_attributes")
    print("=== Corrupted (swap_attributes) ===")
    print(" ".join(corrupted.tokens))
    print(f"Violations: {corrupted.rules_violated}\n")

    corrupted2 = corrupt_sequence(seq, "shuffle")
    print("=== Corrupted (shuffle) ===")
    print(" ".join(corrupted2.tokens))
    print(f"Violations: {corrupted2.rules_violated}\n")

    dataset = SyntheticDataset(num_samples=100)
    print(f"Dataset size: {len(dataset)}")
    pos, neg = dataset[0]
    print(f"Positive shape: {pos.shape}, Negative shape: {neg.shape}")
