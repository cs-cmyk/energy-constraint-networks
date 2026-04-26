"""
Extended coreference and structural corruption strategies.

Goes beyond simple pronoun swaps to include:
1. Name substitution — replace a person's name with a different one
2. Definite description mismatch — swap "The X" references between sentences
3. Demonstrative confusion — break this/that/these/those references
4. Singular/plural entity mismatch — "The committee... he decided"
5. Role/title swap — "The president... the senator announced"
6. Bridging reference break — "opened the book... the screen flickered"
7. Temporal reference break — "last year... next month... yesterday"
8. Repetition injection — repeat a sentence verbatim (common hallucination pattern)
9. Number contradiction — change a specific number to create inconsistency

Each corruption is designed to always produce a visible change (no silent failures).
"""

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CorruptionResult:
    sentences: List[str]
    corruption_type: str
    corrupted_indices: List[int] = field(default_factory=list)


# ============================================================
# Name pools for substitution
# ============================================================

PERSON_NAMES = [
    "James", "Robert", "John", "Michael", "David", "William", "Richard",
    "Joseph", "Thomas", "Charles", "Mary", "Patricia", "Jennifer", "Linda",
    "Barbara", "Elizabeth", "Susan", "Margaret", "Dorothy", "Sarah",
    "Anderson", "Thompson", "Garcia", "Martinez", "Robinson", "Clark",
    "Rodriguez", "Lewis", "Walker", "Hall", "Allen", "Young", "King",
]

ROLES_AND_TITLES = [
    "president", "senator", "director", "chairman", "governor", "minister",
    "secretary", "commander", "general", "professor", "doctor", "captain",
    "manager", "chancellor", "mayor", "ambassador", "commissioner",
    "superintendent", "principal", "dean", "bishop", "admiral",
]

DEFINITE_NOUNS = [
    "system", "program", "project", "study", "report", "plan",
    "proposal", "method", "approach", "technique", "process", "model",
    "theory", "experiment", "analysis", "investigation", "committee",
    "organization", "institution", "department", "team", "group",
]

OBJECTS = [
    "book", "document", "screen", "phone", "car", "building",
    "machine", "device", "instrument", "tool", "weapon", "vehicle",
    "computer", "ship", "aircraft", "bridge", "tower", "camera",
]


# ============================================================
# Helper functions
# ============================================================

def find_capitalized_names(text):
    """Find likely person names (capitalized words not at sentence start)."""
    words = text.split()
    names = []
    for i, w in enumerate(words):
        clean = w.rstrip(".,;:!?'\")")
        if (len(clean) > 1 and clean[0].isupper() and clean[1:].islower()
                and i > 0 and not words[i-1].endswith(".")):
            if clean.lower() not in {
                "the", "this", "that", "these", "those", "there", "then",
                "they", "their", "what", "when", "where", "which", "while",
                "however", "moreover", "furthermore", "although", "because",
                "after", "before", "during", "since", "until", "into",
                "also", "many", "some", "most", "each", "every", "both",
                "several", "various", "certain", "other", "another",
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
                "monday", "tuesday", "wednesday", "thursday", "friday",
                "saturday", "sunday",
                "university", "college", "institute", "school", "hospital",
                "church", "city", "state", "county", "river", "lake",
                "mountain", "island", "street", "avenue", "park", "bridge",
                "building", "museum", "library", "station", "airport",
                "north", "south", "east", "west", "new", "old", "great",
                "saint", "san", "los", "las", "del", "von", "van",
            }:
                names.append((i, clean))
    return names


def find_numbers(text):
    """Find numeric values in text."""
    return [(m.start(), m.group()) for m in re.finditer(r'\b\d[\d,]*\.?\d*\b', text)]


def find_definite_descriptions(text):
    """Find 'the X' patterns."""
    return [(m.start(), m.group()) for m in
            re.finditer(r'\b[Tt]he\s+([a-z]+(?:\s+[a-z]+)?)\b', text)]


# ============================================================
# Corruption strategies
# ============================================================

def corrupt_name_substitution(sentences: List[str]) -> CorruptionResult:
    """
    Replace a person's name with a different name.
    "Dr. Elena Vasquez led the trial. Dr. Park reported the results."
    — introduces a new person where the same one should be referenced.
    """
    result = sentences.copy()
    corrupted = []

    # Find names across all sentences
    name_locations = []
    for idx, sent in enumerate(result):
        names = find_capitalized_names(sent)
        for pos, name in names:
            name_locations.append((idx, name))

    if len(name_locations) < 1:
        # Fallback: pick a sentence and inject a random name change
        idx = random.randint(0, len(result) - 1)
        words = result[idx].split()
        for i, w in enumerate(words):
            clean = w.rstrip(".,;:!?")
            if clean[0:1].isupper() and len(clean) > 2 and i > 0:
                suffix = w[len(clean):]
                new_name = random.choice([n for n in PERSON_NAMES if n != clean])
                words[i] = new_name + suffix
                result[idx] = " ".join(words)
                corrupted.append(idx)
                break

        if not corrupted:
            # Hard fallback: just prepend a contradicting name
            idx = random.randint(1, max(1, len(result) - 1))
            result[idx] = result[idx].replace(
                result[idx].split()[0],
                random.choice(PERSON_NAMES), 1)
            corrupted.append(idx)

        return CorruptionResult(result, "name_substitution", corrupted)

    # Pick a name and replace one occurrence with a different name
    idx, name = random.choice(name_locations)
    new_name = random.choice([n for n in PERSON_NAMES if n.lower() != name.lower()])
    result[idx] = result[idx].replace(name, new_name, 1)
    corrupted.append(idx)

    return CorruptionResult(result, "name_substitution", corrupted)


def corrupt_definite_description_swap(sentences: List[str]) -> CorruptionResult:
    """
    Swap definite descriptions between sentences.
    "The algorithm was tested. The dataset was published."
    → "The dataset was tested. The algorithm was published."
    Breaks what "the X" refers to.
    """
    result = sentences.copy()

    # Find "the X" patterns in different sentences
    descriptions = []
    for idx, sent in enumerate(result):
        for m in re.finditer(r'\b(the\s+)([a-z]+)', sent, re.IGNORECASE):
            noun = m.group(2)
            if noun.lower() not in {"the", "a", "an", "and", "or", "but", "in",
                                     "of", "to", "for", "on", "at", "by", "is",
                                     "was", "are", "were", "be", "been", "being",
                                     "same", "other", "most", "first", "last",
                                     "next", "following", "end", "time"}:
                descriptions.append((idx, m.start(), m.end(), noun))

    # Find two descriptions in different sentences and swap the nouns
    if len(descriptions) >= 2:
        random.shuffle(descriptions)
        for i in range(len(descriptions)):
            for j in range(i + 1, len(descriptions)):
                if descriptions[i][0] != descriptions[j][0]:
                    idx1, _, _, noun1 = descriptions[i]
                    idx2, _, _, noun2 = descriptions[j]
                    if noun1 != noun2:
                        result[idx1] = re.sub(
                            r'\b(the\s+)' + re.escape(noun1) + r'\b',
                            r'\1' + noun2, result[idx1], count=1, flags=re.IGNORECASE)
                        result[idx2] = re.sub(
                            r'\b(the\s+)' + re.escape(noun2) + r'\b',
                            r'\1' + noun1, result[idx2], count=1, flags=re.IGNORECASE)
                        return CorruptionResult(result, "description_swap", [idx1, idx2])

    # Fallback: replace a definite description with a random noun
    if descriptions:
        idx, start, end, noun = random.choice(descriptions)
        new_noun = random.choice([n for n in DEFINITE_NOUNS if n != noun])
        result[idx] = re.sub(
            r'\b(the\s+)' + re.escape(noun) + r'\b',
            r'\1' + new_noun, result[idx], count=1, flags=re.IGNORECASE)
        return CorruptionResult(result, "description_swap", [idx])

    # Hard fallback
    idx = random.randint(0, len(result) - 1)
    result[idx] = "The " + random.choice(DEFINITE_NOUNS) + " " + result[idx][0].lower() + result[idx][1:]
    return CorruptionResult(result, "description_swap", [idx])


def corrupt_demonstrative_confusion(sentences: List[str]) -> CorruptionResult:
    """
    Break demonstrative references (this/that/these/those).
    "The team developed a novel algorithm. This breakthrough led to..."
    → "The team developed a novel algorithm. That failure led to..."
    """
    result = sentences.copy()
    corrupted = []

    demonstrative_swaps = {
        "this": ["that", "another", "some"],
        "that": ["this", "another", "some"],
        "these": ["those", "other", "several"],
        "those": ["these", "other", "several"],
        "This": ["That", "Another", "Some"],
        "That": ["This", "Another", "Some"],
        "These": ["Those", "Other", "Several"],
        "Those": ["These", "Other", "Several"],
    }

    # Also swap the following noun for extra disruption
    counter_nouns = {
        "success": "failure", "failure": "success",
        "increase": "decrease", "decrease": "increase",
        "improvement": "decline", "decline": "improvement",
        "growth": "reduction", "reduction": "growth",
        "advantage": "disadvantage", "disadvantage": "advantage",
        "breakthrough": "setback", "setback": "breakthrough",
        "progress": "regression", "regression": "progress",
        "victory": "defeat", "defeat": "victory",
        "discovery": "loss", "approach": "retreat",
        "expansion": "contraction", "contraction": "expansion",
    }

    for idx in range(1, len(result)):
        words = result[idx].split()
        changed = False
        for i, w in enumerate(words):
            clean = w.rstrip(".,;:!?")
            suffix = w[len(clean):]
            if clean in demonstrative_swaps:
                # Skip if followed by "the" or a verb — it's likely a complementizer
                if i + 1 < len(words):
                    next_word = words[i+1].rstrip(".,;:!?").lower()
                    if next_word in {"the", "a", "an", "is", "was", "are", "were",
                                     "it", "he", "she", "they", "we", "you", "i",
                                     "not", "no", "all", "its", "his", "her"}:
                        continue
                new_dem = random.choice(demonstrative_swaps[clean])
                words[i] = new_dem + suffix
                changed = True
                # Try to also swap the following noun
                if i + 1 < len(words):
                    next_clean = words[i+1].rstrip(".,;:!?")
                    next_suffix = words[i+1][len(next_clean):]
                    if next_clean.lower() in counter_nouns:
                        words[i+1] = counter_nouns[next_clean.lower()] + next_suffix
                break
        if changed:
            result[idx] = " ".join(words)
            corrupted.append(idx)
            break

    if not corrupted:
        # Fallback: inject a demonstrative that doesn't resolve
        idx = random.randint(1, max(1, len(result) - 1))
        result[idx] = "That particular outcome " + result[idx][0].lower() + result[idx][1:]
        corrupted.append(idx)

    return CorruptionResult(result, "demonstrative_confusion", corrupted)


def corrupt_singular_plural_mismatch(sentences: List[str]) -> CorruptionResult:
    """
    Create singular/plural mismatch on entity references.
    "The researchers published their findings. He concluded that..."
    — singular pronoun for plural entity.
    """
    result = sentences.copy()
    corrupted = []

    singular_to_plural = {
        "he": "they", "she": "they", "it": "they",
        "He": "They", "She": "They", "It": "They",
        "his": "their", "her": "their", "its": "their",
        "His": "Their", "Her": "Their", "Its": "Their",
        "him": "them", "was": "were", "is": "are",
        "has": "have",
    }
    plural_to_singular = {
        "they": "he", "They": "He",
        "their": "his", "Their": "His",
        "them": "him", "were": "was",
        "are": "is", "have": "has",
    }

    # Combine both directions
    all_swaps = {**singular_to_plural, **plural_to_singular}

    for idx in range(1, len(result)):
        words = result[idx].split()
        for i, w in enumerate(words):
            clean = w.rstrip(".,;:!?")
            suffix = w[len(clean):]
            if clean in all_swaps:
                words[i] = all_swaps[clean] + suffix
                result[idx] = " ".join(words)
                corrupted.append(idx)
                break
        if corrupted:
            break

    if not corrupted:
        # Fallback
        idx = random.randint(1, max(1, len(result) - 1))
        result[idx] = result[idx].replace(" was ", " were ", 1)
        if result[idx] != sentences[idx]:
            corrupted.append(idx)
        else:
            result[idx] = result[idx].replace(" is ", " are ", 1)
            corrupted.append(idx)

    return CorruptionResult(result, "singular_plural_mismatch", corrupted)


def corrupt_role_title_swap(sentences: List[str]) -> CorruptionResult:
    """
    Swap a role/title with a different one.
    "The president signed the bill. The senator faced opposition."
    — wrong role for the established referent.
    """
    result = sentences.copy()
    corrupted = []

    # Find roles in text — only match "the/The ROLE" pattern to avoid adjective matches
    role_locations = []
    for idx, sent in enumerate(result):
        for role in ROLES_AND_TITLES:
            pattern = r'\b[Tt]he\s+' + re.escape(role) + r'\b'
            if re.search(pattern, sent):
                role_locations.append((idx, role, pattern))

    if role_locations:
        idx, role, pattern = random.choice(role_locations)
        new_role = random.choice([r for r in ROLES_AND_TITLES if r != role])
        result[idx] = re.sub(pattern,
            lambda m: m.group(0).replace(role, new_role),
            result[idx], count=1)
        corrupted.append(idx)
    else:
        # Inject a role where there isn't one
        idx = random.randint(0, len(result) - 1)
        role = random.choice(ROLES_AND_TITLES)
        result[idx] = f"The {role} noted that " + result[idx][0].lower() + result[idx][1:]
        corrupted.append(idx)

    return CorruptionResult(result, "role_title_swap", corrupted)


def corrupt_temporal_reference_break(sentences: List[str]) -> CorruptionResult:
    """
    Break temporal consistency by swapping time references.
    "In 1965, the project began. By 1960, it was completed."
    — completion before start.
    """
    result = sentences.copy()
    corrupted = []

    temporal_swaps = {
        "before": "after", "after": "before",
        "earlier": "later", "later": "earlier",
        "previous": "following", "following": "previous",
        "first": "last", "last": "first",
        "began": "ended", "ended": "began",
        "started": "finished", "finished": "started",
        "initially": "finally", "finally": "initially",
        "recently": "long ago",
    }

    for idx in range(len(result)):
        words = result[idx].split()
        changed = False
        for i, w in enumerate(words):
            clean = w.rstrip(".,;:!?")
            suffix = w[len(clean):]
            if clean.lower() in temporal_swaps:
                new_word = temporal_swaps[clean.lower()]
                if clean[0].isupper():
                    new_word = new_word.capitalize()
                words[i] = new_word + suffix
                changed = True
                break
        if changed:
            result[idx] = " ".join(words)
            corrupted.append(idx)
            break

    if not corrupted:
        # Swap years if present
        for idx in range(len(result)):
            years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', result[idx])
            if years:
                year = years[0]
                offset = random.choice([-30, -20, -10, 10, 20, 30])
                new_year = str(int(year) + offset)
                result[idx] = result[idx].replace(year, new_year, 1)
                corrupted.append(idx)
                break

    if not corrupted:
        idx = random.randint(0, len(result) - 1)
        result[idx] = result[idx] + " This preceded the earlier events."
        corrupted.append(idx)

    return CorruptionResult(result, "temporal_break", corrupted)


def corrupt_repetition_injection(sentences: List[str]) -> CorruptionResult:
    """
    Repeat a sentence verbatim later in the paragraph.
    This mimics a common hallucination pattern where LMs loop.
    """
    result = sentences.copy()
    if len(result) < 3:
        return CorruptionResult(result, "repetition", [])

    # Pick a sentence from the first half, repeat it in the second half
    src = random.randint(0, len(result) // 2)
    dst = random.randint(len(result) // 2 + 1, len(result) - 1)
    result[dst] = result[src]

    return CorruptionResult(result, "repetition", [dst])


def corrupt_number_contradiction(sentences: List[str]) -> CorruptionResult:
    """
    Change a specific number to create inconsistency.
    "The population was 50,000. The city served all 50,000 residents."
    → "The population was 50,000. The city served all 80,000 residents."
    """
    result = sentences.copy()
    corrupted = []

    # Find sentences with numbers
    for idx in range(len(result)):
        numbers = find_numbers(result[idx])
        if numbers:
            pos, num_str = random.choice(numbers)
            try:
                num = float(num_str.replace(",", ""))
                # Change by 20-80%
                factor = random.choice([0.3, 0.5, 0.7, 1.5, 2.0, 3.0])
                new_num = int(num * factor) if num == int(num) else round(num * factor, 1)
                new_str = f"{new_num:,}" if num >= 1000 else str(new_num)
                result[idx] = result[idx][:pos] + new_str + result[idx][pos + len(num_str):]
                corrupted.append(idx)
                break
            except ValueError:
                continue

    if not corrupted:
        # Inject a contradicting number
        idx = random.randint(0, len(result) - 1)
        result[idx] = result[idx].rstrip(".") + ", totalling approximately 47,000 ."
        corrupted.append(idx)

    return CorruptionResult(result, "number_contradiction", corrupted)


def corrupt_bridging_reference_break(sentences: List[str]) -> CorruptionResult:
    """
    Break bridging references by swapping an object with an incompatible one.
    "She opened the book. The pages were yellowed with age."
    → "She opened the book. The screen was yellowed with age."
    """
    result = sentences.copy()
    corrupted = []

    for idx in range(1, len(result)):
        for m in re.finditer(r'\b[Tt]he\s+([a-z]+)', result[idx]):
            noun = m.group(1)
            if noun in DEFINITE_NOUNS or len(noun) < 3:
                continue
            new_noun = random.choice([o for o in OBJECTS if o != noun])
            result[idx] = result[idx][:m.start(1)] + new_noun + result[idx][m.end(1):]
            corrupted.append(idx)
            break
        if corrupted:
            break

    if not corrupted:
        idx = random.randint(1, max(1, len(result) - 1))
        obj = random.choice(OBJECTS)
        result[idx] = f"The {obj} " + result[idx][0].lower() + result[idx][1:]
        corrupted.append(idx)

    return CorruptionResult(result, "bridging_break", corrupted)


# ============================================================
# All extended corruptions
# ============================================================

EXTENDED_CORRUPTIONS = {
    "name_substitution": corrupt_name_substitution,
    "description_swap": corrupt_definite_description_swap,
    "demonstrative_confusion": corrupt_demonstrative_confusion,
    "singular_plural": corrupt_singular_plural_mismatch,
    "role_title_swap": corrupt_role_title_swap,
    "temporal_break": corrupt_temporal_reference_break,
    "repetition": corrupt_repetition_injection,
    "number_contradiction": corrupt_number_contradiction,
    "bridging_break": corrupt_bridging_reference_break,
}

def apply_random_extended_corruption(sentences, donor_sentences=None):
    """Apply a random extended corruption."""
    from nl_adaptation import (
        corrupt_shuffle, corrupt_negate, corrupt_swap_entities,
        corrupt_tense_shift, corrupt_coreference_break, corrupt_topic_splice,
    )

    # Combine original + extended corruptions
    all_corruptions = {
        "shuffle": lambda s: corrupt_shuffle(s).sentences,
        "negate": lambda s: corrupt_negate(s).sentences,
        "entity_swap": lambda s: corrupt_swap_entities(s).sentences,
        "tense_shift": lambda s: corrupt_tense_shift(s).sentences,
        "coref_break": lambda s: corrupt_coreference_break(s).sentences,
        **{k: lambda s, f=v: f(s).sentences for k, v in EXTENDED_CORRUPTIONS.items()},
    }
    if donor_sentences:
        all_corruptions["topic_splice"] = lambda s: corrupt_topic_splice(s, donor_sentences).sentences

    ctype = random.choice(list(all_corruptions.keys()))
    corrupted = all_corruptions[ctype](sentences)
    return corrupted, ctype


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    text = [
        "Dr. Elena Vasquez led the clinical trial at Stanford University.",
        "The study enrolled 500 participants over a period of two years.",
        "She reported that the results exceeded all expectations.",
        "The committee reviewed the findings in their quarterly meeting.",
        "By 2023, the treatment had been approved for general use.",
        "This breakthrough attracted significant attention from the medical community.",
    ]

    print("Original:")
    for i, s in enumerate(text):
        print(f"  {i+1}. {s}")

    print()
    for name, fn in EXTENDED_CORRUPTIONS.items():
        random.seed(42)
        result = fn(text)
        print(f"{result.corruption_type} (corrupted: {result.corrupted_indices}):")
        for i, s in enumerate(result.sentences):
            changed = " <<<" if i in result.corrupted_indices else ""
            print(f"  {i+1}. {s}{changed}")
        print()
