"""Rule-based and fuzzy matching utilities for instrument extraction from transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

CANONICAL_LABELS = [
    "forceps",
    "hemostat",
    "scissors",
    "scalpel",
]

# Deterministic tie-break when multiple instruments are matched in one transcript.
PRIORITY_ORDER = ["forceps", "hemostat", "scissors", "scalpel"]

FORCEPS_PATTERNS = [
    r"\bforceps?\b",
    r"\bforsep(s)?\b",
]
HEMOSTAT_PATTERNS = [
    r"\bhemostat(s)?\b",
]
SCISSORS_PATTERNS = [
    r"\bscissors?\b",
    r"\bscissor\b",
]
SCALPEL_PATTERNS = [
    r"\bscalpel\b",
]

FUZZY_PHRASES = {
    "forceps": ["forceps", "forcep", "forsep"],
    "hemostat": ["hemostat", "hemostats"],
    "scissors": ["scissors", "scissor"],
    "scalpel": ["scalpel"],
}


@dataclass(frozen=True)
class Match:
    """Represents one text match candidate for an instrument label."""

    label: str
    span: Tuple[int, int]
    match: str
    pattern: str


def normalize_transcript(text: str) -> str:
    """Normalize ASR transcript text before rule matching."""
    text = text.lower()
    text = re.sub(r"<epsilon>", "", text)
    text = re.sub(r"</?s>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_pattern_matches(text: str, label: str, pattern: str) -> List[Match]:
    """Return regex matches for one label/pattern pair."""
    return [
        Match(label=label, span=(m.start(), m.end()), match=m.group(0), pattern=pattern)
        for m in re.finditer(pattern, text)
    ]


def _squash_repeated_chars(token: str) -> str:
    """Collapse repeated characters to reduce ASR stutter noise."""
    return re.sub(r"(.)\1+", r"\1", token)


def _squash_repeated_ngrams(token: str, max_n: int = 3) -> str:
    """Collapse repeated n-grams in token text for fuzzy matching robustness."""
    out = token
    changed = True
    while changed:
        changed = False
        for n in range(max_n, 1, -1):
            i = 0
            chunks: List[str] = []
            while i < len(out):
                chunk = out[i : i + n]
                if len(chunk) < n:
                    chunks.append(out[i:])
                    break
                j = i + n
                repeated = False
                while j + n <= len(out) and out[j : j + n] == chunk:
                    j += n
                    repeated = True
                chunks.append(chunk)
                if repeated:
                    changed = True
                i = j
            out = "".join(chunks)
    return out


def _normalize_token_for_fuzzy(token: str) -> str:
    """Normalize a token before approximate string matching."""
    token = re.sub(r"[^a-z0-9]", "", token.lower())
    token = _squash_repeated_chars(token)
    return _squash_repeated_ngrams(token)


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = cur[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (ca != cb)
            cur.append(min(insert_cost, delete_cost, replace_cost))
        prev = cur
    return prev[-1]


def _is_near_token(actual: str, expected: str) -> bool:
    """Heuristic fuzzy token match used for noisy medical ASR transcripts."""
    a = _normalize_token_for_fuzzy(actual)
    e = _normalize_token_for_fuzzy(expected)
    if not a or not e:
        return False
    if a == e:
        return True
    if a[0] != e[0] or a[-1] != e[-1]:
        return False
    max_dist = 1 if len(e) <= 4 else 2 if len(e) <= 7 else 3
    return _levenshtein(a, e) <= max_dist


def _find_fuzzy_phrase_matches(text: str, label: str, phrase: str) -> List[Match]:
    """Find approximate phrase matches with limited skips between tokens."""
    phrase_tokens = phrase.split()
    token_iter = list(re.finditer(r"\S+", text))
    tokens = [m.group(0) for m in token_iter]
    out: List[Match] = []
    if not phrase_tokens or len(token_iter) < len(phrase_tokens):
        return out

    max_skip_per_step = 2
    max_total_skip = 2
    for i in range(len(token_iter)):
        token_pos = i
        total_skips = 0
        matched_indices: List[int] = []

        for expected in phrase_tokens:
            found_pos = None
            upper = min(len(tokens), token_pos + max_skip_per_step + 1)
            for cand in range(token_pos, upper):
                if _is_near_token(tokens[cand], expected):
                    found_pos = cand
                    break
            if found_pos is None:
                matched_indices = []
                break

            total_skips += found_pos - token_pos
            if total_skips > max_total_skip:
                matched_indices = []
                break

            matched_indices.append(found_pos)
            token_pos = found_pos + 1

        if matched_indices:
            start = token_iter[matched_indices[0]].start()
            end = token_iter[matched_indices[-1]].end()
            out.append(Match(label=label, span=(start, end), match=text[start:end], pattern=f"~{phrase}~"))
    return out


def _dedupe(matches: Sequence[Match]) -> List[Match]:
    """Remove duplicate matches while preserving first-seen ordering."""
    seen = set()
    unique: List[Match] = []
    for m in matches:
        key = (m.label, m.span[0], m.span[1], m.match, m.pattern)
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def _sort_matches(matches: Sequence[Match]) -> List[Match]:
    """Sort matches by text span then stable label priority."""
    return sorted(matches, key=lambda m: (m.span[0], m.span[1], PRIORITY_ORDER.index(m.label)))


def extract_instrument(transcript: str) -> Dict[str, object]:
    """Extract canonical instrument label from ASR text using strict then fuzzy matching."""
    normalized = normalize_transcript(transcript)

    strict_matches: List[Match] = []
    for p in FORCEPS_PATTERNS:
        strict_matches.extend(_find_pattern_matches(normalized, "forceps", p))
    for p in HEMOSTAT_PATTERNS:
        strict_matches.extend(_find_pattern_matches(normalized, "hemostat", p))
    for p in SCISSORS_PATTERNS:
        strict_matches.extend(_find_pattern_matches(normalized, "scissors", p))
    for p in SCALPEL_PATTERNS:
        strict_matches.extend(_find_pattern_matches(normalized, "scalpel", p))

    strict_matches = _sort_matches(_dedupe(strict_matches))

    all_matches = list(strict_matches)
    if not all_matches:
        fuzzy_matches: List[Match] = []
        for label, phrases in FUZZY_PHRASES.items():
            for phrase in phrases:
                fuzzy_matches.extend(_find_fuzzy_phrase_matches(normalized, label, phrase))
        all_matches = _sort_matches(_dedupe(fuzzy_matches))

    chosen_label = "unknown"
    chosen_pattern = ""

    if all_matches:
        labels_present = {m.label for m in all_matches}
        for label in PRIORITY_ORDER:
            if label in labels_present:
                chosen_label = label
                break

        chosen = next(m for m in all_matches if m.label == chosen_label)
        chosen_pattern = chosen.pattern

    return {
        "instrument": chosen_label,
        "matched_pattern": chosen_pattern,
        "all_matches": [
            {
                "label": m.label,
                "span": [m.span[0], m.span[1]],
                "match": m.match,
                "pattern": m.pattern,
            }
            for m in all_matches
        ],
    }
