"""Near-duplicate detection on the KOREAN text, before translation.

Why before MT: cheaper (no LLM spend on dups) and MT adds variance that hides dups.
Why char shingles: Korean is agglutinative and spacing is inconsistent ("윤곽3종" vs
"윤곽 3종"), so word shingles under-match. We strip whitespace/punct and use char 4-grams.

Key judgment: a near-dup cluster from the SAME author is a cross-post (fold it).
A near-dup cluster across DIFFERENT authors is a paid template campaign (체험단) —
those are not independent opinions and must not count as consensus.
"""
from __future__ import annotations

import re
from itertools import combinations

from .models import RawReview

_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)
K = 4
THRESHOLD = 0.45          # cross-author: must be near-verbatim to call it a template
SAME_AUTHOR_THRESHOLD = 0.25  # same author re-posting reworded text scores ~0.3 on 4-grams
# O(n^2) here; at scale swap for MinHash+LSH blocked by clinic_id.


def shingles(text: str, k: int = K) -> set[str]:
    t = _STRIP.sub("", text)
    return {t[i : i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def cluster(reviews: list[RawReview]) -> list[list[RawReview]]:
    """Union-find over pairwise Jaccard. Returns clusters (singletons included)."""
    parent = {r.id: r.id for r in reviews}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sh = {r.id: shingles(r.text_ko) for r in reviews}
    for a, b in combinations(reviews, 2):
        t = SAME_AUTHOR_THRESHOLD if a.author == b.author else THRESHOLD
        if jaccard(sh[a.id], sh[b.id]) >= t:
            parent[find(a.id)] = find(b.id)

    groups: dict[str, list[RawReview]] = {}
    for r in reviews:
        groups.setdefault(find(r.id), []).append(r)
    return list(groups.values())


def pick_canonical(group: list[RawReview]) -> RawReview:
    """Prefer receipt-verified, then earliest post (likely the original)."""
    return sorted(group, key=lambda r: (not r.has_receipt, r.posted_at))[0]
