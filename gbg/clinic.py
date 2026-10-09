"""Clinic entity resolution.

Asymmetric cost: a false MERGE attaches one clinic's reviews (incl. safety reports)
to another clinic — that's a trust-destroying, possibly defamatory error. A false
SPLIT just means a thinner profile until a human merges it. So: hard keys first,
specialty must agree, and abstain (-> human queue) whenever more than one clinic fits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .models import ClinicMatch

SPECIALTY = {"성형외과": "ps", "피부과": "derm", "plastic surgery": "ps", "ps": "ps", "dermatology": "derm"}
GENERIC = ["의원", "병원", "클리닉", "clinic", "hospital"]
BRANCH = re.compile(r"(강남|신사|압구정|청담|서초|본)?점$")


def norm_phone(p: str | None) -> str | None:
    """Digits only. Nationwide call-centre numbers (15xx/16xx/18xx) are shared by chains
    and brokers, so they are NOT a hard key: return None and fall through to name matching."""
    d = re.sub(r"\D", "", p) if p else ""
    return None if not d or re.match(r"^1[568]\d{2}", d) else d


def parse_name(raw: str) -> tuple[str, str | None]:
    """'미르 성형외과 본점' -> ('미르', 'ps'); 'Orda PS Clinic Gangnam' -> ('orda', 'ps')."""
    s = raw.strip().lower()
    s = re.sub(r"\bgangnam\b|\bseoul\b", "", s)
    s = re.sub(r"\s+", "", s) if re.search(r"[가-힣]", s) else s
    s = BRANCH.sub("", s)
    spec = None
    for k, v in SPECIALTY.items():
        if k in s:
            spec, s = v, s.replace(k, "")
    for g in GENERIC:
        s = s.replace(g, "")
    return re.sub(r"\s+", "", s).strip(), spec


def bigram_sim(a: str, b: str) -> float:
    A = {a[i : i + 2] for i in range(len(a) - 1)} or {a}
    B = {b[i : i + 2] for i in range(len(b) - 1)} or {b}
    return len(A & B) / len(A | B)


class Registry:
    def __init__(self, path: str | Path):
        self.clinics = json.loads(Path(path).read_text())
        self.by_id = {c["clinic_id"]: c for c in self.clinics}
        self._names = []  # (core, specialty, clinic_id)
        for c in self.clinics:
            for n in [c["name_ko"], c["name_en"], *c["aliases"]]:
                core, spec = parse_name(n)
                spec = spec or parse_name(c["name_ko"])[1]
                self._names.append((core, spec, c["clinic_id"]))

    def resolve(self, raw_name: str, phone: str | None = None) -> ClinicMatch:
        p = norm_phone(phone)
        if p:
            hits = [c["clinic_id"] for c in self.clinics if norm_phone(c["phone"]) == p]
            if len(hits) == 1:
                name_spec = parse_name(raw_name)[1]
                hit_spec = parse_name(self.by_id[hits[0]]["name_ko"])[1]
                if name_spec and hit_spec and name_spec != hit_spec:
                    return ClinicMatch(clinic_id=None, method="abstain", score=0.0, candidates=hits)
                return ClinicMatch(clinic_id=hits[0], method="phone", score=1.0)

        core, spec = parse_name(raw_name)
        best: dict[str, float] = {}
        for ncore, nspec, cid in self._names:
            if spec and nspec and spec != nspec:
                continue  # never merge across specialties (PS vs derm)
            s = 1.0 if core == ncore else bigram_sim(core, ncore)
            if core and (core in ncore or ncore in core):
                s = max(s, 0.8)
            best[cid] = max(best.get(cid, 0.0), s)

        ranked = sorted(best.items(), key=lambda x: -x[1])
        if not ranked or ranked[0][1] < 0.6:
            return ClinicMatch(clinic_id=None, method="abstain", score=ranked[0][1] if ranked else 0.0,
                               candidates=[c for c, _ in ranked[:3]])
        top_id, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if top - runner_up < 0.15:  # ambiguous -> human, don't guess
            return ClinicMatch(clinic_id=None, method="abstain", score=top,
                               candidates=[c for c, s in ranked if s >= 0.6])
        return ClinicMatch(clinic_id=top_id, method="exact_name" if top == 1.0 else "fuzzy_name", score=top)
