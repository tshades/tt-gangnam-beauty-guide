"""Trust scoring. Transparent additive rules (every point has a reason string) so an
editor can see WHY a review ranks where it does. Not an ML model on purpose: at this
stage explainability > accuracy, and the rules double as labelling functions later.

Judgment call: safety signals (ghost surgery, complications) are ESCALATED, never
down-ranked. A platform that buries negative safety reports is the failure mode
that kills trust in this category.
"""
from __future__ import annotations

from .models import ClinicMatch, Extraction, RawReview

SAFETY_TERMS = ["유령수술", "대리수술", "원장님이 다른", "원장님이랑 수술실"]


def score(raw: RawReview, ex: Extraction, clinic: ClinicMatch | None, roster: list[str],
          template_cluster: bool) -> tuple[float, list[str], list[str]]:
    s, why, flags = 0.5, [], []

    if raw.has_receipt:
        s += 0.25; why.append("+0.25 receipt-verified on source app (verified procedure)")
    if ex.self_paid_markers:
        s += 0.05; why.append(f"+0.05 self-paid claim {ex.self_paid_markers} (cheap to fake; small weight)")
    if ex.sponsored_markers:
        s -= 0.35; why.append(f"-0.35 sponsored markers {ex.sponsored_markers}")
    if template_cluster:
        s -= 0.25; why.append("-0.25 near-identical text from multiple authors (template campaign)")

    if ex.surgeon_name_ko:
        if ex.surgeon_name_ko in roster:
            s += 0.10; why.append(f"+0.10 named surgeon {ex.surgeon_name_ko} is on clinic's specialist roster (verified surgeon)")
        else:
            why.append(f"+0.00 named surgeon {ex.surgeon_name_ko} not on clinic roster (no verified-surgeon credit)")

    if any(t in raw.text_ko for t in SAFETY_TERMS):
        flags.append("SAFETY: possible ghost surgery report -> editorial escalation")
    if clinic is None or clinic.clinic_id is None:
        flags.append("clinic unresolved")

    return round(max(0.0, min(1.0, s)), 2), why, flags
