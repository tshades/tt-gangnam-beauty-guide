"""Orchestrator. Deterministic control flow; the LLM is one step, not the driver.

ingest -> dedup (KO text) -> resolve clinic -> extract (LLM, canonical only) -> validate -> trust -> route
Routes: published | quarantined (human queue) | duplicate (folded into canonical)
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import dedup
from .clinic import Registry
from .extract import extract_llm, extract_offline
from .models import ProcessedReview, RawReview, Status
from .trust import score
from .validate import validate


def load_reviews(path: str | Path) -> list[RawReview]:
    return [RawReview(**json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


def run(reviews: list[RawReview], registry: Registry, offline: bool = False, workers: int = 4) -> list[ProcessedReview]:
    out: dict[str, ProcessedReview] = {}

    # 1. dedup on Korean text
    for group in dedup.cluster(reviews):
        ids = [r.id for r in group]
        authors = {r.author for r in group}
        template = len(group) > 1 and len(authors) > 1
        canon = dedup.pick_canonical(group)
        for r in group:
            pr = ProcessedReview(raw=r, status=Status.published, dup_cluster=ids if len(group) > 1 else [],
                                 template_cluster=template)
            if len(group) > 1 and not template and r.id != canon.id:
                pr.status, pr.canonical_id = Status.duplicate, canon.id
                pr.trace.append(f"dedup: same-author cross-post of {canon.id}")
            elif template:
                pr.trace.append(f"dedup: template cluster across {len(authors)} authors {ids}")
            out[r.id] = pr

    live = [p for p in out.values() if p.status != Status.duplicate]

    # 2. clinic resolution (deterministic, abstains when ambiguous)
    for p in live:
        p.clinic = registry.resolve(p.raw.clinic_name_raw, p.raw.clinic_phone)
        p.trace.append(f"clinic: {p.clinic.method} -> {p.clinic.clinic_id} ({p.clinic.score:.2f})")

    def roster(p: ProcessedReview) -> list[str]:
        c = registry.by_id.get(p.clinic.clinic_id) if p.clinic and p.clinic.clinic_id else None
        return c["specialist_surgeons"] if c else []

    # 3. extract (parallel; LLM calls are the only slow/expensive step)
    def do_extract(p: ProcessedReview) -> None:
        try:
            p.extraction = extract_offline(p.raw, roster(p)) if offline else extract_llm(p.raw)
            p.trace.append(f"extract: {'offline' if offline else 'llm'} procs={p.extraction.procedures}")
        except Exception as e:  # one bad review must not sink the batch
            p.issues.append(f"extract failed: {e}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(do_extract, live))

    # 3b. agentic fallback: only for reviews the deterministic resolver abstained on
    abstained = [p for p in live if p.clinic and p.clinic.clinic_id is None]
    if abstained and not offline:
        from .resolver_agent import ResolverAgent
        index: dict[str, list[dict]] = {}
        for p in live:
            if p.clinic and p.clinic.clinic_id:
                index.setdefault(p.raw.author, []).append(
                    {"review_id": p.raw.id, "clinic_id": p.clinic.clinic_id, "via": p.clinic.method,
                     "posted_at": p.raw.posted_at, "text_ko": p.raw.text_ko})
        agent = ResolverAgent(registry, index)
        for p in abstained:
            try:
                p.clinic, steps = agent.resolve(p.raw, p.extraction, p.clinic)
                p.trace += steps
            except Exception as e:
                p.trace.append(f"agent failed: {e} -> stays abstained")

    # 4. validate + 5. trust + route
    for p in live:
        if p.extraction is None:
            p.status = Status.quarantined
            continue
        p.issues += validate(p.raw, p.extraction)
        p.trust_score, p.trust_reasons, flags = score(p.raw, p.extraction, p.clinic, roster(p), p.template_cluster)
        p.issues += flags
        blocking = [i for i in p.issues if not i.startswith("surgeon ")]
        if blocking:
            p.status = Status.quarantined
        p.trace.append(f"route: {p.status.value} trust={p.trust_score}")

    return [out[r.id] for r in reviews]


def summarize(results: list[ProcessedReview]) -> str:
    lines = [f"{'id':4} {'status':12} {'clinic':15} {'trust':5}  procedures / issues"]
    for p in results:
        cid = (p.clinic.clinic_id or "?") if p.clinic else "-"
        procs = ",".join(p.extraction.procedures) if p.extraction else ""
        t = f"{p.trust_score:.2f}" if p.trust_score is not None else "  -  "
        lines.append(f"{p.raw.id:4} {p.status.value:12} {cid:15} {t:5}  {procs}")
        if p.extraction and p.extraction.prices:
            lines.append(f"{'':38}₩ " + "; ".join(f"{x.amount_krw:,} ({x.source_span})" for x in p.extraction.prices))
        for i in p.issues:
            lines.append(f"{'':38}! {i}")
    return "\n".join(lines)
