"""Translate + extract in ONE LLM call (structured output), with a deterministic offline fallback.

One call, not two: translation and extraction share the same reading of slang
(쌍수 = double eyelid), and asking for verbatim source spans lets us validate
the output mechanically afterwards (see validate.py) instead of trusting it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .models import Extraction, Price, RawReview

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "extract"
MODEL = os.environ.get("GBG_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("GBG_EFFORT", "low")  # extraction is routine; raise if evals say so

# Minimal built-in glossary; replaced by data/glossary.json (81 procedures) when present.
_BUILTIN = [
    {"ko": ["쌍수", "쌍꺼풀", "매몰", "절개"], "procedure_id": "double_eyelid", "en": "double eyelid surgery"},
    {"ko": ["코수", "코성형", "코끝", "늑연골"], "procedure_id": "rhinoplasty", "en": "rhinoplasty"},
    {"ko": ["윤곽3종", "윤곽 3종", "광대축소", "사각턱"], "procedure_id": "facial_contouring", "en": "facial contouring (3-part)"},
    {"ko": ["앞트임"], "procedure_id": "epicanthoplasty", "en": "epicanthoplasty"},
    {"ko": ["눈매교정"], "procedure_id": "ptosis_correction", "en": "ptosis correction"},
    {"ko": ["가슴수술", "가슴성형"], "procedure_id": "breast_augmentation", "en": "breast augmentation"},
    {"ko": ["울쎄라"], "procedure_id": "ulthera", "en": "Ulthera HIFU lifting"},
]
SPONSORED = ["협찬", "체험단", "원고료", "제공받아", "제공받았", "#광고"]
SELF_PAID = ["내돈내산", "내돈 내산", "자비로"]


def load_glossary() -> list[dict]:
    p = ROOT / "data" / "glossary.json"
    if p.exists():
        data = json.loads(p.read_text())
        procs = data.get("procedures", data) if isinstance(data, dict) else data
        if isinstance(procs, list) and procs and "procedure_id" in procs[0]:
            return procs
    return _BUILTIN


GLOSSARY = load_glossary()
PROCEDURE_IDS = {g["procedure_id"] for g in GLOSSARY} | {"unknown"}

SYSTEM = """You translate and extract Korean cosmetic-surgery reviews for English-speaking medical tourists.
Rules:
- Translate faithfully; keep the reviewer's tone. Do not add claims that are not in the text.
- procedures: use ONLY procedure_ids from the glossary below; use "unknown" if none fits.
- Prices: Korean reviewers write "180" or "180 나왔고" meaning 180만원 (1,800,000 KRW). Convert to integer KRW.
  source_span MUST be an exact substring of the Korean text.
- surgeon_name_ko: only if a name literally appears in the text. Never infer one.
- sponsored_markers / self_paid_markers: verbatim Korean phrases only.
- confidence: your confidence that the extraction is complete and correct.

Glossary (procedure_id: Korean variants -> English):
{glossary}"""


def _glossary_text() -> str:
    return "\n".join(f"{g['procedure_id']}: {', '.join(g['ko'])} -> {g['en']}" for g in GLOSSARY)


def _cache_key(r: RawReview) -> str:
    return hashlib.sha256(f"{MODEL}|{r.text_ko}".encode()).hexdigest()[:16]


def extract_llm(r: RawReview) -> Extraction:
    import anthropic

    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{_cache_key(r)}.json"
    if cp.exists():  # idempotent re-runs: never pay twice for the same text
        return Extraction.model_validate_json(cp.read_text())

    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        system=[{"type": "text", "text": SYSTEM.format(glossary=_glossary_text()),
                 "cache_control": {"type": "ephemeral"}}],  # glossary prefix is shared by every review
        messages=[{"role": "user", "content": f"Clinic (as written): {r.clinic_name_raw}\n\nReview:\n{r.text_ko}"}],
        output_format=Extraction,
        output_config={"effort": EFFORT},
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"model returned no extraction (stop_reason={resp.stop_reason})")
    cp.write_text(resp.parsed_output.model_dump_json())
    return resp.parsed_output


# ---- offline fallback: deterministic, glossary + regex. Also the baseline to eval the LLM against.
_MANWON = re.compile(r"(\d[\d,]*)\s*만\s*원")
_WON = re.compile(r"(\d{1,3}(?:,\d{3})+|\d{5,})\s*원")
_BARE = re.compile(r"(?:견적은?|가격은?|총|비용)\s*(\d{2,4})(?!\d|,|\s*원|\s*만|샷)|(\d{2,4})\s*(?:주고|들었|나왔)")
_DPLUS = re.compile(r"D\+(\d+)")


def _prices(text: str) -> list[Price]:
    out = []
    for m in _MANWON.finditer(text):
        out.append(Price(amount_krw=int(m.group(1).replace(",", "")) * 10_000, source_span=m.group(0)))
    for m in _WON.finditer(text):
        if "만" not in text[max(0, m.start() - 2) : m.start()]:
            out.append(Price(amount_krw=int(m.group(1).replace(",", "")), source_span=m.group(0)))
    for m in _BARE.finditer(text):
        n = m.group(1) or m.group(2)
        out.append(Price(amount_krw=int(n) * 10_000, source_span=m.group(0)))  # "180" => 180만원
    return out


# Longest variant first, and consume matched text: 사각턱보톡스 (jaw botox) must not
# also fire 사각턱 (jaw-bone surgery). Spaces stripped: "윤곽 3종" == "윤곽3종".
_TERMS = sorted(((k.replace(" ", ""), g["procedure_id"]) for g in GLOSSARY for k in g["ko"]),
                key=lambda x: -len(x[0]))


def match_procedures(text: str) -> list[str]:
    t, found = text.replace(" ", ""), set()
    for term, pid in _TERMS:
        if len(term) >= 2 and term in t:
            found.add(pid)
            t = t.replace(term, "\x00")
    return sorted(found) or ["unknown"]


def extract_offline(r: RawReview, surgeon_roster: list[str] | None = None) -> Extraction:
    t = r.text_ko
    procs = match_procedures(t)
    surgeon = next((n for n in (surgeon_roster or []) if n in t), None)
    neg = any(w in t for w in ["비대칭", "재수술", "걱정", "유령수술", "저하", "부작용으로"])
    pos = any(w in t for w in ["만족", "추천", "효과", "자연스러"])
    d = _DPLUS.search(t)
    return Extraction(
        text_en=f"[offline: no MT] {t}",
        procedures=procs,
        surgeon_name_ko=surgeon,
        prices=_prices(t),
        sponsored_markers=[m for m in SPONSORED if m in t],
        self_paid_markers=[m for m in SELF_PAID if m in t],
        days_post_op=int(d.group(1)) if d else None,
        sentiment="mixed" if pos and neg else "negative" if neg else "positive",
        complications=[],
        confidence=0.5,
    )
