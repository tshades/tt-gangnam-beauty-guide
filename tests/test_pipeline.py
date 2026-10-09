"""Golden-fixture tests. These encode the judgment calls, not just the code paths."""
from pathlib import Path

import pytest

from gbg.clinic import Registry
from gbg.extract import extract_offline, match_procedures
from gbg.models import Extraction, Price, RawReview
from gbg.pipeline import load_reviews, run
from gbg.validate import validate

FX = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def results():
    return {p.raw.id: p for p in run(load_reviews(FX / "reviews.jsonl"), Registry(FX / "clinics.json"), offline=True)}


def test_same_author_crosspost_folded(results):
    assert results["r02"].status == "duplicate" and results["r02"].canonical_id == "r01"


def test_cross_author_template_is_flagged_not_folded(results):
    for rid in ("r03", "r04", "r05"):
        assert results[rid].template_cluster and results[rid].status != "duplicate"
        assert results[rid].trust_score < 0.2


def test_ambiguous_clinic_abstains(results):
    # 하늘의원 could be the PS clinic or the derm clinic: never guess
    assert results["r08"].clinic.clinic_id is None and results["r08"].status == "quarantined"


def test_phone_beats_name(results):
    assert results["r07"].clinic.clinic_id == "c_orda_ps"  # English name, Korean registry


def test_specialty_never_merges():
    reg = Registry(FX / "clinics.json")
    assert reg.resolve("하늘피부과").clinic_id == "c_haneul_derm"
    assert reg.resolve("하늘성형외과").clinic_id == "c_haneul_ps"


def test_bare_number_price_is_manwon(results):
    assert [p.amount_krw for p in results["r01"].extraction.prices] == [1_800_000]


def test_safety_report_escalated_not_buried(results):
    assert any("SAFETY" in i for i in results["r09"].issues)


def test_receipt_and_roster_surgeon_boost(results):
    assert results["r10"].trust_score >= 0.8


def test_longest_match_wins():
    assert match_procedures("사각턱보톡스 맞음") == ["masseter_botox"]


def test_validator_catches_hallucinated_surgeon_and_unit_error():
    raw = RawReview(id="x", source="other", url="u", author="a", posted_at="2026-01-01",
                    clinic_name_raw="미르", text_ko="코수 350 주고 했어요")
    ex = Extraction(text_en="...", procedures=["rhinoplasty"], surgeon_name_ko="최민서",
                    prices=[Price(amount_krw=350, source_span="350")], sentiment="positive", confidence=0.9)
    issues = validate(raw, ex)
    assert any("hallucinated" in i for i in issues) and any("unit error" in i for i in issues)


def test_callcentre_number_is_not_a_hard_key():
    reg = Registry(FX / "clinics.json")
    m = reg.resolve("하늘의원", "1588-0000")
    assert m.method == "abstain"


def test_agent_answer_gate_rejects_invented_or_thin_answers():
    from gbg.models import ClinicMatch
    from gbg.resolver_agent import ResolverAgent
    agent = object.__new__(ResolverAgent)  # gate is pure code; no API client needed
    prior = ClinicMatch(clinic_id=None, method="abstain", score=1.0, candidates=["a", "b"])
    ev2 = ["fact one", "fact two"]
    assert agent._accept({"clinic_id": "zzz", "evidence": ev2, "confidence": 0.99}, {"a"}, prior, []).clinic_id is None
    assert agent._accept({"clinic_id": "a", "evidence": ["one"], "confidence": 0.99}, {"a"}, prior, []).clinic_id is None
    assert agent._accept({"clinic_id": "a", "evidence": ev2, "confidence": 0.5}, {"a"}, prior, []).clinic_id is None
    m = agent._accept({"clinic_id": "a", "evidence": ev2, "confidence": 0.9}, {"a"}, prior, [])
    assert m.clinic_id == "a" and m.method == "agent"
