# Gangnam Beauty Guide — review syndication pipeline

Korean clinic reviews in → deduped, clinic-resolved, translated, validated, trust-scored reviews out.

```
python3 -m venv .venv && .venv/bin/pip install anthropic pydantic pytest python-dotenv
.venv/bin/python -m gbg --offline          # deterministic, no API key
cp .env.example .env  # add key
.venv/bin/python -m gbg --json out.json    # LLM translate+extract (claude-opus-5-5, effort=low)
.venv/bin/python -m pytest -q
```

## Pipeline

```
ingest ─► dedup (KO text) ─► resolve clinic ─► extract (LLM) ─► [resolver agent on abstains] ─► validate ─► trust ─► route
           char 4-grams       phone > name      translate +       spans must    additive,    published
           author-aware       specialty-gated   extract, 1 call   exist in src  explained    quarantined
                              abstains          glossary-bound                               duplicate
```

| Step | LLM? | Why |
|---|---|---|
| dedup | no | cheap, deterministic, runs before MT so we don't pay to translate dups |
| clinic resolution | no | false merge >> false split; hard keys + specialty gate, abstains when ambiguous |
| **resolver agent** | **yes, tool loop** | only on abstains: `search_registry`, `get_clinic`, `reviews_by_author` → `submit_resolution`. **Code gates the answer**: id must come from a tool result, ≥2 evidence items, conf ≥ 0.8, max 6 steps; else human queue |
| translate + extract | yes | slang (쌍수, 코수, 윤곽3종) and context need a model; glossary constrains ids |
| validate | no | the model may only assert what it can point to in the source |
| trust | no | explainable rules an editor can audit; they double as labelling functions later |

## Judgment calls (in code + tests)
- **Same-author near-dup = cross-post → fold. Cross-author near-dup = template campaign (체험단) → keep, flag, down-weight.** Not the same thing.
- **Never merge across specialty** (성형외과 vs 피부과), and abstain when two clinics fit. `하늘의원` goes to a human.
- **Bare numbers are 만원**: "견적 180" = ₩1,800,000. Validator rejects prices outside ₩50k–₩100M as unit errors.
- **Surgeon names must appear verbatim in source**; on-roster surgeon = "verified surgeon" signal.
- **Safety reports (ghost surgery) are escalated, never down-ranked.**

Fixtures are synthetic; clinic names/phones are fictional.

## How this was built
I prepared the skeleton above (pipeline, fixtures, glossary, tests) from the product card **before** the timed window started. That commit is tagged `prep`. Everything after the `clock-start` tag was written during the 40-minute assessment:
`git log --oneline clock-start..HEAD` / `git diff clock-start`.
