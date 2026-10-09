# Gangnam Beauty Guide — review syndication pipeline

Korean clinic reviews in → deduped, clinic-resolved, translated, validated, trust-scored reviews out.
Two LLM agents (translate+extract; a tool-using clinic resolver) sit inside a deterministic pipeline, and **code checks every agent output before anything publishes.**

### Live run on the fixtures ([full output with per-review trace](examples/run_output.json))
```
id   status       clinic          trust  procedures
r01  published    c_haneul_ps     0.65   double_eyelid_buried_suture      ₩1,800,000 ("견적은 180")
r02  duplicate    -                 -    same author cross-posted r01 to a café
r03  published    c_mir_ps        0.00   facial_contouring_3_combo        체험단 template ×3 authors
r04  published    c_mir_ps        0.00   facial_contouring_3_combo        (sponsored markers → trust 0.00)
r05  published    c_mir_ps        0.00   facial_contouring_3_combo
r06  published    c_orda_ps       0.85   rhinoplasty_rib_cartilage        receipt + surgeon on roster
r08  published    c_haneul_derm   0.50   ulthera                          resolved by the AGENT (below)
r09  quarantined  c_haneul_ps     0.50   breast_augmentation              SAFETY: possible ghost surgery → editor
r10  published    c_mir_ps        0.85   facial_contouring_3_combo, ...   ₩12,000,000
```

### The agent handoff, on r08
The review names the clinic only as **"하늘의원"**. The deterministic matcher can't choose between a plastic-surgery clinic and a dermatology clinic with that name, so it **abstains** rather than guess. The resolver agent then investigates:
```
agent.get_clinic(c_haneul_ps) · agent.get_clinic(c_haneul_derm) · agent.reviews_by_author(momo_k) · agent.search_registry(하늘)
agent.submit -> c_haneul_derm  conf=0.90
  - c_haneul_derm lists ulthera in its services; c_haneul_ps is surgical only
  - author momo_k's earlier review r11 (phone-verified to c_haneul_derm) says they planned 울쎄라 there next
  - ...
code gate: id came back from an earlier tool call ✓ · ≥2 distinct evidence items ✓ · conf ≥ 0.8 ✓ · specialty consistent ✓ → accepted
```

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
| **resolver agent** | **yes, tool loop** | only on abstains: `search_registry`, `get_clinic`, `reviews_by_author` → `submit_resolution`. **Code gates the answer**: id must come from a tool result in an earlier turn, ≥2 distinct evidence items, conf ≥ 0.8, specialty consistent, one submission, max 6 steps; else human queue |
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

**Changed during the timed window:**
1. **Stale cache.** The extraction cache was keyed on model+text, so a schema change silently returned old results. It's now keyed on the full rendered request.
2. **Normal recovery reported as complications.** The model listed swelling and bruising as complications; the schema now has a separate `expected_recovery` field.
3. **Cross-model review (Codex on Claude's code) → 7 fixes.** The worst: routing let *hallucinated* surgeon names publish, through a `startswith("surgeon ")` exemption meant only for roster warnings. Also: agent-check holes (lookup and submit in the same response, multiple submissions, blank evidence), a phone match overriding the specialty in the name, and price-parsing boundaries. Each one is pinned by a test (16 total).
4. **Second Codex pass → 3 more fixes:** (a) the roster warning is now a trust note, not an issue, so *every* issue blocks publishing and nothing is filtered by matching message text; (b) the validator re-parses each cited price span and rejects an amount that contradicts it ("350만원" stored as ₩350,000); (c) author history is keyed by (platform, handle), because the same handle on Naver Blog and Daum Café isn't the same person. 17 tests.

## Known limitations (deliberately not rushed in)
- **Agent evidence is free text.** The check confirms *where* the clinic ID came from, but doesn't verify each evidence statement against the tool results. Next step: structured evidence (`{tool, field, value}`) that code re-checks, plus a random 10% of agent-resolved reviews sent to human audit.
- **Procedures and complications aren't tied to source text** the way prices and surgeon names are. A model that invents a procedure from the glossary would pass. Next step: a source span for each, checked like prices.
- At scale: MinHash/LSH instead of O(n²) dedup; a real registry from the government data on clinics registered for foreign patients.
