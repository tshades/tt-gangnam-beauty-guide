"""python -m gbg [reviews.jsonl] [--offline] [--json out.json]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .clinic import Registry
from .pipeline import load_reviews, run, summarize

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("reviews", nargs="?", default=str(ROOT / "fixtures" / "reviews.jsonl"))
    ap.add_argument("--clinics", default=str(ROOT / "fixtures" / "clinics.json"))
    ap.add_argument("--offline", action="store_true", help="no LLM; deterministic glossary/regex extraction")
    ap.add_argument("--json", help="write full results (with trace) to this path")
    a = ap.parse_args()

    offline = a.offline or not os.environ.get("ANTHROPIC_API_KEY")
    if offline and not a.offline:
        print("(no ANTHROPIC_API_KEY -> running offline)\n")
    results = run(load_reviews(a.reviews), Registry(a.clinics), offline=offline)
    print(summarize(results))
    if a.json:
        Path(a.json).write_text(json.dumps([r.model_dump(mode="json") for r in results], ensure_ascii=False, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
