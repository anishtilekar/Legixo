"""Run eval/test_cases.json against the LIVE API and write eval/results.md.

Start the server first:
    python -m uvicorn app.main:app --port 8000

Then:
    python -m scripts.run_eval                # one pass
    python -m scripts.run_eval --repeat 3     # 3x per case, reports stability

Three assertions per case:
  1. status matches (answered / not_found)
  2. every expected source file appears in the citations
  3. every expected fact appears in the answer text (after normalisation)

Out-of-corpus cases additionally require ZERO citations — refusing while still
emitting citations would defeat the point.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

CASES_PATH = Path("eval/test_cases.json")
RESULTS_PATH = Path("eval/results.md")

_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
# The model formats large numbers inconsistently: "1,35,000", "1 35 000", "135000".
# Collapse separators *only* between digits, so "45 days" is untouched.
_DIGIT_SPACE = re.compile(r"(?<=\d)[\s ](?=\d)")
_SPACE_PCT = re.compile(r"\s+%")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Make substring matching robust to formatting the model varies freely.

    Handles: markdown bold, currency symbols, Indian digit grouping (1,35,000),
    '9 %' vs '9%', smart quotes/dashes, and whitespace runs.
    """
    t = text.lower()
    t = t.replace("**", "").replace("*", "")
    for ch in "₹$€£":
        t = t.replace(ch, "")
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("‑", "-")
    t = t.replace(" ", " ").replace(" ", " ")
    t = t.replace(" ", "").replace("​", "")
    # Hyphens become spaces on BOTH sides of the comparison: the model freely
    # writes "billing-head" where the source says "billing head".
    t = t.replace("-", " ")
    t = _DIGIT_COMMA.sub("", t)
    t = _DIGIT_SPACE.sub("", t)
    t = _SPACE_PCT.sub("%", t)
    t = _WS.sub(" ", t)
    return t.strip()


def check_case(case: dict, payload: dict) -> tuple[bool, list[str]]:
    """Returns (passed, list of failure reasons)."""
    failures: list[str] = []

    status = payload.get("status", "")
    if status != case["expected_status"]:
        failures.append(f"status: got {status!r}, expected {case['expected_status']!r}")

    cited_files = {c["source_path"].split("/")[-1] for c in payload.get("citations", [])}
    for want in case["expected_sources"]:
        if want not in cited_files:
            failures.append(f"missing citation: {want} (cited: {sorted(cited_files) or 'none'})")

    if case["expected_status"] == "not_found" and payload.get("citations"):
        failures.append(f"refused but still returned {len(payload['citations'])} citation(s)")

    answer = normalise(payload.get("answer", ""))
    for item in case["expected_contains"]:
        variants = [item] if isinstance(item, str) else item
        if not any(normalise(v) in answer for v in variants):
            failures.append(f"answer missing any of: {variants}")

    return (not failures), failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case (flakiness check)")
    args = parser.parse_args()

    try:
        health = httpx.get(f"{args.url}/healthz", timeout=30).json()
    except Exception as exc:
        print(f"ERROR: API not reachable at {args.url} — start the server first.\n  {exc}")
        return 2
    if not health.get("vector_count"):
        print(f"ERROR: index '{health.get('index')}' has 0 vectors — run `python -m scripts.ingest` first.")
        return 2

    doc = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = doc["cases"]

    results = []
    per_case_pass = defaultdict(int)

    for case in cases:
        runs = []
        for _ in range(args.repeat):
            resp = httpx.post(
                f"{args.url}/ask",
                json={"question": case["question"]},
                timeout=180,
            )
            payload = resp.json()
            ok, failures = check_case(case, payload)
            runs.append({"ok": ok, "failures": failures, "payload": payload})
            if ok:
                per_case_pass[case["id"]] += 1

        # Show a FAILING run's detail when one exists — otherwise an intermittent
        # failure records a passing transcript and is undiagnosable afterwards.
        detail = next((r for r in runs if not r["ok"]), runs[-1])
        results.append(
            {
                "case": case,
                "ok": all(r["ok"] for r in runs),
                "passes": per_case_pass[case["id"]],
                "runs": args.repeat,
                "failures": detail["failures"],
                "payload": detail["payload"],
            }
        )
        mark = "PASS" if results[-1]["ok"] else "FAIL"
        stability = "" if args.repeat == 1 else f" [{per_case_pass[case['id']]}/{args.repeat}]"
        print(f"  {mark}{stability}  {case['id']:>2}. {case['question'][:66]}")
        for f in detail["failures"]:
            print(f"          - {f}")

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    ooc = [r for r in results if r["case"]["expected_status"] == "not_found"]
    ooc_passed = sum(1 for r in ooc if r["ok"])

    print(f"\n  {passed}/{total} passed   (out-of-corpus refusals: {ooc_passed}/{len(ooc)})")

    write_results_md(results, health, args)
    print(f"  wrote {RESULTS_PATH}")
    return 0 if passed == total else 1


def write_results_md(results: list[dict], health: dict, args) -> None:
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    ooc = [r for r in results if r["case"]["expected_status"] == "not_found"]
    ooc_passed = sum(1 for r in ooc if r["ok"])
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Eval results",
        "",
        f"Generated by `python -m scripts.run_eval` on {ts}.",
        "",
        f"- **Score: {passed}/{total} passed**",
        f"- Out-of-corpus refusals: **{ooc_passed}/{len(ooc)}**",
        f"- Runs per case: {args.repeat}",
        f"- Model: `{health.get('llm_model')}` · Embeddings: `{health.get('embed_model')}`",
        f"- Index: `{health.get('index')}` · vectors: {health.get('vector_count')}",
        "",
        "A case counts as PASS only if **every** repeat passed; the `(n/m)` column shows"
        " how many did. Where a case failed intermittently, the transcript below is from"
        " a **failing** run, not a passing one.",
        "",
        "| # | Question | Expected | Got | Citations | Result |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        c, p = r["case"], r["payload"]
        cites = ", ".join(sorted({x["source_path"].split("/")[-1][:2] for x in p.get("citations", [])})) or "—"
        stability = "" if r["runs"] == 1 else f" ({r['passes']}/{r['runs']})"
        mark = "PASS" if r["ok"] else "**FAIL**"
        q = c["question"].replace("|", "\\|")
        lines.append(
            f"| {c['id']} | {q} | {c['expected_status']} | {p.get('status')} | {cites} | {mark}{stability} |"
        )

    lines += ["", "## Notes per case", ""]
    for r in results:
        c, p = r["case"], r["payload"]
        lines.append(f"**{c['id']}. {c['question']}**")
        lines.append("")
        lines.append(f"- Expected: `{c['expected_status']}`, sources: {c['expected_sources'] or 'none'}")
        lines.append(f"- Got: `{p.get('status')}` after {p.get('attempts')} attempt(s)")
        answer = (p.get("answer") or "").replace("\n", " ")
        lines.append(f"- Answer: {answer[:300]}")
        if p.get("citations"):
            for x in p["citations"]:
                lines.append(f"    - {x['marker']} `{x['source_path']}` (score {x['score']})")
        if r["failures"]:
            lines.append(f"- **Failures:** {'; '.join(r['failures'])}")
        lines.append(f"- Note: {c['notes']}")
        lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
