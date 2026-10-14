"""
Human-readable substitution commentary for the consensus candidates,
LLM-generated from the battery-KG evidence layer with per-claim DOI
citations.

Successor to battery_kg/llm_querying_of_candidates.py (which prompted
from salt-KG triples without provenance): the knowledge profiles here are
built from output/candidate_claims.csv, so every claim handed to the
model carries its supporting papers, and the prompt requires the model to
carry those citations into the commentary. The model is forbidden from
adding anything outside the knowledge base: an earlier revision let it
contribute untagged-then-tagged [general knowledge] statements, but those
turned out to carry the headline claims (cost, PFAS status) on no
evidence at all, so they are now excluded rather than labelled.

Claims are ranked by (n_papers, n_mentions); single-paper claims are
included but the model is told to treat them as weak evidence.

Needs RITS_API_KEY and RITS_BASE_URL in the environment.

Reads  output/candidate_claims.csv
Writes output/candidate_commentary_report.txt
       output/candidate_commentary.json
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime

from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
CLAIMS_CSV = os.path.join(OUT_DIR, "candidate_claims.csv")
REPORT_TXT = os.path.join(OUT_DIR, "candidate_commentary_report.txt")
RAW_JSON = os.path.join(OUT_DIR, "candidate_commentary.json")

RITS_BASE_URL = (os.environ.get("RITS_BASE_URL", "").rstrip("/")
                 + "/mistral-small-3-1-24b-2503/v1")
RITS_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

REFERENCE = "LiTFSI"
CANDIDATES = ["LiNO3", "LiClO4", "Li2SO4", "LiBOB", "LiPF6", "LiOTf"]

BUCKETS = [
    ("Effects on cell / property behaviour", {"AFFECTS PROPERTY OF"}),
    ("Measured properties, windows, temperatures",
     {"HAS PROPERTY", "HAS VALUE", "STABLE IN WINDOW",
      "OPERATING TEMP RANGE"}),
    ("Applications", {"USED IN", "USED WITH", "COMPATIBLE WITH"}),
    ("Solvents, co-salts and formulation context",
     {"DISSOLVED IN", "CO OCCURS WITH", "CONTAINS", "HAS COMPOSITION"}),
]
MAX_PER_BUCKET = 12
MAX_DOIS_PER_CLAIM = 3


def load_claims():
    by_salt = defaultdict(list)
    for r in csv.DictReader(open(CLAIMS_CSV)):
        r["n_papers"] = int(r["n_papers"])
        r["n_mentions"] = int(r["n_mentions"])
        by_salt[r["salt"]].append(r)
    return by_salt


def claim_line(r, salt):
    if r["direction"] == "as_subject":
        stmt = f"{salt} {r['relation']} {r['neighbor']}"
    else:
        stmt = f"{r['neighbor']} {r['relation']} {salt}"
    dois = r["dois"].split(";")
    cite = "; ".join(dois[:MAX_DOIS_PER_CLAIM])
    if len(dois) > MAX_DOIS_PER_CLAIM:
        cite += f"; +{len(dois) - MAX_DOIS_PER_CLAIM} more"
    return (f"  - {stmt} ({r['n_papers']} papers, "
            f"{r['n_mentions']} mentions) [{cite}]")


def format_profile(salt, claims):
    papers = set()
    for r in claims:
        papers.update(r["dois"].split(";"))
    lines = [f"**{salt} evidence profile** "
             f"({sum(r['n_mentions'] for r in claims)} mentions across "
             f"{len(papers)} papers; claims ranked by number of "
             f"supporting papers):"]
    for bucket_name, rels in BUCKETS:
        rows = sorted((r for r in claims if r["relation"] in rels),
                      key=lambda r: (-r["n_papers"], -r["n_mentions"]))
        if not rows:
            continue
        lines.append(f"\n{bucket_name}:")
        lines.extend(claim_line(r, salt) for r in rows[:MAX_PER_BUCKET])
    return "\n".join(lines)


def build_prompt(candidate, cand_profile, ref_profile):
    return f"""\
You are an expert in battery electrochemistry and electrolyte materials.

**TASK**: Compare {candidate} to {REFERENCE} (the reference standard) as
a lithium-battery electrolyte salt, and assess the potential of
{candidate} as a replacement for {REFERENCE}. {REFERENCE} is a PFAS; the
motivation is finding substitutes for fluorinated salts.

**KNOWLEDGE BASE** (claims extracted from a literature knowledge graph;
each claim lists its supporting paper DOIs):

{ref_profile}

{cand_profile}

**CITATION RULES (mandatory)**:
- Every statement you derive from the knowledge base above MUST end with
  the bracketed DOI(s) of the claim(s) it rests on, copied verbatim,
  e.g. "improves coulombic efficiency [10.1016/j.cej.2022.135283]".
- Weigh evidence by paper count: claims supported by one paper are weak
  evidence -- say so when you use them.
- Write ONLY what the knowledge base supports. Do not add statements
  from your own general electrochemistry knowledge, however well
  established -- no claims about cost, toxicity, PFAS status, safety or
  supply chain unless a claim above supports them. If the evidence does
  not settle a question, write that it is unsettled and move on.
- Do not invent DOIs. Do not attach a DOI to a statement the cited claim
  does not support.
- Before writing that evidence is absent, check the relation names in the
  profile above and refer to them by those names. Absence of a phrase is
  not absence of evidence: solvent evidence appears under DISSOLVED IN,
  application evidence under USED IN, and so on.

**ANALYSIS REQUIRED** (clear text with bullet points, no JSON):

**1. SIMILARITY ASSESSMENT**
- How similar are {candidate} and {REFERENCE} in usage contexts, based
  on the profiles?

**2. KEY DIFFERENCES**
- 3-5 main differences.

**3. ADVANTAGES OF {candidate} OVER {REFERENCE}**
- Electrochemical, performance and safety only, and only where the
  evidence supports it. The knowledge base has no cost or manufacturing
  relations, so do not discuss cost.

**4. LIMITATIONS OF {candidate} COMPARED TO {REFERENCE}**
- Electrochemical, performance and safety only, and only where the
  evidence supports it. The knowledge base has no cost or manufacturing
  relations, so do not discuss cost.

**5. APPLICATION SUITABILITY**
- Best-fit applications; challenging applications; reasoning.

**6. REPLACEMENT CONSIDERATIONS**
- What the evidence implies about required formulation changes and
  compatibility. Do NOT give a numeric feasibility score or an overall
  drop-in verdict -- neither is derivable from this evidence base.

**7. RESEARCH GAPS**
- What the knowledge base does NOT cover for {candidate} (relations or
  contexts present for {REFERENCE} but missing for {candidate}).

**8. OVERALL ASSESSMENT**
- Is {candidate} a viable {REFERENCE} alternative, and under what
  conditions?
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", "-c", nargs="+", default=CANDIDATES,
                    help=f"salts to comment on (default: {' '.join(CANDIDATES)})")
    ap.add_argument("--report", default=REPORT_TXT, help="output .txt path")
    ap.add_argument("--json", dest="json_path", default=RAW_JSON,
                    help="output .json path")
    args = ap.parse_args()

    api_key = os.environ.get("RITS_API_KEY")
    if not api_key:
        raise SystemExit("RITS_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=RITS_BASE_URL,
                    default_headers={"RITS_API_KEY": api_key})

    by_salt = load_claims()
    if REFERENCE not in by_salt:
        raise SystemExit(f"{REFERENCE} missing from {CLAIMS_CSV} -- rerun "
                         "appkg_candidate_profiles.py")
    ref_profile = format_profile(REFERENCE, by_salt[REFERENCE])

    missing = [c for c in args.candidates if c not in by_salt]
    if missing:
        raise SystemExit(f"not in {CLAIMS_CSV}: {', '.join(missing)}")

    results = []
    for cand in args.candidates:
        prompt = build_prompt(cand, format_profile(cand, by_salt[cand]),
                              ref_profile)
        print(f"{cand}: querying {RITS_MODEL_ID} "
              f"({len(prompt)} prompt chars)")
        t0 = time.time()
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=RITS_MODEL_ID, temperature=0.1, max_tokens=4096)
        text = completion.choices[0].message.content.strip()
        dt = time.time() - t0
        print(f"{cand}: {len(text)} chars in {dt:.0f}s")
        results.append({
            "candidate": cand, "reference": REFERENCE,
            "model": RITS_MODEL_ID, "prompt": prompt, "commentary": text,
            "seconds": round(dt, 1),
            "timestamp": datetime.now().isoformat()})
        with open(args.json_path, "w") as f:
            json.dump(results, f, indent=2)

    with open(args.report, "w") as f:
        f.write(f"Candidate commentary vs {REFERENCE} -- generated "
                f"{datetime.now():%Y-%m-%d %H:%M} by {RITS_MODEL_ID}\n"
                f"from battery-KG claims with per-claim DOI attribution "
                f"(candidate_claims.csv).\n"
                f"Every statement is corpus-derived; the model was "
                f"forbidden from adding outside knowledge, and no "
                f"feasibility score is requested.\n")
        for r in results:
            f.write(f"\n{'=' * 72}\n{r['candidate']}\n{'=' * 72}\n")
            f.write(r["commentary"] + "\n")
    print(f"\nWrote {args.report}, {args.json_path}")


if __name__ == "__main__":
    main()
