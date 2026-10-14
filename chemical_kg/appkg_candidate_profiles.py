"""
Battery-KG candidate profiles with per-claim DOI attribution.

For the consensus substitution candidates (LiNO3, LiClO4, Li2SO4, LiBOB,
plus the fluorinated references LiPF6 and LiOTf), collect every battery-KG
triple whose subject or object is an exact-tier entity of the salt, and
aggregate them into claims: (direction, relation, normalized neighbor)
with the list of supporting papers.

The battery KG's doi column is a paragraph id, not a paper id: every value
ends in _p<i>_<j> (31,640 paragraph ids -> 2,088 distinct papers). Claims
are attributed to papers after stripping that suffix; a claim backed by
several paragraphs of one paper is one paper's worth of evidence.

Mapping policy is inherited from appkg_battery_audit: exact tier only,
unambiguous matches only; composite formulation strings ("PEO/LiTFSI")
are not merged into profiles.

Reads  battery_ontology_triples_with_doi_clean_20250819_214710.csv
       output/synonyms_clean.csv, output/final_electrolyte_components.csv
Writes output/candidate_claims.csv      (one row per claim, all DOIs)
       output/candidate_evidence.csv    (one row per triple, with paragraph)
       output/candidate_profiles_report.txt
"""

import csv
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from appkg_battery_audit import BATTERY_KG, load_synonyms, map_entities
from textnorm import norm_key

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
CLAIMS_CSV = os.path.join(OUT_DIR, "candidate_claims.csv")
EVIDENCE_CSV = os.path.join(OUT_DIR, "candidate_evidence.csv")
REPORT_TXT = os.path.join(OUT_DIR, "candidate_profiles_report.txt")

CANDIDATES = {
    # reference salt: profiled the same way so downstream comparisons
    # (llm_candidate_commentary.py) cite the same evidence layer
    "LiTFSI": "QSZMZKBZAYQGRS-UHFFFAOYSA-N",
    # consensus (both top-10s), non-fluorinated first
    "LiNO3": "IIPYXGDZVMZOAP-UHFFFAOYSA-N",
    "LiClO4": "MHCFAGZWMAWTNR-UHFFFAOYSA-M",
    "Li2SO4": "INHCSSUBVCNVSK-UHFFFAOYSA-L",
    "LiBOB": "NVQAYVUCVASGDK-UHFFFAOYSA-N",
    # fluorinated consensus references
    "LiPF6": "AXPLOJNSKRXQPA-UHFFFAOYSA-N",
    "LiOTf": "MCVFFRWZNYZUIJ-UHFFFAOYSA-M",
}

PARAGRAPH_SUFFIX = re.compile(r"_p\d+_\d+$")
DOI_PREFIX = "https://doi.org/"

REPORT_MAX_GROUPS = 12   # (direction, relation) groups per candidate
REPORT_MAX_CLAIMS = 8    # claims per group
REPORT_MAX_DOIS = 3      # example papers per claim line


def paper_of(doi_paragraph_id):
    return PARAGRAPH_SUFFIX.sub("", doi_paragraph_id)


def short_doi(paper):
    return paper[len(DOI_PREFIX):] if paper.startswith(DOI_PREFIX) else paper


def main():
    df = pd.read_csv(
        BATTERY_KG, dtype=str,
        usecols=["subject", "relation", "object", "doi", "paragraph"],
    ).dropna(subset=["subject", "relation", "object", "doi"])
    print(f"{len(df)} triples, "
          f"{df['doi'].map(paper_of).nunique()} distinct papers")

    ents = pd.unique(pd.concat([df["subject"], df["object"]]))
    syn, titles = load_synonyms()
    mapped = map_entities(ents, syn)
    ent_ik = {e: next(iter(iks)) for e, (t, iks) in mapped.items()
              if t == "exact" and len(iks) == 1}
    target_iks = set(CANDIDATES.values())
    salt_of_ik = {ik: name for name, ik in CANDIDATES.items()}
    for name, ik in CANDIDATES.items():
        forms = sorted(e for e, k in ent_ik.items() if k == ik)
        if not forms:
            raise SystemExit(f"{name}: no exact-tier entity in battery KG")
        print(f"{name}: {len(forms)} surface forms, e.g. {forms[:4]}")

    # ---- triple-level evidence ----
    subj_ik = df["subject"].map(ent_ik)
    obj_ik = df["object"].map(ent_ik)
    evidence = []  # dict rows
    for direction, side_ik, nbr_col in (
            ("as_subject", subj_ik, "object"),
            ("as_object", obj_ik, "subject")):
        hit = df[side_ik.isin(target_iks)].copy()
        hit["salt"] = side_ik[hit.index].map(salt_of_ik)
        for row in hit.itertuples(index=False):
            nbr = getattr(row, nbr_col)
            if ent_ik.get(nbr) == CANDIDATES[row.salt]:
                continue  # self-loop between two surface forms of the salt
            evidence.append({
                "salt": row.salt,
                "direction": direction,
                "relation": row.relation,
                "neighbor": nbr,
                "neighbor_key": norm_key(nbr),
                "subject": row.subject,
                "object": row.object,
                "doi": paper_of(row.doi),
                "doi_paragraph_id": row.doi,
                "paragraph": row.paragraph,
            })
    print(f"{len(evidence)} evidence triples across "
          f"{len({r['salt'] for r in evidence})} candidates")

    with open(EVIDENCE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(evidence[0].keys()))
        w.writeheader()
        w.writerows(evidence)

    # ---- claims: (salt, direction, relation, neighbor_key) ----
    claims = {}
    for r in evidence:
        key = (r["salt"], r["direction"], r["relation"], r["neighbor_key"])
        c = claims.setdefault(key, {
            "surface": Counter(), "papers": set(), "n_mentions": 0})
        c["surface"][r["neighbor"]] += 1
        c["papers"].add(r["doi"])
        c["n_mentions"] += 1

    claim_rows = []
    for (salt, direction, relation, nkey), c in claims.items():
        claim_rows.append({
            "salt": salt,
            "inchikey": CANDIDATES[salt],
            "direction": direction,
            "relation": relation,
            "neighbor": c["surface"].most_common(1)[0][0],
            "neighbor_key": nkey,
            "n_mentions": c["n_mentions"],
            "n_papers": len(c["papers"]),
            "dois": ";".join(sorted(short_doi(p) for p in c["papers"])),
        })
    claim_rows.sort(key=lambda r: (r["salt"], -r["n_papers"],
                                   -r["n_mentions"], r["relation"]))
    with open(CLAIMS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(claim_rows[0].keys()))
        w.writeheader()
        w.writerows(claim_rows)

    # ---- report ----
    lines = [f"Candidate profiles from the battery KG "
             f"({len(df)} triples, {df['doi'].map(paper_of).nunique()} "
             f"papers; exact-tier entities only)",
             "Claim = (direction, relation, normalized neighbor); "
             "papers counted after stripping the _pN_N paragraph suffix.",
             "Corroborated = claim supported by >= 2 distinct papers."]
    for salt in CANDIDATES:
        rows = [r for r in claim_rows if r["salt"] == salt]
        ev = [r for r in evidence if r["salt"] == salt]
        papers = {r["doi"] for r in ev}
        corro = [r for r in rows if r["n_papers"] >= 2]
        lines.append(f"\n{'=' * 72}")
        lines.append(f"{salt}  ({CANDIDATES[salt]})")
        lines.append(f"  {len(ev)} mentions | {len(rows)} claims | "
                     f"{len(papers)} papers | {len(corro)} corroborated "
                     f"claims")

        groups = defaultdict(list)
        for r in rows:
            groups[(r["direction"], r["relation"])].append(r)
        ordered = sorted(
            groups.items(),
            key=lambda kv: -sum(r["n_mentions"] for r in kv[1]))
        for (direction, relation), rs in ordered[:REPORT_MAX_GROUPS]:
            arrow = ("salt -> x" if direction == "as_subject"
                     else "x -> salt")
            n_ment = sum(r["n_mentions"] for r in rs)
            lines.append(f"\n  [{arrow}] {relation}  "
                         f"({n_ment} mentions, {len(rs)} claims)")
            shown = sorted(rs, key=lambda r: (-r["n_papers"],
                                              -r["n_mentions"]))
            for r in shown[:REPORT_MAX_CLAIMS]:
                dois = r["dois"].split(";")
                ex = ", ".join(dois[:REPORT_MAX_DOIS])
                more = (f" (+{len(dois) - REPORT_MAX_DOIS} papers)"
                        if len(dois) > REPORT_MAX_DOIS else "")
                lines.append(f"    {r['neighbor'][:58]:<60}"
                             f"{r['n_mentions']:>4}x "
                             f"{r['n_papers']:>3}p  {ex}{more}")
            if len(rs) > REPORT_MAX_CLAIMS:
                lines.append(f"    ... +{len(rs) - REPORT_MAX_CLAIMS} "
                             f"more claims (see candidate_claims.csv)")
        if len(ordered) > REPORT_MAX_GROUPS:
            lines.append(f"\n  ... +{len(ordered) - REPORT_MAX_GROUPS} more "
                         f"relation groups (see candidate_claims.csv)")

    report = "\n".join(lines)
    with open(REPORT_TXT, "w") as f:
        f.write(report + "\n")
    print(f"\nWrote {CLAIMS_CSV}, {EVIDENCE_CSV}, {REPORT_TXT}")


if __name__ == "__main__":
    main()
