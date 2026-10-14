"""
Stage 5 of the Jchem pipeline: MatKG entity matching + weighted Jaccard.

Replaces jaccard_calculation_lithium_salt_pubchem_paralellized.py, which
re-scanned all 5.4M MatKG triples with a substring regex per formula.
Here the 39k unique entities of the allowed subgraph are matched once
against all component synonyms, then neighbor profiles come from a
prebuilt adjacency map.

Matching (both sides normalized with NFKC/casefold; MatKG inserts spaces
and title-case freely, e.g. "Lithium Bis(Trifluoromethanesulfonyl) Imide"):
- exact:       entity norm_key == synonym norm_key (whitespace/hyphens
               stripped), allowed for every kept synonym
- containment: the synonym's token sequence appears as a contiguous run
               of the entity's tokens ("(B) Acetonitrile" contains
               "acetonitrile"). This is the word-boundary rule: no match
               inside a longer token. Only for match_tier == containment.
Match-side ambiguity guard: a synonym that matches more than
MAX_ENTITIES_PER_SYNONYM entities is excluded from scoring and reported.

Scoring: neighbor profile of a component = union of the MatKG neighbors
of its matched entities, weighted by summed triple Count (both
directions); entities matched to the same component are excluded from
its own neighbor set. Weighted Jaccard vs the reference = sum of
per-neighbor min weight / sum of max weight.

Reads  SUBRELOBJ.csv (repo root)
       output/final_electrolyte_components.csv
       output/synonyms_clean.csv
Writes output/jchem_results.csv           (per component, ranked)
       output/jchem_matched_entities.csv  (match provenance)
       output/jchem_report.txt
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

from textnorm import cased_squash, tokens_of, squash_bounds

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT_DIR = os.path.join(BASE, "output")
SUBRELOBJ = os.path.join(ROOT, "SUBRELOBJ.csv")
COMPONENTS_CSV = os.path.join(OUT_DIR, "final_electrolyte_components.csv")
CLEAN_CSV = os.path.join(OUT_DIR, "synonyms_clean.csv")

RESULTS_CSV = os.path.join(OUT_DIR, "jchem_results.csv")
MATCHES_CSV = os.path.join(OUT_DIR, "jchem_matched_entities.csv")
REPORT_TXT = os.path.join(OUT_DIR, "jchem_report.txt")

ALLOWED_RELS = {"CHM-CHM", "PRO-CHM", "CHM-PRO", "APL-CHM", "CHM-APL",
                "DSC-CHM", "CHM-DSC"}
MAX_ENTITIES_PER_SYNONYM = 50
# at or below this squash length a match must also agree on case:
# "SN" is succinonitrile, "Sn" is tin; "FEC" is not "FeC"
CASED_MATCH_MAX_LEN = 4
DEFAULT_REF_INCHIKEY = "QSZMZKBZAYQGRS-UHFFFAOYSA-N"  # LiTFSI


def load_matkg():
    df = pd.read_csv(SUBRELOBJ, dtype={"Subject": str, "Object": str,
                                       "Rel": str, "Count": int})
    df = df[df["Rel"].isin(ALLOWED_RELS)].reset_index(drop=True)
    print(f"{len(df)} allowed-relation triples")
    ents = pd.unique(pd.concat([df["Subject"], df["Object"]]).astype(str))
    eid = {e: i for i, e in enumerate(ents)}
    print(f"{len(ents)} unique entities")

    adj = defaultdict(lambda: defaultdict(int))
    subs = df["Subject"].map(eid).to_numpy()
    objs = df["Object"].map(eid).to_numpy()
    cnts = df["Count"].to_numpy()
    for s, o, c in zip(subs, objs, cnts):
        adj[s][o] += c
        adj[o][s] += c
    return ents, eid, adj


def match_key(row):
    """A synonym's match identity: squash + tier, plus the case-preserving
    form when the string is short enough for case to carry meaning.
    Short cased matches are necessarily full-entity (a <=4-char synonym
    only ever matches whole short entities under the tier rules), so the
    cased requirement compares against the entity's full cased squash."""
    squash = "".join(row["tokens"].split())
    cased = row["cased"] if len(squash) <= CASED_MATCH_MAX_LEN else ""
    return (squash, row["match_tier"], cased)


def match_entities(ents):
    """Match synonyms to entities on squashed token strings.

    A synonym matches an entity iff its squash occurs in the entity's
    squash starting at a token start and ending at a token end. This is
    word-boundary containment that survives MatKG's token fusion
    ("Trifluoromethylsulfonyl" vs "trifluoromethyl)sulfonyl").
    exact_only synonyms must span the whole entity squash.

    Every candidate substring of every entity (token-start to token-end)
    is enumerated and looked up in the synonym dict -- entities are short
    (median 2-3 tokens), synonyms are many, so this direction is cheap.
    """
    syn_rows = list(csv.DictReader(open(CLEAN_CSV)))
    print(f"{len(syn_rows)} kept synonyms")

    syn_map = defaultdict(set)  # squash -> {(squash, tier, cased-req)}
    for r in syn_rows:
        squash = "".join(r["tokens"].split())
        if squash:
            syn_map[squash].add(match_key(r))

    hits = defaultdict(set)  # match key -> set(eid)
    for i, e in enumerate(ents):
        squash, starts, ends = squash_bounds(tokens_of(e))
        if not squash:
            continue
        n = len(squash)
        e_cased = cased_squash(e)
        for a in starts:
            for b in ends:
                if b <= a:
                    continue
                keys = syn_map.get(squash[a:b])
                if not keys:
                    continue
                for key in keys:
                    if key[1] == "exact_only" and (a != 0 or b != n):
                        continue
                    if key[2] and key[2] != e_cased:
                        continue
                    hits[key].add(i)
    print(f"{len(hits)} synonyms matched at least one entity")
    return syn_rows, dict(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-inchikey", default=DEFAULT_REF_INCHIKEY)
    args = ap.parse_args()

    components = {r["component_id"]: r
                  for r in csv.DictReader(open(COMPONENTS_CSV))}
    ref_comp = next((cid for cid, r in components.items()
                     if r["inchikey"] == args.ref_inchikey), None)
    if ref_comp is None:
        sys.exit(f"reference InChIKey {args.ref_inchikey} not in components")
    print(f"reference component: {ref_comp} "
          f"({components[ref_comp]['title']})")

    ents, _eid, adj = load_matkg()
    syn_rows, hits = match_entities(ents)

    # ---- match-side ambiguity guard ----
    promiscuous = {}
    for key, eids in hits.items():
        if len(eids) > MAX_ENTITIES_PER_SYNONYM:
            promiscuous[key] = len(eids)

    comp_entities = defaultdict(set)
    provenance = []
    for r in syn_rows:
        key = match_key(r)
        if key not in hits or key in promiscuous:
            continue
        for i in hits[key]:
            if i not in comp_entities[r["component_id"]]:
                provenance.append((r["component_id"], r["synonym"],
                                   ents[i], r["match_tier"]))
            comp_entities[r["component_id"]].add(i)
    print(f"{len(comp_entities)} components matched >=1 MatKG entity")

    # ---- neighbor profiles ----
    def profile(comp):
        own = comp_entities[comp]
        prof = defaultdict(int)
        for i in own:
            for j, w in adj[i].items():
                if j not in own:
                    prof[j] += w
        return prof

    if ref_comp not in comp_entities:
        sys.exit("reference component matched no MatKG entities")
    ref_prof = profile(ref_comp)
    ref_set = set(ref_prof)
    print(f"reference profile: {len(ref_prof)} neighbors")

    results = []
    for comp, eids in comp_entities.items():
        prof = profile(comp)
        pset = set(prof)
        inter = pset & ref_set
        union = pset | ref_set
        inter_w = sum(min(prof.get(j, 0), ref_prof.get(j, 0)) for j in inter)
        union_w = sum(max(prof.get(j, 0), ref_prof.get(j, 0)) for j in union)
        row = components[comp]
        results.append({
            "component_id": comp, "title": row["title"],
            "category": row["category"], "family": row["family"],
            "mf": row["mf"], "inchikey": row["inchikey"], "cid": row["cid"],
            "flags": row["flags"],
            "n_entities_matched": len(eids),
            "matched_entities": "; ".join(sorted(ents[i] for i in eids)[:8]),
            "n_neighbors": len(prof),
            "intersection_count": len(inter),
            "union_count": len(union),
            "jaccard_unweighted": (len(inter) / len(union)) if union else 0.0,
            "jaccard_weighted": (inter_w / union_w) if union_w else 0.0,
        })
    results.sort(key=lambda r: -r["jaccard_weighted"])

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    with open(MATCHES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component_id", "synonym", "entity", "match_tier"])
        w.writerows(provenance)

    lines = []
    lines.append(f"Reference: {ref_comp} ({components[ref_comp]['title']}), "
                 f"{len(ref_prof)} neighbors")
    lines.append(f"Components scored: {len(results)} of {len(components)} "
                 f"({len(components) - len(results)} matched no MatKG entity)")
    lines.append(f"Entity matches recorded: {len(provenance)}")
    lines.append(f"\nSynonyms excluded as promiscuous at match time "
                 f"(> {MAX_ENTITIES_PER_SYNONYM} entities):")
    for (squash, tier, _cased), n in sorted(promiscuous.items(),
                                            key=lambda kv: -kv[1]):
        lines.append(f"  {squash} [{tier}]: {n} entities")

    # abbreviation ambiguity audit: 2-3 char matches ride on strings like
    # "EC"/"MA"/"PC" whose MatKG sense may not be this compound at all --
    # neighbor counts far above the full-name entity are the tell
    lines.append("\nShort-abbreviation matches (<= 3 chars) -- audit for "
                 "sense collisions (e.g. MA = methylammonium in "
                 "perovskite papers):")
    for comp, syn, ent, _tier in provenance:
        squash = "".join(tokens_of(syn))
        if len(squash) <= 3:
            eid = next(i for i in comp_entities[comp]
                       if ents[i] == ent)
            lines.append(f"  {comp} ({components[comp]['title'][:40]}): "
                         f"{syn!r} -> {ent!r} [{len(adj[eid])} neighbors]")
    fam = Counter(r["family"] for r in results)
    lines.append("\nScored components by family:")
    for f_, n in fam.most_common(20):
        lines.append(f"  {f_}: {n}")
    lines.append(f"\nTop 50 by weighted Jaccard vs "
                 f"{components[ref_comp]['title']}:")
    lines.append(f"{'rank':<5}{'component':<11}{'w-Jaccard':<11}"
                 f"{'u-Jaccard':<11}{'ents':<6}{'nbrs':<7}title")
    for k, r in enumerate(results[:50], 1):
        lines.append(f"{k:<5}{r['component_id']:<11}"
                     f"{r['jaccard_weighted']:<11.4f}"
                     f"{r['jaccard_unweighted']:<11.4f}"
                     f"{r['n_entities_matched']:<6}{r['n_neighbors']:<7}"
                     f"{r['title'][:60]}")
    report = "\n".join(lines)
    with open(REPORT_TXT, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nWrote {RESULTS_CSV}, {MATCHES_CSV}, {REPORT_TXT}")


if __name__ == "__main__":
    main()
