"""
Application-KG (battery KG) entity mapping investigation + LiTFSI ranking.

The battery KG (battery_ontology_triples_with_doi_clean_*.csv, 316,846
triples, 131 relations) is a general graph, not ego-centric: salts,
solvents, electrodes and formulations all appear as subjects AND objects,
and the entity space is dominated by composite formulation phrases
("PEO/LiTFSI-4 wt%SiO2", "CS-LiTFSI-PEO"), with ~1,000 entity strings
containing TFSI alone.

Mapping policy:
- exact tier: an entity IS the salt (full-squash match against the
  Stage-4 synonym index, cased rule for short strings) -> used for
  identity and for ego-profile scoring
- composite tier: an entity CONTAINS the salt at token boundaries
  ("PEO/LiTFSI electrolytes") -> counted and reported per salt, but NOT
  merged into profiles by default; formulation properties belong to the
  formulation
- neighbor keys are squash-normalized (case/space/hyphen variants of
  free-text objects merge) - deliberate canonicalization instead of
  pairwise fuzzy scoring, so the Jaccard stays a real set overlap

Reads  battery_ontology_triples_with_doi_clean_20250819_214710.csv
       output/synonyms_clean.csv, output/final_electrolyte_components.csv
Writes output/appkg_battery_entity_mapping.csv
       output/appkg_battery_ranking.csv
       output/appkg_battery_report.txt
"""

import csv
import os
from collections import Counter, defaultdict

import pandas as pd

from textnorm import cased_squash, norm_key, tokens_of, squash_bounds

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT_DIR = os.path.join(BASE, "output")
BATTERY_KG = os.path.join(
    ROOT, "battery_ontology_triples_with_doi_clean_20250819_214710.csv")
CLEAN_CSV = os.path.join(OUT_DIR, "synonyms_clean.csv")
COMPONENTS_CSV = os.path.join(OUT_DIR, "final_electrolyte_components.csv")

MAPPING_CSV = os.path.join(OUT_DIR, "appkg_battery_entity_mapping.csv")
RANKING_CSV = os.path.join(OUT_DIR, "appkg_battery_ranking.csv")
REPORT_TXT = os.path.join(OUT_DIR, "appkg_battery_report.txt")

REF_INCHIKEY = "QSZMZKBZAYQGRS-UHFFFAOYSA-N"  # LiTFSI
CASED_MATCH_MAX_LEN = 4
MIN_PROFILE_MENTIONS = 5  # salts with fewer exact-entity triples are
                          # listed but flagged as thin


def load_synonyms():
    comps = {r["component_id"]: r for r in csv.DictReader(open(COMPONENTS_CSV))}
    syn = defaultdict(set)   # squash -> {(inchikey, tier, cased-req)}
    for r in csv.DictReader(open(CLEAN_CSV)):
        squash = "".join(r["tokens"].split())
        if not squash:
            continue
        ik = comps[r["component_id"]]["inchikey"]
        cased = r["cased"] if len(squash) <= CASED_MATCH_MAX_LEN else ""
        syn[squash].add((ik, r["match_tier"], cased))
    titles = {}
    for r in comps.values():
        titles.setdefault(r["inchikey"], r["title"])
    return syn, titles


def map_entities(ents, syn):
    """entity -> (tier, {inchikeys}); tier is exact > composite > none."""
    result = {}
    for e in ents:
        toks = tokens_of(e)
        squash, starts, ends = squash_bounds(toks)
        if not squash:
            continue
        n = len(squash)
        e_cased = cased_squash(e)
        exact, composite = set(), set()
        for a in starts:
            for b in ends:
                if b <= a:
                    continue
                for ik, tier, cased in syn.get(squash[a:b], ()):
                    full = (a == 0 and b == n)
                    if cased and (not full or cased != e_cased):
                        continue
                    if full:
                        exact.add(ik)
                    elif tier == "containment":
                        composite.add(ik)
        if exact:
            result[e] = ("exact", exact)
        elif composite:
            result[e] = ("composite", composite)
    return result


def main():
    df = pd.read_csv(BATTERY_KG, usecols=["subject", "relation", "object"],
                     dtype=str).dropna()
    print(f"{len(df)} triples")
    ents = pd.unique(pd.concat([df["subject"], df["object"]]))
    print(f"{len(ents)} unique entities")

    syn, titles = load_synonyms()
    mapped = map_entities(ents, syn)
    n_exact = sum(1 for t, _ in mapped.values() if t == "exact")
    n_comp = sum(1 for t, _ in mapped.values() if t == "composite")
    print(f"mapped: {n_exact} exact, {n_comp} composite, "
          f"{len(ents) - len(mapped)} unmatched")

    with open(MAPPING_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity", "tier", "inchikeys", "titles"])
        for e, (tier, iks) in sorted(mapped.items()):
            w.writerow([e, tier, ";".join(sorted(iks)),
                        ";".join(titles.get(ik, "") for ik in sorted(iks))])

    # ---- ego profiles from exact-tier entities ----
    # identity requires an unambiguous exact match
    ent_ik = {e: next(iter(iks)) for e, (t, iks) in mapped.items()
              if t == "exact" and len(iks) == 1}
    comp_entities = defaultdict(set)
    for e, ik in ent_ik.items():
        comp_entities[ik].add(e)

    profiles = defaultdict(Counter)
    subj_ik = df["subject"].map(ent_ik)
    obj_ik = df["object"].map(ent_ik)
    for side, other in ((subj_ik, df["object"]), (obj_ik, df["subject"])):
        sub = pd.DataFrame({"ik": side, "nbr": other}).dropna(subset=["ik"])
        for ik, nbr in zip(sub["ik"], sub["nbr"]):
            # skip neighbors that are themselves this salt's entities
            if ent_ik.get(nbr) == ik:
                continue
            profiles[ik][norm_key(nbr)] += 1

    ref = profiles.get(REF_INCHIKEY)
    if not ref:
        raise SystemExit("LiTFSI has no exact-tier profile")
    ref_set = set(ref)
    print(f"LiTFSI profile: {sum(ref.values())} mentions, "
          f"{len(ref)} distinct neighbor keys")

    results = []
    for ik, prof in profiles.items():
        pset = set(prof)
        inter = pset & ref_set
        union = pset | ref_set
        inter_w = sum(min(prof[o], ref[o]) for o in inter)
        union_w = sum(max(prof.get(o, 0), ref.get(o, 0)) for o in union)
        n_composite = sum(1 for e, (t, iks) in mapped.items()
                          if t == "composite" and ik in iks)
        results.append({
            "inchikey": ik, "title": titles.get(ik, ""),
            "n_exact_entities": len(comp_entities[ik]),
            "exact_entities": "; ".join(sorted(comp_entities[ik])[:5]),
            "n_composite_entities": n_composite,
            "n_mentions": sum(prof.values()),
            "n_neighbors": len(prof),
            "intersection_count": len(inter),
            "jaccard_unweighted": len(inter) / len(union) if union else 0.0,
            "jaccard_weighted": inter_w / union_w if union_w else 0.0,
        })
    results.sort(key=lambda r: -r["jaccard_weighted"])

    with open(RANKING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ---- report ----
    lines = []
    lines.append(f"Battery KG: {len(df)} triples, {len(ents)} unique "
                 f"entities, {df['relation'].nunique()} relations")
    lines.append(f"Entity mapping: {n_exact} exact, {n_comp} composite "
                 f"(salt inside formulation string), "
                 f"{len(ents) - len(mapped)} unmatched")
    lines.append(f"Components with an exact-tier presence: {len(profiles)}")

    lines.append("\nLiTFSI surface forms merged by exact matching:")
    for e in sorted(comp_entities.get(REF_INCHIKEY, []))[:12]:
        lines.append(f"  {e!r}")
    n_ref_comp = sum(1 for e, (t, iks) in mapped.items()
                     if t == "composite" and REF_INCHIKEY in iks)
    lines.append(f"  (+{n_ref_comp} composite formulation entities NOT "
                 "merged, e.g.:)")
    for e, (t, iks) in list(mapped.items()):
        if t == "composite" and REF_INCHIKEY in iks:
            lines.append(f"    {e!r}")
            if sum(1 for l in lines if l.startswith("    ")) >= 5:
                break

    lines.append("\nMost frequent unmatched entities (top 25 by triple "
                 "count) - electrode materials and formulations are "
                 "expected here:")
    freq = Counter()
    for col in ("subject", "object"):
        freq.update(df[col].value_counts().to_dict())
    for e, n in freq.most_common(60):
        if e not in mapped:
            lines.append(f"  {e!r}: {n}")
            if len([l for l in lines if l.startswith("  '")]) >= 25:
                break

    lines.append(f"\nTop 40 by weighted Jaccard vs LiTFSI (battery-KG "
                 f"ego profiles, exact-tier only):")
    lines.append(f"{'rank':<5}{'w-Jaccard':<11}{'u-Jaccard':<11}"
                 f"{'mentions':<9}{'nbrs':<7}{'cmpst':<6}title")
    for i, r in enumerate(results[:40], 1):
        thin = " [thin]" if r["n_mentions"] < MIN_PROFILE_MENTIONS else ""
        lines.append(f"{i:<5}{r['jaccard_weighted']:<11.4f}"
                     f"{r['jaccard_unweighted']:<11.4f}"
                     f"{r['n_mentions']:<9}{r['n_neighbors']:<7}"
                     f"{r['n_composite_entities']:<6}"
                     f"{r['title'][:50]}{thin}")
    report = "\n".join(lines)
    with open(REPORT_TXT, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nWrote {MAPPING_CSV}, {RANKING_CSV}, {REPORT_TXT}")


if __name__ == "__main__":
    main()
