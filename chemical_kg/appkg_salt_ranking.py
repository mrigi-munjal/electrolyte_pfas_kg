"""
Application-KG (salt KG) mention mapping + LiTFSI ranking.

The LLM-extracted salt KG (salt_analysis_output/canonical/
salt_relationships_canonical_with_source.csv) is ego-centric: subjects
are salt mentions, objects are applications/properties/solvents. Its
name-based PubChem canonicalization left the same diseases the MatKG
side had:
- fragmentation: LiTFSI triples split across LiTFSI / LiNTf2 / LiTFSA
  canonicals; LiDFOB split as LiODFB vs "Lithium difluoro(oxalato)-
  borate"; triflate split by a wrong name resolution of LiSO3CF3
- generic pooling: "Li salt"/"lithium salt" -> pseudo-entity "lithium"
- junk resolutions: "SM salt" -> samarium

This script remaps every subject mention onto InChIKey-keyed components
using the Stage-4 synonym index (squash matching, cased rule for short
strings), keeps unmatched-but-real novel salts (LiTDI, LiFOP, ...) as
named pseudo-components, drops generic mentions, then ranks all salts
against LiTFSI by weighted Jaccard over object_canonical neighbor
profiles (weight = mention count).

Reads  salt_analysis_output/canonical/salt_relationships_canonical_with_source.csv
       output/synonyms_clean.csv, output/final_electrolyte_components.csv
Writes output/appkg_subject_mapping.csv   (per-mention audit)
       output/appkg_salt_ranking.csv      (ranked vs LiTFSI)
       output/appkg_report.txt
"""

import csv
import os
from collections import Counter, defaultdict

import pandas as pd

from textnorm import cased_squash, tokens_of

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT_DIR = os.path.join(BASE, "output")
SALT_KG = os.path.join(ROOT, "salt_analysis_output", "canonical",
                       "salt_relationships_canonical_with_source.csv")
CLEAN_CSV = os.path.join(OUT_DIR, "synonyms_clean.csv")
COMPONENTS_CSV = os.path.join(OUT_DIR, "final_electrolyte_components.csv")

MAPPING_CSV = os.path.join(OUT_DIR, "appkg_subject_mapping.csv")
RANKING_CSV = os.path.join(OUT_DIR, "appkg_salt_ranking.csv")
REPORT_TXT = os.path.join(OUT_DIR, "appkg_report.txt")

REF_INCHIKEY = "QSZMZKBZAYQGRS-UHFFFAOYSA-N"  # LiTFSI
CASED_MATCH_MAX_LEN = 4
MIN_TRIPLES_PSEUDO = 10   # unmatched mentions this frequent stay as
                          # named pseudo-components (novel salts)
MAX_PHRASE_TOKENS = 4     # longer mentions are formulation phrases

# mentions of these squashes are generic, not a specific salt
GENERIC = {"lithium", "li", "lisalt", "lisalts", "lithiumsalt",
           "lithiumsalts", "salt", "salts", "carbonate", "carbonatesalt",
           "electrolyte", "electrolytesalt", "conductingsalt", "smsalt",
           "liquidelectrolyte", "anion", "cation", "additive"}


def build_synonym_lookup():
    comps = {r["component_id"]: r for r in csv.DictReader(open(COMPONENTS_CSV))}
    lookup = defaultdict(set)   # squash -> {(inchikey, cased-req)}
    for r in csv.DictReader(open(CLEAN_CSV)):
        squash = "".join(r["tokens"].split())
        if not squash:
            continue
        ik = comps[r["component_id"]]["inchikey"]
        cased = r["cased"] if len(squash) <= CASED_MATCH_MAX_LEN else ""
        lookup[squash].add((ik, cased))
    titles = {}
    for r in comps.values():
        titles.setdefault(r["inchikey"], r["title"])
    return lookup, titles


def match_mention(s, lookup):
    """Full-string squash match of one mention -> set of InChIKeys."""
    squash = "".join(tokens_of(s))
    if not squash:
        return set(), squash
    found = set()
    for ik, cased in lookup.get(squash, ()):
        if cased and cased != cased_squash(s):
            continue
        found.add(ik)
    return found, squash


def main():
    df = pd.read_csv(SALT_KG)
    df["object_key"] = df["object_canonical"].fillna(df["object"]).astype(str)
    print(f"{len(df)} triples, {df['subject'].nunique()} unique subjects")

    lookup, titles = build_synonym_lookup()

    # ---- map every distinct (subject, subject_canonical) mention ----
    mention_counts = df.groupby(["subject", "subject_canonical"],
                                dropna=False).size()
    mapping = []
    key_of = {}   # (subject, subject_canonical) -> salt_key or None
    for (subj, canon), n in mention_counts.items():
        canon = "" if pd.isna(canon) else str(canon)
        iks, squash = match_mention(str(subj), lookup)
        via = "subject"
        if not iks and canon:
            iks, _ = match_mention(canon, lookup)
            via = "canonical"
        if squash in GENERIC or (not iks and "".join(tokens_of(canon)) in GENERIC):
            status, key = "generic_dropped", None
        elif len(iks) == 1:
            status, key = f"matched_{via}", next(iter(iks))
        elif len(iks) > 1:
            status, key = "ambiguous_dropped", None
        elif len(tokens_of(str(subj))) > MAX_PHRASE_TOKENS:
            status, key = "phrase_dropped", None
        elif n >= MIN_TRIPLES_PSEUDO:
            status, key = "pseudo_component", f"pseudo:{canon or subj}"
        else:
            status, key = "rare_unmatched", None
        key_of[(subj, canon)] = key
        mapping.append({"subject": subj, "subject_canonical": canon,
                        "n_triples": n, "status": status,
                        "salt_key": key or "",
                        "title": titles.get(key, "") if key else ""})

    with open(MAPPING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mapping[0].keys()))
        w.writeheader()
        w.writerows(sorted(mapping, key=lambda r: -r["n_triples"]))

    # ---- neighbor profiles per salt_key ----
    profiles = defaultdict(Counter)
    variants = defaultdict(set)
    for (subj, canon), grp in df.groupby(["subject", "subject_canonical"],
                                         dropna=False):
        canon = "" if pd.isna(canon) else str(canon)
        key = key_of.get((subj, canon))
        if key is None:
            continue
        profiles[key].update(grp["object_key"].value_counts().to_dict())
        variants[key].add(subj)

    ref = profiles.get(REF_INCHIKEY)
    if not ref:
        raise SystemExit("LiTFSI reference has no profile")
    ref_set = set(ref)
    print(f"{len(profiles)} salts with profiles; "
          f"LiTFSI profile: {sum(ref.values())} mentions, "
          f"{len(ref)} distinct neighbors")

    results = []
    for key, prof in profiles.items():
        pset = set(prof)
        inter = pset & ref_set
        union = pset | ref_set
        inter_w = sum(min(prof[o], ref[o]) for o in inter)
        union_w = sum(max(prof.get(o, 0), ref.get(o, 0)) for o in union)
        display = (titles.get(key) or key.replace("pseudo:", "")
                   if not key.startswith("pseudo:")
                   else key.replace("pseudo:", "") + " [novel/unmatched]")
        results.append({
            "salt_key": key, "display_name": display,
            "n_subject_variants": len(variants[key]),
            "subject_variants": "; ".join(sorted(variants[key])[:6]),
            "n_mentions": sum(prof.values()),
            "n_neighbors": len(prof),
            "intersection_count": len(inter),
            "jaccard_unweighted": len(inter) / len(union) if union else 0.0,
            "jaccard_weighted": inter_w / union_w if union_w else 0.0,
            "top_shared_neighbors": "; ".join(
                o for o, _ in sorted(((o, min(prof[o], ref[o]))
                                      for o in inter),
                                     key=lambda kv: -kv[1])[:8]),
        })
    results.sort(key=lambda r: -r["jaccard_weighted"])

    with open(RANKING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ---- report ----
    status_counts = Counter(m["status"] for m in mapping)
    healed = {k: sorted({m["subject_canonical"] for m in mapping
                         if m["salt_key"] == k})
              for k in profiles if not k.startswith("pseudo:")}
    healed = {k: v for k, v in healed.items() if len(v) > 1}

    lines = []
    lines.append(f"Salt KG triples: {len(df)}; mentions mapped: "
                 f"{len(mapping)} distinct (subject, canonical) pairs")
    lines.append("\nMention mapping outcomes:")
    for s, n in status_counts.most_common():
        n_triples = sum(m["n_triples"] for m in mapping if m["status"] == s)
        lines.append(f"  {s}: {n} mentions / {n_triples} triples")
    lines.append("\nFragmentation healed (one InChIKey <- multiple "
                 "canonical forms of the old pipeline):")
    for k, forms in sorted(healed.items(),
                           key=lambda kv: -len(kv[1]))[:15]:
        lines.append(f"  {titles.get(k, k)}: {forms}")
    lines.append("\nTop unmatched mentions (candidates for "
                 "manual_synonyms.csv):")
    for m in sorted((m for m in mapping
                     if m["status"] in ("rare_unmatched", "phrase_dropped")),
                    key=lambda m: -m["n_triples"])[:20]:
        lines.append(f"  {m['subject']!r} (canon {m['subject_canonical']!r}, "
                     f"{m['n_triples']} triples)")
    lines.append(f"\nTop 40 by weighted Jaccard vs LiTFSI "
                 f"(application-KG neighborhoods):")
    lines.append(f"{'rank':<5}{'w-Jaccard':<11}{'u-Jaccard':<11}"
                 f"{'mentions':<9}{'nbrs':<7}name")
    for i, r in enumerate(results[:40], 1):
        lines.append(f"{i:<5}{r['jaccard_weighted']:<11.4f}"
                     f"{r['jaccard_unweighted']:<11.4f}"
                     f"{r['n_mentions']:<9}{r['n_neighbors']:<7}"
                     f"{r['display_name'][:55]}")
    report = "\n".join(lines)
    with open(REPORT_TXT, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nWrote {MAPPING_CSV}, {RANKING_CSV}, {REPORT_TXT}")


if __name__ == "__main__":
    main()
