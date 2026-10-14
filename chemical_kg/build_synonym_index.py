"""
Stage 4 of the Jchem pipeline: synonym canonicalization + ambiguity audit.

Fixes the diagnosed failures of the legacy synonym step:
- formula keying pooled isomer synonyms -> everything here is keyed by
  component_id / InChIKey / CID
- bare "lithium"/"Li" synonyms substring-matched everything in MatKG ->
  a manual blocklist plus a data-driven promiscuity rule (a normalized
  synonym claimed by more than MAX_SHARED_BLOCKS distinct InChIKey
  connectivity blocks is dropped everywhere)
- registry identifiers (CAS, EINECS, UNII, SCHEMBL, ...) polluted the
  match patterns -> pattern-based cleaning with per-reason counts

Normalization for matching (norm_key): NFKC, casefold, strip all
whitespace and hyphens. MatKG entity names insert spaces and title-case
freely ("Lithium Bis(Trifluoromethanesulfonyl) Imide"), so raw-string
comparison would miss LiTFSI itself. Token sequences (alnum runs) are
kept alongside for word-boundary containment matching in Stage 5.

Short synonyms (norm_key < MIN_CONTAINMENT_LEN and single-token) such as
"EC" or "LiF" are kept but marked exact_only: they may match a MatKG
entity exactly, never as a substring of a longer entity.

Reads  output/final_electrolyte_components.csv
       output/synonyms_raw.jsonl
       output/curated_cid_map.json
       electrolyte_solvents_additives.csv   (abbrev + name extras)
Writes output/synonyms_clean.csv  (kept rows: component_id, cid, synonym,
                                   norm_key, tokens, match_tier, shared_blocks)
       output/synonyms_dropped.csv
       output/synonym_audit.txt
"""

import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from rdkit import Chem
from rdkit import RDLogger

from textnorm import cased_squash, norm_key, tokens_of

RDLogger.DisableLog("rdApp.*")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
COMPONENTS_CSV = os.path.join(OUT_DIR, "final_electrolyte_components.csv")
RAW_JSONL = os.path.join(OUT_DIR, "synonyms_raw.jsonl")
CURATED_CID_MAP = os.path.join(OUT_DIR, "curated_cid_map.json")
CURATED_CSV = os.path.join(BASE, "electrolyte_solvents_additives.csv")
MANUAL_CSV = os.path.join(BASE, "manual_synonyms.csv")
MANUAL_BLOCK_CSV = os.path.join(BASE, "manual_blocklist.csv")
UNIVERSE_CSV = os.path.join(OUT_DIR, "li_universe.csv")

CLEAN_CSV = os.path.join(OUT_DIR, "synonyms_clean.csv")
DROPPED_CSV = os.path.join(OUT_DIR, "synonyms_dropped.csv")
AUDIT_TXT = os.path.join(OUT_DIR, "synonym_audit.txt")

# A normalized synonym claimed by more than this many distinct InChIKey
# connectivity blocks is too promiscuous to attribute ("lithium" spans
# dozens). <= this many blocks is legitimate sharing (anhydrous/hydrate
# pairs, isotopologues, acid/salt pairs) and the synonym is kept for all
# of its owners.
MAX_SHARED_BLOCKS = 3

# Below this norm_key length a single-token synonym is exact-match-only.
MIN_CONTAINMENT_LEN = 5

# Generic terms that must never match, regardless of sharing stats
# (norm_key space: casefolded, whitespace/hyphens removed).
BLOCKLIST = {
    "lithium", "li", "liion", "lithiumion", "lithiumcation", "lithiumatom",
    "lithiumsalt", "lithiumsalts", "lithiumcompound", "lithiumcompounds",
    "salt", "salts", "water", "acid", "acids", "base", "ion", "ions",
    "anion", "anions", "cation", "cations", "hydrate", "anhydrous",
    "solution", "solvent", "electrolyte", "additive", "monomer", "polymer",
    "compound", "reagent", "standard", "mixture", "component",
    # bare functional-group / anion words that PubChem attaches to one
    # specific record but that name a whole class in MatKG
    "ether", "aether", "ester", "ketone", "amide", "amine", "alcohol",
    "carbonate", "bicarbonate", "sulfate", "sulfonate", "sulfite",
    "phosphate", "phosphite", "fluoride", "chloride", "bromide", "iodide",
    "nitrate", "nitrite", "hydroxide", "oxide", "peroxide", "formate",
    "acetate", "oxalate", "borate", "silicate", "carboxylate", "glycol",
}

REGISTRY_PATTERNS = [
    ("cid_number", re.compile(r"^cid ?\d+$")),
    ("cas_number", re.compile(r"^(cas[- ]?)?\d{2,7}-\d{2}-\d$")),
    ("einecs_ec", re.compile(r"^(einecs|ec|ecno\.?)[- ]?\d{3}-\d{3}-\d$")),
    ("unii", re.compile(r"^unii[- ]\w+$")),
    ("dsstox", re.compile(r"^dtx[sc]id\w+$")),
    ("schembl", re.compile(r"^schembl\d+$")),
    ("chebi_chembl", re.compile(r"^(chebi:\d+|chembl\d+)$")),
    ("inchi", re.compile(r"^inchi=", re.I)),
    ("inchikey", re.compile(r"^[a-z]{14}-[a-z]{10}-[a-z]$")),
    ("wikidata", re.compile(r"^q\d+$")),
    # AKOS..., MFCD..., NSC 1234, ZINC..., D02627, EN300-..., HY-12345 etc:
    # short letter prefix glued to a long digit run, optionally dashed
    ("catalog_code", re.compile(r"^[a-z]{1,7}[- ]?\d{4,}(-\d+)?$")),
    ("nsc_style", re.compile(r"^(nsc|brn|hsdb|ccris|caswell no\.?|ai3)[- ]?\d+$")),
    # bare UNII (10 alnum chars incl. a digit, e.g. 9Y730392A5)
    ("unii_bare", re.compile(r"^(?=.*\d)[a-z0-9]{10}$")),
    ("refchem", re.compile(r"^refchem:\d+$")),
    # CP-15,467-61 style vendor codes
    ("catalog_dashed", re.compile(r"^[a-z]{1,4}-[\d,]{3,}(-\d+)?$")),
]


# trailing parenthetical abbreviations carry the formula orderings MatKG
# actually uses: "LITHIUM TRIFLUOROMETHANESULFONATE (LISO3CF3)" -> LiSO3CF3
PAREN_ABBREV = re.compile(r"^(.{4,}?)\s*\(([A-Za-z0-9+/,'-]{2,12})\)\s*$")
# pharmacopoeia/source/analytical tags that appear in the same position
DOC_TAGS = {"tn", "jan", "usp", "usan", "inn", "ban", "mi", "hsdb", "vandf",
            "who-dd", "who-ip", "mart.", "ep monograph", "usp monograph",
            "orange book", "usp-rs", "jp17", "jp18", "nf", "dsc",
            "nmr", "tlc", "hplc", "gc", "ms", "ir", "uv", "cp", "icp",
            "xrd", "aas", "acs", "9ci", "8ci", "van", "gmp", "gc-ms",
            "lc-ms"}


def clean_one(raw):
    """Return (norm, tokens, drop_reason|None) for one raw synonym."""
    s = raw.strip()
    if len(s) < 2:
        return None, None, "too_short"
    if len(s) > 120:
        return None, None, "too_long"
    if "%" in s:
        return None, None, "purity_qualifier"
    low = unicodedata.normalize("NFKC", s).casefold().strip()
    for reason, pat in REGISTRY_PATTERNS:
        if pat.match(low):
            return None, None, reason
    nk = norm_key(s)
    if not re.search(r"[a-z]", nk):
        return None, None, "no_letters"
    toks = tokens_of(s)
    if not toks:
        return None, None, "no_tokens"
    # check the blocklist against the squash too: "Li+" squashes to "li"
    if nk in BLOCKLIST or "".join(toks) in BLOCKLIST:
        return None, None, "blocklist"
    return nk, toks, None


def curated_extras():
    """InChIKey -> [name, abbrev] from the curated solvent/additive seed,
    plus the reserved map: those names/abbrevs may only be claimed by the
    seed compound itself. PubChem adduct/solution records graft solvent
    names onto salts ("THF" as a Lithium Chloride synonym)."""
    extras = defaultdict(list)
    reserved = {}
    with open(CURATED_CSV) as f:
        for row in csv.DictReader(f):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue
            ik = Chem.MolToInchiKey(mol)
            extras[ik].extend([row["name"], row["abbrev"]])
            for s in (row["name"], row["abbrev"]):
                reserved[norm_key(s)] = ik.split("-")[0]
    return extras, reserved


def main():
    with open(COMPONENTS_CSV) as f:
        components = list(csv.DictReader(f))
    with open(CURATED_CID_MAP) as f:
        curated_map = json.load(f)
    syn_by_cid = {}
    with open(RAW_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            syn_by_cid[rec["cid"]] = rec["synonyms"]
    extras, reserved = curated_extras()
    with open(MANUAL_CSV) as f:
        for row in csv.DictReader(f):
            extras[row["inchikey"]].append(row["synonym"])
    manual_block = defaultdict(set)  # inchikey -> {norm_key}: known-wrong
    with open(MANUAL_BLOCK_CSV) as f:
        for row in csv.DictReader(f):
            manual_block[row["inchikey"]].add(norm_key(row["synonym"]))
    # all universe CIDs per InChIKey: a component's synonyms are the union
    # over its duplicate PubChem records (see harvest_synonyms.py)
    siblings = defaultdict(list)
    with open(UNIVERSE_CSV) as f:
        for row in csv.DictReader(f):
            if row["InChIKey"]:
                siblings[row["InChIKey"]].append(int(row["CID"]))
    print(f"{len(components)} components, {len(syn_by_cid)} harvested CIDs")

    # ---- collect + clean candidates per component ----
    candidates = []   # (component_id, cid, raw, nk, toks)
    dropped = []      # (component_id, cid, raw, reason)
    drop_counts = Counter()
    block_of = {}     # component_id -> InChIKey connectivity block

    for row in components:
        comp = row["component_id"]
        ik = row["inchikey"]
        block_of[comp] = ik.split("-")[0] if ik else comp
        cid = row["cid"] or curated_map.get(ik) or ""
        cids = {int(cid)} if cid else set()
        cids.update(siblings.get(ik, []))
        raws = [s for c in sorted(cids) for s in syn_by_cid.get(c, [])]
        raws.append(row["title"])
        raws.extend(extras.get(ik, []))
        # trailing parenthetical abbreviations become synonyms of their own;
        # only cased/numbered forms qualify (LiCl, GBL, LISO3CF3) -- plain
        # lowercase parentheticals are qualifiers ("(powder)", "(natural)"),
        # as is anything on a purity-qualified vendor string
        for raw in list(raws):
            m = PAREN_ABBREV.match(raw or "")
            if (m and "%" not in raw
                    and re.search(r"[A-Z0-9]", m.group(2))
                    and re.search(r"[A-Za-z]", m.group(2))
                    and m.group(2).casefold() not in DOC_TAGS):
                raws.append(m.group(2))

        seen = set()
        for raw in raws:
            if not raw:
                continue
            nk, toks, reason = clean_one(raw)
            if reason:
                drop_counts[reason] += 1
                dropped.append((comp, cid, raw, reason))
                continue
            if nk in manual_block.get(ik, ()):
                drop_counts["manual_blocklist"] += 1
                dropped.append((comp, cid, raw, "manual_blocklist"))
                continue
            if reserved.get(nk, block_of[comp]) != block_of[comp]:
                drop_counts["reserved_curated_name"] += 1
                dropped.append((comp, cid, raw, "reserved_curated_name"))
                continue
            if nk in seen:
                continue
            seen.add(nk)
            candidates.append((comp, cid, raw, nk, toks))

    # ---- promiscuity rule over InChIKey connectivity blocks ----
    blocks_by_nk = defaultdict(set)
    for comp, _cid, _raw, nk, _toks in candidates:
        blocks_by_nk[nk].add(block_of[comp])

    kept = []
    promiscuous = Counter()
    for comp, cid, raw, nk, toks in candidates:
        n_blocks = len(blocks_by_nk[nk])
        if n_blocks > MAX_SHARED_BLOCKS:
            drop_counts["promiscuous"] += 1
            promiscuous[nk] = n_blocks
            dropped.append((comp, cid, raw, f"promiscuous:{n_blocks}_blocks"))
            continue
        # containment (substring-of-entity) rights require a real word:
        # one token of >= MIN_CONTAINMENT_LEN chars, or a multi-word name
        # whose tokens are all substantial. Formula fragments like
        # "O(Et)2" (tokens o/et/2) stay exact-only so they cannot hit
        # "BF3-OEt2"-style entities of different compounds.
        tier = ("containment"
                if (max(len(t) for t in toks) >= MIN_CONTAINMENT_LEN
                    or (len(toks) >= 2 and all(len(t) >= 3 for t in toks)))
                else "exact_only")
        kept.append({
            "component_id": comp, "cid": cid, "synonym": raw,
            "norm_key": nk, "tokens": " ".join(toks),
            "cased": cased_squash(raw),
            "match_tier": tier, "shared_blocks": n_blocks,
        })

    # ---- write outputs ----
    with open(CLEAN_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component_id", "cid", "synonym",
                                          "norm_key", "tokens", "cased",
                                          "match_tier", "shared_blocks"])
        w.writeheader()
        w.writerows(kept)
    with open(DROPPED_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component_id", "cid", "synonym", "reason"])
        w.writerows(dropped)

    comps_with_syn = {r["component_id"] for r in kept}
    orphans = [c for c in block_of if c not in comps_with_syn]
    n_exact_only = sum(1 for r in kept if r["match_tier"] == "exact_only")

    lines = []
    lines.append(f"Components: {len(components)}")
    lines.append(f"Raw candidate synonyms: {len(candidates) + len(dropped)}")
    lines.append(f"Kept: {len(kept)} ({n_exact_only} exact-only)")
    lines.append(f"Dropped: {len(dropped)}")
    lines.append(f"Components with >=1 usable synonym: {len(comps_with_syn)}")
    lines.append(f"Components with none (unmatchable): {len(orphans)}")
    lines.append("\nDrop reasons:")
    for reason, n in drop_counts.most_common():
        lines.append(f"  {reason}: {n}")
    lines.append(f"\nTop promiscuous synonyms (> {MAX_SHARED_BLOCKS} "
                 "connectivity blocks, dropped everywhere):")
    for nk, n_blocks in sorted(promiscuous.items(),
                               key=lambda kv: -kv[1])[:40]:
        lines.append(f"  {nk}: {n_blocks} blocks")
    shared = Counter(r["shared_blocks"] for r in kept)
    lines.append("\nKept synonyms by number of sharing blocks:")
    for n_blocks in sorted(shared):
        lines.append(f"  {n_blocks} block(s): {shared[n_blocks]}")
    report = "\n".join(lines)
    with open(AUDIT_TXT, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nWrote {CLEAN_CSV}, {DROPPED_CSV}, {AUDIT_TXT}")


if __name__ == "__main__":
    sys.exit(main())
