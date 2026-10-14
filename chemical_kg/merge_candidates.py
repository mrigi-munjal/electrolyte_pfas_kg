"""
Final step of the Jchem candidate filtering funnel.

Merges:
  1. accepted PubChem-derived lithium salts / anion fragments
     (chemical_kg/output/electrolyte_candidates_accepted.csv)
  2. the curated solvent/additive/diluent seed table
     (chemical_kg/electrolyte_solvents_additives.csv)

into one deduplicated candidate list for the synonym/canonicalization phase:
  chemical_kg/output/final_electrolyte_components.csv

Also runs the recall check against the 30 curated salts
(20241217_saltCandidates.csv) and reports where each one landed.
"""

import csv
import os

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
ACCEPTED = os.path.join(OUT_DIR, "electrolyte_candidates_accepted.csv")
REVIEW = os.path.join(OUT_DIR, "electrolyte_candidates_review.csv")
SEED = os.path.join(HERE, "electrolyte_solvents_additives.csv")
CURATED_30 = os.path.join(HERE, "..", "20241217_saltCandidates.csv")
FINAL = os.path.join(OUT_DIR, "final_electrolyte_components.csv")

FIELDNAMES = ["component_id", "source", "category", "family", "title",
              "mf", "mw", "smiles", "inchikey", "cid", "flags"]


def main():
    import re
    by_key = {}

    with open(ACCEPTED) as f:
        for r in csv.DictReader(f):
            key = r["inchikey"] or r["cid"]
            row = {
                "component_id": "", "source": "pubchem_funnel",
                "category": r["category"], "family": r["family"],
                "title": r["title"], "mf": r["mf"], "mw": r["mw"],
                "smiles": r["smiles"], "inchikey": r["inchikey"],
                "cid": r["cid"], "flags": r["flags"],
            }
            kept = by_key.get(key)
            # among InChIKey twins prefer the record with a real name over
            # a synonym-less "CID nnn" placeholder (e.g. LiClO4 151488)
            if kept is None or (re.fullmatch(r"CID \d+", kept["title"])
                                and not re.fullmatch(r"CID \d+", row["title"])):
                by_key[key] = row
    rows_out = list(by_key.values())
    seen_keys = set(by_key)

    with open(SEED) as f:
        for r in csv.DictReader(f):
            mol = Chem.MolFromSmiles(r["smiles"])
            if mol is None:
                print(f"WARNING: seed SMILES failed to parse: {r['abbrev']}")
                continue
            ik = Chem.MolToInchiKey(mol)
            if ik in seen_keys:
                continue
            seen_keys.add(ik)
            rows_out.append({
                "component_id": "", "source": "curated_seed",
                "category": r["category"], "family": r["family"],
                "title": f"{r['name']} ({r['abbrev']})",
                "mf": rdMolDescriptors.CalcMolFormula(mol),
                "mw": f"{Descriptors.MolWt(mol):.2f}",
                "smiles": r["smiles"], "inchikey": ik,
                "cid": "", "flags": "",
            })

    # stable component ids: salts first by MW, then solvents/additives
    rows_out.sort(key=lambda r: (r["source"] != "pubchem_funnel", float(r["mw"] or 0)))
    for i, r in enumerate(rows_out, 1):
        r["component_id"] = f"EC_{i:05d}"

    with open(FINAL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows_out)

    n_salt = sum(1 for r in rows_out if r["source"] == "pubchem_funnel")
    n_seed = sum(1 for r in rows_out if r["source"] == "curated_seed")
    print(f"{FINAL}: {len(rows_out)} components "
          f"({n_salt} from PubChem funnel, {n_seed} curated solvents/additives)")

    # ---- recall check against the 30 curated salts ----
    review_keys = set()
    with open(REVIEW) as f:
        review_keys = {r["inchikey"] for r in csv.DictReader(f)}

    print("\nRecall check vs 20241217_saltCandidates.csv:")
    n_ok = n_review = n_miss = 0
    with open(CURATED_30) as f:
        for i, row in enumerate(csv.DictReader(f)):
            mol = Chem.MolFromSmiles(row["SMILES"].strip())
            ik = Chem.MolToInchiKey(mol) if mol else None
            sid = f"Salt_{i + 1:02d}"
            if ik and ik in seen_keys:
                n_ok += 1
            elif ik and ik in review_keys:
                n_review += 1
                print(f"  {sid}: in REVIEW bucket")
            else:
                n_miss += 1
                print(f"  {sid}: MISSING ({row['SMILES'].strip()[:60]})")
    print(f"  accepted: {n_ok}/30, review: {n_review}/30, missing: {n_miss}/30")


if __name__ == "__main__":
    main()
