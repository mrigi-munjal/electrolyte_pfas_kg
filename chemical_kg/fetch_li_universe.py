"""
Stage 0 of the Jchem candidate filtering funnel.

Fetches the full universe of lithium-containing compounds from PubChem
(via Entrez esearch + PUG REST property batches) so that downstream
filtering can be done on real structures (SMILES) instead of molecular
formulas.

Replaces the old manually exported PubChem_compound_smiles_substructure.csv.

Outputs:
  chemical_kg/output/li_universe_cids.json  - cached CID list from esearch
  chemical_kg/output/li_universe.csv        - cid, title, mf, mw, charge, smiles, inchikey

Resumable: already-fetched CIDs are skipped on rerun.
"""

import csv
import io
import json
import os
import sys
import time

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Slightly wider than the funnel's 40-600 Da window; the funnel applies
# the strict cutoff so edge cases are decided locally, not by the query.
# (The legacy 50-250 window excluded LiTFSI itself at MW 287.)
MW_MIN, MW_MAX = 35, 620

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# cache is per-MW-window so changing the window never requires deleting files
CID_CACHE = os.path.join(OUT_DIR, f"li_universe_cids_mw{MW_MIN}_{MW_MAX}.json")
UNIVERSE_CSV = os.path.join(OUT_DIR, "li_universe.csv")

BATCH_SIZE = 500
SLEEP_BETWEEN_REQUESTS = 0.25  # stay well under PubChem's 5 req/s limit
MAX_RETRIES = 5

PROPERTIES = "MolecularFormula,MolecularWeight,Charge,SMILES,InChIKey,Title"
FIELDNAMES = ["CID", "MolecularFormula", "MolecularWeight", "Charge",
              "SMILES", "InChIKey", "Title"]


def fetch_cids():
    """Get all PubChem CIDs containing Li within the MW window."""
    if os.path.exists(CID_CACHE):
        with open(CID_CACHE) as f:
            cids = json.load(f)
        print(f"Loaded {len(cids)} CIDs from cache {CID_CACHE}")
        return cids

    term = f'"Li"[Element] AND {MW_MIN}:{MW_MAX}[MolecularWeight]'
    print(f"esearch term: {term}")
    resp = requests.get(EUTILS, params={
        "db": "pccompound", "term": term,
        "retmax": 100000, "retmode": "json",
    }, timeout=120)
    resp.raise_for_status()
    result = resp.json()["esearchresult"]
    cids = sorted(int(i) for i in result["idlist"])
    print(f"esearch reports Count={result['count']}, retrieved {len(cids)} CIDs")
    if int(result["count"]) > len(cids):
        raise RuntimeError("CID list truncated at retmax; add retstart pagination")

    with open(CID_CACHE, "w") as f:
        json.dump(cids, f)
    return cids


def load_done_cids():
    done = set()
    if os.path.exists(UNIVERSE_CSV):
        with open(UNIVERSE_CSV) as f:
            for row in csv.DictReader(f):
                done.add(int(row["CID"]))
        print(f"Resuming: {len(done)} CIDs already fetched")
    return done


def fetch_property_batch(cids):
    url = f"{PUG}/compound/cid/property/{PROPERTIES}/CSV"
    body = {"cid": ",".join(str(c) for c in cids)}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, data=body, timeout=120)
            if resp.status_code == 200:
                return list(csv.DictReader(io.StringIO(resp.text)))
            # 503 = server busy; back off and retry
            print(f"  HTTP {resp.status_code} (attempt {attempt}), backing off")
        except requests.RequestException as exc:
            print(f"  request failed (attempt {attempt}): {exc}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Batch starting at CID {cids[0]} failed after {MAX_RETRIES} retries")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cids = fetch_cids()
    done = load_done_cids()
    remaining = [c for c in cids if c not in done]
    print(f"{len(remaining)} CIDs to fetch in batches of {BATCH_SIZE}")

    write_header = not os.path.exists(UNIVERSE_CSV)
    n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    with open(UNIVERSE_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i:i + BATCH_SIZE]
            rows = fetch_property_batch(batch)
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
            f.flush()
            print(f"batch {i // BATCH_SIZE + 1}/{n_batches}: +{len(rows)} rows")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    with open(UNIVERSE_CSV) as f:
        total = sum(1 for _ in f) - 1
    print(f"Done. {UNIVERSE_CSV} has {total} compounds")


if __name__ == "__main__":
    # optional MW window override, e.g. `fetch_li_universe.py 24 35` for a
    # supplemental low-MW fetch (LiF at 25.9 Da sat below the original 35
    # floor); the CID cache is per-window and the universe CSV accumulates
    if len(sys.argv) == 3:
        MW_MIN, MW_MAX = int(sys.argv[1]), int(sys.argv[2])
        CID_CACHE = os.path.join(OUT_DIR,
                                 f"li_universe_cids_mw{MW_MIN}_{MW_MAX}.json")
    sys.exit(main())
