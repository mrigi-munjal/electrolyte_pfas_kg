"""
Stage 3 of the Jchem pipeline: synonym harvesting keyed by CID.

Replaces the old formula-keyed synonym tables (salt_synonyms_loader*.py),
which pooled isomer synonyms under one formula and let bare "lithium"/"Li"
synonyms substring-match everything in MatKG.

Reads  chemical_kg/output/final_electrolyte_components.csv
Writes chemical_kg/output/synonyms_raw.jsonl   (one {"cid", "synonyms"} per line)
       chemical_kg/output/curated_cid_map.json (InChIKey -> CID for curated seeds)

Curated-seed rows carry no CID (they were merged from
electrolyte_solvents_additives.csv), so their CIDs are resolved from
InChIKey first. Resumable: already-harvested CIDs are skipped on rerun;
CIDs with no PubChem synonyms are recorded with an empty list so they are
not refetched.
"""

import csv
import json
import os
import sys
import time

import requests

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
COMPONENTS_CSV = os.path.join(OUT_DIR, "final_electrolyte_components.csv")
UNIVERSE_CSV = os.path.join(OUT_DIR, "li_universe.csv")
RAW_JSONL = os.path.join(OUT_DIR, "synonyms_raw.jsonl")
CURATED_CID_MAP = os.path.join(OUT_DIR, "curated_cid_map.json")

BATCH_SIZE = 100
SLEEP_BETWEEN_REQUESTS = 0.25  # stay well under PubChem's 5 req/s limit
MAX_RETRIES = 5
MAX_SYNONYMS_PER_CID = 150  # PubChem lists best-known names first


def post_with_retry(url, body):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, data=body, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None  # no records for any id in the batch
            print(f"  HTTP {resp.status_code} (attempt {attempt}), backing off")
        except requests.RequestException as exc:
            print(f"  request failed (attempt {attempt}): {exc}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"POST {url} failed after {MAX_RETRIES} retries")


def resolve_curated_cids(components):
    """InChIKey -> CID for rows that came without one (curated seeds)."""
    if os.path.exists(CURATED_CID_MAP):
        with open(CURATED_CID_MAP) as f:
            return json.load(f)

    missing = sorted({row["inchikey"] for row in components if not row["cid"]})
    print(f"Resolving CIDs for {len(missing)} curated components by InChIKey")
    mapping = {}
    for ik in missing:
        data = post_with_retry(f"{PUG}/compound/inchikey/cids/JSON",
                               {"inchikey": ik})
        cids = (data or {}).get("IdentifierList", {}).get("CID", [])
        if cids:
            mapping[ik] = int(cids[0])  # first CID is the canonical record
        else:
            print(f"  no CID found for {ik}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    with open(CURATED_CID_MAP, "w") as f:
        json.dump(mapping, f, indent=1)
    print(f"Resolved {len(mapping)}/{len(missing)}; cached to {CURATED_CID_MAP}")
    return mapping


def load_done_cids():
    done = set()
    if os.path.exists(RAW_JSONL):
        with open(RAW_JSONL) as f:
            for line in f:
                done.add(json.loads(line)["cid"])
        print(f"Resuming: {len(done)} CIDs already harvested")
    return done


def main():
    with open(COMPONENTS_CSV) as f:
        components = list(csv.DictReader(f))
    print(f"{len(components)} components loaded")

    curated_map = resolve_curated_cids(components)

    # PubChem keeps duplicate records per structure and the merge dedupes
    # by InChIKey, sometimes keeping a synonym-less "CID nnn" record while
    # discarding its well-named twin (LiClO4: kept 151488, lost 23665649).
    # Harvest every universe CID that shares an InChIKey with a component.
    siblings = {}
    with open(UNIVERSE_CSV) as f:
        for row in csv.DictReader(f):
            if row["InChIKey"]:
                siblings.setdefault(row["InChIKey"], []).append(int(row["CID"]))

    all_cids = set()
    for row in components:
        cid = row["cid"] or curated_map.get(row["inchikey"])
        if cid:
            all_cids.add(int(cid))
        all_cids.update(siblings.get(row["inchikey"], []))
    print(f"{len(all_cids)} unique CIDs to harvest (incl. InChIKey siblings)")

    done = load_done_cids()
    remaining = sorted(all_cids - done)
    n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"{len(remaining)} CIDs remaining in {n_batches} batches")

    with open(RAW_JSONL, "a") as out:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i:i + BATCH_SIZE]
            data = post_with_retry(
                f"{PUG}/compound/cid/synonyms/JSON",
                {"cid": ",".join(str(c) for c in batch)})
            found = {}
            for info in (data or {}).get("InformationList", {}).get("Information", []):
                found[int(info["CID"])] = info.get("Synonym", [])[:MAX_SYNONYMS_PER_CID]
            for cid in batch:
                out.write(json.dumps(
                    {"cid": cid, "synonyms": found.get(cid, [])}) + "\n")
            out.flush()
            n_with = sum(1 for c in batch if found.get(c))
            print(f"batch {i // BATCH_SIZE + 1}/{n_batches}: "
                  f"{n_with}/{len(batch)} CIDs had synonyms")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    with open(RAW_JSONL) as f:
        rows = [json.loads(line) for line in f]
    n_nonempty = sum(1 for r in rows if r["synonyms"])
    print(f"Done. {len(rows)} CIDs in {RAW_JSONL}, "
          f"{n_nonempty} with at least one synonym")


if __name__ == "__main__":
    sys.exit(main())
