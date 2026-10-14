"""
Stages 1-2 of the Jchem candidate filtering funnel.

Takes the raw PubChem lithium universe (chemical_kg/output/li_universe.csv,
produced by fetch_li_universe.py) and reduces it to plausible electrolyte
components using deterministic structure rules instead of an LLM:

  Stage 1 - structure rules (RDKit):
    * MW window 50-250 Da
    * element whitelist (extended elements As/Al/Sb allowed but flagged)
    * net charge 0  -> must be a real Li salt (ionic [Li+] + molecular anion,
                       or an ionic-equivalent covalent drawing); organolithium
                       and amide reagents rejected
    * net charge <0 -> kept as anion_fragment (can pair with Li+)
    * net charge >0 -> rejected (cationic fragments)

  Stage 2 - anion family classification (SMARTS) with an allowlist of
    electrolyte-plausible families. Families that pass structure rules but
    are not on the allowlist land in a human-reviewable bucket instead of
    being silently dropped.

Outputs (chemical_kg/output/):
  electrolyte_candidates_accepted.csv
  electrolyte_candidates_review.csv
  electrolyte_candidates_rejected.csv
  funnel_report.txt
"""

import csv
import os
import re
from collections import Counter, defaultdict

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
UNIVERSE_CSV = os.path.join(OUT_DIR, "li_universe.csv")

# NOTE: the legacy pipeline used 50-250, which silently excluded LiTFSI (287),
# LiBETI (387) and other real candidates. Widened after the 30-salt recall test.
MW_MIN, MW_MAX = 24.0, 600.0  # floor lowered from 40: it silently cut LiF (25.9)

CORE_ELEMENTS = {"Li", "C", "H", "B", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"}
# Real electrolyte salts exist for these (LiAsF6, LiAlCl4, LiSbF6) - allowed but flagged
EXTENDED_ELEMENTS = {"As", "Al", "Sb"}

# --- Stage 2: anion family SMARTS, most specific first ---------------------
# verdict: accept = electrolyte-plausible family; review = structurally clean
# but questionable as an electrolyte component; reject = reagent/junk anion.
FAMILY_RULES = [
    ("sulfonimide",      "[N-](S(=O)=O)S(=O)=O",              "accept"),
    ("sulfonimide",      "N(S(=O)=O)S(=O)=O",                 "accept"),  # covalent drawing
    ("sulfonylmethanide","[C-](S(=O)=O)S(=O)=O",              "accept"),  # e.g. TTM-type
    ("sulfonamide_anion","[N-]S(=O)=O",                        "accept"),  # single-sulfonyl amide
    ("sulfate",          "[O-]S(=O)(=O)O",                     "accept"),
    ("sulfonate",        "[O-]S(=O)=O",                        "accept"),
    ("sulfinate",        "[O-][S;X3]=O",                       "review"),
    ("hexafluorophosphate","[P-](F)(F)(F)(F)(F)F",             "accept"),
    ("fluorophosphate",  "[O-]P(=O)F",                         "accept"),  # PO2F2-, PO3F2-
    ("phosphate",        "[O-]P(=O)(O)O",                      "accept"),
    ("phosphate",        "[O-]P(=O)=O",                        "accept"),  # metaphosphate
    ("phosphinate",      "[O-][P;X4]=O",                       "review"),
    ("fluoroborate",     "[B-](F)(F)(F)F",                     "accept"),
    ("oxalatoborate",    "[B-]1(OC(=O)C(=O)O1)",               "accept"),
    ("fluoro_oxalatoborate","[B-](F)(F)OC=O",                  "accept"),  # LiDFOB-type
    ("metaborate",       "[O-]B=O",                            "accept"),  # LiBO2 (curated Salt_13)
    ("borate",           "[B-]",                               "review"),  # other borates
    # RDKit charge-separates hypervalent Cl oxyanions: ClO4- -> [Cl+3](O-)4
    ("perchlorate",      "[Cl+3]([O-])([O-])([O-])[O-]",       "accept"),
    ("perchlorate",      "[O-]Cl(=O)(=O)=O",                   "accept"),
    ("chlorate",         "[Cl+2]([O-])([O-])[O-]",             "review"),
    ("chlorate",         "[O-]Cl(=O)=O",                       "review"),
    ("nitrate",          "[N+](=O)([O-])[O-]",                 "accept"),
    ("nitrite",          "[O-]N=O",                            "review"),
    ("hexafluoroarsenate","[As-](F)(F)(F)(F)(F)F",             "accept"),
    ("hexafluoroantimonate","[Sb-](F)(F)(F)(F)(F)F",           "accept"),
    ("tetrachloroaluminate","[Al-](Cl)(Cl)(Cl)Cl",             "accept"),
    ("azolide",          "[n-]",                               "accept"),  # LiTDI-type imidazolides
    ("dicyanamide",      "[N-](C#N)C#N",                       "accept"),
    ("tricyanomethanide","[C-](C#N)(C#N)C#N",                  "accept"),
    ("nitrile_carbanion","[C-]C#N",                            "review"),
    ("thiocyanate",      "[S-]C#N",                            "accept"),  # LiSCN
    ("cyanide",          "[C-]#N",                             "reject"),
    ("azide",            "[N-]=[N+]=[N-]",                     "review"),
    ("oxalate",          "[O-]C(=O)C(=O)[O-]",                 "accept"),
    ("carbonate",        "[O-]C(=O)[O-]",                      "accept"),
    ("bicarbonate",      "[O-]C(=O)O",                         "accept"),
    ("carboxylate",      "[O-]C=O",                            "accept"),
    ("enolate",          "[O-]C=C",                            "review"),
    ("phenoxide",        "[O-]c",                              "review"),
    ("alkoxide",         "[O-][C;X4]",                         "review"),
    ("siloxide",         "[O-][Si]",                           "review"),
    ("hydroxide",        "[OX1H1-]",                           "review"),
    ("thiolate",         "[S-]",                               "review"),
    ("silylamide",       "[N-]([Si])[Si]",                     "review"),  # LiHMDS - reagent but salt-like
    ("amide_anion",      "[N-]",                               "reject"),  # plain N- reagents
    ("halide",           "[F-,Cl-,Br-,I-]",                    "accept"),
    ("hydride",          "[H-]",                               "reject"),
    ("oxide",            "[O-2]",                              "reject"),
    ("peroxide",         "[O-][O-]",                           "reject"),
    ("sulfide",          "[S-2]",                              "reject"),
    ("carbanion",        "[C-]",                               "reject"),
    ("phosphide",        "[P-]",                               "reject"),
    ("silicide",         "[Si-]",                              "reject"),
    ("boranuide",        "[BH4-]",                              "reject"),
]
COMPILED_RULES = [(name, Chem.MolFromSmarts(sma), verdict)
                  for name, sma, verdict in FAMILY_RULES]
assert all(m is not None for _, m, _ in COMPILED_RULES), "bad SMARTS in FAMILY_RULES"

PFAS_PATTERN = Chem.MolFromSmarts("[CX4](F)F")  # >=CF2: PFAS-like moiety


def classify_anion_family(mol):
    """Return (family, verdict) for an anion fragment / Li-stripped residue."""
    for name, patt, verdict in COMPILED_RULES:
        if mol.HasSubstructMatch(patt):
            return name, verdict
    return "unknown", "review"


def li_covalent_partner_elements(mol):
    """Elements directly bonded to covalent Li atoms."""
    partners = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "Li":
            for nbr in atom.GetNeighbors():
                partners.add(nbr.GetSymbol())
    return partners


def ionize_covalent_li(mol):
    """
    Convert a covalent Li drawing (e.g. CCO[Li], FS(=O)(=O)N([Li])S(=O)(=O)F)
    into its ionic equivalent so the anion residue can be classified.
    Returns the residue mol (Li removed, partner atom charged -1) or None.
    """
    rw = Chem.RWMol(mol)
    li_idxs = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "Li"]
    for idx in sorted(li_idxs, reverse=True):
        atom = rw.GetAtomWithIdx(idx)
        for nbr in list(atom.GetNeighbors()):
            nbr.SetFormalCharge(nbr.GetFormalCharge() - 1)
            nbr.SetNoImplicit(True)
        rw.RemoveAtom(idx)
    try:
        residue = rw.GetMol()
        Chem.SanitizeMol(residue)
        return residue
    except Exception:
        return None


def process_row(row):
    """
    Classify one universe row. Returns a dict with:
      verdict: accept | review | reject
      category, family, reason, flags
    """
    result = {
        "cid": row["CID"], "title": row["Title"], "mf": row["MolecularFormula"],
        "mw": row["MolecularWeight"], "smiles": row["SMILES"],
        "inchikey": row["InChIKey"],
        "category": "", "family": "", "verdict": "", "reason": "", "flags": [],
    }

    def finish(verdict, reason, category="", family=""):
        result.update(verdict=verdict, reason=reason,
                      category=category, family=family)
        result["flags"] = ";".join(result["flags"])
        return result

    # MW window
    try:
        mw = float(row["MolecularWeight"])
    except (ValueError, TypeError):
        return finish("reject", "no_mw")
    if not (MW_MIN <= mw <= MW_MAX):
        return finish("reject", "mw_out_of_window")

    mol = Chem.MolFromSmiles(row["SMILES"])
    if mol is None:
        return finish("review", "rdkit_parse_error")

    # element whitelist
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    bad = elements - CORE_ELEMENTS - EXTENDED_ELEMENTS - {"H"}
    if bad:
        return finish("reject", "disallowed_element:" + ",".join(sorted(bad)))
    if elements & EXTENDED_ELEMENTS:
        result["flags"].append("extended_element")

    if mol.HasSubstructMatch(PFAS_PATTERN):
        result["flags"].append("pfas_like")

    # fragments; drop waters (hydrate records) but flag them
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    nonwater = []
    for fr in frags:
        if Chem.MolToSmiles(fr) == "O":
            result["flags"].append("hydrate")
        else:
            nonwater.append(fr)
    if not nonwater:
        return finish("reject", "only_water")
    frags = nonwater

    net_charge = sum(a.GetFormalCharge() for fr in frags for a in fr.GetAtoms())
    frag_charges = [sum(a.GetFormalCharge() for a in fr.GetAtoms()) for fr in frags]

    li_ionic = [fr for fr in frags
                if fr.GetNumAtoms() == 1 and fr.GetAtomWithIdx(0).GetSymbol() == "Li"
                and fr.GetAtomWithIdx(0).GetFormalCharge() == 1]
    li_covalent = [fr for fr in frags
                   if any(a.GetSymbol() == "Li" for a in fr.GetAtoms())
                   and fr not in li_ionic and fr.GetNumAtoms() > 1]
    bare_li_neutral = [fr for fr in frags
                       if fr.GetNumAtoms() == 1
                       and fr.GetAtomWithIdx(0).GetSymbol() == "Li"
                       and fr.GetAtomWithIdx(0).GetFormalCharge() == 0]
    anions = [fr for fr, ch in zip(frags, frag_charges) if ch < 0]
    organic_cations = [fr for fr, ch in zip(frags, frag_charges)
                       if ch > 0 and not (fr.GetNumAtoms() == 1
                                          and fr.GetAtomWithIdx(0).GetSymbol() == "Li")]

    if bare_li_neutral and len(frags) == len(bare_li_neutral):
        return finish("reject", "bare_metal")

    # --- cationic fragments: reject ---
    if net_charge > 0:
        return finish("reject", "net_cationic_fragment")

    # --- anionic fragments: keep if family is plausible (user request) ---
    if net_charge < 0:
        family_votes = [classify_anion_family(fr) for fr in anions] or \
                       [classify_anion_family(fr) for fr in frags]
        family, verdict = family_votes[0]
        for fam, ver in family_votes:  # worst verdict wins
            order = {"accept": 0, "review": 1, "reject": 2}
            if order[ver] > order[verdict]:
                family, verdict = fam, ver
        return finish(verdict, "anion_family_" + verdict,
                      category="anion_fragment", family=family)

    # --- neutral records ---
    if organic_cations:
        return finish("review", "mixed_organic_cation", category="mixed_salt")

    if li_ionic and anions:
        # proper ionic Li salt drawing
        fams = [classify_anion_family(fr) for fr in anions]
        family, verdict = fams[0]
        order = {"accept": 0, "review": 1, "reject": 2}
        for fam, ver in fams:
            if order[ver] > order[verdict]:
                family, verdict = fam, ver
        return finish(verdict, "salt_family_" + verdict,
                      category="lithium_salt", family=family)

    if li_covalent:
        partners = set()
        for fr in li_covalent:
            partners |= li_covalent_partner_elements(fr)
        if "C" in partners:
            return finish("reject", "organolithium_C-Li")
        # ionize the covalent drawing and classify the residue
        residue = ionize_covalent_li(li_covalent[0])
        if residue is None:
            return finish("review", "covalent_li_unparseable")
        family, verdict = classify_anion_family(residue)
        return finish(verdict, "covalent_salt_family_" + verdict,
                      category="lithium_salt", family=family)

    if not li_ionic and not li_covalent:
        # Li only as isotope labels or lost in parsing; or Li-free record snuck in
        return finish("reject", "no_lithium_found")

    # [Li+] present but no anion fragment (charge-imbalanced record)
    return finish("review", "unbalanced_ionic_record")


def main():
    counts = Counter()
    reason_counts = Counter()
    family_counts = defaultdict(Counter)
    outputs = {v: [] for v in ("accept", "review", "reject")}

    with open(UNIVERSE_CSV) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} compounds from {UNIVERSE_CSV}")

    for row in rows:
        res = process_row(row)
        counts[res["verdict"]] += 1
        reason_counts[res["reason"]] += 1
        if res["family"]:
            family_counts[res["verdict"]][res["family"]] += 1
        outputs[res["verdict"]].append(res)

    # dedup accepted+review by isotope-stripped InChIKey; among twins
    # prefer a record with a real title over a "CID nnn" placeholder
    # (LiClO4 lives at 23665649 with all its synonyms; 151488 has none),
    # then the lowest CID
    def dedup(rows_):
        placeholder = re.compile(r"CID \d+$")
        best = {}
        for r in sorted(rows_, key=lambda r: (bool(placeholder.fullmatch(r["title"])),
                                              int(r["cid"]))):
            key = r["inchikey"] or r["cid"]
            if key not in best:
                best[key] = r
        return list(best.values())

    fieldnames = ["cid", "title", "mf", "mw", "smiles", "inchikey",
                  "category", "family", "verdict", "reason", "flags"]
    names = {
        "accept": "electrolyte_candidates_accepted.csv",
        "review": "electrolyte_candidates_review.csv",
        "reject": "electrolyte_candidates_rejected.csv",
    }
    for verdict, fname in names.items():
        rows_ = dedup(outputs[verdict]) if verdict != "reject" else outputs[verdict]
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_)
        print(f"{verdict}: {len(outputs[verdict])} rows"
              + (f" -> {len(rows_)} after dedup" if verdict != "reject" else "")
              + f" -> {path}")

    report = os.path.join(OUT_DIR, "funnel_report.txt")
    with open(report, "w") as f:
        f.write(f"Input compounds: {len(rows)}\n\n")
        f.write("Verdicts:\n")
        for v, c in counts.most_common():
            f.write(f"  {v}: {c}\n")
        f.write("\nTop reject/review reasons:\n")
        for r, c in reason_counts.most_common(30):
            f.write(f"  {r}: {c}\n")
        for verdict in ("accept", "review"):
            f.write(f"\n{verdict.capitalize()}ed families:\n")
            for fam, c in family_counts[verdict].most_common():
                f.write(f"  {fam}: {c}\n")
    print(f"Report -> {report}")


if __name__ == "__main__":
    main()
