# Knowledge-graph-based ranking of LiTFSI substitute electrolyte components

Methodology and results for identifying PFAS-free substitutes for LiTFSI
(lithium bis(trifluoromethanesulfonyl)imide) by comparing co-occurrence
neighborhoods in two independent knowledge graphs:

- **Jchem** — weighted Jaccard similarity to LiTFSI over the **MatKG**
  materials-science KG (literature-scale chemistry co-occurrence).
- **Japp** — weighted Jaccard similarity to LiTFSI over an **LLM-extracted,
  ego-centric salt application KG** (salt → application/property/formulation
  relations with source-text provenance).

**Headline result:** among salts ranked in the top 10 of *both* views, after
excluding fluorinated candidates (LiPF6 — the fluorinated incumbent — and
LiOTf, itself a PFAS), the top two substitution candidates are
**LiNO3 and LiClO4** (Li2SO4 third). This is independently confirmed by a
true hold-out review (Wang et al., *npj Materials Sustainability*, 2025;
see §7), which names LiBOB, LiClO4, LiNO3 and cyano-based anions as the
leading fluorine-free substitution routes. LiBOB — in the review's list —
is Japp rank 4 and absent from Jchem's top 10 only because it is rare in
the MatKG corpus.

---

## 1. Candidate universe (stages 0–2)

Replaces the legacy formula-keyed candidate list (13,646 formulas), whose
diagnosed failures were: ~20% second-metal compounds, organolithium
reagents and charged fragments included, formula keying that pooled isomer
synonyms, and a MW 50–250 window that silently excluded LiTFSI itself
(287 Da).

1. `fetch_li_universe.py` (Stage 0) — all Li-containing PubChem compounds,
   MW 24–620 (Entrez esearch + PUG REST property batches; resumable;
   per-window CID cache). **83,740 compounds** with CID, formula, MW,
   charge, SMILES, InChIKey, title. The MW floor is 24 because the first
   fix's floor of 40 silently cut LiF (25.9 Da) — the same failure class
   as the legacy window cutting LiTFSI. A window override
   (`fetch_li_universe.py 24 35`) appends supplemental ranges.

2. `filter_funnel.py` (Stages 1–2) — deterministic RDKit rules, no LLM:
   - element whitelist (As/Al/Sb allowed but flagged `extended_element`)
   - net-cationic fragments rejected; net-anionic kept as `anion_fragment`
   - neutral records must be real Li salts (ionic [Li+] + molecular anion
     or the covalent-drawn equivalent); organolithiums (C–Li) and reagent
     anions (hydride, plain amide, carbanion) rejected
   - anion-family SMARTS allowlist (sulfonimide, sulfonate,
     fluorophosphate, oxalatoborate, azolide, perchlorate, …) decides
     acceptance; plausible-but-unusual families (alkoxide, enolate,
     LiHMDS-likes) go to a review bucket — never silently dropped
   - flags: `pfas_like` (≥ CF2), `hydrate`, `extended_element`
   - dedup by InChIKey; among twins, records with a real title beat
     "CID nnn" placeholders (LiClO4's good record lives at CID 23665649;
     the placeholder twin 151488 has no synonyms), then lowest CID

   Result: **15,926 accepted / 16,355 review / 50,784 rejected** (with
   per-reason counts in `output/funnel_report.txt`).

3. `merge_candidates.py` — accepted salts + 35 curated solvents/additives/
   diluents (`electrolyte_solvents_additives.csv`; solvents contain no Li
   and cannot come from the universe), deduplicated by InChIKey →
   **15,961 components** (`output/final_electrolyte_components.csv`).
   Recall vs the 30 curated reference salts: **26 accepted, 3 review**
   (a photoinitiator phosphinate, an enolate, LiHMDS — defensible calls),
   **1 missing** (tetralithium quinone; not in PubChem).

   Note: `component_id`s are positional and shift when the funnel reruns —
   key anything durable on InChIKey or CID.

## 2. Synonym layer (stages 3–4)

The matching vocabulary is keyed by **CID/InChIKey, never by formula**
(formula keying pooled isomer synonyms in the legacy pipeline).

4. `harvest_synonyms.py` (Stage 3) — PubChem synonyms per CID, resumable.
   Curated-seed CIDs resolved from InChIKey. Harvests every universe CID
   sharing a component's InChIKey (sibling-CID union), because PubChem
   keeps duplicate records per structure and dedup can keep a synonym-less
   twin. **16,462 CIDs harvested; 10,153 with ≥1 synonym.**

5. `build_synonym_index.py` (Stage 4) — cleaning, canonicalization and the
   ambiguity audit:
   - registry/vendor IDs dropped by pattern (CAS, SCHEMBL, UNII incl. bare
     10-char codes, EINECS, DSSTox, CID numbers, catalog codes, …)
   - generic-word blocklist checked against both the normalized string and
     the token squash ("Li+" squashes to "li")
   - **promiscuity rule**: a normalized synonym claimed by >3 InChIKey
     connectivity blocks is dropped everywhere — data-driven, catches
     "trilithium", "hplc", "cp" without list maintenance
   - **reserved curated names**: PubChem adduct/solution records graft
     solvent names onto salts ("THF" as a Lithium Chloride synonym);
     curated seed names/abbrevs may only be claimed by the seed itself
   - trailing parenthetical abbreviations become synonyms of their own
     ("… (LISO3CF3)") but only cased/numbered forms — "(powder)",
     "(natural)" are qualifiers, and purity-qualified vendor strings are
     skipped entirely
   - `manual_synonyms.csv` adds spellings the KGs use but PubChem lacks:
     LiN(CF3SO2)2, LiCF3SO3, LiTf, LiI, LiDFOB, LiNTf2, LiTFSA
   - `manual_blocklist.csv` kills known-wrong PubChem entries (DMF listed
     under ethylene carbonate; "GMP" the manufacturing grade under DMSO)
   - match tiers: full names get containment rights; short strings
     (EC, LiF, Et2O, O(Et)2) are exact-match-only

   Result: **20,303 kept synonyms** across **14,538 matchable components**
   (audit trail in `output/synonym_audit.txt` and
   `output/synonyms_dropped.csv` with per-reason counts).

### Matching rules (`textnorm.py`, shared by all scorers)

Both synonym and entity strings are reduced to a **squashed token string**
with token-boundary offsets (NFKC, casefold, sulph→sulf, alnum token
extraction). A synonym matches an entity iff its squash occurs in the
entity's squash **starting at a token start and ending at a token end**
(the word-boundary rule); exact = the full-span case. This survives
MatKG's inserted spaces, title-casing and token fusion
("Lithium Bis(Trifluoromethanesulfonyl) Imide" — the literal string
"LiTFSI" does not exist in MatKG's subgraph). Matches of ≤4 squash
characters must additionally agree on case: SN is succinonitrile, Sn is
tin; FEC is not FeC; PC is not the phthalocyanine Pc.

## 3. Jchem — MatKG weighted Jaccard (stage 5)

6. `jchem_weighted_jaccard.py` — MatKG (`SUBRELOBJ.csv`, 5,396,084
   triples) filtered to chemistry-relevant relations
   (CHM-CHM, CHM↔PRO, CHM↔APL, CHM↔DSC): **1,727,194 triples over
   39,374 unique entities**. Component synonyms are matched to entities
   once (per-synonym guard: >50 matched entities = excluded and
   reported); each component's **ego profile** is the union of its
   matched entities' neighbors, weighted by summed triple `Count` in both
   directions, excluding the component's own entities.

   **Jchem score — the metric of record for every ranking below**
   (column `jaccard_weighted` in `output/jchem_results.csv`) = weighted
   Jaccard of the component's ego profile A against LiTFSI's profile B:

   ```
   Jchem = Σ_{j ∈ A∩B} min(w_A(j), w_B(j))
         / Σ_{j ∈ A∪B} max(w_A(j), w_B(j))
   ```

   where `w_X(j)` = summed MatKG triple `Count` linking X to neighbor j
   over both directions. (Summing `min` over the intersection rather
   than the union is equivalent — `min` is 0 for any non-shared
   neighbor.) The unweighted form `|A∩B| / |A∪B|` is written to the same
   CSV as `jaccard_unweighted` for reference only; it is **not** ranked
   on, and neither are the size-corrected metrics of §10. Ranking is
   `jaccard_weighted` descending; the salts-only table filters
   `category == lithium_salt` and drops LiTFSI itself (self-similarity
   1.0).

   LiTFSI reference profile: 174 neighbors. **50 of 15,961 components
   match MatKG at all** — that sparsity is the intended relevance gate (the ~10k pharma
   lithium carboxylates never appear in a materials corpus). The report
   includes a short-abbreviation sense audit (e.g. PES also means
   polyethersulfone; MA means methylammonium in perovskite papers).

## 4. Japp — application-KG (salt KG) ranking

7. `appkg_salt_ranking.py` — the LLM-extracted salt KG
   (`salt_analysis_output/canonical/salt_relationships_canonical_with_source.csv`,
   27,993 triples, 302 unique subjects, 10 relations) is **ego-centric**:
   subjects are salt mentions; objects are applications, properties,
   solvents and formulations extracted from paper paragraphs.

   Every subject mention is remapped onto InChIKey components with the
   Stage-4 index, healing the old name-based canonicalization:
   - **fragmentation healed** — LiTFSI had been split across five
     canonical identities (LiTFSI / LiNTf2 / LiTFSA / Li[NTf2] /
     LiN(CF3SO2)2; +292 mentions recovered), LiDFOB across three
     (LiODFB / Li-DFOB / full name), lithium triflate across three
     (including a wrong PubChem name-resolution of LiSO3CF3 to
     "trifluoro(sulfinatooxy)methane")
   - generic mentions ("Li salt" → pseudo-entity "lithium") dropped, not
     pooled; formulation phrases ("PEO-LiTFSI solid polymer electrolyte")
     excluded and reported
   - **novel salts with no PubChem record survive as named
     pseudo-components** (LiTDI, LiFOP, LiFNFSI, LiFPFSI, LiSFSI,
     LiPDSFSI, …) — a name-resolution-only pipeline can never rank these

   Mapping coverage: **90% of triples** (25,170) land on InChIKey
   components; 2,134 on novel pseudo-components; the remainder are
   generic/phrase/rare mentions (fully itemized in
   `output/appkg_subject_mapping.csv`).

   **Japp score** (column `jaccard_weighted` in
   `output/appkg_salt_ranking.csv`) = the *same* weighted-Jaccard formula
   as Jchem, Σ min(w) / Σ max(w), on a different graph and a different
   weight:

   ```
   Japp = Σ_{o ∈ A∩B} min(w_A(o), w_B(o))
        / Σ_{o ∈ A∪B} max(w_A(o), w_B(o))
   ```

   where the profile of a salt is its **subject-side (directed) ego set**
   — the `object_canonical` values of triples whose subject maps to that
   salt — and `w_X(o)` = how many times X is stated with object o. No
   reverse direction is added (unlike Jchem, this KG is ego-centric by
   construction: subjects are always salt mentions).
   LiTFSI profile: 8,835 mentions over 4,676 distinct neighbors.

## 5. Results

### Jchem top 10 (all components; LiTFSI itself = 1.0, excluded)

| # | Component | Type | Jchem |
|---|---|---|---|
| 1 | Lithium nitrate (LiNO3) | salt | 0.458 |
| 2 | Fluoroethylene carbonate (FEC) | additive | 0.408 |
| 3 | Lithium carbonate (Li2CO3) | salt | 0.369 |
| 4 | Tetraglyme (TEGDME/G4) | solvent | 0.347 |
| 5 | Propylene carbonate (PC) | solvent | 0.329 |
| 6 | 1,3-Dioxolane (DOL) | solvent | 0.314 |
| 7 | Ethylene carbonate (EC) | solvent | 0.310 |
| 8 | Triglyme (G3) | solvent | 0.297 |
| 9 | Diethyl carbonate (DEC) | solvent | 0.292 |
| 10 | Dimethyl carbonate (DMC) | solvent | 0.249 |

### Jchem top 10, salts only

| # | Salt | Family | Jchem |
|---|---|---|---|
| 1 | LiNO3 | nitrate | 0.458 |
| 2 | Li2CO3 | carbonate | 0.369 |
| 3 | Li2SO4 | sulfate | 0.235 |
| 4 | LiClO4 | perchlorate | 0.233 |
| 5 | LiF | halide | 0.224 |
| 6 | Li3PO4 | phosphate | 0.221 |
| 7 | LiPF6 | hexafluorophosphate | 0.221 |
| 8 | LiOTf (LiCF3SO3) | sulfonate | 0.178 |
| 9 | LiCl | halide | 0.164 |
| 10 | Lithium citrate | carboxylate | 0.116 |

(Next battery-relevant salts, same metric: LiBr 0.090, LiBOB 0.061,
LiFSI 0.028, LiBF4 0.015, LiAsF6 0.009, LiI 0.008, LiOAc 0.006, LiDFOB
0.005. Every entry after LiBr has a MatKG profile of ≤ 13 neighbors, so
these tail scores are coverage artifacts rather than dissimilarity —
see §10. Two pharma carboxylates that survived the funnel
(L-argininosuccinic acid 0.011, an adenylic-acid anhydride 0.006) are
omitted here but present in `jchem_results.csv`.)

### Japp top 10 (salt KG; LiTFSI = 1.0, excluded)

| # | Salt | Japp |
|---|---|---|
| 1 | LiPF6 | 0.0800 |
| 2 | LiFSI | 0.0743 |
| 3 | LiClO4 | 0.0570 |
| 4 | LiBOB | 0.0308 |
| 5 | LiBF4 | 0.0268 |
| 6 | LiNO3 | 0.0222 |
| 7 | LiDFOB | 0.0221 |
| 8 | LiOTf | 0.0189 |
| 9 | Li2SO4 | 0.0135 |
| 10 | LiFTFSI (novel; no PubChem record) | 0.0113 |

The absolute scales are **not comparable across the two KGs** (§9); the
ordering is the signal.

## 6. Consensus substitution candidates

Five salts appear in **both** top-10 lists:

| Salt | Jchem rank (score) | Japp rank (score) | Fluorine status |
|---|---|---|---|
| LiNO3 | 1 (0.458) | 6 (0.0222) | fluorine-free |
| LiClO4 | 4 (0.233) | 3 (0.0570) | fluorine-free |
| LiPF6 | 7 (0.221) | 1 (0.0800) | fluorinated incumbent |
| LiOTf | 8 (0.178) | 8 (0.0189) | PFAS (CF3) |
| Li2SO4 | 3 (0.235) | 9 (0.0135) | fluorine-free |

For PFAS substitution, LiPF6 (the fluorinated incumbent being co-regulated
alongside PFAS binders) and LiOTf (itself a PFAS) are excluded, leaving
LiNO3, LiClO4 and Li2SO4 — of which **LiNO3 and LiClO4 are the top two by
combined rank**. The two views also complement each other's blind spots:
Jchem's top salts reflect materials-literature ubiquity (LiNO3/Li2CO3 as
additives and SEI components), while Japp — built from application
contexts — puts the true electrolyte-salt substitutes (LiPF6, LiFSI,
LiClO4, LiBOB) on top.

## 7. Hold-out validation

**Hold-out:** Wang, Qing, Zhang & Li, "Fluorine-free electrolytes for
sustainable lithium batteries: a review," *npj Materials Sustainability*
(2025), DOI `10.1038/s44296-025-00074-8`
(https://www.nature.com/articles/s44296-025-00074-8). The review discusses
the 2023 ECHA PFAS restriction proposal (covering LiTFSI and the PVDF/PTFE
binders) and identifies **LiBOB, LiClO4, LiNO3 and cyano-based anions** as
the leading fluorine-free substitution routes, including for LFP cathodes.

**Verified as a true hold-out** (none of the ranking inputs contain it):
- DOI string (`s44296-025-00074-8`, and any `s44296`): **0 occurrences**
  in the battery-KG source corpus (`battery_kg/t2br_data/
  raw_paragraphs_v2.xlsx`, checked via sheet shared-strings), in the
  battery triples DOI column (31,640 unique paragraph-level DOI ids,
  2,088 distinct papers), and in all extraction
  JSONs (`battery_extraction_results.json`,
  `battery_salt_extraction_results.json`,
  `salt_ontology_extraction_results_v2.json`,
  `enhanced_salt_extraction_raw_*.json`). The only "44296" substrings are
  incidental floats (processing times) and row indices.
- Title text ("Fluorine-free electrolytes for sustainable …"):
  **0 occurrences** in any corpus file.
- MatKG's corpus was assembled and released years before 2025 and cannot
  contain the review.

**Agreement:** the review's fluorine-free salt candidates and the
consensus ranking coincide — LiNO3 and LiClO4 (our consensus top two) are
both named, and LiBOB (Japp rank 4; absent from Jchem's top 10 only
because MatKG contains just 13 LiBOB-adjacent entities) is the review's
third named salt. Since the review shares no source documents with either
KG, this constitutes independent confirmation that the KG-derived
neighborhood similarity recovers chemically meaningful substitution
candidates.

## 8. Battery-KG cross-check

8. `appkg_battery_audit.py` — the general (non-ego-centric) battery KG
   (`battery_ontology_triples_with_doi_clean_*.csv`: 316,846 triples,
   129 relations, 31,640 paragraph-level DOI ids from 2,088 distinct
   papers, 120,779 unique entities — dominated by
   formulation phrases). Entity mapping: **241 exact-tier** entities
   (LiTFSI alone unifies 12+ surface forms: 'LITFSI', 'Li NTf2',
   'Li [NTf2]', 'Li+, TFSI–', 'Li-TFSI', 'LiTFSA', …) across 62
   components, plus **1,824 composite-tier** entities
   ('PEO/LiTFSI electrolytes') counted per salt but deliberately not
   merged into profiles — formulation properties belong to the
   formulation. Ego-profile ranking vs LiTFSI reproduces the Japp salt
   ordering (LiPF6, LiFSI, LiClO4, LiNO3, LiBOB, LiDFOB, LiBF4 top,
   interleaved with solvents since this graph does not separate them),
   cross-checking the salt-KG result on an independent extraction. This
   KG carries per-triple DOI provenance, enabling human-readable candidate
   profiles with source attribution (script 9).

9. `appkg_candidate_profiles.py` — per-claim DOI attribution for the
   consensus candidates (LiNO3, LiClO4, Li2SO4, LiBOB) plus the
   fluorinated consensus references (LiPF6, LiOTf). A **claim** is
   (direction, relation, squash-normalized neighbor) aggregated over
   exact-tier triples only; every claim carries the full list of
   supporting papers. The KG's `doi` column is a *paragraph* id
   (`…_p2_8`); papers are counted after stripping the `_p<i>_<j>` suffix
   (31,640 paragraph ids → 2,088 distinct papers), so a claim repeated
   across paragraphs of one paper counts as one paper of evidence.
   18,919 evidence triples total (including the LiTFSI reference
   profile); **corroborated** (≥ 2 papers) claims —
   LiPF6 653 (1,342 papers), LiClO4 77 (111), LiBOB 63 (40), LiNO3 60
   (159), Li2SO4 24 (35), LiOTf 8 (22). Signature corroborated claims:
   **LiNO3** — co-salt/additive alongside LiTFSI in DOL/DME ether
   electrolytes (96×/81 papers), improves coulombic efficiency and
   suppresses dendrites; **LiClO4** — PEO / PC / acetonitrile
   electrolytes, ionic-conductivity claims; **LiBOB** — SEI formation,
   capacity retention, suppresses Fe dissolution from LiFePO4;
   **LiPF6** — the carbonate-electrolyte workhorse (EC/DMC 435×/313
   papers); **Li2SO4** — almost entirely aqueous-electrolyte and
   synthesis-bath contexts (dissolved in water 44×/19 papers), which
   weakens it as an aprotic drop-in substitution candidate; **LiOTf** —
   thin evidence base (22 papers). LiTFSI itself is profiled the same
   way (5,635 exact-tier mentions) so downstream comparisons cite the
   same evidence layer. Outputs: `candidate_claims.csv` (one
   row per claim with all DOIs), `candidate_evidence.csv` (one row per
   triple with source paragraph), `candidate_profiles_report.txt`.

10. `llm_candidate_commentary.py` — human-readable substitution
    commentary per candidate vs LiTFSI, generated by
    Mistral-Small-3.1-24B (RITS) from the script-9 claims. Successor to
    `battery_kg/llm_querying_of_candidates.py` (which prompted from
    salt-KG triples without provenance): here the knowledge profiles
    handed to the model are corroboration-ranked claims each carrying
    its paper DOIs, and the prompt makes citations mandatory — every
    corpus-derived statement must end with its bracketed DOI(s), and
    anything from the model's prior must be tagged `[general
    knowledge]`, so a reader can tell corpus-backed pros/cons from LLM
    prior. Eight sections per candidate (similarity, differences,
    advantages, limitations, application suitability, replacement
    feasibility, research gaps, overall). Outputs:
    `candidate_commentary_report.txt`, `candidate_commentary.json`
    (raw prompts + responses). Needs `RITS_API_KEY` and `RITS_BASE_URL`.

## 9. Why Japp absolute scores are low (diagnostics)

Measured on LiTFSI's salt-KG profile (8,835 mentions, 4,676 distinct
neighbors):
- **85% of neighbors are singletons** (appear exactly once in the KG);
- **53% contain digits** — paper-specific values ("1.2×10⁻⁴ S cm⁻¹ at
  60 °C") that no other salt can share;
- only **20% (941)** of LiTFSI's neighbors co-occur with *any* other
  salt — a hard ceiling on any intersection;
- profile-size asymmetry caps the maximum achievable unweighted Jaccard:
  LiPF6 0.44, LiFSI 0.22, LiClO4 0.11, LiBOB 0.11, LiBF4 0.07. Against
  these ceilings the observed overlaps are substantial (LiClO4 reaches
  50% of its ceiling; LiBF4 37%; LiFSI 34%; LiBOB 28%).

MatKG, by contrast, is a canonicalized shared vocabulary (39k entities
reused across compounds; LiTFSI profile only 174 neighbors), so raw
overlaps run ~5× higher. Surface-variant normalization of the neighbor
vocabulary would merge only ~4% of keys (11,163→10,728) — the sparsity is
genuine phrasing diversity, not formatting noise.

## 10. Size-corrected Jchem companion ranking

11. `jchem_normalized_ranking.py` — extra analysis on top of
    `jchem_results.csv` (does **not** modify the primary Jchem outputs).
    Motivation: the weighted Jaccard punishes small MatKG profiles —
    LiBOB's 13 neighbors are a *perfect subset* of LiTFSI's 174, so its
    unweighted Jaccard (13/174 = 0.075) is already the ceiling for a
    13-neighbor profile, yet it ranks 30/50. Two set-size-corrected
    companion columns:
    - **ceiling-normalized Jaccard**: `jaccard_unweighted` divided by
      the best Jaccard attainable given the two profile sizes
      (min/max); 1.0 = "scored its ceiling" (one profile subsumes the
      other);
    - **overlap coefficient** (Szymkiewicz–Simpson):
      |A∩B| / min(|A|,|B|).

    Both metrics inflate tiny profiles (any 2-neighbor component whose
    neighbors sit in LiTFSI's set scores 1.0), so profiles with < 10
    neighbors are flagged `[thin]` and the report shows top-10 lists
    with and without them. Headline (≥ 10-neighbor profiles): **LiBOB
    is rank 1 by both metrics** (1.0 — perfect subset), LiOTf 2
    (overlap 0.97, 29/30 neighbors shared), then EMC, **LiClO4** and
    **LiPF6** (overlap 0.879 each), LiF, DMC/AN/THF/VC/PC/DOL
    interleaved. The size correction confirms §7's reading: LiBOB's low
    primary-Jchem rank is a MatKG-coverage artifact, and the
    fluorine-free consensus salts (LiBOB, LiClO4) sit at the top once
    profile size is controlled for. Thin-profile salts to note, not
    rank: LiFSI (5/5 shared), LiBF4 (4/4), LiDFOB (2/2). Outputs:
    `jchem_normalized_ranking.csv`, `jchem_normalized_report.txt`.

## 11. Upstream ontology and canonicalization (feeds Japp, §4/§8)

Japp and the battery-KG cross-check (§8) consume triples extracted by an
LLM (Mistral, via `battery_kg/`) against two ontologies, then cleaned by a
canonicalization pass (`battery_ontology_analysis/battery_ontology_cleaning_with_doi.py`)
before this pipeline's InChIKey remapping (§4) runs on top.

- **Battery-system ontology** (ego-network-agnostic — captures the full
  battery-system context per paragraph, not just one compound's
  neighborhood), organized into 5 functional categories: (i) composition &
  formulation, e.g. `(EC/DMC/EMC, CONTAINS, LiPF6)`; (ii) stability &
  operating limits, e.g. `(BATTERY_CELL, STABLE IN WINDOW, 2.5–4.0 V)`;
  (iii) device & cell context, e.g. `(BATTERY_CELL, HAS FORM FACTOR, pouch
  cell)`; (iv) properties & measurements, e.g. `(PP, HAS PROPERTY,
  dimensional stability)`; (v) causal/molecular-level, e.g. `(metallic
  lithium, DEPOSITS ON, graphite)`. Raw extraction: **316,846 triples, 131
  relation types** (`battery_ontology_triples_with_doi_clean_20250819_214710.csv`).
- **Salt-level ontology** (lighter-weight, ego-centric on the salt as
  subject; this is the source for Japp's `salt_relationships_canonical_with_source.csv`),
  10 relation types: `CO OCCURS WITH`, `DISSOLVED IN`, `USED IN`, `HAS
  CONCENTRATION IN ELECTROLYTE`, `HAS PROPERTY`, `HAS ROLE`, `AFFECTS
  PROPERTY OF`, `STABLE IN WINDOW`, `OPERATING TEMP RANGE`, `HAS CHEMICAL
  STRUCTURE`. Raw extraction: **27,993 triples, 302 unique subjects**, e.g.
  `(LiPF6, DISSOLVED IN, 1:1 ethylene carbonate and dimethyl carbonate)`.
- **Canonicalization** (lowercase normalization, alias-table abbreviation
  expansion, PubChem CID-based dedup) resolves synonym proliferation
  ("LiTFSI" / "Li-TFSI" / "LiN(SO2CF3)2" / "bis(trifluoromethanesulfonyl)imide
  lithium") before it inflates apparent entity diversity and suppresses
  Jaccard scores: batteryKG **316,846 → 231,658 triples (−26.9%)**; saltKG
  **27,993 → 14,123 triples (−49.5%)**, with subject-entity count reduced
  302 → 288 via 710 entity mappings (326 automated PubChem lookups, 311
  manual). Full DOI-level provenance is preserved per triple throughout,
  which is what makes the per-claim DOI attribution in §8 (script 9)
  possible. Note: this canonicalization pass is upstream of and distinct
  from this pipeline's own InChIKey-based synonym healing (§4), which
  operates on the *already*-canonicalized saltKG CSV.

## 12. Limitations and known hazards

- **MatKG ubiquity bias:** Jchem rewards compounds that are common across
  materials literature in any role (LiNO3, Li2CO3, LiF as additives, SEI
  components, precursors), and starves genuinely good but rare substitutes
  (LiFSI: 5 MatKG neighbors; LiBOB: 13). Japp compensates.
- **Abbreviation sense collisions** survive in free text even after the
  cased rule ('PES' = prop-1-ene-sultone or polyethersulfone; 'TMP',
  'DMC' have non-battery senses; 'MA+' is methylammonium). Every short
  match is itemized in the Jchem report for audit.
- **Japp absolute scores** are structurally deflated (§9); compare ranks,
  not magnitudes, and never across KGs.
- The salt-KG object side is raw LLM phrasing; consolidating it
  (polarity-aware property canonicalization) would raise absolute scores
  but is deliberately not applied in the current numbers.
- Isotopologues (6Li/7Li records) are separate components by design;
  the ambiguity rules operate on InChIKey connectivity blocks so shared
  names survive.
- The accepted set still contains ~10k pharma lithium carboxylates; they
  are retained because the family column makes exclusion trivial and the
  KG-match gate prunes them naturally (only 2 carboxylates score in
  Jchem).

## 13. Reproduction

```
python chemical_kg/fetch_li_universe.py        # stage 0 (+ optional "24 35" style window)
python chemical_kg/filter_funnel.py            # stages 1-2
python chemical_kg/merge_candidates.py
python chemical_kg/harvest_synonyms.py         # stage 3 (network; resumable)
python chemical_kg/build_synonym_index.py      # stage 4
python chemical_kg/jchem_weighted_jaccard.py   # stage 5 -> Jchem
python chemical_kg/appkg_salt_ranking.py       # -> Japp
python chemical_kg/appkg_battery_audit.py      # battery-KG cross-check
```

Needs rdkit, requests, pandas; login-node friendly. Editable inputs:
`electrolyte_solvents_additives.csv` (curated seeds),
`manual_synonyms.csv` (alias additions), `manual_blocklist.csv`
(known-wrong synonym kills).

## 14. Output inventory (`chemical_kg/output/`)

| File | Contents |
|---|---|
| `li_universe.csv` | Stage-0 PubChem Li universe (83,740) |
| `electrolyte_candidates_{accepted,review,rejected}.csv`, `funnel_report.txt` | funnel verdicts + per-reason counts |
| `final_electrolyte_components.csv` | 15,961 merged components |
| `synonyms_raw.jsonl`, `curated_cid_map.json` | harvested synonyms per CID |
| `synonyms_clean.csv`, `synonyms_dropped.csv`, `synonym_audit.txt` | matching vocabulary + audit |
| `jchem_results.csv`, `jchem_matched_entities.csv`, `jchem_report.txt` | Jchem ranking, match provenance, sense audit |
| `appkg_subject_mapping.csv`, `appkg_salt_ranking.csv`, `appkg_report.txt` | Japp mention mapping + ranking |
| `appkg_battery_entity_mapping.csv`, `appkg_battery_ranking.csv`, `appkg_battery_report.txt` | battery-KG audit + cross-check |
| `candidate_claims.csv`, `candidate_evidence.csv`, `candidate_profiles_report.txt` | per-claim DOI-attributed candidate profiles (§8, script 9) |
| `candidate_commentary_report.txt`, `candidate_commentary.json` | LLM substitution commentary with DOI citations (§8, script 10) |
| `jchem_normalized_ranking.csv`, `jchem_normalized_report.txt` | size-corrected Jchem companion ranking (§10, script 11) |
