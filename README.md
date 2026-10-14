# PFAS-SPARK: Substitution Pipeline using Application-level and Chemical-level Relational Knowledge Graphs for PFAS Components in Battery Electrolytes

This repository identifies PFAS-free substitutes for **LiTFSI** (lithium
bis(trifluoromethanesulfonyl)imide) in battery electrolytes by comparing
co-occurrence neighborhoods across two independent knowledge graphs — a
literature-scale **chemical KG** (MatKG) and an **LLM-extracted application
KG** built from battery-electrolyte paper paragraphs — then cross-validates
the result against a battery-system-level ontology and an independent
literature hold-out.

**Note**: Large json/csv/xlsx outputs are not tracked in this repository.
See [Setup](#setup) for the environment variables the LLM steps need.

---

## Headline result

Among salts ranked in the top 10 of *both* KG views (**Jchem** — MatKG
weighted Jaccard; **Japp** — application-KG weighted Jaccard), after
excluding fluorinated candidates (LiPF6, the fluorinated incumbent, and
LiOTf, itself a PFAS), the top two substitution candidates are **LiNO3 and
LiClO4** (Li2SO4 third). This is independently confirmed by a true hold-out
review (Wang et al., *npj Materials Sustainability*, 2025 — §6), which
names LiBOB, LiClO4, LiNO3 and cyano-based anions as the leading
fluorine-free substitution routes. LiBOB (absent from Jchem's top 10 only
because it's rare in MatKG) is Japp rank 4.

---

## Which artefact is current

Several results exist in more than one version. **Cite the "current" column;
the superseded files are kept for provenance only.**

| Result | Current script | Current output |
|---|---|---|
| Jchem ranking (§1.3, §4) | `chemical_kg/jchem_weighted_jaccard.py` | `chemical_kg/output/jchem_results.csv` |
| Japp ranking (§3, §4) | `chemical_kg/appkg_salt_ranking.py` | `chemical_kg/output/appkg_salt_ranking.csv` |
| Candidate profiles (§7.2) | `chemical_kg/appkg_candidate_profiles.py` | `candidate_claims.csv`, `candidate_evidence.csv`, `candidate_profiles_report.txt` |
| LLM commentary (§8) | `chemical_kg/llm_candidate_commentary.py` | `candidate_commentary_report_v2.txt` / `_v2.json` |
| Judge evaluation (§12.1) | `battery_kg/llm_triple_evaluation.py` | `evaluation_output/evaluation_combined_*.csv` |
| Human arm (§12.3) | `battery_kg/analyze_human_agreement.py` | `human_worksheet_20260814_172329_mrigi_judge_v2.xlsx` |

**Superseded, do not cite:** the root-level `jaccard_calculation*.py`
scripts and `battery_kg/jaccard_calculation_application_kg.py` (all
predate the InChIKey-remapped Japp of §3.1);
`battery_kg/llm_querying_of_candidates.py` (prompted from salt-KG triples
with no provenance); and `candidate_commentary_report.txt` / `.json`
without the `_v2` suffix (allowed the model untethered
`[general knowledge]` claims and a feasibility score — see §8).

### Which KG version feeds which step

Each KG exists in more than one form, and different steps deliberately use
different ones. Verified from the input paths in each script.

| Step | Input file | Rows |
|---|---|---|
| **Japp ranking** (§3, §4) | `salt_analysis_output/canonical/salt_relationships_canonical_with_source.csv` | **27,993** entity-canonicalized |
| Battery-KG cross-check (§7.1) | `battery_ontology_triples_with_doi_clean_20250819_214710.csv` | **316,841** raw valid triples |
| Candidate evidence profiles (§7.2) | same raw file as §7.1 | **316,841** |
| Judge evaluation, batteryKG (§12) | `battery_ontology_analysis/relationships_data_canonical_with_doi_20250820_140958.csv` | **231,658** canonicalized |
| Judge evaluation, saltKG (§12) | same 27,993 file as Japp | **27,993** |

Two points that are easy to get wrong:

- **Japp does *not* run on the triple-deduplicated saltKG**
  (`salt_relationships_canonical.csv`, 14,123 rows). It cannot: Japp
  weights are mention counts, and the 27,993-row file contains only 15,305
  distinct (subject, relation, object) triples — 2,336 of them repeat, one
  as many as 777 times. Deduplicating would flatten every weight to 1 and
  collapse the weighted Jaccard onto the unweighted one. The §3.1 mapping
  figures confirm the input: 25,035 + 135 = 25,170 mentions land on
  InChIKey components (89.9% of 27,993) and 2,134 on pseudo-components.
- **The battery KG appears in two different forms.** §7's cross-check and
  evidence profiles read the **raw 316,841** triples with their own
  exact-tier InChIKey mapping; the §12 judge sample reads the
  **canonicalized 231,658**. This is intentional — §7 needs the unmerged
  surface forms for provenance — but any statement of the form "we used
  the batteryKG" has to say which.

### Models used

All via the RITS API (IBM). **"Mistral" alone is ambiguous in this repo —
two different Mistral models are used**, so the tables below and the
evaluation sections always qualify it.

| Task | Model ID | Short name | Script |
|---|---|---|---|
| Battery-ontology extraction (§2.1) | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral-Small-3.1-24B | `battery_kg/ontology_extraction_triples_battery.py` |
| Salt-ontology extraction (§2.2) | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral-Small-3.1-24B | `battery_kg/ontology_extraction_triples_salt.py`, `enhanced_salt_ontology_extraction_with_doi.py` |
| Judge 1 (§12.1) | `mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` | **Mistral-Large-3-675B** | `battery_kg/llm_triple_evaluation.py` |
| Judge 2 (§12.1) | `meta-llama/llama-3-1-405b-instruct-fp8` | LLaMA-3.1-405B | `battery_kg/llm_triple_evaluation.py` |
| Substitution commentary (§8) | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral-Small-3.1-24B | `chemical_kg/llm_candidate_commentary.py` |
| *(superseded)* candidate querying | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral-Small-3.1-24B | `battery_kg/llm_querying_of_candidates.py` |

Note for the write-up: **both judges are different models from the
extractor**, and both are far larger (675B and 405B against the
extractor's 24B). The triple evaluation is therefore not self-assessment —
worth stating explicitly, since an extractor grading its own output would
be the obvious objection. Judges run at temperature 0, seed 42; the
commentary runs at temperature 0.1.

---

## Overview

Two independent pipelines, cross-validated against each other and against
an external hold-out:

1. **Chemical Knowledge Graph Pipeline** (`chemical_kg/`) — molecular-level
   similarity using MatKG's materials-science literature co-occurrence
   graph (**Jchem**, §1–2).
2. **Application Knowledge Graph Pipeline** (`battery_kg/`) — LLM-extracted
   (Mistral-Small-3.1-24B, via RITS) battery-system and salt-application
   relationships
   from paper paragraphs, with full DOI provenance (**Japp**, §3–4).
3. **Cross-validation** — consensus ranking (§5), a true literature
   hold-out (§6), a general (non-ego-centric) battery-KG cross-check with
   per-claim DOI-attributed candidate profiles (§7), and LLM-generated,
   citation-mandatory substitution commentary (§8).
4. **BONUS: Advanced link-prediction analysis** (`battery_ontology_analysis/`,
   §11) — relationship-prediction benchmarking over the canonicalized
   battery ontology.

---

## 1. Chemical Knowledge Graph Pipeline — candidate universe (`chemical_kg/`)

### 1.1 Candidate universe (stages 0–2)

Replaces a legacy formula-keyed candidate list (13,646 PubChem formulas,
MW 50–250 Da) whose diagnosed failures were: ~20% second-metal compounds,
organolithium reagents and charged fragments included, formula keying that
pooled isomer synonyms, and an MW window that silently excluded LiTFSI
itself (287 Da).

- **`fetch_li_universe.py`** (Stage 0) — all Li-containing PubChem
  compounds, MW 24–620 (Entrez esearch + PUG REST property batches;
  resumable; per-window CID cache). **83,740 compounds** with CID,
  formula, MW, charge, SMILES, InChIKey, title. MW floor is 24 because a
  floor of 40 silently cut LiF (25.9 Da) — the same failure class as the
  legacy window cutting LiTFSI.
- **`filter_funnel.py`** (Stages 1–2) — deterministic RDKit rules, no LLM:
  element whitelist (As/Al/Sb allowed but flagged `extended_element`);
  net-cationic fragments rejected, net-anionic kept as `anion_fragment`;
  neutral records must be real Li salts (organolithiums and reagent anions
  rejected); anion-family SMARTS allowlist (sulfonimide, sulfonate,
  fluorophosphate, oxalatoborate, azolide, perchlorate, …) decides
  acceptance, with plausible-but-unusual families routed to a review
  bucket rather than silently dropped; flags for `pfas_like` (≥ CF2),
  `hydrate`, `extended_element`; dedup by InChIKey (real-titled record
  beats "CID nnn" placeholder, then lowest CID).
  Result: **15,926 accepted / 16,355 review / 50,784 rejected**
  (`output/funnel_report.txt`).
- **`merge_candidates.py`** — accepted salts + 35 curated
  solvents/additives/diluents, deduplicated by InChIKey →
  **15,961 components** (`output/final_electrolyte_components.csv`).

`component_id`s are positional and shift when the funnel reruns — key
anything durable on InChIKey or CID.

### 1.2 Synonym layer (stages 3–4)

The matching vocabulary is keyed by **CID/InChIKey, never by formula**
(formula keying pooled isomer synonyms in the legacy pipeline).

- **`harvest_synonyms.py`** (Stage 3) — PubChem synonyms per CID,
  resumable; harvests every universe CID sharing a component's InChIKey
  (sibling-CID union), because PubChem keeps duplicate records per
  structure. **16,462 CIDs harvested; 10,153 with ≥1 synonym.**
- **`build_synonym_index.py`** (Stage 4) — cleaning, canonicalization, and
  an ambiguity audit: registry/vendor ID stripping (CAS, SCHEMBL, UNII,
  EINECS, DSSTox, …); generic-word blocklist on both normalized string and
  token squash; a data-driven **promiscuity rule** (a synonym claimed by
  >3 InChIKey connectivity blocks is dropped everywhere); reserved
  curated names (PubChem adduct/solution records can't graft solvent
  names onto salts); trailing parenthetical abbreviations become
  synonyms; `manual_synonyms.csv` / `manual_blocklist.csv` for
  KG-specific spellings and known-wrong PubChem entries; match tiers
  (short strings like EC, LiF are exact-match-only).
  Result: **20,303 kept synonyms** across **14,538 matchable components**
  (`output/synonym_audit.txt`, `output/synonyms_dropped.csv`).

**Matching rule** (`textnorm.py`, shared by all scorers): both synonym and
entity strings are reduced to a squashed token string (NFKC, casefold,
sulph→sulf, alnum tokens); a synonym matches iff its squash occurs in the
entity's squash at token boundaries. This survives MatKG's inserted
spaces, title-casing and token fusion ("Lithium Bis(Trifluoromethanesulfonyl)
Imide" — the literal string "LiTFSI" never appears in MatKG). Matches
≤4 squash characters must additionally agree on case (SN = succinonitrile,
Sn = tin; FEC ≠ FeC; PC ≠ phthalocyanine Pc).

### 1.3 Jchem — MatKG weighted Jaccard (stage 5)

**`jchem_weighted_jaccard.py`** — MatKG (`SUBRELOBJ.csv`, 5,396,084
triples) filtered to chemistry-relevant relations (CHM-CHM, CHM↔PRO,
CHM↔APL, CHM↔DSC): **1,727,194 triples over 39,374 unique entities**.
Each component's ego profile is the union of its matched entities'
neighbors, weighted by summed triple `Count` in both directions.

**Jchem score — the metric of record for every ranking in §4** (column
`jaccard_weighted` in `chemical_kg/output/jchem_results.csv`) = weighted
Jaccard of the component's ego profile A against LiTFSI's profile B:

```
Jchem = Σ_{j ∈ A∩B} min(w_A(j), w_B(j))
      / Σ_{j ∈ A∪B} max(w_A(j), w_B(j))
```

where `w_X(j)` = MatKG triple `Count` linking X to neighbor j, summed
over both edge directions and over every MatKG entity matched to the
component. **`Count` is the number of distinct papers in which the two
entities co-occur** — MatKG builds it as `COUNT(DISTINCT DOI)`, not a raw
occurrence tally. Two consequences worth stating in a methods section:
MatKG stores each pair in both directions with identical `Count` (verified:
1,664,538 both-direction entries, 100.0% identical), so summing both
directions makes the implemented weight **2×** the paper count for
symmetric pairs — immaterial to the score, since weighted Jaccard is a
ratio and the factor cancels; and the minimum `Count` anywhere in
`SUBRELOBJ.csv` is **30**, so MatKG publishes only pairs co-occurring in
≥30 papers and Jchem therefore runs on an already-pruned graph. (Summing `min` over the intersection rather than the
union is equivalent — `min` is 0 for any non-shared neighbor.) The
unweighted form `|A∩B| / |A∪B|` is written alongside as
`jaccard_unweighted` for reference only; it is **not** ranked on, and
neither are the size-corrected metrics of §10. Ranking is
`jaccard_weighted` descending; the salts-only table filters
`category == lithium_salt` and drops LiTFSI itself (self-similarity 1.0).

LiTFSI reference profile: **174 neighbors**.
**50 of 15,961 components match MatKG at all** — that sparsity is the
intended relevance gate (~10k pharma lithium carboxylates never appear in
a materials corpus).

---

## 2. Application Knowledge Graph Pipeline (`battery_kg/`)

LLM-extracted (Mistral-Small-3.1-24B, via RITS API) battery-system and
salt relationships
from `battery_kg/t2br_data/raw_paragraphs_v2.xlsx` (T2BR dataset), against
two purpose-built ontologies.

### 2.1 Battery-system ontology

Ego-network-agnostic: rather than extracting a graph centered on one
compound, it captures the full battery-system context per paragraph,
enabling downstream multi-hop comparison queries. Five functional
categories, with example triples pulled from the actual extraction:

| Category | Example triple |
|---|---|
| (i) Composition & formulation | `(EC/DMC/EMC, CONTAINS, LiPF6)`, `(PP separator, COMPATIBLE WITH, EC/DMC/EMC)` |
| (ii) Stability & operating limits | `(BATTERY_CELL, STABLE IN WINDOW, 2.5–4.0 V)`, `(PP, OPERATING TEMP RANGE, 150 °C)` |
| (iii) Device & cell context | `(BATTERY_CELL, HAS FORM FACTOR, pouch cell)`, `(LiFePO4, HAS ELECTRODE LOADING, 8.0 mg cm⁻²)` |
| (iv) Properties & measurements | `(PP, HAS PROPERTY, dimensional stability)`, `(thermal shrinkage, HAS VALUE, 29.5%)` |
| (v) Causal / molecular-level | `(PP, AFFECTS PROPERTY OF, dimensional stability)`, `(metallic lithium, DEPOSITS ON, graphite)`, `(electrolyte, FORMS, solid electrolyte interphase (SEI))` |

Extracted by `ontology_extraction_triples_battery.py`
(explored with `explore_battery_ontology_extraction_results.py`). Raw result
(`battery_ontology_triples_with_doi_clean_20250819_214710.csv`):
**316,841 triples, 129 relation types**, dominated by `HAS PROPERTY`
(87,197), `AFFECTS PROPERTY OF` (82,056), `USED IN` (40,893),

> **Triple count — use 316,841.** The raw CSV has **316,846 rows**, but
> **5 carry a blank `object`** and are not triples: three are
> `LiMn2O4 / LiCoO2 / LiCoxMnyNizO2 — USED IN — <blank>`, and two are
> malformed extractions where the model put a predicate phrase in the
> relation slot and emitted no object (`iron content on the lithium
> sites — reduces electrochemical capacity — <blank>`). Dropping those 5
> accounts for **every** downstream figure at once: 316,846 − 5 =
> **316,841**; the two junk relation phrases exist *only* on those rows, so
> 131 − 2 = **129** relation types; 3 of the 5 are `USED IN`, so
> 40,896 − 3 = **40,893**; and 316,841 − 231,658 = **85,183** (−26.9%).
> A third figure, **316,839**, appears in `candidate_profiles_report.txt`
> because `appkg_candidate_profiles.py` loads with pandas' default NA
> handling, which additionally coerces two objects of the literal string
> `N/A` to NaN. All three counts are defensible; 316,841 is the one the
> rest of this README and the paper are consistent with.
`HAS COMPOSITION` (25,824), `CO OCCURS WITH` (15,852).

#### 2.1.1 Extraction prompt

One paragraph of battery-electrolyte literature text
(`battery_kg/t2br_data/raw_paragraphs_v2.xlsx`, T2BR dataset) in, one
`{"relationships": [...]}` JSON object out, via
`create_battery_analysis_prompt()`
(`battery_kg/ontology_extraction_triples_battery.py:20-129`). The prompt
lists the full ~30-template relation schema across the five categories
above, gives a worked JSON example, and instructs strict JSON-only
output:

```
You are an expert in battery materials. Analyze the following paragraph and
extract information about battery components. Here is a list of
relationships I want you to extract:
# ─── composition & formulation ────────────────────────────────
("SALT", "DISSOLVED IN", "SOLVENT"), ("SALT", "CO OCCURS WITH", "SOLVENT"),
("ELECTROLYTE", "CONTAINS", "SALT"), ...
# ─── electrode structure & materials / device context / stability &
#     operating limits / properties & measurements / molecular detail ──
...

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include
any explanation or additional text. Not all relationships will be present
in every paragraph, so only include those that are relevant. No
relationships should be inferred or made up; they must be explicitly
stated in the text. Some paragraphs might not mention any of these
relationships, in which case you should return an empty JSON object.

Here is an example of the JSON format I expect:
{
    "relationships": [
        {"subject": "LiPF6", "relation": "DISSOLVED IN", "object": "EC/DMC"},
        {"subject": "LiPF6", "relation": "USED IN", "object": "lithium-ion battery"},
        ...
    ]
}

Don't extract entities like "subject": "ANODE" — only specific materials
or components mentioned in the text. For "AFFECTS PROPERTY OF" and
"HAS PROPERTY", extract direction where stated ("increases ionic
conductivity" not just "ionic conductivity").

Paragraph to analyze:
{abstract}

Remember: Respond ONLY with valid JSON. Make sure the JSON is well-formed
and does not have missing commas, missing keys, extra commas, or any
other errors that will make it difficult to parse.
```

Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` via RITS,
`temperature=0.0`, `max_tokens=2048`. `parse_json_response()` strips
markdown code fences, locates the outermost `{...}` span, and
`json.loads`s it; failure (exception, or no `{` found) sets
`parsed_analysis: None` for that entry.

#### 2.1.2 Parse-failure breakdown

**2,029 of 46,602 paragraphs (4.35%) have `parsed_analysis: None`** in
`battery_ontology_extraction_results_v2.json` — this is the number SI
reports as "44,563 (95.6%) parsed" (46,602 − 2,029 = 44,573; independently
confirmed in `output/battery_ontology_multi_cpu_1503036.out`:
"Successfully parsed: 44573/46602 (95.6%)"). Re-parsing all 2,029 failed
`raw_response` strings directly gives the failure-mode breakdown:

| Cause | Count | % of failures |
|---|---|---|
| **Truncated JSON** (response cut off mid-object, no closing `}`) | 2,007 | 98.9% |
| API timeout (`Error: Request timed out.`) | 10 | 0.5% |
| Other malformed JSON (present but syntactically broken) | 12 | 0.6% |
| Empty response / no `{...}` found at all | 0 | 0% |

This is essentially a single root cause: **`max_tokens=2048` is too low
for paragraphs with many extractable relationships.** 1,942 of the 2,019
non-timeout failures have raw responses over 5,500 characters (consistent
with hitting the token cap), and 2,007/2,019 stop mid-list without a
closing `}` — e.g.:

```
{
    "relationships": [
        {"subject": "Li-Zn@NC@CC", "relation": "USED IN", "object": "BATTERY_CELL"},
        {"subject": "Li-NC@CC", "relation": "USED IN", "object": "BATTERY_CELL"},
        {"subject": "Li-CC", "relation": "USED IN", "object": "BATTERY_CELL"},
        {"subject": "          ← response ends here, no closing brace (char 6445)
```

This is why the failure rate is ~2,000× higher than the salt-only
extraction (§2.2): that prompt has only 10 relation types and
correspondingly much shorter expected outputs, so it almost never hits
the cap.

**Two different pipelines, two different metrics — do not conflate them:**
- **Salt-ontology JSON parse failure**: 1/46,602 = **2.1×10⁻⁵**
  (`salt_ontology_extraction_results_v2.json`; matches
  `salt_analysis_output/salt_ontology_analysis_report.txt`:
  "Successfully parsed: 46,601 / Success rate: 100.0%").
- **Battery-ontology JSON parse failure**: 2,029/46,602 = **4.35%**
  (SI's 44,563/46,602, 95.6% — the number backing the 316,841-triple KG
  used throughout §4–§7).

A Discussion-section claim of ~4.3×10⁻⁵ failure rate describes the
salt-ontology pipeline (off from the true 2.1×10⁻⁵ by ~2×, plausibly a
different run), **not** the battery-ontology pipeline that the paper's
headline results and candidate profiles (§7) actually depend on — that
one fails 4.4% of the time, ~1,000× higher. State both numbers
explicitly and attribute each to its pipeline; don't let one stand in
for "extraction reliability" generally.

### 2.2 Salt-level ontology

A separate, lighter-weight, **ego-centric** ontology focused exclusively on
the salt as subject — 10 relation types: `CO OCCURS WITH`,
`DISSOLVED IN`, `USED IN`, `HAS CONCENTRATION IN ELECTROLYTE`,
`HAS PROPERTY`, `HAS ROLE`, `AFFECTS PROPERTY OF`, `STABLE IN WINDOW`,
`OPERATING TEMP RANGE`, `HAS CHEMICAL STRUCTURE`. Example triples:
`(LiPF6, DISSOLVED IN, 1:1 ethylene carbonate and dimethyl carbonate)`,
`(LiPF6, CO OCCURS WITH, PP polymer matrix)`, `(LiPF6, USED IN, Li-ion
cells)`.

Extracted by `ontology_extraction_triples_salt_with_doi.py`
with DOI provenance
tracking, generic-mention filtering ("Li salt" not extracted), and
directional property extraction (e.g. "increases" vs "decreases ionic
conductivity"). Raw result
(`enhanced_salt_extraction_raw_20250815_225432.json`):
**27,993 triples, 302 unique salt subjects, 5,567 source paragraphs.**
Relation frequency: `CO OCCURS WITH` (5,524), `DISSOLVED IN` (4,643),
`HAS CONCENTRATION IN` (4,249), `USED IN` (4,041), `HAS PROPERTY` (3,766).

### 2.3 Canonicalization

`battery_ontology_cleaning_with_doi.py` resolves synonym proliferation
(e.g. "LiTFSI" / "Li-TFSI" / "bis(trifluoromethanesulfonyl)imide lithium"
/ "LiN(SO2CF3)2" all naming the same compound) before it inflates apparent
entity diversity and suppresses downstream similarity scores. Pipeline:
lowercase normalization → abbreviation expansion via a domain-specific
alias table → PubChem CID-based deduplication, with full DOI-level
provenance preserved per triple throughout (enabling the per-claim
attribution in §7).

**Results:**
- **batteryKG**: 316,841 → **231,658 triples (−26.9%)**, a reduction of
  85,183.
- **saltKG**: 27,993 → **14,123 triples (−49.5%)**, after collapsing
  exact-duplicate (subject, relation, object) triples produced by synonym
  merging. Entity-level resolution: 302 raw subjects → 288 canonical
  subjects via 710 entity mappings (326 automated PubChem lookups, 311
  manual).

This canonicalization pass is **upstream of and distinct from** the
chemical-KG pipeline's own InChIKey-based synonym healing (§3.1), which
runs on top of the *already*-canonicalized saltKG CSV.

---

## 3. Japp — application-KG (salt KG) ranking

### 3.1 InChIKey remapping

**`appkg_salt_ranking.py`** — the canonicalized salt KG
(`salt_analysis_output/canonical/salt_relationships_canonical_with_source.csv`,
27,993 triples, 302 unique subjects, 10 relations) is **ego-centric**:
subjects are salt mentions, objects are applications/properties/solvents/
formulations extracted from paper paragraphs.

Every subject mention is remapped onto InChIKey components with the
chemical-KG Stage-4 synonym index, healing further fragmentation the
name-based canonicalization in §2.3 left behind:
- **LiTFSI** had been split across five canonical identities (LiTFSI /
  LiNTf2 / LiTFSA / Li[NTf2] / LiN(CF3SO2)2; +292 mentions recovered),
  **LiDFOB** across three (LiODFB / Li-DFOB / full name), **lithium
  triflate** across three (including a wrong PubChem name-resolution of
  LiSO3CF3 to "trifluoro(sulfinatooxy)methane").
- Generic mentions ("Li salt" → pseudo-entity "lithium") dropped, not
  pooled; formulation phrases ("PEO-LiTFSI solid polymer electrolyte")
  excluded and reported.
- **Novel salts with no PubChem record survive as named
  pseudo-components** (LiTDI, LiFOP, LiFNFSI, LiFPFSI, LiSFSI, LiPDSFSI,
  …) — a name-resolution-only pipeline can never rank these.

Mapping coverage: **90% of triples** (25,170) land on InChIKey components;
2,134 on novel pseudo-components; the remainder are generic/phrase/rare
mentions (`output/appkg_subject_mapping.csv`).

**Japp score** (column `jaccard_weighted` in
`chemical_kg/output/appkg_salt_ranking.csv`) = the *same* weighted-Jaccard
formula as Jchem, Σ min(w) / Σ max(w), on a different graph and a
different weight:

```
Japp = Σ_{o ∈ A∩B} min(w_A(o), w_B(o))
     / Σ_{o ∈ A∪B} max(w_A(o), w_B(o))
```

where a salt's profile is its **subject-side (directed) ego set** — the
`object_canonical` values of triples whose subject maps to that salt — and
`w_X(o)` = how many times X is stated with object o. No reverse direction
is added (unlike Jchem, this KG is ego-centric by construction: subjects
are always salt mentions). LiTFSI profile: **8,835 mentions over 4,676
distinct neighbors.**

---

## 4. Results

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

---

## 5. Consensus substitution candidates

Five salts appear in **both** top-10 lists:

| Salt | Jchem rank (score) | Japp rank (score) | Fluorine status |
|---|---|---|---|
| LiNO3 | 1 (0.458) | 6 (0.0222) | fluorine-free |
| LiClO4 | 4 (0.233) | 3 (0.0570) | fluorine-free |
| LiPF6 | 7 (0.221) | 1 (0.0800) | fluorinated incumbent |
| LiOTf | 8 (0.178) | 8 (0.0189) | PFAS (CF3) |
| Li2SO4 | 3 (0.235) | 9 (0.0135) | fluorine-free |

For PFAS substitution, LiPF6 (the fluorinated incumbent, co-regulated
alongside PFAS binders) and LiOTf (itself a PFAS) are excluded on
chemistry grounds — neither can serve as a *substitute* for a
PFAS-containing salt. Li2SO4 is **not** excluded on those grounds (it is
fluorine-free) and remains in the consensus set, but it is deprioritized
behind LiNO3 and LiClO4: per-claim DOI evidence (§7.2) shows its
corroborated claims are almost entirely aqueous-electrolyte / synthesis-
bath chemistry (top claim `DISSOLVED IN water`, 44×/19 papers), not
aprotic battery-electrolyte use — the same solvent regime LiTFSI actually
operates in — which weakens it as a realistic drop-in candidate despite
its Jchem/Japp ranks. That leaves **LiNO3 and LiClO4 as the top two
substitution candidates by combined rank**, with Li2SO4 third. The two
KG views complement each other's blind spots:
Jchem's top salts reflect materials-literature ubiquity (LiNO3/Li2CO3 as
additives and SEI components), while Japp — built from application
contexts — puts the true electrolyte-salt substitutes (LiPF6, LiFSI,
LiClO4, LiBOB) on top.

---

## 6. Hold-out validation

**Hold-out:** Wang, Qing, Zhang & Li, "Fluorine-free electrolytes for
sustainable lithium batteries: a review," *npj Materials Sustainability*
(2025), DOI `10.1038/s44296-025-00074-8`. The review discusses the 2023
ECHA PFAS restriction proposal (covering LiTFSI and PVDF/PTFE binders)
and identifies **LiBOB, LiClO4, LiNO3 and cyano-based anions** as the
leading fluorine-free substitution routes, including for LFP cathodes.

**Verified as a true hold-out** (none of the ranking inputs contain it):
- DOI string (`s44296-025-00074-8`) and title text: **0 occurrences** in
  the battery-KG source corpus (`battery_kg/t2br_data/raw_paragraphs_v2.xlsx`),
  in the battery triples DOI column (31,640 unique paragraph-level DOI
  ids, 2,088 distinct papers), and in all extraction JSONs.
- MatKG's corpus was assembled and released years before 2025 and cannot
  contain the review.

**Agreement:** the review's fluorine-free candidates and the consensus
ranking coincide — LiNO3 and LiClO4 (our consensus top two) are both
named, and LiBOB (Japp rank 4; absent from Jchem's top 10 only because
MatKG contains just 13 LiBOB-adjacent entities) is the review's third
named salt. Since the review shares no source documents with either KG,
this constitutes independent confirmation that KG-derived neighborhood
similarity recovers chemically meaningful substitution candidates.

---

## 7. Battery-KG cross-check & DOI-attributed candidate profiles

### 7.1 Battery-KG cross-check

**`appkg_battery_audit.py`** — the general (non-ego-centric) battery KG
(**316,839 triples as loaded**, 129 relations, 31,640 paragraph-level DOI
ids from 2,088 distinct papers, **120,779 unique entities**, dominated by
formulation phrases).

> **Why 316,839 here and 316,841 in §2.1.** Both are right; they are
> different bases. §2.1 counts the KG itself: 316,846 CSV rows minus the 5
> blank-object rows = **316,841 valid triples**. §7's two consumers
> (`appkg_battery_audit.py`, `appkg_candidate_profiles.py`) both load with
> pandas' default NA handling and `.dropna()`, which additionally removes
> **2 rows whose object is the literal string `N/A`**, leaving
> **316,839**. Every §7 figure — including the 120,779 entities — is on
> that 316,839 basis, so quote 316,839 whenever you quote a §7 number. Entity mapping: **241 exact-tier** entities (LiTFSI alone
unifies 12+ surface forms: 'LITFSI', 'Li NTf2', 'Li [NTf2]', 'Li+, TFSI–',
'Li-TFSI', 'LiTFSA', …) across 62 components, plus **1,824 composite-tier**
entities ('PEO/LiTFSI electrolytes') counted per salt but deliberately not
merged into profiles. Ego-profile ranking vs LiTFSI is **broadly
concordant with, but not identical to, Japp**: the same top three
(LiPF6, LiFSI, LiClO4), then LiNO3, LiBOB, LiDFOB, LiF, Li2CO3, LiBF4 —
i.e. LiNO3 rises (Japp 6 → 4) and LiBF4 falls (Japp 5 → 9), interleaved
with solvents since this graph does not separate them. That the top three
and the broad ordering survive an independent extraction is the
cross-check; the exact permutation does not transfer.

### 7.2 Per-claim DOI-attributed candidate profiles — **status: done (2026-07-24)**

**`appkg_candidate_profiles.py`** generates per-claim DOI attribution for
the consensus candidates (LiNO3, LiClO4, Li2SO4, LiBOB) plus the
fluorinated consensus references (LiPF6, LiOTf), with LiTFSI itself
profiled the same way as the reference (5,635 exact-tier mentions) so
downstream comparisons cite the same evidence layer.

A **claim** is `(direction, relation, squash-normalized neighbor)`
aggregated over exact-tier triples only; every claim carries the full
list of supporting DOIs. The KG's `doi` column is a *paragraph* id
(`…_p2_8`); papers are counted after stripping the `_p<i>_<j>` suffix
(31,640 paragraph ids → 2,088 distinct papers), so a claim repeated across
paragraphs of one paper counts as one paper of evidence.

**18,919 evidence triples total.** Corroborated (≥2 papers) claims:
**LiPF6 653 claims (1,342 papers)**, **LiClO4 77 (111)**, **LiBOB 63
(40)**, **LiNO3 60 (159)**, **Li2SO4 24 (35)**, **LiOTf 8 (22)**.

Signature corroborated claims:
- **LiNO3** — co-salt/additive alongside LiTFSI in DOL/DME ether
  electrolytes (98×/82 papers for `CO OCCURS WITH LiTFSI`), improves
  coulombic efficiency, reduces interface impedance, suppresses
  dendrites.
- **LiClO4** — PEO / PC / acetonitrile electrolytes, ionic-conductivity
  claims.
- **LiBOB** — SEI formation, capacity retention, suppresses Fe
  dissolution from LiFePO4.
- **LiPF6** — the carbonate-electrolyte workhorse (EC/DMC 435×/313
  papers).
- **Li2SO4** — almost entirely aqueous-electrolyte and synthesis-bath
  contexts (`DISSOLVED IN water`, 44×/19 papers), which weakens it as an
  aprotic drop-in substitution candidate.
- **LiOTf** — thin evidence base (22 papers).

**What the output looks like** (excerpt from
`candidate_profiles_report.txt`, one block per candidate, one sub-block
per relation direction, ranked by mention count):

```
LiNO3  (IIPYXGDZVMZOAP-UHFFFAOYSA-N)
  888 mentions | 427 claims | 159 papers | 60 corroborated claims

  [salt -> x] CO OCCURS WITH  (170 mentions, 92 claims)
    LiTFSI                             14x  14p  10.1016/j.carbon.2019.11.022, 10.1016/j.carbon.2020.06.073, ... (+11 papers)
    DOL/DME                            13x  13p  10.1016/j.electacta.2021.138301, ... (+10 papers)
    ...

  [x -> salt] CO OCCURS WITH  (132 mentions, 30 claims)
    LiTFSI                             96x  81p  10.1016/j.apsusc.2021.150589, 10.1016/j.carbon.2019.11.002, ... (+78 papers)
    ...
```

Outputs (`chemical_kg/output/`): `candidate_claims.csv` (one row per
claim, all DOIs), `candidate_evidence.csv` (one row per triple with
source paragraph, 27.6 MB), `candidate_profiles_report.txt` (human-
readable, 94 KB). Last generated 2026-07-24 18:14; current with the rest
of the pipeline.

---

### 7.3 Worked example — the complete LiClO4 profile

Reproduced verbatim from `candidate_profiles_report.txt` (lines 260–391)
as a paper-ready specimen of the evidence layer. Every candidate is
profiled in this format; LiClO4 is shown in full because it is the
best-corroborated non-fluorinated candidate (77 corroborated claims from
111 papers, second only to the LiPF6 reference).

Reading the format: each sub-block is one `(direction, relation)` pair,
ordered by mention count. `[salt -> x]` means the salt is the triple
subject, `[x -> salt]` the object. Within a sub-block each row is one
claim — normalized neighbor, `N×` mentions, `Np` distinct papers after
stripping paragraph suffixes, then the supporting DOIs. Claims are
ranked by paper count, not mention count, so a phrase repeated in one
paper cannot outrank a result replicated across several.

```
LiClO4  (MHCFAGZWMAWTNR-UHFFFAOYSA-M)
  853 mentions | 408 claims | 111 papers | 77 corroborated claims

  [salt -> x] CO OCCURS WITH  (222 mentions, 94 claims)
    PEO                                                           22x  16p  10.1016/j.carbon.2022.02.034, 10.1016/j.cej.2020.127274, 10.1016/j.electacta.2014.04.099 (+13 papers)
    PC                                                            18x  14p  10.1016/j.ceja.2021.100163, 10.1016/j.ceramint.2015.01.121, 10.1016/j.ceramint.2016.11.149 (+11 papers)
    acetonitrile                                                  14x  10p  10.1016/j.compscitech.2021.108837, 10.1016/j.electacta.2018.11.206, 10.1016/j.ensm.2019.06.029 (+7 papers)
    propylene carbonate                                           11x  10p  10.1016/j.ceramint.2007.09.037, 10.1016/j.electacta.2010.11.050, 10.1016/j.electacta.2017.10.123 (+7 papers)
    EC/DMC                                                         9x   7p  10.1016/j.electacta.2011.11.080, 10.1016/j.jallcom.2007.02.103, 10.1016/j.jpowsour.2007.09.085 (+4 papers)
    NMP                                                            5x   5p  10.1016/j.carbon.2022.02.034, 10.1016/j.jelechem.2019.03.034, 10.1016/j.memsci.2019.01.025 (+2 papers)
    PAN                                                            8x   4p  10.1016/j.electacta.2017.08.162, 10.1016/j.jallcom.2021.159890, 10.1016/j.ssi.2017.11.026 (+1 papers)
    LLZTO                                                          6x   4p  10.1016/j.cej.2019.02.148, 10.1016/j.electacta.2017.08.162, 10.1016/j.nanoen.2016.09.002 (+1 papers)
    ... +86 more claims (see candidate_claims.csv)

  [salt -> x] DISSOLVED IN  (149 mentions, 53 claims)
    PC                                                            19x  15p  10.1016/j.ceja.2021.100163, 10.1016/j.ceramint.2015.01.121, 10.1016/j.ceramint.2016.11.149 (+12 papers)
    propylene carbonate                                           11x  10p  10.1016/j.ceramint.2007.09.037, 10.1016/j.electacta.2010.11.050, 10.1016/j.electacta.2017.10.123 (+7 papers)
    EC/DMC                                                        15x   9p  10.1016/j.electacta.2011.04.105, 10.1016/j.electacta.2011.11.080, 10.1016/j.electacta.2019.135567 (+6 papers)
    acetonitrile                                                  12x   8p  10.1016/j.compscitech.2021.108837, 10.1016/j.electacta.2018.11.206, 10.1016/j.ensm.2019.06.029 (+5 papers)
    PEO                                                            7x   5p  10.1016/j.electacta.2014.04.099, 10.1016/j.electacta.2017.08.162, 10.1016/j.jallcom.2021.159890 (+2 papers)
    EC/DEC                                                         4x   4p  10.1016/j.electacta.2009.03.007, 10.1016/j.jpowsour.2005.03.060, 10.1016/j.jpowsour.2007.05.055 (+1 papers)
    EC:DEC                                                         4x   4p  10.1016/j.jpowsour.2009.06.046, 10.1016/j.jpowsour.2009.11.147, 10.1016/j.materresbull.2008.02.014 (+1 papers)
    DMF                                                            4x   4p  10.1016/j.cej.2019.02.148, 10.1016/j.memsci.2019.01.025, 10.1016/j.ssi.2017.11.026 (+1 papers)
    ... +45 more claims (see candidate_claims.csv)

  [salt -> x] USED IN  (141 mentions, 8 claims)
    BATTERY_CELL                                                 134x  85p  10.1016/j.cej.2019.02.148, 10.1016/j.cej.2020.127274, 10.1016/j.ceja.2021.100163 (+82 papers)
    Li-ion batteries                                               1x   1p  10.1016/j.jhazmat.2020.122614
    lithium-ion batteries                                          1x   1p  10.1016/j.jpowsour.2010.03.038
    lithium-ion battery                                            1x   1p  10.1016/j.jpowsour.2013.05.178
    lithium batteries                                              1x   1p  10.1016/j.electacta.2019.135133
    coin cell                                                      1x   1p  10.1016/j.jpowsour.2020.228033
    CR2016 coin-type cell                                          1x   1p  10.1016/j.mee.2015.04.029
    polymer electrolyte                                            1x   1p  10.1016/j.electacta.2012.10.027

  [x -> salt] CONTAINS  (71 mentions, 49 claims)
    electrolyte                                                    7x   7p  10.1016/j.est.2019.02.020, 10.1016/j.jpowsour.2011.01.072, 10.1016/j.jpowsour.2013.04.109 (+4 papers)
    acetonitrile                                                   4x   4p  10.1016/j.compscitech.2021.108837, 10.1016/j.ensm.2022.01.029, 10.1016/j.jpowsour.2010.03.029 (+1 papers)
    EC/DEC                                                         3x   3p  10.1016/j.jpowsour.2005.03.060, 10.1016/j.jpowsour.2007.05.055, 10.1016/j.jpowsour.2013.05.178
    ethylene carbonate and dimethyl carbonate                      3x   3p  10.1016/j.jallcom.2010.05.173, 10.1016/j.jallcom.2015.04.094, 10.1016/j.mee.2015.04.029
    LiClO4/PC                                                      3x   2p  10.1016/j.electacta.2013.07.219, 10.1016/j.mseb.2016.10.001
    PEO-LiClO4                                                     2x   2p  10.1016/j.jmat.2019.04.002, 10.1016/j.jpowsour.2019.05.008
    solid polymer electrolytes                                     2x   2p  10.1016/j.cej.2020.124993, 10.1016/j.electacta.2017.08.162
    PC/EC/DMC                                                      3x   1p  10.1016/j.jpowsour.2020.228033
    ... +41 more claims (see candidate_claims.csv)

  [salt -> x] AFFECTS PROPERTY OF  (68 mentions, 56 claims)
    ionic conductivity                                             7x   5p  10.1016/j.electacta.2019.135133, 10.1016/j.jpowsour.2010.09.121, 10.1016/j.jpowsour.2011.04.062 (+2 papers)
    conductivity                                                   2x   2p  10.1016/j.electacta.2011.11.080, 10.1016/j.jpowsour.2010.03.029
    interfacial resistance                                         2x   2p  10.1016/j.electacta.2017.08.162, 10.1016/j.jpowsour.2013.05.178
    thermal stability                                              2x   2p  10.1016/j.electacta.2017.08.162, 10.1016/j.electacta.2019.135133
    viscosity                                                      2x   2p  10.1016/j.electacta.2011.11.080, 10.1016/j.electacta.2019.135133
    increases viscosity                                            2x   2p  10.1016/j.electacta.2019.135133, 10.1016/j.memsci.2020.118132
    capacity                                                       2x   2p  10.1016/j.electacta.2011.11.080, 10.1016/j.jpowsour.2020.228033
    reduces parasitic current density                              1x   1p  10.1016/j.jpowsour.2007.09.085
    ... +48 more claims (see candidate_claims.csv)

  [salt -> x] CONTAINS  (46 mentions, 28 claims)
    PC                                                             8x   8p  10.1016/j.ceramint.2015.01.121, 10.1016/j.ceramint.2016.11.149, 10.1016/j.jelechem.2009.03.006 (+5 papers)
    propylene carbonate                                            5x   5p  10.1016/j.electacta.2010.11.050, 10.1016/j.jpowsour.2020.228033, 10.1016/j.materresbull.2004.07.003 (+2 papers)
    EC/DEC                                                         3x   3p  10.1016/j.electacta.2009.03.007, 10.1016/j.jpowsour.2005.03.060, 10.1016/j.jpowsour.2007.05.055
    1 M LiClO4                                                     3x   3p  10.1016/j.electacta.2019.135567, 10.1016/j.jpowsour.2009.06.046, 10.1016/j.jpowsour.2009.11.147
    EC/DMC                                                         2x   2p  10.1016/j.jpowsour.2007.09.085, 10.1016/j.jpowsour.2008.01.079
    SALT                                                           2x   2p  10.1016/j.jpowsour.2010.03.029, 10.1016/j.powtec.2013.06.021
    Li+                                                            2x   2p  10.1016/j.ceja.2021.100163, 10.1016/j.ssi.2018.02.040
    EC                                                             1x   1p  10.1016/j.jpowsour.2009.06.042
    ... +20 more claims (see candidate_claims.csv)

  [salt -> x] HAS PROPERTY  (42 mentions, 32 claims)
    1 M concentration                                              3x   3p  10.1016/j.ceramint.2015.01.121, 10.1016/j.electacta.2014.08.141, 10.1016/j.jpowsour.2009.11.147
    ionic conductivity                                             3x   3p  10.1016/j.electacta.2019.135133, 10.1016/j.jpowsour.2014.04.014, 10.1016/j.nanoen.2018.01.028
    concentration                                                  2x   2p  10.1016/j.jelechem.2009.03.006, 10.1016/j.ssi.2003.12.033
    EO/Li+ ratio of 16                                             3x   1p  10.1016/j.memsci.2019.01.025
    decomposition temperature                                      2x   1p  10.1016/j.electacta.2019.135133
    mass fraction                                                  2x   1p  10.1016/j.memsci.2020.118132
    1 M                                                            2x   1p  10.1016/j.ssc.2003.10.015
    conductivity                                                   1x   1p  10.1016/j.jpowsour.2010.09.121
    ... +24 more claims (see candidate_claims.csv)

  [salt -> x] STABLE IN WINDOW  (32 mentions, 28 claims)
    2.0–4.5 V                                                      2x   2p  10.1016/j.jpowsour.2005.03.060, 10.1016/j.jpowsour.2009.06.042
    2.5 to 4.3 V                                                   2x   2p  10.1016/j.jpowsour.2008.01.079, 10.1016/j.jpowsour.2009.11.147
    2.5 to 4.5 V                                                   2x   2p  10.1016/j.jpowsour.2009.06.046, 10.1016/j.powtec.2008.09.013
    2.7 to 4.1 V                                                   2x   2p  10.1016/j.materresbull.2004.07.003, 10.1016/j.ssc.2003.10.015
    2.0 to 4.5 V                                                   1x   1p  10.1016/j.jpowsour.2005.03.117
    3.0 to 4.0 V                                                   1x   1p  10.1016/j.jpowsour.2005.03.117
    −0.5 to 1.5 V                                                  1x   1p  10.1016/j.jpowsour.2010.03.029
    3V to 4V                                                       1x   1p  10.1016/j.jpowsour.2018.05.023
    ... +20 more claims (see candidate_claims.csv)

  [x -> salt] CO OCCURS WITH  (31 mentions, 25 claims)
    PEO                                                            3x   3p  10.1016/j.cej.2020.127274, 10.1016/j.ensm.2022.01.029, 10.1016/j.jpowsour.2019.05.008
    LiBOB                                                          3x   2p  10.1016/j.jpowsour.2016.10.105, 10.1016/j.nanoen.2020.105562
    Poly(ethylene oxide)                                           2x   2p  10.1016/j.cej.2020.127274, 10.1016/j.electacta.2018.11.206
    LiDFOB                                                         2x   1p  10.1016/j.nanoen.2020.105562
    OLiSSE                                                         1x   1p  10.1016/j.jechem.2021.09.011
    C-LiFePO4                                                      1x   1p  10.1016/j.jpowsour.2010.03.029
    PPy                                                            1x   1p  10.1016/j.jpowsour.2010.03.029
    stainless steel                                                1x   1p  10.1016/j.jpowsour.2010.03.029
    ... +17 more claims (see candidate_claims.csv)

  [salt -> x] OPERATING TEMP RANGE  (28 mentions, 14 claims)
    room temperature                                              12x  12p  10.1016/j.ceramint.2007.09.037, 10.1016/j.ceramint.2016.11.149, 10.1016/j.electacta.2014.08.141 (+9 papers)
    25 °C                                                          4x   4p  10.1016/j.materresbull.2009.09.031, 10.1016/j.mtener.2021.100836, 10.1016/j.powtec.2008.09.013 (+1 papers)
    125–500 K                                                      1x   1p  10.1016/j.jpowsour.2007.05.055
    150–320 K                                                      1x   1p  10.1016/j.jpowsour.2007.05.055
    350–1300 K                                                     1x   1p  10.1016/j.jpowsour.2007.05.055
    120 °C                                                         1x   1p  10.1016/j.jpowsour.2008.01.079
    105 °C                                                         1x   1p  10.1016/j.jpowsour.2009.06.042
    130 °C                                                         1x   1p  10.1016/j.jpowsour.2013.07.091
    ... +6 more claims (see candidate_claims.csv)

  [x -> salt] COMPATIBLE WITH  (9 mentions, 8 claims)
    microporous polypropylene                                      2x   2p  10.1016/j.jpowsour.2009.06.046, 10.1016/j.jpowsour.2009.11.147
    TPU–PVdF nonwoven films                                        1x   1p  10.1016/j.jpowsour.2011.04.062
    Celguard 2400                                                  1x   1p  10.1016/j.jallcom.2007.02.103
    separator                                                      1x   1p  10.1016/j.jpowsour.2020.228033
    Celgard 2400                                                   1x   1p  10.1016/j.materresbull.2009.09.031
    LiFePO4                                                        1x   1p  10.1016/j.powtec.2008.09.013
    EC:DEC                                                         1x   1p  10.1016/j.powtec.2008.09.013
    microporous polypropylene separator                            1x   1p  10.1016/j.powtec.2008.09.013

  [salt -> x] USED WITH  (7 mentions, 6 claims)
    LiFePO4                                                        2x   2p  10.1016/j.jpowsour.2009.11.147, 10.1016/j.powtec.2008.09.013
    cathode materials                                              1x   1p  10.1016/j.jpowsour.2007.05.055
    lithium anode                                                  1x   1p  10.1016/j.jpowsour.2007.05.055
    LiFePO4/C                                                      1x   1p  10.1016/j.jpowsour.2009.06.046
    cathode                                                        1x   1p  10.1016/j.jpowsour.2020.228033
    lithium foil                                                   1x   1p  10.1016/j.jpowsour.2020.228033

  ... +3 more relation groups (see candidate_claims.csv)

```

**What this profile supports, for the paper:**

- **Aprotic solvent compatibility is directly evidenced**, unlike
  Li2SO4's aqueous-only footprint: PC (19×/15p), propylene carbonate
  (11×/10p), EC/DMC (15×/9p), acetonitrile (12×/8p), EC/DEC, EC:DEC —
  all under `DISSOLVED IN`, the relation that matters for a drop-in
  electrolyte-salt substitution argument.
- **Polymer-electrolyte use is the dominant context** — PEO leads
  `CO OCCURS WITH` (22×/16p), with PAN, LLZTO and PEO-LiClO4 composite
  phrasings behind it.
- **An electrochemical stability window is actually reported**
  (`STABLE IN WINDOW`: 2.0–4.5 V, 2.5–4.3 V, 2.5–4.5 V, 2.7–4.1 V, each
  2 papers), which most candidates lack entirely. Note these are
  cell-level windows from the source papers, not salt-intrinsic limits.
- **Cell integration is corroborated at scale** — `USED IN BATTERY_CELL`
  134×/85 papers, plus named separators and LiFePO4 pairings under
  `COMPATIBLE WITH` and `USED WITH`.

**What it does not support:** the `AFFECTS PROPERTY OF` evidence is thin
(68 mentions, 56 claims, top claim `ionic conductivity` at only 5
papers), so LiClO4's *performance* case rests on far weaker corroboration
than its *usage* case. `HAS PROPERTY` is similarly sparse and dominated
by concentration restatements ("1 M concentration", "1 M"). Any claim
about how LiClO4 performs relative to LiTFSI should be sourced from the
underlying paragraphs in `candidate_evidence.csv`, not from claim counts
alone.

---

## 8. LLM-Based Commentary Generation — **status: done (2026-07-24)**

**`llm_candidate_commentary.py`** (`chemical_kg/`) — human-readable
substitution commentary per candidate vs LiTFSI, generated by
Mistral-Small-3.1-24B (RITS) from the §7.2 claims. Successor to
`battery_kg/llm_querying_of_candidates.py`, which prompted the LLM
directly from salt-KG triples with no provenance; here the knowledge
profiles handed to the model are corroboration-ranked claims each
carrying its paper DOIs, and the prompt makes citations **mandatory** —
every corpus-derived statement must end with its bracketed DOI(s).

**Prompt revision (2026-09-02).** The first version let the model add
statements from its own electrochemistry knowledge provided they were
tagged `[general knowledge]`. That tagging worked, and it exposed a
problem: the tagged statements were carrying the *headline* claims. Both
of LiClO4's stated advantages — "non-fluorinated" and "less expensive to
manufacture" — were `[general knowledge]`, i.e. supported by no evidence
in this pipeline, and the 1–10 feasibility score came from no rubric at
all. The prompt now **forbids** outside knowledge rather than labelling
it, drops the feasibility score and drop-in verdict, and removes the
Cost-Manufacturing heading from the advantages/limitations sections
(the KG has no cost or manufacturing relations, so that heading was
structurally inviting invention). A further rule requires the model to
check the profile's actual relation names before asserting a gap — the
old run claimed LiClO4 "lacks solubility data" while §7.3 shows a large
`DISSOLVED IN` block.

Seven sections per candidate: similarity assessment, key differences,
advantages, limitations, application suitability, replacement
considerations, research gaps, overall assessment. Needs `RITS_API_KEY`
(see [Setup](#setup)); `--candidates` selects which salts to run.

Outputs (`chemical_kg/output/`): **`candidate_commentary_report_v2.txt`**
(18.3 KB, LiNO3/LiClO4/Li2SO4, corpus-only, generated 2026-09-02) and
`candidate_commentary_v2.json`. The superseded first run is retained as
`candidate_commentary_report.txt` (36.8 KB, all six candidates,
2026-07-24) — cite v2.

### 8.1 Commentary — LiNO3, LiClO4 and Li2SO4 vs LiTFSI

The three consensus non-fluorinated candidates (§5), **verbatim** from
`candidate_commentary_report_v2.txt`, generated 2026-09-02 by
`mistralai/Mistral-Small-3.1-24B-Instruct-2503` from `candidate_claims.csv`
(§7.2). Every statement is corpus-derived — the model was forbidden from
adding outside knowledge, and no feasibility score was requested.

> **Verified errors — read before quoting.** The model is reliable about
> what evidence *exists* and unreliable about what evidence is *absent*.
> Every confirmed error below is either an absence claim or a paper-count
> comparison; both require reasoning over the whole profile rather than
> reading one line of it, and the reference profile shown to the model is
> truncated to ~2% of LiTFSI's 2,597 claims (`MAX_PER_BUCKET = 12`).
>
> 1. **Li2SO4 §2 — "no explicit stability window mentioned for LiTFSI":
>    false.** LiTFSI has **55** `STABLE IN WINDOW` claims, including
>    "electrochemical stability window" (3 papers), 5.12 V, 2.0–6.0 V and
>    −2.2 to 3.3 V. They fell below the profile cut. The sentence inverts
>    the real picture, since LiTFSI's windows span a far wider range than
>    Li2SO4's −0.8 to 1.0 V aqueous limit.
> 2. **Li2SO4 §2 vs §7 — self-contradictory on interface impedance.** §2
>    says Li2SO4 affects it; §7 says there is no evidence. The data sides
>    with §2, but on **one claim from one paper** — so the correct reading
>    is "single-paper, weak", which neither section states.
> 3. **LiClO4 §1 — paper-count comparisons unreliable.** "LiTFSI has more
>    mentions (2 papers) compared to LiClO4 (7 papers)" is backwards, and
>    a 2-vs-2 comparison is described as "more". Roughly a dozen such
>    comparisons appear in that block; large gaps (26 vs 1) are stated
>    correctly, near-ties are garbled.
>
> Positive cited claims and their DOIs are sound throughout — no DOI was
> found attached to a claim it does not support. Source every "no evidence
> for X" statement from `candidate_claims.csv` directly.

#### LiNO3

```
### 1. SIMILARITY ASSESSMENT
- **Usage Contexts**:
  - Both LiNO3 and LiTFSI are frequently used in battery cells [10.1016/S0378-7753(01)00727-3; 10.1016/j.apsusc.2021.150589].
  - Both salts are used in coin cells [10.1016/j.ensm.2019.06.010; 10.1016/j.jechem.2019.07.010].
  - Both are compatible with Celgard 2400 [10.1016/j.jechem.2020.06.060; 10.1016/j.mtener.2021.100663].
  - Both are often co-occurring with PEO [10.1016/S0378-7753(01)00727-3; 10.1016/j.cej.2020.124848].
  - Both are dissolved in DOL/DME [10.1016/j.carbon.2019.11.002; 10.1016/j.electacta.2021.138301].
  - Both are used with LiFePO4 [10.1016/j.cej.2021.130017; 10.1016/j.electacta.2019.03.051].

### 2. KEY DIFFERENCES
- **Ionic Conductivity**:
  - LiTFSI is reported to increase ionic conductivity in 26 papers [10.1016/j.cej.2021.132025; 10.1016/j.cej.2022.135926; 10.1016/j.electacta.2019.02.052].
  - LiNO3 has weak evidence of greater ionic conductivity, supported by only one paper [10.1016/j.jpowsour.2012.03.049].
- **Thermal Stability**:
  - LiTFSI has higher thermal stability, supported by 5 papers [10.1016/j.electacta.2015.09.001; 10.1016/j.jpowsour.2016.10.105; 10.1016/j.jpowsour.2018.02.066].
  - There is no evidence for the thermal stability of LiNO3.
- **Glass Transition Temperature**:
  - LiTFSI affects the glass transition temperature in multiple ways, increasing it in 9 papers [10.1016/j.electacta.2012.10.027; 10.1016/j.electacta.2016.01.119; 10.1016/j.electacta.2019.134560] and decreasing it in 5 papers [10.1016/j.electacta.2019.02.052; 10.1016/j.jechem.2021.01.030; 10.1016/j.jpowsour.2018.10.045].
  - There is no evidence for the effect of LiNO3 on the glass transition temperature.
- **Viscosity**:
  - LiTFSI increases viscosity in 8 papers [10.1016/j.electacta.2010.03.043; 10.1016/j.electacta.2016.08.025; 10.1016/j.jechem.2021.01.030].
  - There is no evidence for the effect of LiNO3 on viscosity.
- **Solubility**:
  - Both LiTFSI and LiNO3 have solubility properties mentioned in 5 papers each [10.1016/j.cej.2021.129685; 10.1016/j.jallcom.2020.157122].

### 3. ADVANTAGES OF LiNO3 OVER LiTFSI
- **Coulombic Efficiency**:
  - LiNO3 affects coulombic efficiency positively, supported by 3 papers [10.1016/j.cej.2022.135283; 10.1016/j.electacta.2022.140212; 10.1016/j.ensm.2018.09.021].
- **Capacity Retention**:
  - LiNO3 affects capacity retention positively, supported by 3 papers [10.1016/j.electacta.2022.140212; 10.1016/j.ensm.2021.09.002; 10.1016/j.jpowsour.2021.230099].
- **Interface Impedance**:
  - LiNO3 reduces interface impedance, supported by 2 papers [10.1016/j.ensm.2021.09.002; 10.1016/j.jechem.2020.09.004].
- **Dendrite Formation**:
  - LiNO3 affects dendrite formation positively, supported by 2 papers [10.1016/j.ensm.2018.09.021; 10.1016/j.ssi.2020.115321].
- **Electrolyte Resistance**:
  - LiNO3 increases electrolyte resistance, supported by 1 paper [10.1016/j.ensm.2021.09.002].

### 4. LIMITATIONS OF LiNO3 COMPARED TO LiTFSI
- **Ionic Conductivity**:
  - LiNO3 has weak evidence of greater ionic conductivity, supported by only one paper [10.1016/j.jpowsour.2012.03.049].
- **Thermal Stability**:
  - There is no evidence for the thermal stability of LiNO3.
- **Glass Transition Temperature**:
  - There is no evidence for the effect of LiNO3 on the glass transition temperature.
- **Viscosity**:
  - There is no evidence for the effect of LiNO3 on viscosity.
- **Concentration**:
  - LiNO3 has specific concentration mentions (2 wt% and 1 wt%), but these are supported by only 2 papers each [10.1016/j.mtener.2020.100465; 10.1016/j.mtener.2022.100949; 10.1016/j.cej.2020.124848; 10.1016/j.mtener.2021.100871].

### 5. APPLICATION SUITABILITY
- **Best-Fit Applications**:
  - LiNO3 is suitable for applications where coulombic efficiency, capacity retention, and reduction of interface impedance are critical, such as in lithium metal batteries and aqueous rechargeable lithium batteries [10.1016/j.jpowsour.2020.227710; 10.1016/j.jpowsour.2012.03.049].
- **Challenging Applications**:
  - LiNO3 may face challenges in applications requiring high ionic conductivity and thermal stability, such as high-temperature or high-performance lithium-ion batteries [10.1016/j.electacta.2015.09.001; 10.1016/j.jpowsour.2016.10.105; 10.1016/j.jpowsour.2018.02.066].

### 6. REPLACEMENT CONSIDERATIONS
- **Formulation Changes**:
  - Replacing LiTFSI with LiNO3 may require adjustments in electrolyte formulations to maintain or improve ionic conductivity and thermal stability.
  - Compatibility with existing solvents and polymers (e.g., PEO, DOL/DME) is generally maintained, but specific concentrations and additives may need optimization.
- **Compatibility**:
  - Both salts are compatible with Celgard 2400 and PEO, suggesting that membrane and polymer compatibility issues are unlikely to be significant barriers to replacement [10.1016/j.jechem.2020.06.060; 10.1016/j.mtener.2021.100663].

### 7. RESEARCH GAPS
- **Thermal Stability**:
  - There is no evidence for the thermal stability of LiNO3.
- **Glass Transition Temperature**:
  - There is no evidence for the effect of LiNO3 on the glass transition temperature.
- **Viscosity**:
  - There is no evidence for the effect of LiNO3 on viscosity.
- **Concentration Effects**:
  - More detailed studies on the concentration effects of LiNO3 in various electrolytes are needed.
- **Operating Temperature Range**:
  - There is no evidence for the operating temperature range of LiNO3.

### 8. OVERALL ASSESSMENT
- **Viability as a LiTFSI Alternative**:
  - LiNO3 shows potential as a LiTFSI alternative in specific applications where coulombic efficiency, capacity retention, and interface impedance reduction are critical.
  - However, LiNO3's viability is limited by the lack of evidence on its thermal stability, glass transition temperature effects, and viscosity.
  - Further research is needed to fully assess LiNO3's performance in high-temperature and high-performance applications.
  - Replacement considerations include potential formulation changes to maintain ionic conductivity and thermal stability, but compatibility with existing materials is generally favorable.
```

#### LiClO4

```
### 1. SIMILARITY ASSESSMENT
- **Usage Contexts**:
  - Both LiClO4 and LiTFSI are used in battery cells, with LiTFSI having a significantly higher number of mentions (309 papers) compared to LiClO4 (85 papers) [10.1016/S0378-7753(01)00727-3; 10.1016/j.cej.2019.02.148].
  - Both salts are used with LiFePO4, but LiTFSI has more mentions in this context (2 papers) compared to LiClO4 (2 papers) [10.1016/j.cej.2021.130017; 10.1016/j.jpowsour.2009.11.147].
  - Both salts are compatible with Celgard 2400 separators, but LiTFSI has more mentions (2 papers) compared to LiClO4 (1 paper) [10.1016/j.jechem.2020.06.060; 10.1016/j.jallcom.2007.02.103].
  - Both salts are used in electrolyte solutions, but LiTFSI has more mentions (2 papers) compared to LiClO4 (7 papers) [10.1016/j.electacta.2019.03.051; 10.1016/j.est.2019.02.020].
  - Both salts co-occur with PEO, but LiTFSI has more mentions (66 papers) compared to LiClO4 (16 papers) [10.1016/S0378-7753(01)00727-3; 10.1016/j.carbon.2022.02.034].
  - Both salts are dissolved in acetonitrile, but LiTFSI has more mentions (33 papers) compared to LiClO4 (8 papers) [10.1016/j.cej.2020.126517; 10.1016/j.compscitech.2021.108837].

### 2. KEY DIFFERENCES
- **Ionic Conductivity**:
  - LiTFSI affects ionic conductivity more frequently (38 papers) compared to LiClO4 (5 papers) [10.1016/j.cej.2021.133352; 10.1016/j.electacta.2019.135133].
- **Thermal Stability**:
  - LiTFSI has more mentions of higher thermal stability (5 papers) compared to LiClO4 (2 papers) [10.1016/j.electacta.2015.09.001; 10.1016/j.electacta.2017.08.162].
- **Viscosity**:
  - LiTFSI increases viscosity (8 papers) more frequently than LiClO4 (2 papers) [10.1016/j.electacta.2010.03.043; 10.1016/j.electacta.2011.11.080].
- **Operating Temperature Range**:
  - LiTFSI has more specific operating temperature ranges mentioned (20 °C, 60 °C, room temperature, 25 °C) compared to LiClO4 (room temperature, 25 °C) [10.1016/j.jpowsour.2005.03.231; 10.1016/j.ceramint.2007.09.037].
- **Solvent Compatibility**:
  - LiTFSI is more frequently dissolved in DOL/DME (58 papers) compared to LiClO4, which is more frequently dissolved in PC (15 papers) [10.1016/j.carbon.2019.11.002; 10.1016/j.ceja.2021.100163].

### 3. ADVANTAGES OF LiClO4 OVER LiTFSI
- **Reduces Parasitic Current Density**:
  - LiClO4 reduces parasitic current density [10.1016/j.jpowsour.2007.09.085].
- **Increases Current Beyond 4.2 V**:
  - LiClO4 increases current beyond 4.2 V [10.1016/j.jpowsour.2007.09.085].
- **Reduces Capacity Fading**:
  - LiClO4 reduces capacity fading [10.1016/j.jpowsour.2009.04.033].

### 4. LIMITATIONS OF LiClO4 COMPARED TO LiTFSI
- **Ionic Conductivity**:
  - LiClO4 has fewer mentions of affecting ionic conductivity (5 papers) compared to LiTFSI (38 papers) [10.1016/j.electacta.2019.135133; 10.1016/j.cej.2021.133352].
- **Thermal Stability**:
  - LiClO4 has fewer mentions of thermal stability (2 papers) compared to LiTFSI (6 papers) [10.1016/j.electacta.2017.08.162; 10.1016/j.electacta.2019.135133].
- **Viscosity**:
  - LiClO4 has fewer mentions of affecting viscosity (2 papers) compared to LiTFSI (8 papers) [10.1016/j.electacta.2011.11.080; 10.1016/j.electacta.2010.03.043].

### 5. APPLICATION SUITABILITY
- **Best-Fit Applications**:
  - LiClO4 may be suitable for applications requiring reduced parasitic current density and increased current beyond 4.2 V [10.1016/j.jpowsour.2007.09.085].
  - LiClO4 may be suitable for applications where reducing capacity fading is important [10.1016/j.jpowsour.2009.04.033].
- **Challenging Applications**:
  - LiClO4 may face challenges in applications requiring high ionic conductivity and thermal stability, as these properties are less frequently mentioned for LiClO4 compared to LiTFSI [10.1016/j.electacta.2019.135133; 10.1016/j.cej.2021.133352].

### 6. REPLACEMENT CONSIDERATIONS
- **Formulation Changes**:
  - Replacing LiTFSI with LiClO4 may require changes in solvent formulations, as LiTFSI is more frequently dissolved in DOL/DME, while LiClO4 is more frequently dissolved in PC [10.1016/j.carbon.2019.11.002; 10.1016/j.ceja.2021.100163].
  - Compatibility with separators and other battery components may need to be reassessed, as LiTFSI has more mentions of compatibility with specific separators like Celgard 2400 and 2325 [10.1016/j.jechem.2020.06.060; 10.1016/j.jpowsour.2019.227547].

### 7. RESEARCH GAPS
- **Missing Relations for LiClO4**:
  - There is no evidence for LiClO4 affecting properties like glass transition temperature, crystallinity, or interface impedance, which are mentioned for LiTFSI [10.1016/j.electacta.2012.10.027; 10.1016/j.jechem.2020.04.025].
  - There is no evidence for LiClO4 co-occurring with LiNO3 or being used in SPEs, which are mentioned for LiTFSI [10.1016/j.apsusc.2021.150589; 10.1016/j.electacta.2014.04.099].
  - There is no evidence for LiClO4 being used with specific materials like PAN or BEMA, which are mentioned for LiTFSI [10.1016/j.jmat.2022.02.009; 10.1016/j.jpowsour.2015.12.001].

### 8. OVERALL ASSESSMENT
- **Viability as a LiTFSI Alternative**:
  - LiClO4 shows potential as a LiTFSI alternative in specific applications where reducing parasitic current density, increasing current beyond 4.2 V, and reducing capacity fading are important [10.1016/j.jpowsour.2007.09.085; 10.1016/j.jpowsour.2009.04.033].
  - However, LiClO4 may not be suitable for applications requiring high ionic conductivity and thermal stability, as these properties are less frequently mentioned for LiClO4 compared to LiTFSI [10.1016/j.electacta.2019.135133; 10.1016/j.cej.2021.133352].
  - Replacing LiTFSI with LiClO4 may require significant formulation changes and compatibility assessments, particularly in solvent selection and separator compatibility [10.1016/j.carbon.2019.11.002; 10.1016/j.jechem.2020.06.060].
```

#### Li2SO4

```
### 1. SIMILARITY ASSESSMENT
- **Usage Contexts**:
  - Both Li2SO4 and LiTFSI are used in battery cells [10.1016/S0378-7753(01)00727-3; 10.1016/j.apt.2017.03.031].
  - Both are used with LiFePO4 [10.1016/j.cej.2021.130017; 10.1016/j.electacta.2014.11.168].
  - LiTFSI is used in lithium-ion batteries, lithium batteries, and solid polymer electrolytes (SPEs) [10.1016/j.electacta.2015.09.001; 10.1016/j.cej.2021.132874; 10.1016/j.electacta.2014.04.099].
  - Li2SO4 is used in aqueous lithium-ion batteries and aqueous rechargeable lithium batteries [10.1016/j.jelechem.2007.01.007; 10.1016/j.ssi.2016.10.015].

### 2. KEY DIFFERENCES
- **Solvent Compatibility**:
  - LiTFSI is commonly dissolved in DOL/DME, acetonitrile, and PEO [10.1016/j.carbon.2019.11.002; 10.1016/j.cej.2020.126517; 10.1016/S0378-7753(01)00727-3].
  - Li2SO4 is primarily dissolved in water and aqueous solutions [10.1016/j.cap.2014.09.001; 10.1016/j.electacta.2014.10.148].
- **Operating Temperature Range**:
  - LiTFSI operates at a range of temperatures including 20°C, 60°C, room temperature, and 25°C [10.1016/j.jpowsour.2005.03.231; 10.1016/j.ensm.2018.09.020; 10.1016/j.electacta.2019.06.009; 10.1016/j.cej.2021.130017].
  - Li2SO4 operates at room temperature [10.1016/j.jelechem.2007.01.007; 10.1016/j.jpowsour.2018.11.035].
- **Electrochemical Properties**:
  - LiTFSI affects properties such as ionic conductivity, glass transition temperature, and viscosity [10.1016/j.cej.2021.133352; 10.1016/j.electacta.2012.10.027; 10.1016/j.electacta.2010.03.043].
  - Li2SO4 affects properties such as rate capability, electrochemical performance, and interface impedance [10.1016/j.electacta.2010.11.027; 10.1016/j.electacta.2014.10.148; 10.1016/j.nanoen.2020.104864].
- **Stability Window**:
  - Li2SO4 is stable in the window of -0.8 to 1.0 V [10.1016/j.electacta.2014.12.088; 10.1016/j.electacta.2015.01.139].
  - There is no explicit stability window mentioned for LiTFSI in the knowledge base.
- **Concentration**:
  - LiTFSI has a concentration property mentioned in several papers [10.1016/j.carbon.2019.11.022; 10.1016/j.cej.2022.134647; 10.1016/j.joule.2017.08.012].
  - Li2SO4 has a 1 M concentration mentioned in one paper [10.1016/j.jallcom.2018.01.083].

### 3. ADVANTAGES OF Li2SO4 OVER LiTFSI
- **Chemical Stability**:
  - Li2SO4 has good chemical stability [10.1016/j.hydromet.2012.11.013].
- **Promotes Formation of LFP**:
  - Li2SO4 promotes the formation of LFP [10.1016/j.apt.2017.03.031].

### 4. LIMITATIONS OF Li2SO4 COMPARED TO LiTFSI
- **Ionic Conductivity**:
  - Li2SO4 is not beneficial for Li-ion transport [10.1016/j.nanoen.2020.104864].
- **Operating Temperature Range**:
  - Li2SO4 is only mentioned to operate at room temperature, whereas LiTFSI has a broader operating temperature range [10.1016/j.jelechem.2007.01.007; 10.1016/j.jpowsour.2018.11.035].
- **Solvent Compatibility**:
  - Li2SO4 is primarily dissolved in water and aqueous solutions, which may limit its applicability in non-aqueous electrolyte systems [10.1016/j.cap.2014.09.001; 10.1016/j.electacta.2014.10.148].

### 5. APPLICATION SUITABILITY
- **Best-Fit Applications**:
  - Li2SO4 is well-suited for aqueous lithium-ion batteries and aqueous rechargeable lithium batteries due to its solubility in water and aqueous solutions [10.1016/j.jelechem.2007.01.007; 10.1016/j.ssi.2016.10.015].
- **Challenging Applications**:
  - Li2SO4 may face challenges in applications requiring non-aqueous electrolytes or broader temperature ranges, as it is primarily dissolved in water and operates at room temperature [10.1016/j.jelechem.2007.01.007; 10.1016/j.jpowsour.2018.11.035].

### 6. REPLACEMENT CONSIDERATIONS
- **Formulation Changes**:
  - Replacing LiTFSI with Li2SO4 would require a shift to aqueous electrolyte solutions, which may necessitate changes in battery design and materials compatibility [10.1016/j.cap.2014.09.001; 10.1016/j.electacta.2014.10.148].
  - Compatibility with existing separators and electrodes would need to be assessed, as LiTFSI is compatible with materials like Celgard 2400, Celgard 2325, PAN, and BEMA [10.1016/j.jechem.2020.06.060; 10.1016/j.jpowsour.2019.227547; 10.1016/j.jmat.2022.02.009; 10.1016/j.jpowsour.2015.12.001].

### 7. RESEARCH GAPS
- **Missing Relations for Li2SO4**:
  - There is no evidence for Li2SO4 affecting properties such as ionic conductivity, glass transition temperature, or viscosity, which are well-documented for LiTFSI [10.1016/j.cej.2021.133352; 10.1016/j.electacta.2012.10.027; 10.1016/j.electacta.2010.03.043].
  - There is no evidence for Li2SO4 co-occurring with solvents like DOL/DME, acetonitrile, or PEO, which are common for LiTFSI [10.1016/j.carbon.2019.11.002; 10.1016/j.cej.2020.126517; 10.1016/S0378-7753(01)00727-3].
  - There is no evidence for Li2SO4 affecting properties like Coulombic efficiency, thermal stability, or interface impedance, which are documented for LiTFSI [10.1016/j.electacta.2014.03.185; 10.1016/j.electacta.2019.135133; 10.1016/j.jechem.2020.06.060].

### 8. OVERALL ASSESSMENT
- **Viability as a LiTFSI Alternative**:
  - Li2SO4 shows potential as an alternative in specific applications, particularly in aqueous lithium-ion batteries and aqueous rechargeable lithium batteries. However, its limited operating temperature range and solubility primarily in water may restrict its broader applicability. Further research is needed to fully understand its electrochemical properties and compatibility with non-aqueous electrolyte systems.
```

---

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

---

## 10. Size-corrected Jchem companion ranking

**`jchem_normalized_ranking.py`** — extra analysis on top of
`jchem_results.csv` (does **not** modify the primary Jchem outputs).
Motivation: the weighted Jaccard punishes small MatKG profiles — LiBOB's
13 neighbors are a *perfect subset* of LiTFSI's 174, so its unweighted
Jaccard (13/174 = 0.075) is already the ceiling for a 13-neighbor
profile, yet it ranks 30/50. Two set-size-corrected companion columns:
- **ceiling-normalized Jaccard**: `jaccard_unweighted` divided by the
  best Jaccard attainable given the two profile sizes (min/max); 1.0 =
  "scored its ceiling."
- **overlap coefficient** (Szymkiewicz–Simpson): |A∩B| / min(|A|,|B|).

Both metrics inflate tiny profiles, so profiles with <10 neighbors are
flagged `[thin]`. Headline (≥10-neighbor profiles): **LiBOB is rank 1 by
both metrics** (1.0 — perfect subset), **LiOTf** 2 (overlap 0.97, 29/30
shared), then EMC, **LiClO4** and **LiPF6** (overlap 0.879 each), LiF,
DMC/AN/THF/VC/PC/DOL interleaved. The size correction confirms §6's
reading: LiBOB's low primary-Jchem rank is a MatKG-coverage artifact, and
the fluorine-free consensus salts (LiBOB, LiClO4) sit at the top once
profile size is controlled for. Thin-profile salts to note, not rank:
LiFSI (5/5 shared), LiBF4 (4/4), LiDFOB (2/2). Outputs:
`jchem_normalized_ranking.csv`, `jchem_normalized_report.txt`.

---

## 11. BONUS: Advanced Link Prediction Analysis (`battery_ontology_analysis/`)

#### `battery_ontology_link_plausibility.py` — relationship prediction benchmarking
- **Methods**: PMI baseline (statistical co-occurrence association),
  TF-IDF + k-NN (relationship fingerprints), graph-neighborhood
  prediction (structural similarity via shared neighbors).
- **Features**: memory-efficient processing, evaluation metrics
  (Precision, Recall, F1, AUC), balanced train/test split.
- **Measured scale**: 231,658 canonical relationships → 99,954 after
  filtering; graph of 12,846 nodes, 99,954 edges, average degree 15.56;
  38,697,814 entity pairs analyzed for PMI; TF-IDF over 9,526 entities /
  11,764 feature dimensions.
- **Execution**: `python battery_ontology_analysis/battery_ontology_link_plausibility.py`

#### `create_evaluation_worksheet.py` — manual evaluation setup
- **Input**: `battery_ontology_analysis/relationships_data_canonical_with_doi_*.csv`
- **Output**: `battery_ontology_evaluation_worksheet_*.xlsx`

---

## 12. Triple-quality evaluation — LLM-as-judge + human arm

Both application-KG datasets (battery KG, salt KG) were audited for
**correctness** (is the triple consistent with its source paragraph?).
Two LLM judges scored 300 triples each; one human then re-scored a
stratified 50-triple subsample as a third rater.

Scripts: `battery_kg/llm_triple_evaluation.py` (judges),
`battery_kg/analyze_evaluation_results.py` (per-dataset),
`battery_kg/create_human_agreement_worksheet.py` (blinded worksheet),
`battery_kg/analyze_human_agreement.py` (human arm).

### 12.1 Judge vs judge (n = 600)

| Dataset | Mistral PASS | LLaMA PASS | Agreement | Cohen's κ | Gwet AC1 | PABAK |
|---|---|---|---|---|---|---|
| battery_kg (300) | 86.3% | 90.0% | 90.3% | 0.538 | 0.878 | 0.807 |
| salt_kg (300) | 80.3% | 83.3% | 90.3% | 0.675 | 0.862 | 0.807 |
| **OVERALL (600)** | **83.3%** | **86.7%** | **90.3%** | **0.622** | 0.870 | 0.807 |

Judges: `mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` and
`meta-llama/llama-3-1-405b-instruct-fp8` (RITS), temperature 0, seed 42, no
ERROR responses. **"Mistral" in every evaluation table below means
Mistral-Large-3-675B, not the Mistral-Small-3.1-24B used for extraction and
commentary.** Both judges are larger, different models from the extractor,
so this is not self-assessment.

**κ is prevalence-suppressed, not a reliability problem.** The two
datasets have *identical* 90.3% raw agreement yet κ of 0.538 vs 0.675 —
the difference is entirely an artifact of the ~85% PASS base rate (the
kappa paradox). AC1 ≈ 0.87 on both. Always quote κ with AC1/PABAK here.

Mistral is the stricter judge in both datasets (3–4 pp lower PASS rate).
Judge disagreements: 58/600, split 39 (Mistral FAIL / LLaMA PASS) to 19
(Mistral PASS / LLaMA FAIL).

### 12.2 Errors are systematic, not uniform

Consensus-PASS rate (both judges PASS) by relation type, n ≥ 10:

| battery_kg | rate | salt_kg | rate |
|---|---|---|---|
| HAS COMPOSITION | **59%** | AFFECTS PROPERTY OF | **52%** |
| HAS VALUE | 83% | HAS PROPERTY | 70% |
| HAS PROPERTY | 85% | HAS ROLE | 80% |
| USED IN | 86% | USED IN | 81% |
| CO OCCURS WITH | 93% | HAS CONCENTRATION IN ELECTROLYTE | 82% |
| AFFECTS PROPERTY OF | **93%** | DISSOLVED IN | 83% |
| | | CO OCCURS WITH | 84% |
| | | STABLE IN WINDOW | 90% |

`AFFECTS PROPERTY OF` scores 93% in battery_kg and 52% in salt_kg — the
same relation label, opposite reliability, tracking the two different
extraction prompts (§2.1.1 vs §2.2). `CO OCCURS WITH`, which carries the
most weight in the Japp neighborhood comparison, sits at the reliable end
(84–93%).

### 12.3 Human arm (n = 50, stratified)

A plain random 50 would have contained ~5 judge-disagreement cases, too
few to characterize. The subsample is therefore stratified on the joint
judge cell and balanced across datasets; inclusion probabilities are
recorded so every population-level figure is design-weighted back to 600.

| Judge cell (Mistral/LLaMA) | Population | Sampled | Weight | Human FAIL | Human PASS |
|---|---|---|---|---|---|
| PASS/PASS | 481 | 20 | 24.05 | 0 | 20 |
| FAIL/FAIL | 61 | 10 | 6.10 | 10 | 0 |
| FAIL/PASS | 39 | 10 | 3.90 | 1 | 9 |
| PASS/FAIL | 19 | 10 | 1.90 | 1 | 9 |

**The human agrees with the judge consensus 30/30 — perfectly.** Wherever
both judges reached the same verdict, PASS or FAIL, the human reached it
too (κ = 1.000, 95% CI on agreement 0.886–1.000). *All* human–judge
disagreement is confined to the 20 triples where the judges disagreed
with each other; on those the human split 9 PASS / 1 FAIL in both
directions.

**Design-weighted human precision = 88.9% (95% CI 87.7–90.0%)**, against
Mistral 83.3% and LLaMA 86.7%. Both judges understate precision, but
only modestly (2–6 pp).

| Comparison | Weighted agreement | Weighted κ | Weighted AC1 |
|---|---|---|---|
| Human vs Mistral | 93.8% (92.7–95.0) | 0.744 | 0.919 |
| Human vs LLaMA | 96.5% (95.3–97.7) | 0.837 | 0.955 |
| Judge vs judge | 90.3% (87.7–92.4) | 0.622 | 0.870 |

Per dataset, same design weighting:

| Dataset | Human precision | H–Mistral agr. | H–LLaMA agr. | Judge–judge agr. |
|---|---|---|---|---|
| batteryKG | 92.4% | 92.7% | 97.6% | 90.3% |
| saltKG | 85.4% | 94.9% | 95.4% | 90.3% |
| **Overall** | **88.9%** | **93.8%** | **96.5%** | **90.3%** |

Weighted κ / AC1 per dataset: batteryKG H–Mistral 0.621/0.910, H–LLaMA
0.851/0.971; saltKG H–Mistral 0.822/0.929, H–LLaMA 0.826/0.938.

**Do not quote confidence intervals on the per-dataset human figures.**
Each rests on 25 annotations, and several strata have n=5 with zero
within-cell variation, which drives the stratum variance estimate to
exactly zero and understates the SE badly. The per-dataset κ spread
(0.621 vs 0.822 for H–Mistral) is mostly estimator noise — the agreement
figures move far less on the same data. Report it, do not interpret it.

Each judge agrees with the human *more* than the two judges agree with
each other — the human sits between them, and LLaMA is the closer of the
two. Judge-PASS remains near-perfectly reliable (weighted judge-PASS
precision 0.996 Mistral, 0.993 LLaMA): when a judge passes a triple, the
human effectively always passes it as well.

On the 20 split cases the human sided with Mistral exactly 10 times and
LLaMA 10 times: **no evidence either judge is the better arbiter.**
Fleiss' κ across all three raters is 0.411 in-sample.

In-sample figures (80.0% agreement, κ = 0.554 against each judge) are
reported by the script for transparency but are **not** population
estimates — the design deliberately oversamples the rare cells, which
inflates chance agreement to p_e = 0.552 and depresses raw κ.

### 12.4 What this means for the rankings

Triple precision of 88.9% sits inside the band where an aggregate
co-occurrence statistic is safe, and the relation carrying most of the
Japp signal (`CO OCCURS WITH`) is among the most reliable at 84–93%.

The consensus result is the operationally useful one: **judge consensus
was human-validated at 30/30 in this sample**, so restricting the KG to
consensus-PASS triples (481/600 = 80.2%) yields a subset whose verdicts a
human endorsed without exception. That makes the consensus subset the
natural basis for the sensitivity check below.

Caveats that remain open:

- **Single annotator.** Under a lenient rubric ("FAIL only if the triple
  *clearly* contradicts or cannot be supported"), one rater's precision
  estimate cannot be separated from that rater's calibration. A second
  annotator would give a human–human ceiling.
- **Precision only — recall is unmeasured.** For a Jaccard neighborhood
  comparison, missing edges distort similarity as directly as wrong ones.
- **The ranking has not yet been re-derived on the consensus-PASS
  subset.** Until that sensitivity check is run, "does LiNO3/LiClO4
  survive the error rate?" is unanswered.

---

## 13. Limitations and known hazards

- **MatKG ubiquity bias:** Jchem rewards compounds common across
  materials literature in any role (LiNO3, Li2CO3, LiF as additives, SEI
  components, precursors), and starves genuinely good but rare
  substitutes (LiFSI: 5 MatKG neighbors; LiBOB: 13). Japp compensates.
- **Abbreviation sense collisions** survive in free text even after the
  cased rule ('PES' = prop-1-ene-sultone or polyethersulfone; 'TMP',
  'DMC' have non-battery senses; 'MA+' is methylammonium). Every short
  match is itemized in the Jchem report for audit.
- **Japp absolute scores** are structurally deflated (§9); compare ranks,
  not magnitudes, and never across KGs.
- **The LLM judges mildly under-report precision** (§12.3): human-weighted
  precision is 88.9% vs the judges' 83.3%/86.7%. Treat judge PASS rates as
  a precision *floor*. Judge disagreement is the only place human–judge
  disagreement occurs — where the two judges concur, the human concurred
  30/30 — so consensus verdicts can be relied on and split verdicts cannot.
- **Recall is unmeasured.** All triple-quality numbers are precision-only;
  for a Jaccard neighborhood comparison, missing edges bias similarity as
  much as wrong ones.
- The salt-KG object side is raw LLM phrasing; consolidating it
  (polarity-aware property canonicalization) would raise absolute scores
  but is deliberately not applied in the current numbers.
- Isotopologues (6Li/7Li records) are separate components by design; the
  ambiguity rules operate on InChIKey connectivity blocks so shared
  names survive.
- The accepted set still contains ~10k pharma lithium carboxylates; kept
  because the family column makes exclusion trivial and the KG-match
  gate prunes them naturally (only 2 carboxylates score in Jchem).

---

## Setup

Tested with Python 3.9. Package versions used for the results are frozen
in `env/requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
```

The LLM steps (triple extraction, LLM-as-judge evaluation, candidate
commentary) call an OpenAI-compatible inference endpoint. Set:

```bash
export RITS_API_KEY=...      # API key for the endpoint
export RITS_BASE_URL=...     # base URL; the model path (e.g. /mistral-small-3-1-24b-2503/v1) is appended
```

Model IDs are listed under "Models used" above. All non-LLM steps run
without these variables.

---

## 14. Reproduction

```bash
python chemical_kg/fetch_li_universe.py        # stage 0 (+ optional "24 35" style window)
python chemical_kg/filter_funnel.py            # stages 1-2
python chemical_kg/merge_candidates.py
python chemical_kg/harvest_synonyms.py         # stage 3 (network; resumable)
python chemical_kg/build_synonym_index.py      # stage 4
python chemical_kg/jchem_weighted_jaccard.py   # stage 5 -> Jchem
python chemical_kg/appkg_salt_ranking.py       # -> Japp
python chemical_kg/appkg_battery_audit.py      # battery-KG cross-check
python chemical_kg/appkg_candidate_profiles.py # DOI-attributed candidate profiles
python chemical_kg/llm_candidate_commentary.py # LLM commentary (needs RITS_API_KEY)
python chemical_kg/jchem_normalized_ranking.py # size-corrected Jchem companion
```

Needs rdkit, requests, pandas; login-node friendly. Editable inputs:
`electrolyte_solvents_additives.csv` (curated seeds),
`manual_synonyms.csv` (alias additions), `manual_blocklist.csv`
(known-wrong synonym kills).

Battery/salt ontology extraction and canonicalization (upstream of the
above; §2): `ontology_extraction_triples_salt_with_doi.py`,
`ontology_extraction_triples_battery.py`,
`battery_ontology_cleaning_with_doi.py`.

---

## 15. Output inventory (`chemical_kg/output/`)

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
| `candidate_claims.csv`, `candidate_evidence.csv`, `candidate_profiles_report.txt` | per-claim DOI-attributed candidate profiles (§7.2) |
| `candidate_commentary_report.txt`, `candidate_commentary.json` | LLM substitution commentary with DOI citations (§8) |
| `jchem_normalized_ranking.csv`, `jchem_normalized_report.txt` | size-corrected Jchem companion ranking (§10) |

### Upstream ontology/canonicalization outputs

| File | Contents |
|---|---|
| `battery_ontology_triples_with_doi_clean_20250819_214710.csv/.json` | raw battery-ontology extraction (316,846 CSV rows; **316,841 valid triples** — see §2.1) |
| `salt_analysis_output/enhanced_salt_extraction_raw_20250815_225432.json` | raw salt-ontology extraction (27,993 triples) |
| `salt_analysis_output/canonical/salt_relationships_canonical_with_source.csv` | entity-canonicalized saltKG (27,993 rows; feeds §3) |
| `salt_analysis_output/canonical/salt_relationships_canonical.csv` | triple-deduplicated saltKG (14,123 rows) |
| `salt_analysis_output/canonical/canonicalization_statistics.json`, `salt_canonicalization_report.txt` | saltKG canonicalization stats |
| `battery_ontology_analysis/relationships_data_canonical_with_doi_*.csv` | canonicalized batteryKG (231,658 rows) |
| `battery_ontology_analysis/entity_canonicalization_mappings*.json`, `canonicalization_report*.txt` | batteryKG canonicalization mappings + report |
| `battery_ontology_evaluation_worksheet_*.xlsx` | manual evaluation worksheet (§11) |

### Triple-quality evaluation outputs (`evaluation_output/`, gitignored)

| File | Contents |
|---|---|
| `evaluation_combined_*.csv` | 600 judged triples, both judges' verdicts (§12.1) |
| `evaluation_{mistral_large,llama_405b}_*.json` | per-judge raw responses with reasoning |
| `evaluation_report_*.txt`, `inter_annotator_agreement_*.json` | judge-vs-judge summary |
| `per_dataset_analysis_*.txt/.json` | per-dataset PASS/FAIL + κ |
| `human_worksheet_*.xlsx/.csv` | blinded 50-triple annotation worksheet (§12.3) |
| `human_worksheet_key_*.csv` | judge labels + design weights — **unblind only after annotating** |
| `human_agreement_report_*.txt/.json` | human-arm results (§12.3) |

---

## Next steps

- Optional Jaccard-side improvements currently *not* applied: token-bag
  (order-free) matching tier, plural stemming, near-miss review report,
  polarity-aware neighbor canonicalization. (Ceiling-normalized scores
  now exist as the §10 companion analysis; the primary ranking is
  unchanged.)
- **Re-derive the Japp ranking on the consensus-PASS subset** (481/600
  triples) and confirm LiNO3/LiClO4 hold the top two. This is the
  sensitivity check that answers "does the conclusion survive the error
  rate?" — currently the main open item from §12.4.
- **Second human annotator** on the same 50 triples, to separate judge
  over-strictness from annotator leniency and to give a human–human
  agreement ceiling.
- Hypergeometric significance testing (computed ad hoc, not committed):
  LiNO3 back to rank ~6 (−log10 p ≈ 240), LiBOB drops to ~rank 35
  (≈31) against N=39,374 allowed-subgraph entities — no single scalar
  crowns both agreement and containment; a two-axis presentation
  (agreement vs. containment) plus a significance gate is recommended
  but not yet implemented.
