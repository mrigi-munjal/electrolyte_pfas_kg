# PFAS-SPARK: Substitution Pipeline using Application-level and Chemical-level Relational Knowledge Graphs for PFAS Components in Battery Electrolytes

This repository identifies PFAS-free substitutes for **LiTFSI** (lithium
bis(trifluoromethanesulfonyl)imide) in battery electrolytes. It compares
each candidate's neighbourhood with LiTFSI's in two independent knowledge
graphs:

- **Jchem**: a chemical-level KG, built from MatKG's materials-science literature co-occurrence graph.
- **Japp**: an application-level KG, LLM-extracted from battery-electrolyte paper paragraphs with DOI provenance.

The resulting ranking is cross-checked against a separate battery-system
KG and against an independent literature hold-out.

## Result

Five salts rank in the top 10 of both views:

| Salt | Jchem rank (score) | Japp rank (score) | Fluorine status |
|---|---|---|---|
| LiNO3 | 1 (0.458) | 6 (0.0222) | fluorine-free |
| LiClO4 | 4 (0.233) | 3 (0.0570) | fluorine-free |
| Li2SO4 | 3 (0.235) | 9 (0.0135) | fluorine-free |
| LiPF6 | 7 (0.221) | 1 (0.0800) | fluorinated incumbent |
| LiOTf | 8 (0.178) | 8 (0.0189) | PFAS (CF3) |

LiPF6 and LiOTf are fluorinated, so they cannot be substitutes. That
leaves **LiNO3 and LiClO4** as the top two candidates, with Li2SO4 third.
Li2SO4's supporting claims are mostly aqueous chemistry, not aprotic
electrolyte use.

**Independent check.** The review by Wang et al., *npj Materials
Sustainability* (2025), [doi:10.1038/s44296-025-00074-8](https://doi.org/10.1038/s44296-025-00074-8),
names LiBOB, LiClO4, LiNO3 and cyano-based anions as the leading
fluorine-free routes. It appears in neither KG's source corpus. LiBOB is
Japp rank 4; it is low in Jchem only because MatKG covers it sparsely.

Japp and Jchem scores are on different scales: compare ranks, not
magnitudes.

## Method

**Jchem** (`chemical_kg/`):

1. Collect all lithium-containing PubChem compounds (83,740).
2. Filter them with RDKit rules (no LLM) to 15,926 Li salts, plus 35 curated solvents and additives, giving 15,961 components keyed by InChIKey.
3. Harvest PubChem synonyms and clean them into a matching vocabulary of 20,303 synonyms.
4. Match components to MatKG entities with a token-boundary rule that tolerates MatKG's spacing and casing. Short abbreviations must also agree on case.
5. Score each component against LiTFSI by weighted Jaccard over their MatKG neighbourhoods:

```
J = Σ_{j ∈ A∩B} min(w_A(j), w_B(j)) / Σ_{j ∈ A∪B} max(w_A(j), w_B(j))
```

Here `w(j)` is the MatKG co-occurrence `Count`, which is the number of distinct papers.

**Japp** (`battery_kg/`, `battery_ontology_analysis/`, `chemical_kg/appkg_*.py`):

1. Mistral-Small-3.1-24B extracts triples from literature paragraphs against two ontologies:
   - a battery-system ontology: 316,841 triples, 129 relation types, 2,088 papers;
   - a salt-centred ontology: 27,993 triples, 302 salt subjects, 10 relation types.
2. Entities are canonicalised by name and PubChem CID.
3. Salt mentions are remapped onto the same InChIKey components as Jchem. This merges split identities, for example five separate LiTFSI forms. Novel salts with no PubChem record are kept as named pseudo-components.
4. Japp is the same weighted Jaccard, where `w(o)` counts how often a salt is stated with object `o`.

**Validation:**

- **Battery-KG cross-check** (`appkg_battery_audit.py`): ranking on the battery-system KG gives the same top three as Japp (LiPF6, LiFSI, LiClO4).
- **Candidate profiles** (`appkg_candidate_profiles.py`): per-claim, DOI-attributed evidence for each candidate.
- **Citation-grounded LLM commentary** (`llm_candidate_commentary.py`).
- **Size-corrected Jchem** (`jchem_normalized_ranking.py`): with profile size controlled for, LiBOB and LiClO4 rise to the top.

**Triple quality** (`battery_kg/llm_triple_evaluation.py`, `analyze_human_agreement.py`):

- Two LLM judges, different from and larger than the extractor, scored 600 triples (300 per KG): Mistral-Large-3-675B, 83.3% PASS, and LLaMA-3.1-405B, 86.7% PASS.
- The judges agreed 90.3% of the time. Cohen's κ is 0.622 and Gwet's AC1 is 0.870; κ is depressed by the ~85% PASS prevalence.
- A stratified 50-triple human subsample gives a design-weighted precision of **88.9%**. Where both judges agreed, the human agreed 30/30.

## Models

All LLM calls use an OpenAI-compatible endpoint.

| Task | Model | Script |
|---|---|---|
| Triple extraction | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `battery_kg/ontology_extraction_triples_battery.py`, `battery_kg/enhanced_salt_ontology_extraction_with_doi.py` |
| Judge 1 | `mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` | `battery_kg/llm_triple_evaluation.py` |
| Judge 2 | `meta-llama/llama-3-1-405b-instruct-fp8` | `battery_kg/llm_triple_evaluation.py` |
| Commentary | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `chemical_kg/llm_candidate_commentary.py` |

The judges run at temperature 0 with seed 42. Extraction runs at temperature 0 and commentary at 0.1.

## Repository layout

| Path | Contents |
|---|---|
| `chemical_kg/` | Candidate universe, synonym index, Jchem, Japp, cross-check, candidate profiles, commentary (see `chemical_kg/README.md`) |
| `battery_kg/` | LLM triple extraction and triple-quality evaluation |
| `battery_ontology_analysis/` | Battery-KG canonicalisation and link-prediction analysis |
| `salt_analysis_output/` | Salt-KG summary reports and figures |
| `env/requirements.txt` | Frozen package versions |

The root-level `jaccard_calculation*.py`, `battery_kg/jaccard_calculation_application_kg.py`
and `battery_kg/llm_querying_of_candidates.py` are earlier versions, kept
for provenance. They are superseded by the scripts above.

## Setup

Python 3.9:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
```

The LLM steps need the following environment variables; nothing else does:

```bash
export RITS_API_KEY=...      # API key for the endpoint
export RITS_BASE_URL=...     # base URL; the model path (e.g. /mistral-small-3-1-24b-2503/v1) is appended
```

**Input data are not included.** You need:

- MatKG's `SUBRELOBJ.csv`;
- the source paragraphs, `battery_kg/t2br_data/raw_paragraphs_v2.xlsx`.

PubChem data is fetched by the scripts.

## Reproduction

```bash
# Application KG (needs the LLM endpoint)
python battery_kg/ontology_extraction_triples_battery.py
python battery_kg/enhanced_salt_ontology_extraction_with_doi.py
python battery_ontology_analysis/battery_ontology_cleaning_with_doi.py
python salt_analysis_output/salt_ontology_cleaning_with_source.py

# Chemical KG and rankings
python chemical_kg/fetch_li_universe.py        # stage 0 (PubChem; resumable)
python chemical_kg/filter_funnel.py            # stages 1-2
python chemical_kg/merge_candidates.py
python chemical_kg/harvest_synonyms.py         # stage 3 (PubChem; resumable)
python chemical_kg/build_synonym_index.py      # stage 4
python chemical_kg/jchem_weighted_jaccard.py   # Jchem
python chemical_kg/appkg_salt_ranking.py       # Japp
python chemical_kg/appkg_battery_audit.py      # battery-KG cross-check
python chemical_kg/appkg_candidate_profiles.py # DOI-attributed candidate profiles
python chemical_kg/jchem_normalized_ranking.py # size-corrected Jchem
python chemical_kg/llm_candidate_commentary.py # LLM commentary (needs the endpoint)

# Triple-quality evaluation
python battery_kg/llm_triple_evaluation.py
python battery_kg/analyze_human_agreement.py
```

Outputs are written to `chemical_kg/output/`, `evaluation_output/` and
alongside the inputs.

## Limitations

- **MatKG ubiquity bias.** Jchem favours compounds that are common in any role, such as additives and SEI components. It under-scores rare but relevant salts, such as LiFSI and LiBOB. Japp compensates.
- **Low absolute Japp scores.** Most salt-KG neighbours are paper-specific phrases, so absolute Japp scores are structurally low. Use ranks.
- **Precision only.** Triple quality was measured for precision only; recall is unmeasured.
- **One human annotator.** The human subsample was scored by a single annotator.

## License

MIT; see `LICENSE`.
