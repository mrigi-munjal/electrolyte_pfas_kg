#!/usr/bin/env python3
"""
Analyze human-vs-LLM-judge agreement for the triple evaluation.

Reads the filled-in worksheet produced by create_human_agreement_worksheet.py,
joins it back to the judge labels, and reports:

  1. Judge-vs-judge agreement on the full 600 (raw agreement, Cohen's kappa,
     Gwet's AC1, PABAK, bootstrap CIs, per dataset).
  2. Human-vs-judge agreement on the 50 annotated triples -- both as observed
     in the stratified sample and reweighted to the full 600, since the sample
     deliberately oversamples the rare judge cells.
  3. Three-rater Fleiss' kappa on the 50.
  4. Arbitration: when the two judges disagreed, which one the human sided with.
  5. The human estimate of triple precision (PASS rate) for the KG.

Why both kappa and AC1: the PASS rate here is ~85%, and Cohen's kappa is
strongly depressed by that prevalence (the kappa paradox). AC1 and PABAK are
reported alongside so a modest kappa is not misread as a weak evaluation.

Usage:
    python battery_kg/analyze_human_agreement.py \
        --worksheet evaluation_output/human_worksheet_20260814_120000.xlsx
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "evaluation_output"
N_BOOTSTRAP = 5000
RANDOM_SEED = 42


# ============================================================
# AGREEMENT STATISTICS
# ============================================================

def interpret_kappa(k: float) -> str:
    if np.isnan(k):
        return "Unable to compute"
    if k < 0:
        return "Poor (less than chance)"
    if k < 0.20:
        return "Slight agreement"
    if k < 0.40:
        return "Fair agreement"
    if k < 0.60:
        return "Moderate agreement"
    if k < 0.80:
        return "Substantial agreement"
    return "Almost perfect agreement"


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    """Gwet's AC1 for two raters and binary labels.

    Unlike Cohen's kappa, the chance-agreement term does not collapse when one
    category dominates, so it stays interpretable at high prevalence.
    """
    po = float(np.mean(a == b))
    pi = (np.mean(a) + np.mean(b)) / 2.0
    pe = 2.0 * pi * (1.0 - pi)
    if np.isclose(pe, 1.0):
        return float("nan")
    return (po - pe) / (1.0 - pe)


def kappa_from_table(table: np.ndarray) -> float:
    """Cohen's kappa from a 2x2 table of (possibly weighted) proportions."""
    total = table.sum()
    if total <= 0:
        return float("nan")
    p = table / total
    po = np.trace(p)
    pe = float(p.sum(axis=0) @ p.sum(axis=1))
    if np.isclose(pe, 1.0):
        return float("nan")
    return (po - pe) / (1.0 - pe)


def ac1_from_table(table: np.ndarray) -> float:
    total = table.sum()
    if total <= 0:
        return float("nan")
    p = table / total
    po = np.trace(p)
    pi = (p.sum(axis=0)[1] + p.sum(axis=1)[1]) / 2.0
    pe = 2.0 * pi * (1.0 - pi)
    if np.isclose(pe, 1.0):
        return float("nan")
    return (po - pe) / (1.0 - pe)


def fleiss_kappa_binary(ratings: np.ndarray) -> float:
    """Fleiss' kappa for n items x r raters, binary 0/1 labels."""
    n_items, n_raters = ratings.shape
    if n_items == 0 or n_raters < 2:
        return float("nan")
    counts = np.column_stack([(ratings == 0).sum(axis=1), (ratings == 1).sum(axis=1)])
    p_i = (counts * (counts - 1)).sum(axis=1) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_cat = counts.sum(axis=0) / (n_items * n_raters)
    pe = float((p_cat ** 2).sum())
    if np.isclose(pe, 1.0):
        return float("nan")
    return (p_bar - pe) / (1.0 - pe)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, stat_fn, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Percentile bootstrap CI for a paired agreement statistic."""
    rng = np.random.RandomState(seed)
    n = len(a)
    if n < 2:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            v = stat_fn(a[idx], b[idx])
        except Exception:
            v = np.nan
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score interval -- behaves at the small counts in the rare cells."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def stratified_proportion(df: pd.DataFrame, indicator: np.ndarray):
    """Design-weighted proportion and SE under stratified sampling by judge cell.

    p_hat = sum_h W_h * p_h, with a finite population correction, so the number
    is an estimate over all 600 triples rather than over the 50 sampled ones.
    """
    df = df.assign(_ind=indicator.astype(float))
    total_pop = df.groupby("judge_cell")["cell_population"].first().sum()
    p_hat, var = 0.0, 0.0
    for cell, g in df.groupby("judge_cell"):
        big_n = float(g["cell_population"].iloc[0])
        small_n = len(g)
        w = big_n / total_pop
        p_h = g["_ind"].mean()
        p_hat += w * p_h
        if small_n > 1:
            fpc = 1.0 - small_n / big_n
            var += (w ** 2) * fpc * p_h * (1 - p_h) / (small_n - 1)
    se = float(np.sqrt(var))
    return float(p_hat), se, (max(0.0, p_hat - 1.96 * se), min(1.0, p_hat + 1.96 * se))


def weighted_confusion(df: pd.DataFrame, col_a: str, col_b: str) -> np.ndarray:
    """2x2 table of design-weighted counts, rows = col_a, cols = col_b (FAIL, PASS)."""
    table = np.zeros((2, 2))
    for _, row in df.iterrows():
        i = 1 if row[col_a] == "PASS" else 0
        j = 1 if row[col_b] == "PASS" else 0
        table[i, j] += row["design_weight"]
    return table


# ============================================================
# I/O
# ============================================================

def load_worksheet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return pd.read_excel(path, sheet_name="Annotation")
    return pd.read_csv(path)


def find_key(worksheet_path: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    m = re.search(r"(\d{8}_\d{6})", worksheet_path.name)
    if m:
        candidate = worksheet_path.parent / f"human_worksheet_key_{m.group(1)}.csv"
        if candidate.exists():
            return candidate
    keys = list(OUTPUT_DIR.glob("human_worksheet_key_*.csv"))
    if not keys:
        print("ERROR: no human_worksheet_key_*.csv found; pass --key")
        sys.exit(1)
    return max(keys, key=lambda f: f.stat().st_mtime)


def find_latest_evaluation_file() -> Path:
    files = list(OUTPUT_DIR.glob("evaluation_combined_*.csv"))
    if not files:
        print("ERROR: no evaluation_combined_*.csv found in evaluation_output/")
        sys.exit(1)
    return max(files, key=lambda f: f.stat().st_mtime)


# ============================================================
# ANALYSIS SECTIONS
# ============================================================

def judge_vs_judge(full: pd.DataFrame) -> dict:
    out = {}
    for name, g in list(full.groupby("dataset")) + [("OVERALL", full)]:
        m = (g["mistral_correctness"] == "PASS").to_numpy().astype(int)
        l = (g["llama_correctness"] == "PASS").to_numpy().astype(int)
        po = float(np.mean(m == l))
        k = float(cohen_kappa_score(m, l))
        out[name] = {
            "n": int(len(g)),
            "mistral_pass_rate": float(m.mean()),
            "llama_pass_rate": float(l.mean()),
            "observed_agreement": po,
            "agreement_ci": wilson_ci(int((m == l).sum()), len(g)),
            "cohens_kappa": k,
            "kappa_ci": bootstrap_ci(m, l, cohen_kappa_score),
            "kappa_interpretation": interpret_kappa(k),
            "gwet_ac1": gwet_ac1(m, l),
            "pabak": 2 * po - 1,
        }
    return out


def human_vs_judges(merged: pd.DataFrame) -> dict:
    human = (merged["human_correctness"] == "PASS").to_numpy().astype(int)
    res = {}

    for judge, col in (("mistral", "mistral_correctness"), ("llama", "llama_correctness")):
        j = (merged[col] == "PASS").to_numpy().astype(int)
        po = float(np.mean(human == j))
        k = float(cohen_kappa_score(human, j)) if len(set(human)) > 1 or len(set(j)) > 1 else float("nan")
        w_table = weighted_confusion(merged, "human_correctness", col)
        p_w, se_w, ci_w = stratified_proportion(merged, human == j)
        res[judge] = {
            "sample": {
                "n": int(len(merged)),
                "observed_agreement": po,
                "agreement_ci": wilson_ci(int((human == j).sum()), len(merged)),
                "cohens_kappa": k,
                "kappa_ci": bootstrap_ci(human, j, cohen_kappa_score),
                "kappa_interpretation": interpret_kappa(k),
                "gwet_ac1": gwet_ac1(human, j),
                "confusion": {
                    "human_PASS_judge_PASS": int(((human == 1) & (j == 1)).sum()),
                    "human_PASS_judge_FAIL": int(((human == 1) & (j == 0)).sum()),
                    "human_FAIL_judge_PASS": int(((human == 0) & (j == 1)).sum()),
                    "human_FAIL_judge_FAIL": int(((human == 0) & (j == 0)).sum()),
                },
            },
            "weighted_to_population": {
                "observed_agreement": p_w,
                "se": se_w,
                "ci": ci_w,
                "cohens_kappa": kappa_from_table(w_table),
                "gwet_ac1": ac1_from_table(w_table),
                "judge_precision_vs_human": (
                    float(w_table[1, 1] / w_table[:, 1].sum())
                    if w_table[:, 1].sum() > 0 else float("nan")
                ),
                "judge_recall_vs_human": (
                    float(w_table[1, 1] / w_table[1, :].sum())
                    if w_table[1, :].sum() > 0 else float("nan")
                ),
            },
        }

    # Consensus arm: only the triples where both judges said the same thing.
    consensus = merged[merged["mistral_correctness"] == merged["llama_correctness"]]
    if len(consensus) > 1:
        h = (consensus["human_correctness"] == "PASS").to_numpy().astype(int)
        c = (consensus["mistral_correctness"] == "PASS").to_numpy().astype(int)
        po = float(np.mean(h == c))
        res["consensus"] = {
            "n": int(len(consensus)),
            "observed_agreement": po,
            "agreement_ci": wilson_ci(int((h == c).sum()), len(consensus)),
            "cohens_kappa": float(cohen_kappa_score(h, c)),
            "gwet_ac1": gwet_ac1(h, c),
        }

    ratings = np.column_stack([
        human,
        (merged["mistral_correctness"] == "PASS").to_numpy().astype(int),
        (merged["llama_correctness"] == "PASS").to_numpy().astype(int),
    ])
    res["fleiss_kappa_three_raters"] = fleiss_kappa_binary(ratings)
    return res


def arbitration(merged: pd.DataFrame) -> dict:
    """Where the judges split, whose side does the human take?"""
    split = merged[merged["mistral_correctness"] != merged["llama_correctness"]]
    if len(split) == 0:
        return {"n": 0}
    sides_mistral = int((split["human_correctness"] == split["mistral_correctness"]).sum())
    sides_llama = int((split["human_correctness"] == split["llama_correctness"]).sum())
    n = len(split)
    return {
        "n": n,
        "human_sides_with_mistral": sides_mistral,
        "human_sides_with_mistral_rate": sides_mistral / n,
        "human_sides_with_mistral_ci": wilson_ci(sides_mistral, n),
        "human_sides_with_llama": sides_llama,
        "human_sides_with_llama_rate": sides_llama / n,
        "human_sides_with_llama_ci": wilson_ci(sides_llama, n),
        "by_cell": {
            cell: {
                "n": int(len(g)),
                "human_PASS": int((g["human_correctness"] == "PASS").sum()),
            }
            for cell, g in split.groupby("judge_cell")
        },
    }


def human_precision(merged: pd.DataFrame) -> dict:
    """Human PASS rate, reweighted to the full 600 -- the human precision estimate."""
    ind = (merged["human_correctness"] == "PASS").to_numpy()
    p, se, ci = stratified_proportion(merged, ind)
    return {
        "sample_pass_rate": float(ind.mean()),
        "weighted_pass_rate": p,
        "se": se,
        "ci": ci,
    }


# ============================================================
# REPORT
# ============================================================

def fmt_ci(ci) -> str:
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return "[n/a]"
    return f"[{lo:.3f}, {hi:.3f}]"


def build_report(jvj, hvj, arb, prec, merged, timestamp, worksheet_file, n_full) -> str:
    r = []
    r.append("=" * 78)
    r.append("TRIPLE EVALUATION -- JUDGE AND HUMAN AGREEMENT")
    r.append(f"Generated: {timestamp}")
    r.append(f"Worksheet: {worksheet_file}")
    r.append("=" * 78)

    r.append("")
    r.append("1. JUDGE vs JUDGE (Mistral-Large-3-675B vs LLaMA-3.1-405B)")
    r.append(f"   Full judged set, n = {n_full}")
    r.append("-" * 78)
    r.append(f"{'Dataset':<12} {'n':>5} {'MistralP':>9} {'LlamaP':>8} {'Agree':>7} "
             f"{'Kappa':>7} {'AC1':>7} {'PABAK':>7}")
    for name, s in jvj.items():
        r.append(
            f"{name:<12} {s['n']:>5} {s['mistral_pass_rate']:>9.3f} "
            f"{s['llama_pass_rate']:>8.3f} {s['observed_agreement']:>7.3f} "
            f"{s['cohens_kappa']:>7.3f} {s['gwet_ac1']:>7.3f} {s['pabak']:>7.3f}"
        )
    o = jvj["OVERALL"]
    r.append("")
    r.append(f"   Overall agreement {o['observed_agreement']:.1%} "
             f"(95% CI {fmt_ci(o['agreement_ci'])})")
    r.append(f"   Cohen's kappa {o['cohens_kappa']:.3f} "
             f"(95% CI {fmt_ci(o['kappa_ci'])}) -- {o['kappa_interpretation']}")
    r.append(f"   Gwet's AC1 {o['gwet_ac1']:.3f}, PABAK {o['pabak']:.3f}")
    r.append("   Kappa is held down by the high PASS prevalence (~85%); AC1 and")
    r.append("   PABAK are reported so that is visible rather than implicit.")

    r.append("")
    r.append("2. HUMAN vs LLM JUDGES")
    r.append(f"   Annotated sample n = {len(merged)}, stratified on the judge cell")
    r.append("   and reweighted to the full judged set.")
    r.append("-" * 78)
    for judge in ("mistral", "llama"):
        s = hvj[judge]["sample"]
        w = hvj[judge]["weighted_to_population"]
        r.append("")
        r.append(f"   HUMAN vs {judge.upper()}")
        r.append(f"     In-sample agreement    {s['observed_agreement']:.1%} "
                 f"(95% CI {fmt_ci(s['agreement_ci'])})")
        r.append(f"     Weighted agreement     {w['observed_agreement']:.1%} "
                 f"(95% CI {fmt_ci(w['ci'])})  <- population estimate")
        r.append(f"     Cohen's kappa          {s['cohens_kappa']:.3f} in-sample, "
                 f"{w['cohens_kappa']:.3f} weighted")
        r.append(f"     Gwet's AC1             {s['gwet_ac1']:.3f} in-sample, "
                 f"{w['gwet_ac1']:.3f} weighted")
        c = s["confusion"]
        r.append(f"     Sample confusion       human PASS/judge PASS {c['human_PASS_judge_PASS']}, "
                 f"human PASS/judge FAIL {c['human_PASS_judge_FAIL']},")
        r.append(f"                            human FAIL/judge PASS {c['human_FAIL_judge_PASS']}, "
                 f"human FAIL/judge FAIL {c['human_FAIL_judge_FAIL']}")
        r.append(f"     Judge PASS precision   {w['judge_precision_vs_human']:.3f} "
                 f"(weighted; of triples the judge passed, share the human passed)")

    if "consensus" in hvj:
        s = hvj["consensus"]
        r.append("")
        r.append(f"   HUMAN vs JUDGE CONSENSUS (both judges agreed, n = {s['n']})")
        r.append(f"     Agreement {s['observed_agreement']:.1%} "
                 f"(95% CI {fmt_ci(s['agreement_ci'])}), kappa {s['cohens_kappa']:.3f}, "
                 f"AC1 {s['gwet_ac1']:.3f}")

    r.append("")
    r.append(f"   Fleiss' kappa across all three raters: "
             f"{hvj['fleiss_kappa_three_raters']:.3f} (in-sample)")

    r.append("")
    r.append("3. ARBITRATION -- where the two judges split")
    r.append("-" * 78)
    if arb["n"] == 0:
        r.append("   No judge disagreements in the annotated sample.")
    else:
        r.append(f"   Split cases annotated: {arb['n']}")
        r.append(f"     Human sides with Mistral {arb['human_sides_with_mistral']}/{arb['n']} "
                 f"({arb['human_sides_with_mistral_rate']:.1%}, "
                 f"95% CI {fmt_ci(arb['human_sides_with_mistral_ci'])})")
        r.append(f"     Human sides with LLaMA   {arb['human_sides_with_llama']}/{arb['n']} "
                 f"({arb['human_sides_with_llama_rate']:.1%}, "
                 f"95% CI {fmt_ci(arb['human_sides_with_llama_ci'])})")
        for cell, s in arb["by_cell"].items():
            r.append(f"     cell {cell:<12} n={s['n']:<3} human PASS = {s['human_PASS']}")

    r.append("")
    r.append("4. HUMAN ESTIMATE OF TRIPLE PRECISION")
    r.append("-" * 78)
    r.append(f"   Raw PASS rate in the stratified sample: {prec['sample_pass_rate']:.1%} "
             "(NOT an unbiased estimate -- rare cells are oversampled)")
    r.append(f"   Design-weighted PASS rate over all {n_full} triples: "
             f"{prec['weighted_pass_rate']:.1%} "
             f"(95% CI {fmt_ci(prec['ci'])})")
    r.append(f"   For comparison: Mistral {jvj['OVERALL']['mistral_pass_rate']:.1%}, "
             f"LLaMA {jvj['OVERALL']['llama_pass_rate']:.1%}")

    r.append("")
    r.append("=" * 78)
    r.append("NOTES")
    r.append("- The 50 triples were sampled with unequal probability across the four")
    r.append("  judge cells; every population-level number above is design-weighted.")
    r.append("  In-sample figures are reported next to them for transparency only.")
    r.append("- One annotator, so human-vs-judge numbers carry that annotator's bias.")
    r.append("  A second annotator on the same 50 would let human-human agreement be")
    r.append("  quoted as an upper bound on what any judge can reach.")
    r.append("=" * 78)
    return "\n".join(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze human vs LLM-judge agreement")
    parser.add_argument("--worksheet", "-w", required=True, help="filled worksheet (xlsx or csv)")
    parser.add_argument("--key", "-k", default=None, help="human_worksheet_key_*.csv")
    parser.add_argument("--evaluation", "-e", default=None, help="evaluation_combined CSV")
    parser.add_argument("--output", "-o", default=None, help="report path")
    args = parser.parse_args()

    worksheet_path = Path(args.worksheet)
    if not worksheet_path.exists():
        print(f"ERROR: worksheet not found: {worksheet_path}")
        sys.exit(1)

    key_path = find_key(worksheet_path, args.key)
    eval_path = Path(args.evaluation) if args.evaluation else find_latest_evaluation_file()

    ws = load_worksheet(worksheet_path)
    key = pd.read_csv(key_path)
    full = pd.read_csv(eval_path)
    full = full[
        full["mistral_correctness"].isin(["PASS", "FAIL"])
        & full["llama_correctness"].isin(["PASS", "FAIL"])
    ]

    print(f"Worksheet:  {worksheet_path}")
    print(f"Key:        {key_path}")
    print(f"Judged set: {eval_path} ({len(full)} triples)")

    if "human_correctness" not in ws.columns:
        print("ERROR: worksheet has no human_correctness column")
        sys.exit(1)

    ws["human_correctness"] = (
        ws["human_correctness"].astype(str).str.strip().str.upper()
    )
    n_rows = len(ws)
    ws = ws[ws["human_correctness"].isin(["PASS", "FAIL"])]
    if len(ws) == 0:
        print(
            f"ERROR: no PASS/FAIL values in human_correctness -- the worksheet "
            f"({n_rows} rows) has not been annotated yet."
        )
        sys.exit(1)
    if len(ws) < n_rows:
        print(f"NOTE: {n_rows - len(ws)} of {n_rows} rows left blank/invalid, dropped")

    merged = key.merge(
        ws[["triple_id", "human_correctness"]], on="triple_id", how="inner"
    )
    if len(merged) != len(ws):
        print(f"WARNING: {len(ws) - len(merged)} annotated rows did not match the key")
    print(f"Annotated and matched: {len(merged)} triples\n")

    jvj = judge_vs_judge(full)
    hvj = human_vs_judges(merged)
    arb = arbitration(merged)
    prec = human_precision(merged)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = build_report(
        jvj, hvj, arb, prec, merged, timestamp, worksheet_path.name, len(full)
    )
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"human_agreement_report_{timestamp}.txt"
    out_path.write_text(report)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "timestamp": timestamp,
        "worksheet": str(worksheet_path),
        "key": str(key_path),
        "evaluation_file": str(eval_path),
        "n_annotated": int(len(merged)),
        "n_judged": int(len(full)),
        "judge_vs_judge": jvj,
        "human_vs_judges": hvj,
        "arbitration": arb,
        "human_precision": prec,
    }, indent=2, default=float))

    print(f"\nReport saved to: {out_path}")
    print(f"JSON saved to:   {json_path}")


if __name__ == "__main__":
    main()
