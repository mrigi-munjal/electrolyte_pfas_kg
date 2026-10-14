#!/usr/bin/env python3
"""
Agreement + Cohen's kappa between the human annotator and the two LLM judges.

A focused companion to analyze_human_agreement.py: that script produces the
full report (AC1, PABAK, Fleiss, arbitration, bootstrap CIs); this one shows
the two headline numbers -- percent agreement and Cohen's kappa -- for each
rater pair, with the arithmetic written out so it can be checked by hand.

It also fingerprints the worksheet it read (path, mtime, sha256, verdict
counts, marker column) so there is never any doubt about which file the
numbers came from.

Usage:
    python battery_kg/human_judge_agreement.py \
        -w human_worksheet_20260814_172329_mrigi_judge.xlsx

    # confirm the file carries an expected marker column, e.g. a column of
    # "works" written by the annotator:
    python battery_kg/human_judge_agreement.py \
        -w <worksheet> --expect-marker works
"""

import argparse
import hashlib
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "evaluation_output"
VALID = ("PASS", "FAIL")


# ============================================================
# AGREEMENT ARITHMETIC (written out rather than delegated)
# ============================================================

def confusion(a: pd.Series, b: pd.Series) -> np.ndarray:
    """2x2 counts, rows = rater a, cols = rater b, order (FAIL, PASS)."""
    t = np.zeros((2, 2), dtype=int)
    for x, y in zip(a, b):
        t[1 if x == "PASS" else 0, 1 if y == "PASS" else 0] += 1
    return t


def percent_agreement(t: np.ndarray) -> float:
    return float(np.trace(t) / t.sum()) if t.sum() else float("nan")


def cohens_kappa(t: np.ndarray) -> dict:
    """Cohen's kappa with its intermediate terms exposed."""
    n = t.sum()
    if n == 0:
        return {"kappa": float("nan"), "p_o": float("nan"), "p_e": float("nan")}
    p_o = np.trace(t) / n
    # p_e = sum over categories of (row marginal * col marginal) / n^2
    p_e = float((t.sum(axis=1) @ t.sum(axis=0)) / (n * n))
    kappa = float("nan") if np.isclose(p_e, 1.0) else (p_o - p_e) / (1 - p_e)
    return {"kappa": kappa, "p_o": float(p_o), "p_e": p_e, "n": int(n)}


def interpret(k: float) -> str:
    if np.isnan(k):
        return "undefined"
    for bound, label in ((0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
                         (0.60, "moderate"), (0.80, "substantial")):
        if k < bound:
            return label
    return "almost perfect"


def weighted_agreement(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Percent agreement reweighted to the full judged set.

    The 50 triples were drawn with unequal probability across judge cells, so
    the in-sample rate is biased toward the oversampled cells. Each stratum
    contributes in proportion to its share of the population.
    """
    if "design_weight" not in df.columns:
        return {}
    agree = (df[col_a] == df[col_b]).astype(float)
    total_pop = df.groupby("judge_cell")["cell_population"].first().sum()
    p_hat, var = 0.0, 0.0
    for _, g in df.assign(_a=agree).groupby("judge_cell"):
        big_n = float(g["cell_population"].iloc[0])
        n_h = len(g)
        w = big_n / total_pop
        p_h = g["_a"].mean()
        p_hat += w * p_h
        if n_h > 1:
            var += (w ** 2) * (1 - n_h / big_n) * p_h * (1 - p_h) / (n_h - 1)
    se = float(np.sqrt(var))
    return {"p": float(p_hat), "se": se,
            "ci": (max(0.0, p_hat - 1.96 * se), min(1.0, p_hat + 1.96 * se))}


# ============================================================
# FILE IDENTITY
# ============================================================

def fingerprint(path: Path, sheet: pd.DataFrame, marker_col, expect_marker):
    """Print exactly what was read, so the wrong file cannot pass unnoticed."""
    raw = path.read_bytes()
    print("=" * 72)
    print("FILE READ")
    print("=" * 72)
    print(f"  path    : {path.resolve()}")
    print(f"  modified: {datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
    print(f"  size    : {len(raw):,} bytes")
    print(f"  sha256  : {hashlib.sha256(raw).hexdigest()[:32]}")
    print(f"  columns : {list(sheet.columns)}")
    print(f"  rows    : {len(sheet)}")

    counts = Counter(sheet["human_correctness"].dropna().astype(str).str.strip().str.upper())
    print(f"  verdicts: {dict(counts)}")

    if marker_col is not None:
        vals = Counter(str(v).strip().lower() for v in marker_col.dropna())
        print(f"  marker  : {dict(vals)}")
    else:
        print("  marker  : (no extra marker column found)")

    if expect_marker:
        found = 0
        if marker_col is not None:
            found = sum(1 for v in marker_col.dropna()
                        if str(v).strip().lower() == expect_marker.strip().lower())
        if found == 0:
            print()
            print(f"  *** STOP: expected a column containing '{expect_marker}' and found none.")
            print("  *** This is NOT the file you think it is -- the edits are")
            print("  *** unsaved, or they were saved to a different path.")
            sys.exit(2)
        print(f"  marker OK: {found} cells match '{expect_marker}'")
    print()


# ============================================================
# MAIN
# ============================================================

def load_sheet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return pd.read_excel(path, sheet_name="Annotation")
    return pd.read_csv(path)


def find_key(worksheet: Path, explicit) -> Path:
    if explicit:
        return Path(explicit)
    m = re.search(r"(\d{8}_\d{6})", worksheet.name)
    if m:
        for base in (worksheet.parent, OUTPUT_DIR):
            cand = base / f"human_worksheet_key_{m.group(1)}.csv"
            if cand.exists():
                return cand
    keys = sorted(OUTPUT_DIR.glob("human_worksheet_key_*.csv"))
    if not keys:
        print("ERROR: no key file found; pass --key")
        sys.exit(1)
    return max(keys, key=lambda f: f.stat().st_mtime)


def report_pair(label_a, label_b, a, b, df=None, col_a=None, col_b=None):
    t = confusion(a, b)
    k = cohens_kappa(t)
    pa = percent_agreement(t)
    print("-" * 72)
    print(f"{label_a}  vs  {label_b}")
    print("-" * 72)
    print(f"                {label_b+' FAIL':>16} {label_b+' PASS':>16}")
    print(f"  {label_a+' FAIL':<14}{t[0,0]:>16} {t[0,1]:>16}")
    print(f"  {label_a+' PASS':<14}{t[1,0]:>16} {t[1,1]:>16}")
    print()
    print(f"  n                  = {k['n']}")
    print(f"  observed agreement = {np.trace(t)}/{k['n']} = {pa:.4f}  ({pa:.1%})")
    print(f"  chance agreement   = {k['p_e']:.4f}")
    print(f"  Cohen's kappa      = (p_o - p_e) / (1 - p_e)")
    print(f"                     = ({k['p_o']:.4f} - {k['p_e']:.4f}) / (1 - {k['p_e']:.4f})"
          f" = {k['kappa']:.4f}  [{interpret(k['kappa'])}]")
    if df is not None:
        w = weighted_agreement(df, col_a, col_b)
        if w:
            print(f"  weighted agreement = {w['p']:.1%} "
                  f"(95% CI {w['ci'][0]:.1%} - {w['ci'][1]:.1%})  <- estimate over all 600")
    print()
    return {"percent_agreement": pa, **k}


def main():
    ap = argparse.ArgumentParser(description="Agreement and Cohen's kappa, human vs judges")
    ap.add_argument("--worksheet", "-w", required=True)
    ap.add_argument("--key", "-k", default=None)
    ap.add_argument("--expect-marker", default=None,
                    help="fail loudly unless some column contains this value "
                         "(use to prove the intended file was read)")
    args = ap.parse_args()

    path = Path(args.worksheet)
    if not path.exists():
        print(f"ERROR: worksheet not found: {path}")
        sys.exit(1)

    sheet = load_sheet(path)
    if "human_correctness" not in sheet.columns:
        print("ERROR: no human_correctness column in the worksheet")
        sys.exit(1)

    # Any column beyond the 9 known ones is treated as a marker column.
    known = {"triple_id", "subject", "relation", "object", "source_text",
             "human_correctness", "human_notes",
             "mistral_correctness", "llama_correctness"}
    extras = [c for c in sheet.columns if c not in known and not str(c).startswith("Unnamed")]
    marker = sheet[extras[0]] if extras else None
    if marker is None:
        unnamed = [c for c in sheet.columns if str(c).startswith("Unnamed")]
        if unnamed and sheet[unnamed[0]].notna().any():
            marker = sheet[unnamed[0]]

    fingerprint(path, sheet, marker, args.expect_marker)

    key = pd.read_csv(find_key(path, args.key))
    sheet["human_correctness"] = sheet["human_correctness"].astype(str).str.strip().str.upper()
    graded = sheet[sheet["human_correctness"].isin(VALID)]
    dropped = len(sheet) - len(graded)
    if dropped:
        print(f"NOTE: {dropped} row(s) without a PASS/FAIL verdict were dropped\n")

    df = key.merge(graded[["triple_id", "human_correctness"]], on="triple_id", how="inner")
    if len(df) != len(graded):
        print(f"WARNING: {len(graded) - len(df)} graded row(s) had no match in the key\n")

    print("=" * 72)
    print(f"AGREEMENT AND COHEN'S KAPPA  (n = {len(df)} annotated triples)")
    print("=" * 72)
    print()

    results = {
        "human_vs_mistral": report_pair(
            "human", "mistral", df["human_correctness"], df["mistral_correctness"],
            df, "human_correctness", "mistral_correctness"),
        "human_vs_llama": report_pair(
            "human", "llama", df["human_correctness"], df["llama_correctness"],
            df, "human_correctness", "llama_correctness"),
        "mistral_vs_llama": report_pair(
            "mistral", "llama", df["mistral_correctness"], df["llama_correctness"],
            df, "mistral_correctness", "llama_correctness"),
    }

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'pair':<22}{'n':>5}{'agreement':>12}{'kappa':>9}  interpretation")
    for name, r in results.items():
        print(f"{name:<22}{r['n']:>5}{r['percent_agreement']:>11.1%}"
              f"{r['kappa']:>9.3f}  {interpret(r['kappa'])}")
    print()
    print("Note: in-sample agreement is computed on a sample that oversamples the")
    print("rare judge cells by design. The weighted figures above each table are")
    print("the ones that estimate the full 600-triple population.")


if __name__ == "__main__":
    main()
