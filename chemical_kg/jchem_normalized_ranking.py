"""
Size-bias diagnostics for the Jchem ranking: ceiling-normalized Jaccard
and overlap coefficient, as companion columns.

Extra analysis on top of jchem_results.csv -- does NOT rerun matching and
does NOT overwrite the Jchem outputs. The weighted Jaccard punishes small
MatKG profiles: LiBOB's 13 neighbors are a perfect subset of LiTFSI's
174, so its unweighted Jaccard (13/174 = 0.075) is already the maximum a
13-neighbor profile can score. Two set-size-corrected views:

- ceiling_norm_jaccard: jaccard_unweighted divided by the best Jaccard
  attainable given the two profile sizes (min(|A|,|B|) / max(|A|,|B|)).
  1.0 means "scored its ceiling", i.e. one profile is a subset of the
  other.
- overlap_coef: |A n B| / min(|A|, |B|) (Szymkiewicz-Simpson).

Both inflate tiny profiles -- a 2-neighbor component with both neighbors
in LiTFSI's set also scores 1.0 -- so rows with n_neighbors below
THIN_PROFILE are flagged [thin] and the report shows top-10 lists with
and without them.

Reads  output/jchem_results.csv
Writes output/jchem_normalized_ranking.csv
       output/jchem_normalized_report.txt
"""

import csv
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
RESULTS_CSV = os.path.join(OUT_DIR, "jchem_results.csv")
NORMALIZED_CSV = os.path.join(OUT_DIR, "jchem_normalized_ranking.csv")
REPORT_TXT = os.path.join(OUT_DIR, "jchem_normalized_report.txt")

REF_INCHIKEY = "QSZMZKBZAYQGRS-UHFFFAOYSA-N"  # LiTFSI
THIN_PROFILE = 10  # below this many neighbors the corrected scores are
                   # dominated by chance overlap


def main():
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    ref = next(r for r in rows if r["inchikey"] == REF_INCHIKEY)
    n_ref = int(ref["n_neighbors"])
    print(f"reference {ref['title']}: {n_ref} neighbors")

    out = []
    for r in rows:
        if r["inchikey"] == REF_INCHIKEY:
            continue
        n = int(r["n_neighbors"])
        inter = int(r["intersection_count"])
        jacc = float(r["jaccard_unweighted"])
        lo, hi = min(n, n_ref), max(n, n_ref)
        ceiling = lo / hi if hi else 0.0
        out.append({
            "title": r["title"],
            "inchikey": r["inchikey"],
            "n_neighbors": n,
            "intersection_count": inter,
            "jaccard_weighted": float(r["jaccard_weighted"]),
            "jaccard_unweighted": jacc,
            "ceiling_norm_jaccard": jacc / ceiling if ceiling else 0.0,
            "overlap_coef": inter / lo if lo else 0.0,
            "thin": "thin" if n < THIN_PROFILE else "",
        })

    out.sort(key=lambda r: (-r["ceiling_norm_jaccard"], -r["overlap_coef"],
                            -r["n_neighbors"]))
    with open(NORMALIZED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    def top10(rows, score):
        lines = [f"{'rank':<5}{score:<14}{'overlap':<9}{'nbrs':<6}"
                 f"{'inter':<7}{'w-Jacc':<9}title"]
        for i, r in enumerate(rows[:10], 1):
            flag = " [thin]" if r["thin"] else ""
            lines.append(f"{i:<5}{r[score]:<14.4f}{r['overlap_coef']:<9.4f}"
                         f"{r['n_neighbors']:<6}{r['intersection_count']:<7}"
                         f"{r['jaccard_weighted']:<9.4f}"
                         f"{r['title'][:48]}{flag}")
        return lines

    by_ceiling = sorted(out, key=lambda r: -r["ceiling_norm_jaccard"])
    by_overlap = sorted(out, key=lambda r: -r["overlap_coef"])
    solid = [r for r in out if not r["thin"]]

    lines = [f"Jchem size-corrected companion ranking "
             f"(reference LiTFSI, {n_ref} neighbors; "
             f"thin = fewer than {THIN_PROFILE} neighbors)",
             "\nTop 10 by ceiling-normalized Jaccard (all profiles):"]
    lines += top10(by_ceiling, "ceiling_norm_jaccard")
    lines.append(f"\nTop 10 by ceiling-normalized Jaccard "
                 f"(profiles with >= {THIN_PROFILE} neighbors):")
    lines += top10(sorted(solid, key=lambda r: -r["ceiling_norm_jaccard"]),
                   "ceiling_norm_jaccard")
    lines.append("\nTop 10 by overlap coefficient (all profiles):")
    lines += top10(by_overlap, "overlap_coef")
    lines.append(f"\nTop 10 by overlap coefficient "
                 f"(profiles with >= {THIN_PROFILE} neighbors):")
    lines += top10(sorted(solid, key=lambda r: -r["overlap_coef"]),
                   "overlap_coef")

    report = "\n".join(lines)
    with open(REPORT_TXT, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nWrote {NORMALIZED_CSV}, {REPORT_TXT}")


if __name__ == "__main__":
    main()
