"""tehood compare - combine several `tehood test` result sets for one genome (e.g. count +
density at several windows) and list TE-associated and TE-depleted sets separately,
with the analyses supporting each.

Outputs PREFIX_TE-associated.tsv, PREFIX_TE-depleted.tsv (n_support, supported_by),
PREFIX_comparison.tsv (all sets side by side), PREFIX_summary.tsv, PREFIX_comparison.pdf.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def log(msg):
    print(f"[tehood compare] {msg}", file=sys.stderr)


def compare(files, names, prefix, alpha=0.05, plot=True):
    if len(files) != len(names) or len(set(names)) != len(names):
        raise SystemExit("ERROR: --inputs and --names must have the same length, names unique")
    R = {n: pd.read_csv(f, sep="\t", dtype={"domain": str}) for f, n in zip(files, names)}
    info_cols = ["domain", "species", "description", "n_genes", "likely_TE_gene", "te_viral_domain",
                 "frac_genes_with_TE_domain", "multicopy_domain"]
    common = [c for c in info_cols if all(c in x for x in R.values())]
    m = pd.concat([x[common] for x in R.values()]).drop_duplicates("domain").reset_index(drop=True)
    for n, x in R.items():
        st = np.where(~x["tested"].astype(bool), "not_tested",
                      np.where((x["q"] < alpha) & (x["observed"] > x["null_mean"]), "TE-associated",
                               np.where((x["q"] < alpha) & (x["observed"] < x["null_mean"]), "TE-depleted", "NS")))
        y = pd.DataFrame({"domain": x["domain"], f"status_{n}": st, f"q_{n}": x["q"], f"log2FE_{n}": x["log2FE"],
                          f"observed_{n}": x["observed"], f"null_mean_{n}": x["null_mean"]})
        m = m.merge(y, on="domain", how="left")
    S = m[[f"status_{n}" for n in names]].fillna("absent").to_numpy()
    Q = m[[f"q_{n}" for n in names]].to_numpy(dtype=float)
    nm = np.array(names)
    for lab, col in [("TE-associated", "associated"), ("TE-depleted", "depleted")]:
        hit = S == lab
        m[f"n_support_{col}"] = hit.sum(1); m[f"{col}_in"] = [",".join(nm[h]) for h in hit]
    na, nd = m["n_support_associated"].to_numpy(), m["n_support_depleted"].to_numpy()
    allnt = np.isin(S, ["not_tested", "absent"]).all(1)
    m["consensus"] = np.where((na > 0) & (nd > 0), "conflict",
                     np.where(na > 0, [f"TE-associated ({k}/{len(names)})" for k in na],
                     np.where(nd > 0, [f"TE-depleted ({k}/{len(names)})" for k in nd],
                     np.where(allnt, "not_tested", "NS"))))
    with np.errstate(all="ignore"):
        m["best_q"] = np.nanmin(np.where(np.isnan(Q), np.inf, Q), axis=1)
    m.loc[~np.isfinite(m["best_q"]), "best_q"] = np.nan
    per = [f"{p}_{n}" for n in names for p in ("status", "q", "log2FE", "observed", "null_mean")]
    front = [c for c in ["domain", "description", "species", "n_genes", "likely_TE_gene", "te_viral_domain",
                         "frac_genes_with_TE_domain", "multicopy_domain", "consensus"] if c in m]
    outs = {}
    for col, fname in [("associated", "TE-associated"), ("depleted", "TE-depleted")]:
        x = m[(m[f"n_support_{col}"] > 0) & (m["consensus"] != "conflict")].copy()
        x = x.rename(columns={f"n_support_{col}": "n_support", f"{col}_in": "supported_by"})
        x = x.sort_values(["n_support", "best_q"], ascending=[False, True])[front + ["n_support", "supported_by"] + per]
        x.to_csv(f"{prefix}_{fname}.tsv", sep="\t", index=False, na_rep="NA"); outs[col] = x
    m = m.assign(_o=m["consensus"].isin(["NS", "not_tested"])).sort_values(["_o", "best_q"]).drop(columns="_o")
    m.to_csv(f"{prefix}_comparison.tsv", sep="\t", index=False, na_rep="NA")
    with open(f"{prefix}_summary.tsv", "w") as o:
        o.write(f"# q < {alpha}\nanalysis\ttested\tTE-associated\tTE-depleted\n")
        for i, n in enumerate(names):
            o.write(f"{n}\t{np.isin(S[:, i], ['NS', 'TE-associated', 'TE-depleted']).sum()}\t"
                    f"{(S[:, i] == 'TE-associated').sum()}\t{(S[:, i] == 'TE-depleted').sum()}\n")
        for lab in ["TE-associated", "TE-depleted"]:
            o.write(f"\n# overlap of {lab} lists\n\t" + "\t".join(names) + "\n")
            for i, n in enumerate(names):
                o.write(n + "\t" + "\t".join(str(((S[:, i] == lab) & (S[:, j] == lab)).sum()) for j in range(len(names))) + "\n")
        o.write("\n# consensus\n")
        for k, v in m["consensus"].value_counts().items():
            o.write(f"{k}\t{v}\n")
    log(f"{len(m)} sets, {len(names)} analyses: {', '.join(names)}")
    for i, n in enumerate(names):
        log(f"  {n:16s} TE-associated {(S[:, i] == 'TE-associated').sum():4d}   TE-depleted {(S[:, i] == 'TE-depleted').sum():4d}")
    log(f"TE-associated in >= 1 analysis: {len(outs['associated'])} (all {len(names)}: "
        f"{(outs['associated']['n_support'] == len(names)).sum()})")
    log(f"TE-depleted   in >= 1 analysis: {len(outs['depleted'])} (all {len(names)}: "
        f"{(outs['depleted']['n_support'] == len(names)).sum()})")
    if plot and len(names) >= 2:
        _plot(m, names, prefix)
    return m


def _plot(m, names, prefix):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return
    with PdfPages(f"{prefix}_comparison.pdf") as pdf:
        for i in range(len(names) - 1):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                si, sj = m[f"status_{a}"], m[f"status_{b}"]
                col = np.where((si == "TE-associated") & (sj == "TE-associated"), "darkred",
                      np.where((si == "TE-associated") | (sj == "TE-associated"), "salmon",
                      np.where((si == "TE-depleted") & (sj == "TE-depleted"), "seagreen",
                      np.where((si == "TE-depleted") | (sj == "TE-depleted"), "palegreen", "0.75"))))
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(m[f"log2FE_{a}"], m[f"log2FE_{b}"], s=5, c=col)
                ax.axhline(0, ls="--", c="0.5", lw=0.5); ax.axvline(0, ls="--", c="0.5", lw=0.5)
                rho = m[f"log2FE_{a}"].corr(m[f"log2FE_{b}"], method="spearman")
                ax.set_xlabel(f"log2FE {a}"); ax.set_ylabel(f"log2FE {b}"); ax.set_title(f"rho = {rho:.2f}")
                pdf.savefig(fig); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tehood compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="comma-separated *_results.tsv files")
    ap.add_argument("--names", help="comma-separated short names [file names]")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    a = ap.parse_args(argv)
    files = a.inputs.split(",")
    names = a.names.split(",") if a.names else [os.path.basename(f).replace("_results.tsv", "") for f in files]
    compare(files, names, a.out, a.alpha, a.plot)


if __name__ == "__main__":
    main()
