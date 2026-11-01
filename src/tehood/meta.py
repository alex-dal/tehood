"""tehood meta - combine one analysis across many genomes, to rescue protein domains whose
per-genome evidence is too weak to survive FDR in any single genome (TE-poor genomes,
few genes per domain), and to separate consistent signals from one-genome flukes.

  tehood meta --dirs results/* --out meta/all
  tehood meta --dirs results/* --analyses count,density_w10000 --clades clades.tsv --empirical-null --out meta/all

For every analysis found in every directory (PREFIX_<analysis>_results.tsv written by
`tehood run` / `tehood test`), each domain gets two genome-level tests:

  Stouffer   consistency: sum of per-genome z / sqrt(n). z comes from the one-sided
             mid-p, (1 + p_enr - p_dep) / 2, which is ~uniform under the null for
             count (discrete) and density statistics alike. Genomes where the domain
             has no information (no TEs anywhere near it) give z ~ 0 and do not dilute
             a real signal much; genomes where it is absent are skipped.
  recurrence how many genomes reach p <= 0.05 in one direction, versus Binomial(n, 0.05):
             finds domains that are TE-associated in a subset of genomes (e.g. a clade).

p_meta = Bonferroni of the two tests and two directions; BH over domains -> q, status.
All per-genome rows are used, including sets not tested within a genome (sparse
domains can still add up across genomes).

Related genomes are not independent (orthologous genes inherit flanking TEs). With
--clades (genome<TAB>clade) genomes are first combined within each clade (Stouffer) and
the clade is the unit for both tests. `lambda` (median Z^2 / 0.455) in the log and
PREFIX_summary.tsv measures inflation: > ~1.3 means the theoretical null is too lenient;
use --clades and/or --empirical-null (rescale Z by the spread of all domains' Z).

Outputs, per analysis A:
  PREFIX_A_meta.tsv          all domains: units, recurrence, Stouffer Z, p, q, status,
                             'rescued' (meta-significant but significant in no single genome),
                             p_stouffer_drop1 (strongest genome/clade left out), genomes listed
  PREFIX_A_genome_z.tsv      domain x genome signed z (heatmaps; map onto the phylogeny)
and across the domain-level analyses (count, density_w*):
  PREFIX_TE-associated.tsv, PREFIX_TE-depleted.tsv   (n_support, supported_by)
  PREFIX_summary.tsv, PREFIX_meta_qq.pdf (with matplotlib)
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

from .stats import bh

EPS = 1e-300


def log(msg):
    print(f"[tehood meta] {msg}", file=sys.stderr)


# ---------------------------------------------------------------- input --
def find_results(dirs, files):
    """-> {analysis: [(genome, path), ...]}"""
    paths = list(files or [])
    for d in dirs or []:
        paths += sorted(glob.glob(os.path.join(d, "*_results.tsv")))
    out = {}
    for p in paths:
        with open(p) as fh:
            head = fh.readline().rstrip("\n").split("\t"); row = fh.readline().rstrip("\n").split("\t")
        if "species" not in head or "p_enr" not in head or len(row) < len(head):
            continue
        g = row[head.index("species")]; base = os.path.basename(p)
        if not base.startswith(g + "_"):
            log(f"skipping {p}: file name does not start with '{g}_'"); continue
        an = base[len(g) + 1:-len("_results.tsv")]
        out.setdefault(an, []).append((g, p))
    for an, lst in out.items():
        gs = [g for g, _ in lst]
        dup = {g for g in gs if gs.count(g) > 1}
        if dup:
            raise SystemExit(f"ERROR: analysis {an}: genome(s) {sorted(dup)} found more than once "
                             "(give each run directory once)")
    return out


def load(lst, min_genes):
    cols = ["domain", "description", "n_genes", "observed", "null_mean", "log2FE", "p_enr", "p_dep",
            "status", "tested", "likely_TE_gene"]
    xs = []
    for g, p in lst:
        x = pd.read_csv(p, sep="\t", dtype={"domain": str, "description": str}, usecols=lambda c: c in cols)
        x["genome"] = g; xs.append(x)
    x = pd.concat(xs, ignore_index=True)
    x = x[(x["n_genes"] >= min_genes) & x["p_enr"].notna() & x["p_dep"].notna()].copy()
    pm = np.clip((1 + x["p_enr"].to_numpy() - x["p_dep"].to_numpy()) / 2, 1e-15, 1 - 1e-15)
    x["z"] = stats.norm.isf(pm)                                     # > 0: more TEs than random genes
    return x


# --------------------------------------------------------------- combine --
def to_units(x, clades):
    """Per-unit z and one-sided p. Unit = genome, or clade (Stouffer within clade)."""
    if clades is None:
        u = x[["domain", "genome", "z", "p_enr", "p_dep"]].rename(columns={"genome": "unit"})
        return u
    x = x.assign(unit=x["genome"].map(clades))
    miss = sorted(x.loc[x["unit"].isna(), "genome"].unique())
    if miss:
        log(f"WARNING: {len(miss)} genome(s) not in --clades, used as their own clade: {', '.join(miss[:5])}")
        x.loc[x["unit"].isna(), "unit"] = x.loc[x["unit"].isna(), "genome"]
    u = x.groupby(["domain", "unit"], sort=False)["z"].agg(["sum", "size"]).reset_index()
    u["z"] = u["sum"] / np.sqrt(u["size"])
    u["p_enr"] = stats.norm.sf(u["z"]); u["p_dep"] = stats.norm.cdf(u["z"])
    return u[["domain", "unit", "z", "p_enr", "p_dep"]]


def combine(x, clades, min_units, recur_p, alpha, empirical_null):
    u = to_units(x, clades)
    u["enr05"] = u["p_enr"] <= recur_p; u["dep05"] = u["p_dep"] <= recur_p; u["pos"] = u["z"] > 0
    g = u.groupby("domain", sort=False)
    m = pd.DataFrame({"n_units": g.size(), "sum_z": g["z"].sum(), "max_z": g["z"].max(), "min_z": g["z"].min(),
                      "frac_positive_z": g["pos"].mean(),
                      "units_enr_p05": g["enr05"].sum().astype(int), "units_dep_p05": g["dep05"].sum().astype(int)})
    x = x.assign(_sig=x["status"].isin(["TE-associated", "TE-depleted"]),
                 _tested=x["tested"].astype(str).str.lower().eq("true"),
                 _te=x["likely_TE_gene"].astype(str).str.lower().eq("true") if "likely_TE_gene" in x else False)
    gx = x.groupby("domain", sort=False)
    info = pd.DataFrame({
        "description": gx["description"].first(),
        "n_genomes": gx.size(), "median_n_genes": gx["n_genes"].median(),
        "median_log2FE": gx["log2FE"].median(),
        "n_genomes_tested": gx["_tested"].sum().astype(int),
        "n_genomes_significant": gx["_sig"].sum().astype(int),
        "likely_TE_gene_any": gx["_te"].any(),
    })
    for c, col in [("genomes_enr_p05", "p_enr"), ("genomes_dep_p05", "p_dep")]:
        info[c] = x[x[col] <= recur_p].groupby("domain", sort=False)["genome"].agg(",".join)
        info[c] = info[c].fillna("")
    m = info.join(m, how="inner")
    n = m["n_units"].to_numpy(float)
    Z = m["sum_z"].to_numpy() / np.sqrt(n)
    ok = m["n_units"].to_numpy() >= min_units
    # inflation / empirical null from informative domains only (tested in >= min_units genomes):
    # sparse count domains give z ~ 0 by discreteness and would hide inflation
    inf = ok & (m["n_genomes_tested"].to_numpy() >= min(min_units, 3))
    lam = float(np.median(Z[inf] ** 2) / stats.chi2.ppf(0.5, 1)) if inf.sum() >= 20 else np.nan
    centre, scale = 0.0, 1.0
    if empirical_null and inf.sum() >= 50:
        centre = float(np.median(Z[inf]))
        scale = max(1.0, float(stats.median_abs_deviation(Z[inf], scale="normal")))
    Zc = (Z - centre) / scale
    m["stouffer_Z"] = Zc
    m["p_stouffer_enr"] = np.maximum(stats.norm.sf(Zc), EPS)
    m["p_stouffer_dep"] = np.maximum(stats.norm.cdf(Zc), EPS)
    with np.errstate(divide="ignore", invalid="ignore"):          # robustness: drop the strongest unit
        zd_enr = (m["sum_z"] - m["max_z"]) / np.sqrt(n - 1); zd_dep = (m["sum_z"] - m["min_z"]) / np.sqrt(n - 1)
    m["p_stouffer_drop1"] = np.where(n > 1, np.where(Zc >= 0, stats.norm.sf((zd_enr - centre) / scale),
                                                     stats.norm.cdf((zd_dep - centre) / scale)), np.nan)
    m["expected_p05"] = n * recur_p
    m["p_recur_enr"] = np.maximum(stats.binom.sf(m["units_enr_p05"] - 1, n, recur_p), EPS)
    m["p_recur_dep"] = np.maximum(stats.binom.sf(m["units_dep_p05"] - 1, n, recur_p), EPS)
    pe = np.minimum(1, 2 * np.minimum(m["p_stouffer_enr"], m["p_recur_enr"]))
    pd_ = np.minimum(1, 2 * np.minimum(m["p_stouffer_dep"], m["p_recur_dep"]))
    m["p_meta"] = np.minimum(1, 2 * np.minimum(pe, pd_))
    m["direction"] = np.where(pe <= pd_, "more_TEs", "fewer_TEs")
    m["driven_by"] = np.where(np.where(pe <= pd_, m["p_stouffer_enr"] <= m["p_recur_enr"],
                                       m["p_stouffer_dep"] <= m["p_recur_dep"]), "consistency", "recurrence")
    m["tested"] = ok
    q = np.full(len(m), np.nan); q[ok] = bh(m["p_meta"].to_numpy()[ok]); m["q_meta"] = q
    sig = ok & (q < alpha)
    m["status"] = np.where(~ok, "not_tested", np.where(sig & (m["direction"] == "more_TEs"), "TE-associated",
                           np.where(sig, "TE-depleted", "NS")))
    m["rescued"] = sig & (m["n_genomes_significant"] == 0)
    m = m.reset_index().rename(columns={"index": "domain"})
    cols = ["domain", "description", "status", "q_meta", "p_meta", "direction", "driven_by", "rescued",
            "n_genomes", "n_units", "n_genomes_tested", "n_genomes_significant", "median_n_genes", "median_log2FE",
            "stouffer_Z", "frac_positive_z", "p_stouffer_enr", "p_stouffer_dep", "p_stouffer_drop1",
            "units_enr_p05", "units_dep_p05", "expected_p05", "p_recur_enr", "p_recur_dep",
            "likely_TE_gene_any", "tested", "genomes_enr_p05", "genomes_dep_p05"]
    m = m[cols].sort_values(["tested", "p_meta"], ascending=[False, True])
    return m, dict(lam=lam, centre=centre, scale=scale, n_tested=int(ok.sum()))


# ------------------------------------------------------------------ main --
def is_domain_level(an):
    return an in ("count", "density") or re.fullmatch(r"density_w\d+", an) is not None


def summarise(metas, prefix):
    names = [a for a in metas if is_domain_level(a)]
    if not names:
        return
    allm = None
    for a in names:
        y = metas[a][["domain", "description", "n_genomes", "status", "q_meta", "rescued", "driven_by"]].rename(
            columns={c: f"{c}_{a}" for c in ["status", "q_meta", "rescued", "driven_by", "n_genomes"]})
        allm = y if allm is None else allm.merge(y, on=["domain", "description"], how="outer")
    for lab, fn in [("TE-associated", "TE-associated"), ("TE-depleted", "TE-depleted")]:
        S = allm[[f"status_{a}" for a in names]].fillna("absent").to_numpy()
        hit = S == lab
        x = allm[hit.any(1)].copy()
        x.insert(2, "n_support", hit[hit.any(1)].sum(1))
        x.insert(3, "supported_by", [",".join(np.array(names)[h]) for h in hit[hit.any(1)]])
        R = x[[f"rescued_{a}" for a in names]].fillna(False).astype(bool).to_numpy() & hit[hit.any(1)]
        x.insert(4, "rescued_in", [",".join(np.array(names)[r]) for r in R])
        Q = x[[f"q_meta_{a}" for a in names]].to_numpy(float)
        x.insert(5, "best_q", np.nanmin(np.where(np.isnan(Q), np.inf, Q), axis=1))
        x = x.sort_values(["n_support", "best_q"], ascending=[False, True])
        x.to_csv(f"{prefix}_{fn}.tsv", sep="\t", index=False, na_rep="NA")
        log(f"{lab}: {len(x)} domains in >= 1 domain-level analysis -> {prefix}_{fn}.tsv")


def qq_plot(metas, prefix):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(metas)[:12]
    fig, axs = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2), squeeze=False)
    for ax, a in zip(axs[0], names):
        m = metas[a]; p = np.sort(m.loc[m["tested"], "p_meta"].to_numpy())
        if len(p) == 0: continue
        e = (np.arange(1, len(p) + 1) - 0.5) / len(p)
        ax.plot(-np.log10(e), -np.log10(np.maximum(p, 1e-300)), ".", ms=2, color="k")
        lim = -np.log10(e[0]); ax.plot([0, lim], [0, lim], color="r", lw=0.8)
        ax.set_title(a, fontsize=8); ax.set_xlabel("expected -log10 p"); ax.set_ylabel("observed -log10 p")
    fig.tight_layout(); fig.savefig(f"{prefix}_meta_qq.pdf"); plt.close(fig)


def run(a):
    res = find_results(a.dirs, a.results)
    if a.analyses:
        want = [s.strip() for s in a.analyses.split(",") if s.strip()]
        miss = [w for w in want if w not in res]
        if miss:
            raise SystemExit(f"ERROR: analyses not found: {miss}; available: {sorted(res)}")
        res = {k: res[k] for k in want}
    if not res:
        raise SystemExit("ERROR: no *_results.tsv files found")
    clades = None
    if a.clades:
        c = pd.read_csv(a.clades, sep="\t", header=None, comment="#", dtype=str).iloc[:, :2]
        clades = dict(zip(c[0].str.strip(), c[1].str.strip()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    metas, summ = {}, []
    for an in sorted(res, key=lambda s: (not is_domain_level(s), s)):
        if an in ("count_classes", "density_classes") and not a.analyses:
            continue                      # all-TE part of the by-class runs = count / density_w<extra>
        lst = res[an]
        if len(lst) < a.min_genomes:
            log(f"{an}: only {len(lst)} genome(s), skipped"); continue
        x = load(lst, a.min_set_genes)
        m, d = combine(x, clades, a.min_genomes if clades is None else a.min_units,
                       a.recur_p, a.alpha, a.empirical_null)
        m.to_csv(f"{a.out}_{an}_meta.tsv", sep="\t", index=False, na_rep="NA")
        zt = x.pivot_table(index="domain", columns="genome", values="z", aggfunc="first")
        zt.round(3).to_csv(f"{a.out}_{an}_genome_z.tsv", sep="\t", na_rep="NA")
        na = int((m["status"] == "TE-associated").sum()); nd = int((m["status"] == "TE-depleted").sum())
        nr = int(m["rescued"].sum())
        units = "clades" if clades else "genomes"
        log(f"{an}: {len(lst)} genomes, {d['n_tested']} domains in >= {a.min_genomes if clades is None else a.min_units} "
            f"{units}; lambda {d['lam']:.2f}{' (inflated: consider --clades / --empirical-null)' if d['lam'] > 1.3 else ''}"
            f"; TE-associated {na}, TE-depleted {nd} (q < {a.alpha}); rescued {nr}")
        summ.append(dict(analysis=an, genomes=len(lst), units=units, domains_tested=d["n_tested"], **{"lambda": d["lam"]},
                         empirical_null_centre=d["centre"], empirical_null_scale=d["scale"],
                         TE_associated=na, TE_depleted=nd, rescued=nr,
                         significant_in_any_single_genome=int((m["n_genomes_significant"] > 0).sum())))
        metas[an] = m
    pd.DataFrame(summ).to_csv(f"{a.out}_summary.tsv", sep="\t", index=False, float_format="%.4g")
    summarise(metas, a.out)
    if a.plot:
        qq_plot(metas, a.out)
    log(f"done -> {a.out}_*")


def parser():
    ap = argparse.ArgumentParser(prog="tehood meta", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", help="per-genome result directories (tehood run --outdir)")
    ap.add_argument("--results", nargs="+", help="or: individual *_results.tsv files")
    ap.add_argument("--analyses", help="comma list, e.g. count,density_w10000 [all found]")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--clades", help="genome<TAB>clade: combine within clades first (non-independence)")
    ap.add_argument("--min-genomes", type=int, default=3, help="min genomes carrying the domain [3]")
    ap.add_argument("--min-units", type=int, default=2, help="with --clades: min clades [2]")
    ap.add_argument("--min-set-genes", type=int, default=1, help="min genes per genome to use a row [1]")
    ap.add_argument("--recur-p", type=float, default=0.05, help="per-genome p threshold for recurrence [0.05]")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--empirical-null", action="store_true",
                    help="centre/scale Z by the bulk of domains (never below the theoretical scale)")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if not a.dirs and not a.results:
        parser().error("give --dirs or --results")
    run(a)


if __name__ == "__main__":
    main()
