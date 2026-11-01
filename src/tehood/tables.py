"""tehood tables - per-gene TE tables for one genome.

Writes to OUTDIR:
  obs_gene_TE.tsv               gene  te_id  te_class      TEs within W bp of each gene
                                (TEs overlapping any gene body ignored by default)
  gene_TE_density.tsv           gene window flank_bp te_bp density
                                share of the flanks (W bp each side, own gene body
                                excluded, clipped at sequence ends) covered by MERGED TE
                                sequence - fragmented annotations are not double counted
  gene_TE_density_by_class.tsv  (with --level) the same per TE class
  window_choice.tsv             (with -D auto) per candidate window: share of genes with
                                complete flanks, mean share of flank present, kept or not
  gene_body_TE.tsv              gene body_bp te_bp body_te_frac
  te_family_QC.tsv              per TE family: copies, bp, share of bp inside gene bodies,
                                most common domain of the overlapped genes
  params.txt                    parameters (for methods sections)
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np
import pandas as pd

from . import __version__
from .intervals import Coverage, build_track, merge, window_pairs


def log(msg):
    print(f"[tehood tables {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


# --------------------------------------------------------------------- input --
def read_table(path, ncol_min=1):
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) >= ncol_min:
                rows.append(f)
    return rows


def read_chroms(path):
    return {f[0].strip(): int(f[1]) for f in read_table(path, 2)}


def read_genes(path, chroms):
    rows = [(f[0], int(f[1]), int(f[2]), f[3].strip(), f[5].strip() if len(f) > 5 else "+")
            for f in read_table(path, 4) if f[0] in chroms]
    g = pd.DataFrame(rows, columns=["chr", "start", "end", "gene", "strand"])
    return g.sort_values(["chr", "start", "end"], kind="stable").reset_index(drop=True)


def read_tes(path, chroms, id_col=4, fam_col=5):
    rows = []
    for f in read_table(path, 3):
        if f[0] not in chroms:
            continue
        tid = f[id_col - 1].strip() if id_col and len(f) >= id_col else f"{f[0]}:{f[1]}"
        fam = f[fam_col - 1].strip().replace(" ", "_") if fam_col and len(f) >= fam_col else "."
        rows.append((f[0], int(f[1]), int(f[2]), tid, fam))
    t = pd.DataFrame(rows, columns=["chr", "start", "end", "te", "family"])
    return t.sort_values(["chr", "start", "end"], kind="stable").reset_index(drop=True)


def read_domains(path):
    rows = []
    for f in read_table(path, 2):
        g, d = f[0].strip(), f[1].strip()
        if g and d:
            rows.append((g, d, f[2].strip() if len(f) > 2 else ""))
    return pd.DataFrame(rows, columns=["gene", "domain", "description"]).drop_duplicates()


def te_class(label, level):
    """DNA/DTM/TE_00000063 -> 'DNA' (order) or 'DNA/DTM' (superfamily)."""
    if not level:
        return "."
    p = label.split("/")
    return p[0] if level == "order" or len(p) < 2 else p[0] + "/" + p[1]


# -------------------------------------------------------------------- window --
def intergenic_window(gene_tables, stat="mean"):
    """mean/median distance between consecutive genes on a sequence (overlaps skipped)."""
    d = []
    for g in gene_tables:
        for _, x in g.sort_values(["chr", "start"]).groupby("chr", sort=False):
            s = x["start"].to_numpy(); e = np.maximum.accumulate(x["end"].to_numpy())
            gap = s[1:] - e[:-1]
            d.append(gap[gap > 0])
    d = np.concatenate(d) if d else np.array([])
    if not len(d):
        raise SystemExit("ERROR: no intergenic distances could be computed")
    return int(round(d.mean() if stat == "mean" else np.median(d))), len(d)


# --------------------------------------------------------------------- steps --
def family_qc(tes, genes, domains, gene_track):
    cov = Coverage(gene_track)
    ln = (tes["end"] - tes["start"]).to_numpy()
    ov = cov.covered(tes["chr"].to_numpy(), tes["start"].to_numpy(), tes["end"].to_numpy())
    q = pd.DataFrame({"te_family": tes["family"], "bp": ln, "gbp": np.minimum(ov, ln),
                      "mostly": ov >= 0.5 * ln})
    agg = q.groupby("te_family").agg(n_copies=("bp", "size"), bp=("bp", "sum"), gbp=("gbp", "sum"),
                                     n_copies_mostly_in_genes=("mostly", "sum")).reset_index()
    agg["frac_bp_in_genes"] = (agg["gbp"] / agg["bp"]).round(4)
    # most common domain among genes each family overlaps
    gi, ti = window_pairs(genes, tes, 0)
    top = {}
    if len(gi):
        desc = (domains.assign(lab=lambda x: x["domain"] + ":" + x["description"].str.replace(" ", "_"))
                .groupby("gene")["lab"].apply(list).to_dict())
        fam_gene = pd.DataFrame({"f": tes["family"].to_numpy()[ti], "g": genes["gene"].to_numpy()[gi]}).drop_duplicates()
        cnt = collections.defaultdict(collections.Counter)
        for f, g in zip(fam_gene["f"], fam_gene["g"]):
            for lab in desc.get(g, []):
                cnt[f][lab] += 1
        top = {f: f"{c.most_common(1)[0][0]} ({c.most_common(1)[0][1]} genes)" for f, c in cnt.items() if c}
    agg["top_domain_of_overlapped_genes"] = agg["te_family"].map(top).astype(object).fillna("-")
    agg = agg.sort_values(["frac_bp_in_genes", "n_copies"], ascending=[False, False])
    return agg[["te_family", "n_copies", "bp", "frac_bp_in_genes", "n_copies_mostly_in_genes",
                "top_domain_of_overlapped_genes"]]


def flanks(genes, chroms, w):
    L = genes["chr"].map(chroms).to_numpy()
    s = genes["start"].to_numpy(); e = genes["end"].to_numpy()
    return np.maximum(0, s - w), s, e, np.minimum(L, e + w)


def density_table(genes, chroms, w, cov):
    ls, le, rs, re_ = flanks(genes, chroms, w)
    c = genes["chr"].to_numpy()
    fl = (le - ls) + (re_ - rs)
    te_bp = cov.covered(c, ls, le) + cov.covered(c, rs, re_)
    dens = np.where(fl > 0, te_bp / np.maximum(fl, 1), np.nan)
    return pd.DataFrame({"gene": genes["gene"], "window": w, "flank_bp": fl, "te_bp": te_bp,
                         "density": np.round(dens, 6)})


AUTO_GRID = (2500, 5000, 10000, 20000)


def choose_windows(genes, chroms, W, grid=AUTO_GRID, min_full=0.6):
    """Density windows adapted to assembly contiguity.

    A window is kept if >= min_full of genes have both flanks complete (not cut by a
    sequence end). In fragmented assemblies large windows mostly measure "up to the
    contig end", are noisy and not comparable with the same window in a contiguous
    genome, so they are dropped. The count window W is always kept.
    """
    rows = []
    for w in sorted(set(grid) | {W}):
        ls, le, rs, re_ = flanks(genes, chroms, w)
        fl = (le - ls) + (re_ - rs)
        full = float(np.mean(fl >= 2 * w)); present = float(np.mean(fl / (2 * w)))
        keep = (w == W) or full >= min_full
        why = "count window" if w == W else (f"{full:.0%} genes with full flanks >= {min_full:.0%}" if keep
                                             else f"only {full:.0%} genes with full flanks (< {min_full:.0%})")
        rows.append(dict(window=w, genes_full_flanks=round(full, 4), mean_flank_present=round(present, 4),
                         kept=keep, reason=why))
    t = pd.DataFrame(rows)
    return sorted(t.loc[t.kept, "window"].astype(int)), t


def build_tables(genes_path, tes_path, domains_path, chroms_path, outdir, window="mean", pool=(),
                 dwins=("w",), id_col=4, fam_col=None, level=None, min_class=20,
                 exclude_frac=None, count_overlapping=False, auto_min_full=0.6):
    """Writes the per-gene tables; returns (count window W, density windows used)."""
    os.makedirs(outdir, exist_ok=True)
    chroms = read_chroms(chroms_path)
    genes = read_genes(genes_path, chroms)
    if fam_col is None:
        fam_col = {4: 5, 5: 4}.get(id_col, 0)
    tes = read_tes(tes_path, chroms, id_col, fam_col)
    domains = read_domains(domains_path)
    if not len(tes):
        raise SystemExit("ERROR: no TEs on the sequences of the chromosome file")
    gene_ids = set(genes["gene"])
    n_unm = int((~domains["gene"].isin(gene_ids)).sum())
    if n_unm:
        log(f"WARNING: {n_unm} gene-domain pairs refer to genes absent from the gene file")
    # window
    src = "user-defined"
    if str(window) in ("mean", "median"):
        tabs = [genes] + [pd.DataFrame([(f[0], int(f[1]), int(f[2])) for f in read_table(p, 3)],
                                       columns=["chr", "start", "end"]) for p in pool]
        W, n = intergenic_window(tabs, str(window))
        src = f"{'pooled over %d gene sets' % len(tabs) if pool else 'this genome'}: {n} intergenic intervals"
        log(f"window = {W} bp ({src})")
    else:
        W = int(window)
    dwins = [str(x).strip() for x in dwins if str(x).strip()]
    dw = sorted({W if x == "w" else int(x) for x in dwins if x != "auto"})
    win_note = "user-defined"
    if "auto" in dwins:
        auto, wt = choose_windows(genes, chroms, W, min_full=auto_min_full)
        wt.to_csv(os.path.join(outdir, "window_choice.tsv"), sep="\t", index=False)
        dropped = wt.loc[~wt.kept, "window"].astype(int).tolist()
        dw = sorted(set(dw) | set(auto))
        win_note = (f"auto (kept if >= {auto_min_full:.0%} of genes have full flanks"
                    + (f"; dropped {', '.join(map(str, dropped))} bp" if dropped else "") + "; see window_choice.tsv)")
        log(f"density windows (auto): {', '.join(map(str, dw))} bp" + (f"; dropped {', '.join(map(str, dropped))} bp "
            f"(too few genes with complete flanks: fragmented assembly)" if dropped else ""))
    gene_track = build_track(genes)
    # TE family QC (+ optional exclusion)
    fam_note = "no family column"
    if fam_col:
        qc = family_qc(tes, genes, domains, gene_track)
        qc.to_csv(os.path.join(outdir, "te_family_QC.tsv"), sep="\t", index=False)
        sus = qc[(qc["n_copies"] >= 5) & (qc["frac_bp_in_genes"] >= 0.5)]
        fam_note = f"{len(sus)} families (>= 5 copies) with >= 50% of their bp inside genes ({int(sus['bp'].sum())} bp)"
        if len(sus):
            log(f"TE QC: {fam_note}; see te_family_QC.tsv (classify step separates artefacts from capture)")
        if exclude_frac:
            drop = set(qc.loc[qc["frac_bp_in_genes"] >= exclude_frac, "te_family"])
            n0 = len(tes); tes = tes[~tes["family"].isin(drop)].reset_index(drop=True)
            fam_note += f"; excluded {len(drop)} families with >= {exclude_frac} in genes ({n0 - len(tes)} copies)"
            log(f"TE QC: excluded {len(drop)} families ({n0 - len(tes)} copies)")
    log(f"{len(genes)} genes, {domains['domain'].nunique()} domains, {len(tes)} TEs")
    # counts
    if count_overlapping:
        tes_c = tes
    else:
        cov_g = Coverage(gene_track)
        tes_c = tes[cov_g.covered(tes["chr"].to_numpy(), tes["start"].to_numpy(), tes["end"].to_numpy()) == 0].reset_index(drop=True)
    gi, ti = window_pairs(genes, tes_c, W)
    pairs = pd.DataFrame({"gene": genes["gene"].to_numpy()[gi], "te": tes_c["te"].to_numpy()[ti],
                          "te_class": [te_class(x, level) for x in tes_c["family"].to_numpy()[ti]]}).drop_duplicates()
    pairs.sort_values(["gene", "te"]).to_csv(os.path.join(outdir, "obs_gene_TE.tsv"), sep="\t", index=False, header=False)
    log(f"counts: {len(pairs)} gene-TE pairs within {W} bp ({len(tes_c)} TEs not touching genes)")
    # density (merged TE sequence)
    track = build_track(tes); cov = Coverage(track)
    te_bp = sum(int((e - s).sum()) for s, e in track.values()); g_bp = sum(chroms.values())
    dens = pd.concat([density_table(genes, chroms, w, cov) for w in dw])
    dens.sort_values(["window", "gene"]).to_csv(os.path.join(outdir, "gene_TE_density.tsv"), sep="\t", index=False, na_rep="NA")
    for w in dw:
        log(f"density: flanks of {w} bp, mean flank TE density {dens.loc[dens.window == w, 'density'].mean():.3f} "
            f"(genome TE fraction {te_bp / g_bp:.3f})")
    # density by class
    if level:
        tes = tes.assign(te_class=[te_class(x, level) for x in tes["family"]])
        n_cls = tes["te_class"].value_counts()
        cls = [c for c in n_cls.index if n_cls[c] >= min_class]
        out = []
        for c in sorted(cls):
            cv = Coverage(build_track(tes[tes["te_class"] == c]))
            for w in dw:
                t = density_table(genes, chroms, w, cv); t.insert(2, "te_class", c); out.append(t)
        if out:
            pd.concat(out).to_csv(os.path.join(outdir, "gene_TE_density_by_class.tsv"), sep="\t", index=False, na_rep="NA")
        log(f"TE classes ({level}, >= {min_class} copies): " + ", ".join(f"{c} ({n_cls[c]})" for c in sorted(cls)))
    # gene bodies
    body = cov.covered(genes["chr"].to_numpy(), genes["start"].to_numpy(), genes["end"].to_numpy())
    bl = (genes["end"] - genes["start"]).to_numpy()
    bt = pd.DataFrame({"gene": genes["gene"], "body_bp": bl, "te_bp": body,
                       "body_te_frac": np.round(body / np.maximum(bl, 1), 4)})
    bt.to_csv(os.path.join(outdir, "gene_body_TE.tsv"), sep="\t", index=False)
    log(f"gene bodies: {(bt.body_te_frac >= 0.5).sum()} genes are >= 50% TE sequence")
    with open(os.path.join(outdir, "params.txt"), "w") as o:
        for k, v in [("tehood", __version__), ("date", time.ctime()), ("genes", genes_path),
                     ("tes", f"{tes_path} (id column {id_col}, family column {fam_col or 'none'})"),
                     ("domains", domains_path), ("chroms", chroms_path),
                     ("count_window_bp", f"{W} ({src})"),
                     ("count_TEs", f"{len(tes_c)} of {len(tes)}" + ("" if count_overlapping else " (TEs overlapping genes ignored)")),
                     ("density_flanks_bp", " ".join(map(str, dw)) + " (own gene body excluded; merged TE bp / flank bp)"),
                     ("density_window_choice", win_note),
                     ("te_class_level", level or "none"), ("te_family_QC", fam_note),
                     ("genome_TE_fraction", f"{te_bp / g_bp:.4f}")]:
            o.write(f"{k}\t{v}\n")
    log(f"done -> {outdir}")
    return W, dw


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tehood tables", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-g", "--genes", required=True, help="genes BED (chr start end gene_id . strand)")
    ap.add_argument("-t", "--tes", required=True, help="TE BED (chr start end te_id class/family strand)")
    ap.add_argument("-d", "--domains", required=True, help="gene <TAB> domain [<TAB> description]")
    ap.add_argument("-c", "--chroms", required=True, help="chrom sizes (chr <TAB> length)")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("-w", "--window", default="mean", help="count window: INT bp, mean or median [mean]")
    ap.add_argument("-G", "--pool", action="append", default=[], help="extra genes BED pooled for mean/median")
    ap.add_argument("-D", "--density-windows", default="w",
                    help="comma list of flank sizes; w = count window; auto = those of 2500,5000,10000,20000 "
                         "that the assembly supports (see -A) [w]")
    ap.add_argument("-A", "--auto-min-full", type=float, default=0.6,
                    help="auto windows: min share of genes with both flanks complete [0.6]")
    ap.add_argument("-k", "--id-col", type=int, default=4, help="TE id column [4]; BED with class in column 4 and id in column 5: 5")
    ap.add_argument("-F", "--family-col", type=int, default=None, help="TE family/class column [5 if -k 4, 4 if -k 5]")
    ap.add_argument("-L", "--level", choices=["order", "superfamily"], help="TE class level for class tables")
    ap.add_argument("-M", "--min-class", type=int, default=20, help="min copies for a class table [20]")
    ap.add_argument("-X", "--exclude-frac", type=float, help="drop TE families with >= FRAC of bp in genes")
    ap.add_argument("-I", "--count-overlapping", action="store_true", help="also count TEs overlapping genes")
    a = ap.parse_args(argv)
    if a.window not in ("mean", "median") and not a.window.isdigit():
        ap.error("--window must be an integer, mean or median")
    build_tables(a.genes, a.tes, a.domains, a.chroms, a.outdir, a.window, a.pool,
                 a.density_windows.split(","), a.id_col, a.family_col, a.level, a.min_class,
                 a.exclude_frac, a.count_overlapping, a.auto_min_full)


if __name__ == "__main__":
    main()
