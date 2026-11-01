"""tehood run - the whole analysis for one genome, or for genomes listed in a samples table.

From raw annotations:
  tehood run --gff genomic.gff --ipr protein.faa.tsv --fasta genome.fna \\
             --te genome.fna.out --name sp1 --outdir results/sp1
From already prepared inputs (BED files and a gene<TAB>domain table):
  tehood run --genes-bed g.bed --tes-bed t.bed --domains d.tsv --chroms c.tsv \\
             [--rm-out g.out] --name sp1 --outdir results/sp1
Many genomes (samples.tsv: sample gff ipr fasta te [te_format] [window]):
  tehood run --samples samples.tsv --outdir results [--row N | --jobs 4]
  (--row N runs only row N, e.g. --row $SLURM_ARRAY_TASK_ID in a SLURM array)

Steps: prepare -> classify (TE copies inside genes; removes only artefact copies)
       -> tables -> test (count; density at each window; per TE class; domain pairs and
       architectures) -> compare -> explain.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys

from . import associate, classify, compare, explain, prepare, tables


def log(msg):
    print(f"[tehood run] {msg}", file=sys.stderr)


def run_one(a, name, outdir, gff=None, ipr=None, fasta=None, te=None, te_format="rm-out", window=None):
    os.makedirs(outdir, exist_ok=True)
    inp = os.path.join(outdir, "inputs")
    rm_out = a.rm_out if a.rm_out else (te if te_format == "rm-out" else None)
    if a.genes_bed:                                       # prepared inputs
        genes, tes, doms, chroms = a.genes_bed, a.tes_bed, a.domains, a.chroms
        id_col, fam_col = a.id_col, a.family_col
    else:
        args = ["all", "--gff", gff, "--ipr", ipr, "--fasta", fasta, f"--{te_format}", te,
                "--type", a.domain_type, "--outdir", inp]
        if a.extra: args += ["--extra", a.extra]
        prepare.main(args)
        genes, tes, doms, chroms = (os.path.join(inp, f) for f in ("genes.bed", "tes.bed", "domains.tsv", "chrom.sizes"))
        id_col, fam_col = 4, 5
    if rm_out and not a.no_genic_filter:
        filt = os.path.join(outdir, "tes.genic_filtered.bed")
        classify.main(["--rm-out", rm_out, "--genes", genes, "--domains", doms,
                       "--te-domains", associate.data_file("te_viral_domains_S1A.tsv"),
                       "--out", os.path.join(outdir, "genic_TE"), "--filtered-bed", filt])
        tes, id_col, fam_col = filt, 4, 5
    dwins = a.density_windows.split(",") + ([] if a.extra_window == "auto" else [a.extra_window])
    W, wins = tables.build_tables(genes, tes, doms, chroms, outdir, window or a.window, [], dwins, id_col, fam_col,
                                  a.level, 20, a.exclude_frac, False, a.auto_min_full)
    if a.extra_window == "auto":            # class / set analyses: largest density window <= 10 kb
        ex = max([w for w in wins if w <= 10000] or [min(wins)])
    else:
        ex = int(a.extra_window)
    log(f"{name}: density windows {', '.join(map(str, wins))} bp; class and set analyses at {ex} bp")
    t = lambda f: os.path.join(outdir, f)
    body = ["--body-te", t("gene_body_TE.tsv")] + (["--max-body-te", str(a.max_body_te)] if a.max_body_te is not None else [])
    common = ["--genes", genes, "--domains", doms, "--name", name, "--nperm", str(a.nperm), "--seed", str(a.seed)] + body
    if not a.plot: common.append("--no-plot")
    T = lambda extra: associate.main(common + extra)
    T(["--stat", "count", "--pairs", t("obs_gene_TE.tsv"), "--out", t(f"{name}_count")])
    if len(wins) > 1:
        T(["--stat", "density", "--density", t("gene_TE_density.tsv"), "--out", t(f"{name}_density")])
    else:                                   # keep the density_w<bp> name (comparable across genomes)
        T(["--stat", "density", "--density", t("gene_TE_density.tsv"), "--out", t(f"{name}_density_w{wins[0]}")])
    res = [t(f"{name}_count_results.tsv")] + [t(f"{name}_density_w{w}_results.tsv") for w in wins]
    names = ["count"] + [f"density_w{w}" for w in wins]
    compare.main(["--inputs", ",".join(res), "--names", ",".join(names), "--out", t(name)] + ([] if a.plot else ["--no-plot"]))
    for f, n, w in zip(res, names, [W] + wins):
        explain.main(["--results", f, "--genes", genes, "--tes", tes, "--domains", doms,
                      "--window", str(w), "--out", f.replace("_results.tsv", "_hit_TE_families.tsv")])
    if a.level:
        T(["--stat", "density", "--density", t("gene_TE_density.tsv"), "--window", str(ex),
           "--by-class", "--out", t(f"{name}_density_classes")])
        T(["--stat", "count", "--pairs", t("obs_gene_TE.tsv"), "--by-class", "--out", t(f"{name}_count_classes")])
    for s in [x for x in a.sets.split(",") if x]:
        T(["--stat", "density", "--density", t("gene_TE_density.tsv"), "--window", str(ex),
           "--sets", s, "--out", t(f"{name}_density_{s}")])
        if s == "pair":
            T(["--stat", "count", "--pairs", t("obs_gene_TE.tsv"), "--sets", "pair", "--out", t(f"{name}_count_pair")])
    log(f"{name}: done -> {outdir}")


def read_samples(path):
    rows = []
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if line.strip() and not line.startswith("#"):
                f = line.rstrip("\n").split("\t"); f += [""] * (7 - len(f))
                rows.append(dict(name=f[0], gff=f[1], ipr=f[2], fasta=f[3], te=f[4],
                                 te_format=f[5] or "rm-out", window=f[6] or None))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tehood run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("raw inputs (one genome)")
    g.add_argument("--gff"); g.add_argument("--ipr"); g.add_argument("--fasta")
    g.add_argument("--te", help="RepeatMasker .out (EDTA or RepeatModeler libraries) or EDTA TEanno.gff3")
    g.add_argument("--te-format", choices=["rm-out", "edta-gff"], default="rm-out")
    g.add_argument("--domain-type", choices=["ipr", "pfam", "both", "all-db"], default="ipr")
    g.add_argument("--extra", help="custom gene sets: gene<TAB>set_id[<TAB>description], e.g. CSEPs")
    p = ap.add_argument_group("prepared inputs (one genome)")
    p.add_argument("--genes-bed"); p.add_argument("--tes-bed"); p.add_argument("--domains"); p.add_argument("--chroms")
    p.add_argument("--id-col", type=int, default=4, help="TE id column of --tes-bed [4]; use 5 for BEDs with the class in column 4")
    p.add_argument("--family-col", type=int, default=None, help="TE family/class column [5 if --id-col 4, else 4]")
    p.add_argument("--rm-out", help="RepeatMasker .out matching --tes-bed (enables the classify step)")
    m = ap.add_argument_group("many genomes")
    m.add_argument("--samples", help="samples.tsv: sample gff ipr fasta te [te_format] [window]")
    m.add_argument("--row", type=int, help="run only this 1-based row (SLURM array index)")
    m.add_argument("--jobs", type=int, default=1, help="genomes in parallel [1]")
    o = ap.add_argument_group("analysis")
    o.add_argument("--name", default="sample"); o.add_argument("--outdir", required=True)
    o.add_argument("--window", default="mean", help="count window: INT, mean or median [mean]")
    o.add_argument("--density-windows", default="w,auto",
                   help="density flank sizes: bp, w (= count window), auto (those of 2500,5000,10000,20000 "
                        "the assembly supports, see --auto-min-full) [w,auto]")
    o.add_argument("--auto-min-full", type=float, default=0.6,
                   help="auto windows: keep a window if >= this share of genes have both flanks complete [0.6]")
    o.add_argument("--extra-window", default="auto",
                   help="density window for class/set analyses: bp, or auto = largest density window <= 10 kb [auto]")
    o.add_argument("--level", default="superfamily", help="TE class level: order, superfamily or '' [superfamily]")
    o.add_argument("--sets", default="pair,architecture", help="extra gene-set analyses [pair,architecture]; '' = none")
    o.add_argument("--max-body-te", type=float, default=0.5, help="drop genes >= this share TE [0.5]; -1 = keep all")
    o.add_argument("--exclude-frac", type=float, help="drop whole TE families with >= FRAC bp in genes [off]")
    o.add_argument("--no-genic-filter", action="store_true", help="skip the classify step")
    o.add_argument("--nperm", type=int, default=10000); o.add_argument("--seed", type=int, default=1)
    o.add_argument("--no-plot", dest="plot", action="store_false")
    a = ap.parse_args(argv)
    a.level = a.level or None
    if a.max_body_te is not None and a.max_body_te < 0: a.max_body_te = None
    if a.samples:
        rows = read_samples(a.samples)
        if a.row: rows = [rows[a.row - 1]]
        jobs = [(r, os.path.join(a.outdir, r["name"])) for r in rows]
        if a.jobs > 1 and len(jobs) > 1:
            with cf.ProcessPoolExecutor(a.jobs) as ex:
                for f in [ex.submit(run_one, a, r["name"], o_, r["gff"], r["ipr"], r["fasta"], r["te"],
                                    r["te_format"], r["window"]) for r, o_ in jobs]:
                    f.result()
        else:
            for r, o_ in jobs:
                run_one(a, r["name"], o_, r["gff"], r["ipr"], r["fasta"], r["te"], r["te_format"], r["window"])
    elif a.genes_bed:
        if not (a.tes_bed and a.domains and a.chroms):
            ap.error("--genes-bed needs --tes-bed, --domains and --chroms")
        run_one(a, a.name, a.outdir)
    else:
        if not (a.gff and a.ipr and a.fasta and a.te):
            ap.error("give --gff --ipr --fasta --te, or --genes-bed ..., or --samples")
        run_one(a, a.name, a.outdir, a.gff, a.ipr, a.fasta, a.te, a.te_format)


if __name__ == "__main__":
    main()
