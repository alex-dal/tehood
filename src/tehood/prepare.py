"""
tehood prepare - turn raw annotations into the 4 standard inputs used by
`tehood tables` / `tehood test`. Python standard library only.

Sub-commands
  genes     GFF3 or GTF  -> genes.bed (protein-coding genes) + protein_map.tsv
  domains   InterProScan TSV (+ protein_map.tsv) -> domains.tsv (gene, domain, description)
  tes       RepeatMasker .out (from EDTA or RepeatModeler/RepeatMasker) or EDTA TEanno.gff3 -> tes.bed
  chroms    genome FASTA -> chrom.sizes (+ optional N-gap bed)
  window    one or more genes.bed -> mean/median intergenic distance (window size)
  all       run genes+domains+tes+chroms for one genome into an output folder

Standard outputs (all tab-separated, no header):
  genes.bed     chr  start0  end  gene_id  .  strand
  domains.tsv   gene_id  domain_id  description
  tes.bed       chr  start0  end  te_id  class/family  strand   (te_id = unique element id)
  chrom.sizes   chr  length
"""
import argparse
import gzip
import os
import re
import statistics
import sys
from collections import defaultdict

DEFAULT_EXCLUDED_TE_CLASSES = [
    "Simple_repeat", "Low_complexity", "Satellite", "rRNA", "tRNA", "snRNA",
    "scRNA", "srpRNA", "ARTEFACT", "Unspecified", "Unknown",
]


def log(msg):
    print(f"[tehood prepare] {msg}", file=sys.stderr)


def xopen(path, mode="rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


# --------------------------------------------------------------------------- #
# GFF3 / GTF
# --------------------------------------------------------------------------- #
def parse_attrs(s):
    """Parse GFF3 (key=value;) or GTF (key "value";) attributes."""
    attrs = {}
    s = s.strip()
    if "=" in s.split(";")[0] or ("=" in s and '"' not in s):
        for part in s.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                attrs[k.strip()] = v.strip()
    else:  # GTF
        for m in re.finditer(r'(\S+)\s+"([^"]*)"', s):
            attrs.setdefault(m.group(1), m.group(2))
        for m in re.finditer(r'(\S+)\s+([^";\s]+)\s*;', s):
            attrs.setdefault(m.group(1), m.group(2))
    return attrs


def strip_prefix(x):
    return re.sub(r"^(gene|rna|cds|mrna|transcript|exon)[-:]", "", x, flags=re.I)


def read_gff(path):
    """Return list of (seqid, type, start1, end, strand, attrs) and detected format."""
    feats, fmt = [], None
    with xopen(path) as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            if fmt is None:
                fmt = "gff3" if re.search(r"(^|;)\s*[A-Za-z_]+=", f[8]) else "gtf"
            feats.append((f[0], f[2], int(f[3]), int(f[4]), f[6], parse_attrs(f[8])))
    return feats, fmt


def genes_from_gff(path, gene_types=("gene",), all_genes=False):
    """Build gene table + protein/transcript alias -> gene map.

    Works for NCBI GFF3, JGI GFF3/GTF, funannotate/BRAKER/AUGUSTUS GFF3/GTF.
    Only genes with a CDS descendant are kept unless all_genes=True.
    """
    feats, fmt = read_gff(path)
    log(f"{path}: {len(feats)} features, format={fmt}")
    genes = {}            # gene_id -> [chr, start1, end, strand]
    aliases = defaultdict(set)   # alias -> {gene_id}
    has_cds = set()

    def add_alias(a, g):
        if a:
            aliases[a].add(g)
            aliases[strip_prefix(a)].add(g)

    if fmt == "gff3":
        by_id, parent = {}, {}
        for i, (c, t, s, e, st, a) in enumerate(feats):
            fid = a.get("ID")
            if fid:
                by_id[fid] = i
            if "Parent" in a:
                parent[i] = a["Parent"].split(",")[0]

        def top_gene(i, depth=0):
            c, t, s, e, st, a = feats[i]
            if t in gene_types or depth > 10:
                return i
            p = parent.get(i)
            if p is None or p not in by_id:
                return i  # top-level feature without gene parent (e.g. mRNA only)
            return top_gene(by_id[p], depth + 1)

        for i, (c, t, s, e, st, a) in enumerate(feats):
            if t in gene_types and "ID" in a:
                gid = strip_prefix(a["ID"])
                genes[gid] = [c, s, e, st]
                for k in ("ID", "Name", "locus_tag", "gene_id", "old_locus_tag"):
                    add_alias(a.get(k), gid)
        for i, (c, t, s, e, st, a) in enumerate(feats):
            if t not in ("CDS", "mRNA", "transcript"):
                continue
            gi = top_gene(i)
            gc, gt, gs, ge, gst, ga = feats[gi]
            gid = strip_prefix(ga.get("ID", f"{gc}_{gs}_{ge}"))
            if gid not in genes:
                genes[gid] = [gc, gs, ge, gst]
            if t == "CDS":
                has_cds.add(gid)
            for k in ("ID", "Name", "protein_id", "proteinId", "transcript_id",
                      "transcriptId", "name", "orig_protein_id", "orig_transcript_id"):
                v = a.get(k)
                if v:
                    add_alias(v.split("|")[-1] if k.startswith("orig_") else v, gid)
            # mRNA parent of a CDS
            p = parent.get(i)
            if p:
                add_alias(p, gid)
    else:  # GTF: no gene lines required; gene span = min/max over features
        for c, t, s, e, st, a in feats:
            gid = a.get("gene_id") or a.get("name") or a.get("transcript_id")
            if not gid:
                continue
            if gid in genes:
                g = genes[gid]
                g[1], g[2] = min(g[1], s), max(g[2], e)
            else:
                genes[gid] = [c, s, e, st]
            if t in ("CDS", "start_codon", "stop_codon"):
                has_cds.add(gid)
            for k in ("gene_id", "gene_name", "transcript_id", "transcriptId",
                      "protein_id", "proteinId", "name"):
                add_alias(a.get(k), gid)

    keep = genes if all_genes else {g: v for g, v in genes.items() if g in has_cds}
    log(f"{len(genes)} genes found, {len(keep)} kept "
        f"({'all' if all_genes else 'protein-coding, i.e. with CDS'})")
    amap = {}
    n_amb = 0
    for al, gs in aliases.items():
        gs = gs & keep.keys()
        if len(gs) == 1:
            amap[al] = next(iter(gs))
        elif len(gs) > 1:
            n_amb += 1
    if n_amb:
        log(f"{n_amb} ambiguous aliases (map to >1 gene) were dropped")
    return keep, amap


def write_genes(genes, amap, out_bed, out_map):
    rows = sorted(((v[0], v[1] - 1, v[2], g, ".", v[3]) for g, v in genes.items()),
                  key=lambda r: (r[0], r[1], r[2]))
    with open(out_bed, "w") as fh:
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    with open(out_map, "w") as fh:
        for al in sorted(amap):
            fh.write(f"{al}\t{amap[al]}\n")
    log(f"wrote {out_bed} ({len(rows)} genes) and {out_map} ({len(amap)} aliases)")


# --------------------------------------------------------------------------- #
# InterProScan
# --------------------------------------------------------------------------- #
def domains_from_ipr(ipr_tsv, map_file, dtype="ipr", extra=None, min_map=0.5):
    amap = {}
    with open(map_file) as fh:
        for line in fh:
            a, g = line.rstrip("\n").split("\t")
            amap[a] = g
    pairs, prots, unmapped = {}, set(), set()

    def lookup(p):
        for cand in (p, strip_prefix(p), p.split("|")[-1], re.sub(r"\.\d+$", "", p),
                     re.sub(r"[-.]T?\d+$", "", p), re.sub(r"[-.]p\d+$", "", p)):
            if cand in amap:
                return amap[cand]
        return None

    with xopen(ipr_tsv) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5 or line.startswith("#"):
                continue
            prot = f[0]
            prots.add(prot)
            g = lookup(prot)
            if g is None:
                unmapped.add(prot)
                continue
            ids = []
            if dtype in ("ipr", "both") and len(f) > 11 and f[11].startswith("IPR"):
                ids.append((f[11], f[12] if len(f) > 12 else ""))
            if dtype in ("pfam", "both") and f[3].lower() == "pfam":
                ids.append((f[4], f[5] if len(f) > 5 else ""))
            if dtype == "all-db":
                ids.append((f[4], f[5] if len(f) > 5 else ""))
            for d, desc in ids:
                key = (g, d)
                if key not in pairs or (not pairs[key] and desc):
                    pairs[key] = desc.replace("\t", " ")
    rate = 1 - len(unmapped) / max(1, len(prots))
    log(f"{len(prots)} proteins in InterProScan file, {rate:.1%} mapped to genes")
    if unmapped:
        log("examples of unmapped protein IDs: " + ", ".join(sorted(unmapped)[:5]))
    if rate < min_map:
        sys.exit(f"ERROR: only {rate:.1%} of proteins could be mapped to genes. "
                 "Check that the protein FASTA given to InterProScan comes from the "
                 "same annotation as the GFF (or supply a 2-column protein->gene "
                 "table with --map).")
    if extra:  # custom gene sets, e.g. CSEPs: gene <tab> set_id [<tab> description]
        n = 0
        with open(extra) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 2:
                    g = lookup(f[0]) or f[0]
                    pairs[(g, f[1])] = f[2] if len(f) > 2 else ""
                    n += 1
        log(f"added {n} custom gene-set annotations from {extra}")
    return pairs


def write_domains(pairs, out):
    with open(out, "w") as fh:
        for (g, d) in sorted(pairs):
            fh.write(f"{g}\t{d}\t{pairs[(g, d)]}\n")
    log(f"wrote {out}: {len(pairs)} gene-domain pairs, "
        f"{len({d for _, d in pairs})} domains, {len({g for g, _ in pairs})} genes")


# --------------------------------------------------------------------------- #
# TEs
# --------------------------------------------------------------------------- #
def class_excluded(cls, excluded):
    top = cls.split("/")[0]
    return cls in excluded or top in excluded


def tes_from_rm_out(path, excluded, min_len=0, max_div=None):
    """RepeatMasker .out (EDTA's RM run, or RepeatModeler+RepeatMasker)."""
    rows, n_in = [], 0
    with xopen(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) < 15 or not f[0].isdigit():
                continue  # header lines
            n_in += 1
            div = float(f[1])
            chrom, s, e = f[4], int(f[5]) - 1, int(f[6])
            strand = "-" if f[8] == "C" else "+"
            fam, cls, rid = f[9], f[10], f[14]
            if class_excluded(cls, excluded) or (e - s) < min_len:
                continue
            if max_div is not None and div > max_div:
                continue
            # RM ID is shared by fragments of the same element -> element-level id
            rows.append((chrom, s, e, f"{chrom}:{rid}", f"{cls}/{fam}", strand))
    log(f"{path}: {n_in} RepeatMasker hits, {len(rows)} kept after filtering")
    return rows


EDTA_SKIP_TYPES = {"repeat_region", "target_site_duplication", "long_terminal_repeat",
                   "primer_binding_site", "RR_tract"}


def tes_from_edta_gff(path, excluded, min_len=0):
    """EDTA *.EDTA.TEanno.gff3 (EDTA run with --anno 1)."""
    rows, n_in = [], 0
    with xopen(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            n_in += 1
            if f[2] in EDTA_SKIP_TYPES:
                continue
            a = parse_attrs(f[8])
            cls = a.get("Classification", f[2])
            if class_excluded(cls, excluded):
                continue
            s, e = int(f[3]) - 1, int(f[4])
            if e - s < min_len:
                continue
            tid = a.get("Parent") or a.get("ID") or f"{f[0]}_{s}_{e}"
            rows.append((f[0], s, e, f"{f[0]}:{tid}", f"{cls}/{a.get('Name', '.')}",
                         f[6] if f[6] in "+-" else "+"))
    log(f"{path}: {n_in} features, {len(rows)} TE features kept")
    return rows


def write_bed(rows, out):
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    with open(out, "w") as fh:
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    log(f"wrote {out} ({len(rows)} records)")


# --------------------------------------------------------------------------- #
# FASTA
# --------------------------------------------------------------------------- #
def chroms_from_fasta(path, out, gaps_out=None, min_gap=10):
    sizes, gaps = [], []
    name, L, gap_start = None, 0, None

    def close_gap(pos):
        nonlocal gap_start
        if gap_start is not None and pos - gap_start >= min_gap:
            gaps.append((name, gap_start, pos))
        gap_start = None

    with xopen(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    close_gap(L)
                    sizes.append((name, L))
                name, L, gap_start = line[1:].split()[0], 0, None
                continue
            seq = line.strip()
            if gaps_out:
                for m in re.finditer(r"[^Nn]+|[Nn]+", seq):
                    if m.group()[0] in "Nn":
                        if gap_start is None:
                            gap_start = L + m.start()
                    else:
                        close_gap(L + m.start())
            L += len(seq)
    if name is not None:
        close_gap(L)
        sizes.append((name, L))
    with open(out, "w") as fh:
        for n, l in sizes:
            fh.write(f"{n}\t{l}\n")
    log(f"wrote {out} ({len(sizes)} sequences, {sum(l for _, l in sizes):,} bp)")
    if gaps_out:
        write_bed([(c, s, e) for c, s, e in gaps], gaps_out)


# --------------------------------------------------------------------------- #
# intergenic distance -> window size
# --------------------------------------------------------------------------- #
def intergenic(beds):
    dists = []
    for b in beds:
        rows = []
        with open(b) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.split("\t")
                rows.append((f[0], int(f[1]), int(f[2])))
        rows.sort()
        prev_c, prev_end = None, None
        for c, s, e in rows:
            if c == prev_c:
                d = s - prev_end
                if d > 0:  # overlapping/nested genes are ignored
                    dists.append(d)
                prev_end = max(prev_end, e)
            else:
                prev_c, prev_end = c, e
    if not dists:
        sys.exit("no intergenic distances found")
    print(f"n_intervals\t{len(dists)}")
    print(f"mean_intergenic_bp\t{statistics.mean(dists):.0f}")
    print(f"median_intergenic_bp\t{statistics.median(dists):.0f}")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog='tehood prepare', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("genes", help="GFF3/GTF -> genes.bed + protein_map.tsv")
    g.add_argument("--gff", required=True)
    g.add_argument("--out-bed", default="genes.bed")
    g.add_argument("--out-map", default="protein_map.tsv")
    g.add_argument("--gene-types", default="gene",
                   help="comma-separated feature types that define genes [gene]")
    g.add_argument("--all-genes", action="store_true",
                   help="keep genes without CDS (ncRNA, pseudogenes)")

    d = sub.add_parser("domains", help="InterProScan TSV -> domains.tsv")
    d.add_argument("--ipr", required=True, help="InterProScan TSV (-f TSV -iprlookup)")
    d.add_argument("--map", required=True,
                   help="protein_map.tsv from 'genes', or any 2-col protein->gene table")
    d.add_argument("--type", default="ipr", choices=["ipr", "pfam", "both", "all-db"])
    d.add_argument("--extra", help="custom gene sets: gene<TAB>set_id[<TAB>desc] (e.g. CSEPs)")
    d.add_argument("--out", default="domains.tsv")
    d.add_argument("--min-mapped", type=float, default=0.5)

    t = sub.add_parser("tes", help="RepeatMasker .out or EDTA TEanno.gff3 -> tes.bed")
    src = t.add_mutually_exclusive_group(required=True)
    src.add_argument("--rm-out", help="RepeatMasker .out (EDTA or RepeatModeler libs)")
    src.add_argument("--edta-gff", help="EDTA *.EDTA.TEanno.gff3")
    t.add_argument("--exclude-classes", default=",".join(DEFAULT_EXCLUDED_TE_CLASSES))
    t.add_argument("--min-len", type=int, default=0)
    t.add_argument("--max-div", type=float, default=None,
                   help="max %% divergence (RepeatMasker only)")
    t.add_argument("--out", default="tes.bed")

    c = sub.add_parser("chroms", help="FASTA -> chrom.sizes (+ N-gaps bed)")
    c.add_argument("--fasta", required=True)
    c.add_argument("--out", default="chrom.sizes")
    c.add_argument("--gaps", help="also write runs of N to this bed (use with --exclude)")

    w = sub.add_parser("window", help="mean/median intergenic distance from genes.bed files")
    w.add_argument("beds", nargs="+")

    a = sub.add_parser("all", help="genes+domains+tes+chroms for one genome")
    a.add_argument("--gff", required=True)
    a.add_argument("--ipr", required=True)
    a.add_argument("--fasta", required=True)
    s2 = a.add_mutually_exclusive_group(required=True)
    s2.add_argument("--rm-out")
    s2.add_argument("--edta-gff")
    a.add_argument("--type", default="ipr", choices=["ipr", "pfam", "both", "all-db"])
    a.add_argument("--extra")
    a.add_argument("--exclude-classes", default=",".join(DEFAULT_EXCLUDED_TE_CLASSES))
    a.add_argument("--min-len", type=int, default=0)
    a.add_argument("--outdir", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "genes":
        genes, amap = genes_from_gff(args.gff, tuple(args.gene_types.split(",")),
                                     args.all_genes)
        write_genes(genes, amap, args.out_bed, args.out_map)
    elif args.cmd == "domains":
        write_domains(domains_from_ipr(args.ipr, args.map, args.type, args.extra,
                                       args.min_mapped), args.out)
    elif args.cmd == "tes":
        ex = set(x for x in args.exclude_classes.split(",") if x)
        rows = (tes_from_rm_out(args.rm_out, ex, args.min_len, args.max_div)
                if args.rm_out else tes_from_edta_gff(args.edta_gff, ex, args.min_len))
        write_bed(rows, args.out)
    elif args.cmd == "chroms":
        chroms_from_fasta(args.fasta, args.out, args.gaps)
    elif args.cmd == "window":
        intergenic(args.beds)
    elif args.cmd == "all":
        os.makedirs(args.outdir, exist_ok=True)
        o = lambda f: os.path.join(args.outdir, f)
        genes, amap = genes_from_gff(args.gff)
        write_genes(genes, amap, o("genes.bed"), o("protein_map.tsv"))
        write_domains(domains_from_ipr(args.ipr, o("protein_map.tsv"), args.type,
                                       args.extra), o("domains.tsv"))
        ex = set(x for x in args.exclude_classes.split(",") if x)
        rows = (tes_from_rm_out(args.rm_out, ex, args.min_len) if args.rm_out
                else tes_from_edta_gff(args.edta_gff, ex, args.min_len))
        write_bed(rows, o("tes.bed"))
        chroms_from_fasta(args.fasta, o("chrom.sizes"), o("gaps.bed"))
        intergenic([o("genes.bed")])


if __name__ == "__main__":
    main()
