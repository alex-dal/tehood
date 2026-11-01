"""
tehood explain - which TE families make a domain look TE-associated?

For every hit (TE-associated / TE-depleted in a `tehood test` results file) it
reports the TE sequence within W bp of the domain's genes by family, the top
family, and the SELF-FAMILY share: the share of that flanking TE sequence that
belongs to families which also overlap the gene bodies of >= 2 genes of the same
domain.

A high self-family share (e.g. > 0.5) means the association is circular: the
genes sit next to copies of a TE family that itself contains (parts of) these
genes. Possible causes, which annotation alone cannot separate:
  - gene capture: a TE carrying the gene amplified (Pack-MULE, Helitron, Starship)
  - segmental / tandem duplication of the gene with its flank, flank called "TE"
  - a host gene family in the TE library (run `tehood classify` first)
Check with sequence: TIRs/TSDs at copy ends, transposase ORF in the consensus,
synteny/presence-absence in relatives (README section 5).

usage: tehood explain --results SP_density_w10000_results.tsv --genes genes.bed \
          --tes tes.bed --domains domains.tsv --window 10000 --out SP_hit_TE_families.tsv
       (tes.bed: `tehood prepare` / `tehood classify` format, family in column 5)
"""
import argparse, bisect, collections


def main(argv=None):
    ap = argparse.ArgumentParser(prog='tehood explain', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("results", "genes", "tes", "domains", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--window", type=int, default=10000)
    ap.add_argument("--all", action="store_true", help="all tested domains, not only hits")
    a = ap.parse_args(argv)

    genes = {}
    with open(a.genes) as fh:
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) >= 4 and not l.startswith("#"):
                genes[f[3].strip()] = (f[0], int(f[1]), int(f[2]))
    dom_genes = collections.defaultdict(set)
    with open(a.domains) as fh:
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) >= 2 and f[0].strip() in genes:
                dom_genes[f[1].strip()].add(f[0].strip())
    tes = collections.defaultdict(list)
    with open(a.tes) as fh:
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if len(f) >= 5:
                tes[f[0]].append((int(f[1]), int(f[2]), f[3], f[4]))
    for c in tes: tes[c].sort()
    starts = {c: [x[0] for x in v] for c, v in tes.items()}
    maxlen = {c: max((x[1] - x[0] for x in v), default=0) for c, v in tes.items()}

    def tes_in(c, s, e):
        v = tes.get(c, []); i0 = bisect.bisect_left(starts.get(c, []), s - maxlen.get(c, 0))
        i1 = bisect.bisect_right(starts.get(c, []), e)
        return [x for x in v[i0:i1] if x[0] < e and x[1] > s]

    hdr = None; rows = []
    with open(a.results) as fh:
        hdr = fh.readline().rstrip("\n").split("\t"); H = {k: i for i, k in enumerate(hdr)}
        for l in fh:
            f = l.rstrip("\n").split("\t")
            if a.all and f[H["tested"]] != "TRUE": continue
            if not a.all and f[H["status"]] not in ("TE-associated", "TE-depleted"): continue
            rows.append(f)
    W = a.window
    out = open(a.out, "w")
    out.write("domain\tdescription\tstatus\tn_genes\tflank_TE_bp\ttop_family\ttop_family_share\t"
              "self_family_share\tself_families\tn_families\n")
    for f in rows:
        d = f[H["domain"]]; gs = dom_genes.get(d, set())
        # families inside the gene bodies of this domain (>= 2 genes)
        body_fam = collections.Counter()
        for g in gs:
            c, s, e = genes[g]
            for fam in {x[3] for x in tes_in(c, s, e)}: body_fam[fam] += 1
        selff = {k for k, v in body_fam.items() if v >= 2}
        # TE bp in the flanks (own gene body excluded), each TE counted once per domain
        seen, bp = set(), collections.Counter()
        for g in gs:
            c, s, e = genes[g]
            for lo, hi in ((max(0, s - W), s), (e, e + W)):
                for x in tes_in(c, lo, hi):
                    key = (c, x[0], x[1], x[2])
                    if key in seen: continue
                    seen.add(key); bp[x[3]] += min(hi, x[1]) - max(lo, x[0])
        tot = sum(bp.values())
        top, topbp = (bp.most_common(1)[0] if bp else ("-", 0))
        self_bp = sum(v for k, v in bp.items() if k in selff)
        out.write(f"{d}\t{f[H['description']]}\t{f[H['status']]}\t{f[H['n_genes']]}\t{tot}\t{top}\t"
                  f"{topbp / tot if tot else 0:.3f}\t{self_bp / tot if tot else 0:.3f}\t"
                  f"{','.join(sorted(selff)) or '-'}\t{len(bp)}\n")
    out.close()


if __name__ == "__main__":
    main()
