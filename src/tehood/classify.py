"""
tehood classify - for TE families whose copies sit inside gene models,
collect the evidence that distinguishes four explanations:

Every COPY is first classified:
  young_unit_with_gene    near-identical to the consensus (div < --young-div) over
                          most of it (>= 80%) and containing / overlapping a gene:
                          a recently amplified unit that carries a gene ->
                          gene capture by a TE (e.g. Pack-MULE, Helitron, Starship)
                          OR recent segmental duplication; needs TIR/TSD check
  paralog_homology        old (div >= --old-div), inside a gene, covering only part
                          of it: RepeatMasker hit to a host paralog of a gene that
                          is part of the consensus -> annotation ARTEFACT
  other_genic / intergenic

and every FAMILY gets a call (first rule that fits):
  TE_ORF_gene             overlapped "genes" carry only TE/viral domains (Table S1A)
  TE_gene_fusion          overlapped genes carry TE/viral + host domains
  gene_in_mobile_unit     >= 2 young full-length copies with a gene (capture candidate)
                          [+ paralog hits, reported separately]
  paralog_homology        most genic copies are old partial hits inside genes, and
                          no young gene-carrying units: library contains a host
                          gene; the "TE-association" of that gene family is an artefact
  consensus_is_gene       genic copies match the whole consensus within gene bounds,
                          consensus ~ gene length (contamination, or a TE = gene)
  ambiguous               none of the above

--filtered-bed writes the TE annotation WITHOUT the paralog_homology copies, so
real (and possibly gene-carrying) TE copies are kept for the association test.

This is an ANNOTATION-ONLY triage: it cannot see TIRs, TSDs, transposase ORFs in
the consensus, synteny, or transcripts. See the 'decisive checks' printed at the
end and README section 5.

Inputs : RepeatMasker .out (with the library used for annotation), genes BED6,
         domains (gene, domain[, description]), TE/viral domain list (IPR ids,
         e.g. Table S1A of Almeida et al. 2025, MBE 42:msaf097)
Outputs: PREFIX_families.tsv  one row per family with >= --min-copies copies and
                              >= --min-genic of its bp inside genes
         PREFIX_copies.tsv    one row per element of those families (gene relation,
                              consensus coverage, divergence)
usage: tehood classify --rm-out g.out --genes genes.bed --domains domains.tsv \
           --te-domains te_viral_domains_S1A.tsv --out PREFIX
"""
import argparse, bisect, collections, statistics, sys

EXCL = {"Simple_repeat", "Low_complexity", "Satellite", "rRNA", "tRNA", "snRNA",
        "scRNA", "srpRNA", "ARTEFACT", "Unspecified", "Unknown"}


def read_rm(path):
    """element-level records: fragments sharing an RM ID are merged"""
    el = {}
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) < 15 or not f[0].isdigit():
                continue
            cls = f[10]
            if cls in EXCL or cls.split("/")[0] in EXCL:
                continue
            if f[8] == "+":
                rb, re_, rl = int(f[11]), int(f[12]), int(f[13].strip("()"))
            else:
                rl, re_, rb = int(f[11].strip("()")), int(f[12]), int(f[13])
            key = (f[4], f[14])
            e = el.setdefault(key, dict(chr=f[4], id=f[14], fam=f[9], cls=cls, s=10**12, e=0,
                                        cons_len=re_ + rl, segs=[], div=[], bp=0))
            s, t = int(f[5]) - 1, int(f[6])
            e["s"], e["e"] = min(e["s"], s), max(e["e"], t)
            e["segs"].append((s, t, min(rb, re_), max(rb, re_)))
            e["div"].append((float(f[1]), t - s)); e["bp"] += t - s
    return list(el.values())


def union_len(iv):
    iv = sorted(iv); tot, cs, ce = 0, None, None
    for a, b in iv:
        if cs is None or a > ce:
            if cs is not None: tot += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return tot + (ce - cs if cs is not None else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='tehood classify', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rm-out", required=True); ap.add_argument("--genes", required=True)
    ap.add_argument("--domains", required=True); ap.add_argument("--te-domains", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-copies", type=int, default=3)
    ap.add_argument("--min-genic", type=float, default=0.5, help="min share of family bp inside genes")
    ap.add_argument("--flank", type=int, default=200, help="bp beyond a gene end to call 'extends'")
    ap.add_argument("--young-div", type=float, default=5.0, help="max %%div for a young copy")
    ap.add_argument("--old-div", type=float, default=15.0, help="min %%div for an old paralog hit")
    ap.add_argument("--filtered-bed", help="write TE BED (`tehood prepare` format) without paralog_homology copies")
    a = ap.parse_args(argv)

    te_dom = set()
    with open(a.te_domains) as fh:
        for line in fh:
            x = line.split("\t")[0].strip()
            if x.startswith(("IPR", "PF")): te_dom.add(x)
    genes, by_chr = {}, collections.defaultdict(list)
    with open(a.genes) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4 or line.startswith("#"): continue
            g = f[3].strip(); genes[g] = (f[0], int(f[1]), int(f[2]))
            by_chr[f[0]].append((int(f[1]), int(f[2]), g))
    for c in by_chr: by_chr[c].sort()
    starts = {c: [x[0] for x in v] for c, v in by_chr.items()}
    gdom, gdesc = collections.defaultdict(set), {}
    with open(a.domains) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2: continue
            g, d = f[0].strip(), f[1].strip(); gdom[g].add(d)
            if len(f) > 2: gdesc[d] = f[2].strip()

    def overlapping(c, s, e):
        v = by_chr.get(c, []); i = bisect.bisect_right(starts.get(c, []), e)
        return [x for x in v[max(0, i - 50):i] if x[0] < e and x[1] > s]

    els = read_rm(a.rm_out)
    fam = collections.defaultdict(list)
    for e in els:
        e["genes"] = overlapping(e["chr"], e["s"], e["e"])
        fam[e["fam"]].append(e)

    rows, crow = [], []
    for f, E in fam.items():
        bp = sum(x["bp"] for x in E)
        genic_bp = 0
        for x in E:
            for s, t, _, _ in x["segs"]:
                genic_bp += sum(max(0, min(t, ge) - max(s, gs)) for gs, ge, _ in x["genes"]) if x["genes"] else 0
        genic_frac = min(1.0, genic_bp / bp) if bp else 0
        if len(E) < a.min_copies or genic_frac < a.min_genic:
            continue
        cons = statistics.median(x["cons_len"] for x in E)
        n_gen = n_cont = n_span = n_part = 0
        cov_g, gl, cons_out, divs_g, divs_o = [], [], [], [], []
        ov_genes = set()
        for x in E:
            divx = sum(d * l for d, l in x["div"]) / max(1, sum(l for _, l in x["div"]))
            cons_cov = union_len([(rb, re_) for _, _, rb, re_ in x["segs"]]) / max(1, x["cons_len"])
            rel, gcov, outside_cons = "outside", 0.0, 0.0
            if x["genes"]:
                n_gen += 1; divs_g.append(divx)
                gs, ge, g = max(x["genes"], key=lambda y: min(x["e"], y[1]) - max(x["s"], y[0]))
                ov_genes.update(y[2] for y in x["genes"]); gl.append(ge - gs)
                left, right = gs - x["s"] > a.flank, x["e"] - ge > a.flank
                rel = "spans_gene_both_flanks" if left and right else \
                      "extends_one_flank" if left or right else "within_gene"
                if rel == "within_gene": n_cont += 1
                elif rel == "spans_gene_both_flanks": n_span += 1
                else: n_part += 1
                gcov = union_len([(max(s, gs), min(t, ge)) for s, t, _, _ in x["segs"] if t > gs and s < ge]) / max(1, ge - gs)
                cov_g.append(gcov)
                # share of the consensus matched OUTSIDE the gene body (flank part of the element)
                out_c = [(rb, re_) for s, t, rb, re_ in x["segs"] if t <= gs or s >= ge]
                outside_cons = union_len(out_c) / max(1, x["cons_len"]); cons_out.append(outside_cons)
            else:
                divs_o.append(divx)
            if x["genes"] and divx < a.young_div and cons_cov >= 0.8:
                cc = "young_unit_with_gene"
            elif x["genes"] and divx >= a.old_div and rel == "within_gene" and gcov < 0.8:
                cc = "paralog_homology"
            else:
                cc = "other_genic" if x["genes"] else "intergenic"
            x["copy_class"] = cc
            crow.append((f, x["cls"], x["chr"], x["s"], x["e"], x["e"] - x["s"], x["cons_len"],
                         round(cons_cov, 3), round(divx, 1), rel, ",".join(y[2] for y in x["genes"]) or "-",
                         round(gcov, 3), round(outside_cons, 3), cc))
        # domains of the overlapped genes
        host_cnt, te_genes, te_only, fused = collections.Counter(), 0, 0, 0
        for g in ov_genes:
            ds = gdom.get(g, set()); t = ds & te_dom; h = ds - te_dom
            for d in h: host_cnt[d] += 1
            if t: te_genes += 1
            if t and not h: te_only += 1
            if t and h: fused += 1
        ng = max(1, len(ov_genes))
        top = host_cnt.most_common(1)
        top_s = f"{top[0][0]}:{gdesc.get(top[0][0], '')} ({top[0][1]})" if top else "-"
        med_gl = statistics.median(gl) if gl else 0
        ratio = cons / med_gl if med_gl else float("nan")
        frac_in = n_gen / len(E); frac_cont = n_cont / max(1, n_gen); frac_span = n_span / max(1, n_gen)
        med_out = statistics.median(cons_out) if cons_out else 0
        med_gcov = statistics.median(cov_g) if cov_g else 0
        n_young = sum(x.get("copy_class") == "young_unit_with_gene" for x in E)
        n_par = sum(x.get("copy_class") == "paralog_homology" for x in E)
        # ---- call (ordered; first rule that fits)
        why = []
        if te_only / ng >= 0.5:
            call = "TE_ORF_gene"; why.append(f"{te_only}/{len(ov_genes)} genes carry only TE/viral domains")
        elif fused / ng >= 0.25:
            call = "TE_gene_fusion"; why.append(f"{fused}/{len(ov_genes)} genes carry TE/viral + host domains")
        elif n_young >= 2:
            call = "gene_in_mobile_unit"
            why.append(f"{n_young} near-identical full-length copies contain genes (capture or segmental "
                       f"duplication - check TIRs/TSDs); {n_par} old partial hits on paralogs")
        elif n_par / max(1, n_gen) >= 0.5:
            call = "paralog_homology"
            why.append(f"{n_par}/{n_gen} genic copies are old (>= {a.old_div:g}% div) partial hits inside genes; "
                       f"no young gene-carrying units")
        elif frac_cont >= 0.6 and 0.5 <= ratio <= 1.6 and med_gcov >= 0.6:
            call = "consensus_is_gene"
            why.append(f"{n_cont}/{n_gen} genic copies inside gene bounds, cover median {med_gcov:.0%} of gene; consensus {ratio:.1f}x gene")
        else:
            call = "ambiguous"
            why.append(f"{n_young} young units, {n_par}/{n_gen} paralog-like hits, contained {frac_cont:.0%}, "
                       f"consensus {ratio:.1f}x gene")
        rows.append((f, E[0]["cls"], len(E), bp, round(genic_frac, 3), int(cons), int(med_gl),
                     round(ratio, 2), round(frac_in, 2), n_cont, n_part, n_span, round(med_gcov, 2),
                     round(med_out, 2), round(statistics.median(divs_g), 1) if divs_g else "",
                     round(statistics.median(divs_o), 1) if divs_o else "",
                     n_young, n_par, len(ov_genes), te_genes, fused, top_s, call, "; ".join(why)))
    rows.sort(key=lambda r: -r[3])
    hdr = ["te_family", "class", "n_copies", "bp", "frac_bp_in_genes", "consensus_len", "median_gene_len",
           "consensus_to_gene", "frac_copies_genic", "n_within_gene", "n_extend_one_flank",
           "n_span_both_flanks", "median_gene_cov", "median_consensus_outside_gene",
           "median_div_genic_copies", "median_div_other_copies", "n_young_units_with_gene", "n_paralog_homology_hits",
           "n_genes", "n_genes_TE_domain", "n_genes_TE_plus_host_domain", "top_host_domain", "call", "evidence"]
    with open(a.out + "_families.tsv", "w") as o:
        o.write("\t".join(hdr) + "\n")
        for r in rows: o.write("\t".join(map(str, r)) + "\n")
    with open(a.out + "_copies.tsv", "w") as o:
        o.write("te_family\tclass\tchr\tstart\tend\tlen\tconsensus_len\tconsensus_cov\tdiv\t"
                "relation_to_gene\tgenes\tgene_cov\tconsensus_outside_gene\tcopy_class\n")
        for r in crow: o.write("\t".join(map(str, r)) + "\n")
    if a.filtered_bed:   # `tehood prepare`-style TE BED without paralog_homology copies
        n_drop = 0
        with open(a.filtered_bed, "w") as o:
            for x in sorted(els, key=lambda y: (y["chr"], y["s"])):
                if x.get("copy_class") == "paralog_homology":
                    n_drop += 1; continue
                for s_, t, _, _ in sorted(x["segs"]):
                    o.write(f"{x['chr']}\t{s_}\t{t}\t{x['chr']}:{x['id']}\t{x['cls']}/{x['fam']}\t+\n")
        print(f"[classify] wrote {a.filtered_bed}: removed {n_drop} paralog_homology copies "
              f"of {len(els)} TE copies", file=sys.stderr)
    cnt = collections.Counter(r[-2] for r in rows)
    bpc = collections.Counter(); [bpc.update({r[-2]: r[3]}) for r in rows]
    print(f"[classify] {len(rows)} genic TE families (>= {a.min_copies} copies, >= {a.min_genic:.0%} bp in genes)", file=sys.stderr)
    for k in ["gene_in_mobile_unit", "paralog_homology", "consensus_is_gene", "TE_gene_fusion", "TE_ORF_gene", "ambiguous"]:
        print(f"[classify]   {k:22s} {cnt[k]:4d} families  {bpc[k]:>9,} bp", file=sys.stderr)


if __name__ == "__main__":
    main()
