"""Synthetic genome for testing tehood null models.

Realistic features: TEs only in intergenic space, a few large TE islands
(gap chosen with weight length^1.5), TE length clamped to the gap it occupies
(TEs create their own intergenic space). Two planted signals:
  IPR999998  30 genes, +1 TE inside each flanking gap      -> truly TE-associated
  IPR999999  60 genes with no TE within the window          -> truly TE-depleted
  IPR999997  25 genes next to a large TE island (gap > 5 kb) -> genes in TE islands
  IPR999996  30 genes whose largest nearby TE is annotated as
             6 fragments with different IDs (TE content unchanged)
                                                            -> should NOT be called;
             counting TEs is fooled by fragmentation, TE density is not
  IPR999995  30 genes with 2 extra DNA/Helitron TEs per flank -> TE-associated,
             and ONLY for the Helitron class (class-specific test)
  IPR999990 + IPR999991   A on 150 genes, B on 150 genes, 30 of them carry BOTH;
             only the A+B genes sit next to TE islands -> the PAIR is TE-associated
             and combination-specific (more than genes with only A or only B)
Background TEs are LTR/Gypsy (50%), LTR/Copia (30%) and DNA/DTM (20%).
All other domains (IPR1000xx) are assigned at random, plus 10 random domain
pairs co-assigned to 15 random genes each                  -> should be NS
usage: tehood simulate OUTDIR [--seed N]
"""
import bisect, collections, os, random, sys



def simulate(out, seed=7):
    random.seed(seed)

    os.makedirs(out, exist_ok=True); P = lambda f: os.path.join(out, f)
    chroms = {f"chr{i}": 1_000_000 for i in range(1, 5)}
    with open(P("genome.fna"), "w") as fa:
        for c, L in chroms.items():
            fa.write(f">{c}\n"); seq = "".join(random.choice("ACGT") for _ in range(L))
            fa.write("\n".join(seq[i:i + 80] for i in range(0, L, 80)) + "\n")
    genes, gaps, left_gap = [], [], {}
    with open(P("genomic.gff"), "w") as gff:
        gff.write("##gff-version 3\n"); gid = 0
        for c, L in chroms.items():
            pos, prev = 1000, 0
            while pos < L - 5000:
                glen = random.randint(800, 3000); s, e = pos, pos + glen; st = random.choice("+-"); gid += 1
                g, prot = f"FOO_{gid:05d}", f"XP_{gid:06d}.1"
                gff.write(f"{c}\tsim\tgene\t{s}\t{e}\t.\t{st}\t.\tID=gene-{g}\n"
                          f"{c}\tsim\tmRNA\t{s}\t{e}\t.\t{st}\t.\tID=rna-{g};Parent=gene-{g}\n"
                          f"{c}\tsim\tCDS\t{s}\t{e}\t.\t{st}\t0\tID=cds-{prot};Parent=rna-{g};protein_id={prot}\n")
                left_gap[g] = len(gaps); gaps.append((c, prev, s - 1)); prev = e
                genes.append((c, s - 1, e, prot, g))
                pos = e + (random.randint(5000, 20000) if random.random() < 0.04 else random.randint(300, 2500))
            gaps.append((c, prev, L))
    W = 1800
    cum, t = [], 0
    for c, s, e in gaps:
        t += (e - s) ** 1.5; cum.append(t)
    tes = []; te_cls = {}
    def bg_cls():
        r = random.random(); return "LTR/Gypsy" if r < 0.5 else "LTR/Copia" if r < 0.8 else "DNA/DTM"
    for _ in range(3500):
        c, s, e = gaps[bisect.bisect(cum, random.random() * t)]
        ln = min(random.randint(200, 4000), e - s - 2)
        if ln >= 50:
            st = random.randint(s + 1, e - ln); tes.append((c, st, st + ln))
    bychr = collections.defaultdict(list)
    for x in tes: bychr[x[0]].append(x)
    nte = {g: sum(1 for _, ts, te in bychr[c] if ts < e + W and te > s - W) for c, s, e, p, g in genes}
    enr = random.sample(genes, 30)
    for c, s, e, p, g in enr:                      # one extra TE in each flanking gap
        for gi in (left_gap[g], left_gap[g] + 1):
            gc, gs, ge = gaps[gi]
            if gc != c or ge - gs < 300: continue
            ln = random.randint(100, min(1000, ge - gs - 50)); st = random.randint(gs + 1, ge - ln)
            tes.append((c, st, st + ln))
    enr_ids = {x[4] for x in enr}
    dep = random.sample([g for g, v in nte.items() if v == 0 and g not in enr_ids], 60)
    used = enr_ids | set(dep)
    # island genes: a flanking gap > 5 kb (these gaps are TE-rich by construction)
    isl_c = [x for x in genes if x[4] not in used and
             max(gaps[left_gap[x[4]]][2] - gaps[left_gap[x[4]]][1],
                 gaps[left_gap[x[4]] + 1][2] - gaps[left_gap[x[4]] + 1][1]) > 5000]
    isl = random.sample(isl_c, min(25, len(isl_c))); used |= {x[4] for x in isl}
    # fragmented: split the largest TE near each chosen gene into 6 fragments
    by_gene = {}
    for x in genes:
        c, s, e = x[0], x[1], x[2]
        by_gene[x[4]] = [k for k, (tc, ts, te) in enumerate(tes) if tc == c and ts < e + W and te > s - W
                         and not (ts < e and te > s)]
    frag_c = [x for x in genes if x[4] not in used and by_gene[x[4]]]
    random.shuffle(frag_c)
    frag, split = [], set()
    for x in frag_c:
        k = max(by_gene[x[4]], key=lambda k: tes[k][2] - tes[k][1])
        c, ts, te = tes[k]
        if k in split or te - ts < 600: continue
        split.add(k); frag.append(x)
        cuts = sorted(random.sample(range(ts + 1, te), 5)); b = [ts] + cuts + [te]
        for i in range(6):
            tes.append((c, b[i], b[i + 1]))
        if len(frag) == 30: break
    tes = [t for k, t in enumerate(tes) if k not in split]
    used |= {x[4] for x in frag}
    def plant_flank(x, n_per_flank, ln_rng, cls):
        c, s, e, p, gname = x
        for gi in (left_gap[gname], left_gap[gname] + 1):
            gc, gs, ge = gaps[gi]
            for _ in range(n_per_flank):
                if gc != c or ge - gs < ln_rng[0] + 100: continue
                ln = random.randint(ln_rng[0], min(ln_rng[1], ge - gs - 50)); st = random.randint(gs + 1, ge - ln)
                tes.append((c, st, st + ln)); te_cls[len(tes) - 1] = cls
    free = [x for x in genes if x[4] not in used]; random.shuffle(free)
    hel = free[:30]                                   # class-specific plant
    for x in hel: plant_flank(x, 2, (300, 1200), "DNA/Helitron")
    used |= {x[4] for x in hel}
    # pair plant: the A+B genes are the remaining genes next to large TE islands
    ab = [x for x in isl_c if x[4] not in used][:30]; used |= {x[4] for x in ab}
    free = [x for x in genes if x[4] not in used]; random.shuffle(free)
    a_only, b_only = free[:120], free[120:240]
    prot_of = {g: p for c, s, e, p, g in genes}
    doms = [f"IPR{100000 + i}" for i in range(120)]
    with open(P("protein.faa.tsv"), "w") as ipr:
        row = lambda p, d, desc: f"{p}\tm\t300\tPfam\tPF{d[3:]}\tx\t1\t100\t1e-10\tT\t01-01-2024\t{d}\t{desc}\n"
        for c, s, e, p, g in genes:
            for d in random.sample(doms, random.choice([0, 1, 1, 2])): ipr.write(row(p, d, f"Random domain {d}"))
        for k in range(10):                           # random co-occurring pairs (null for --sets pair)
            dA, dB = random.sample(doms, 2)
            for x in random.sample(genes, 15):
                ipr.write(row(x[3], dA, f"Random domain {dA}")); ipr.write(row(x[3], dB, f"Random domain {dB}"))
        for x in enr: ipr.write(row(x[3], "IPR999998", "PLANTED TE-associated"))
        for g in dep: ipr.write(row(prot_of[g], "IPR999999", "PLANTED TE-depleted"))
        for x in isl: ipr.write(row(x[3], "IPR999997", "PLANTED TE-island genes"))
        for x in frag: ipr.write(row(x[3], "IPR999996", "PLANTED fragmented annotation (null)"))
        for x in hel: ipr.write(row(x[3], "IPR999995", "PLANTED Helitron-specific"))
        for x in ab + a_only: ipr.write(row(x[3], "IPR999990", "PLANTED pair component A"))
        for x in ab + b_only: ipr.write(row(x[3], "IPR999991", "PLANTED pair component B"))
    with open(P("genome.fna.out"), "w") as o:
        o.write("header\nheader\n\n")
        random.seed(99)
        for i, (c, s, e) in enumerate(tes, 1):
            cl = te_cls.get(i - 1) or bg_cls()
            o.write(f"  1000 10.0 1.0 1.0 {c} {s + 1} {e} (100) + fam{i % 40} {cl} 1 {e - s} (0) {i}\n")
    print(f"{len(genes)} genes, {len(tes)} TEs, planted: 30 TE-associated, 60 TE-depleted, "
          f"{len(isl)} TE-island, {len(frag)} fragmented-annotation, {len(hel)} Helitron, {len(ab)} A+B pair genes")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="tehood simulate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir"); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    simulate(a.outdir, a.seed)
