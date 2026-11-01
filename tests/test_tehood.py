"""tehood test suite.

  pytest tests/                         unit tests + 3 synthetic genomes (~2 min)
  TEHOOD_TEST_GENOMES=5 pytest tests/   more synthetic genomes

Synthetic genomes (tehood.synthetic) carry planted truth; the pass criteria are the
ones used to validate the method (README "Validation").
"""
import os

import numpy as np
import pandas as pd
import pytest

from tehood import associate, intervals, prepare, stats, synthetic, tables

N_GENOMES = int(os.environ.get("TEHOOD_TEST_GENOMES", "3"))


# ------------------------------------------------------------ unit: intervals --
def brute_cover(iv, a, b):
    cov = np.zeros(b - a, dtype=bool)
    for s, e in iv:
        lo, hi = max(s, a), min(e, b)
        if hi > lo: cov[lo - a:hi - a] = True
    return int(cov.sum())


def test_merge_and_coverage_match_brute_force():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5000, 300); e = s + rng.integers(1, 200, 300)
    ms, me = intervals.merge(s, e)
    assert np.all(ms[1:] > me[:-1])                       # disjoint, not touching
    cov = intervals.Coverage({"c": (ms, me)})
    for _ in range(200):
        a = int(rng.integers(0, 5000)); b = a + int(rng.integers(0, 800))
        assert cov.covered(np.array(["c"]), np.array([a]), np.array([b]))[0] == brute_cover(list(zip(s, e)), a, b)


def test_window_pairs_match_brute_force():
    rng = np.random.default_rng(1)
    g = pd.DataFrame({"chr": "c", "start": rng.integers(0, 10000, 50)}); g["end"] = g.start + rng.integers(100, 900, 50)
    t = pd.DataFrame({"chr": "c", "start": rng.integers(0, 10000, 200)}); t["end"] = t.start + rng.integers(10, 3000, 200)
    gi, ti = intervals.window_pairs(g, t, 500)
    got = set(zip(gi, ti))
    want = {(i, j) for i in g.index for j in t.index if t.start[j] < g.end[i] + 500 and t.end[j] > g.start[i] - 500}
    assert got == want


# ---------------------------------------------------------------- unit: stats --
def test_bh_matches_reference():
    from scipy.stats import false_discovery_control
    p = np.array([0.01, 0.04, 0.03, 0.2, np.nan])
    q = stats.bh(p)
    assert np.allclose(q[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.2]) and np.isnan(q[4])
    r = np.random.default_rng(5).random(500)
    assert np.allclose(stats.bh(r), false_discovery_control(r, method="bh"))


def test_sample_sets_are_distinct():
    rng = np.random.default_rng(2)
    for G, n in [(1000, 3), (1000, 200), (50, 49)]:
        x = stats.sample_sets(rng, G, n, 500)
        assert x.shape == (500, n) and all(len(set(r)) == n for r in x) and x.max() < G


def test_prefix_sampler_is_uniform():
    rng = np.random.default_rng(4)
    ps = stats.PrefixSampler(rng, 50, 10, 20000)
    for n in (1, 3, 10):
        x = ps.sets(n)
        assert all(len(set(r)) == n for r in x[:500])
        freq = np.bincount(x.ravel(), minlength=50) / x.size          # each gene equally likely
        assert np.abs(freq - 1 / 50).max() < 0.003


def test_saddlepoint_matches_monte_carlo():
    rng = np.random.default_rng(3)
    dv = np.where(rng.random(5000) < 0.6, 0, rng.beta(0.5, 3, 5000))     # skewed, zero-inflated
    n = 20; m = dv[rng.integers(0, len(dv), (200000, n))].mean(1)
    x = np.quantile(m, 0.999)
    assert abs(np.log10(stats.sp_tail([x], n, dv)[0]) - np.log10(np.mean(m >= x))) < 0.15


# ------------------------------------------------------- synthetic ground truth --
@pytest.fixture(scope="module", params=range(1, N_GENOMES + 1))
def genome(request, tmp_path_factory):
    seed = request.param; d = tmp_path_factory.mktemp(f"g{seed}")
    synthetic.simulate(str(d / "raw"), seed)
    prepare.main(["all", "--gff", str(d / "raw/genomic.gff"), "--ipr", str(d / "raw/protein.faa.tsv"),
                  "--fasta", str(d / "raw/genome.fna"), "--rm-out", str(d / "raw/genome.fna.out"),
                  "--outdir", str(d / "in")])
    tables.build_tables(str(d / "in/genes.bed"), str(d / "in/tes.bed"), str(d / "in/domains.tsv"),
                        str(d / "in/chrom.sizes"), str(d / "out"), "mean", [], ["5000", "10000"], 4, 5, "superfamily")
    o = lambda f: str(d / "out" / f)
    base = ["--domains", str(d / "in/domains.tsv"), "--no-plot", "--seed", str(seed)]
    associate.main(base + ["--stat", "count", "--pairs", o("obs_gene_TE.tsv"), "--genes", str(d / "in/genes.bed"), "--out", o("count")])
    associate.main(base + ["--stat", "density", "--density", o("gene_TE_density.tsv"), "--out", o("density")])
    associate.main(base + ["--stat", "density", "--density", o("gene_TE_density.tsv"), "--window", "10000",
                           "--sets", "pair", "--out", o("pair")])
    associate.main(base + ["--stat", "density", "--density", o("gene_TE_density.tsv"), "--window", "10000",
                           "--by-class", "--out", o("cls")])
    return d / "out"


def load(d, f):
    return pd.read_csv(d / f"{f}_results.tsv", sep="\t", dtype={"domain": str}).set_index("domain")


@pytest.mark.parametrize("analysis", ["count", "density_w5000", "density_w10000"])
def test_calibration_bias_and_depletion(genome, analysis):
    x = load(genome, analysis)
    rnd = x[x.index.str.startswith("IPR1000")]
    assert rnd.status.isin(["TE-associated", "TE-depleted"]).sum() <= 3          # false positives / 120
    assert abs(x.loc[x.tested, "log2FE"].median()) < 0.25                        # no bias
    assert x.loc["IPR999999", "status"] == "TE-depleted"                         # planted depletion


@pytest.mark.parametrize("analysis", ["density_w5000", "density_w10000"])
def test_density_finds_islands_and_ignores_fragmentation(genome, analysis):
    x = load(genome, analysis)
    assert x.loc["IPR999997", "status"] == "TE-associated"                       # TE-island genes
    assert x.loc["IPR999996", "status"] != "TE-associated"                       # fragmented annotation


def test_pairs(genome):
    p = load(genome, "pair")
    assert p.loc["IPR999990&IPR999991", "status"] == "TE-associated"
    assert bool(p.loc["IPR999990&IPR999991", "combination_specific"])
    rnd = p[~p.index.str.contains("IPR9999")]
    assert rnd.status.isin(["TE-associated", "TE-depleted"]).sum() <= 1


def test_classes(genome):
    s = pd.read_csv(genome / "cls_by_class_summary.tsv", sep="\t").set_index("domain")
    h = s.loc["IPR999995"]
    assert "Helitron" in h["classes_significant"] and bool(h["class_specific"])
    for f in genome.glob("cls_class-*_results.tsv"):
        x = pd.read_csv(f, sep="\t", dtype={"domain": str})
        assert (x.domain.str.startswith("IPR1000") & x.status.isin(["TE-associated", "TE-depleted"])).sum() <= 3


# ------------------------------------------------------------- unit: meta --
def fake_results(d, genome, rng, n_dom=400, weak=(), strong=()):
    """Per-genome result table: null domains with uniform p (count-like ties for sparse ones)."""
    rows = []
    for i in range(n_dom):
        dom = f"IPR{500000 + i}"
        if rng.random() < 0.4:                       # sparse: tie at 0 TEs, P(X=0) = 0.8
            pe, pdp = 1.0, 0.8 if rng.random() < 0.8 else 1.0
        else:
            u = rng.random(); pe, pdp = u, 1 - u
        rows.append((dom, pe, pdp))
    for dom, p in list(weak) + list(strong):
        rows.append((dom, p, 1.0))
    x = pd.DataFrame(rows, columns=["domain", "p_enr", "p_dep"])
    x["species"] = genome; x["description"] = "d"; x["n_genes"] = 5; x["observed"] = 1.0; x["null_mean"] = 1.0
    x["log2FE"] = 0.0; x["tested"] = True; x["likely_TE_gene"] = False
    x["status"] = np.where(x["p_enr"] < 1e-6, "TE-associated", "NS")
    x.to_csv(d / f"{genome}_count_results.tsv", sep="\t", index=False)


def test_meta_calibration_and_rescue(tmp_path):
    from tehood import meta
    rng = np.random.default_rng(11)
    for k in range(30):
        g = f"G{k}"; (tmp_path / g).mkdir()
        weak = [("IPR777777", 0.02)]                               # p = 0.02 everywhere: never significant alone
        rec = [("IPR777775", 0.001 if k < 6 else float(rng.random()))]  # strong in 6 of 30 genomes only
        fake_results(tmp_path / g, g, rng, weak=weak + rec)
    meta.main(["--dirs", *[str(p) for p in tmp_path.glob("G*")], "--out", str(tmp_path / "m"), "--no-plot"])
    m = pd.read_csv(tmp_path / "m_count_meta.tsv", sep="\t").set_index("domain")
    null = m[m.index.str.startswith("IPR5")]
    assert null.status.isin(["TE-associated", "TE-depleted"]).sum() <= 2          # 400 null domains
    assert abs(np.mean(null.p_meta < 0.05) - 0.05) < 0.04 or np.mean(null.p_meta < 0.05) < 0.05
    assert m.loc["IPR777777", "status"] == "TE-associated" and bool(m.loc["IPR777777", "rescued"])
    assert m.loc["IPR777775", "status"] == "TE-associated"
    assert m.loc["IPR777775", "units_enr_p05"] >= 6


# ------------------------------------------------------ unit: auto windows --
def test_auto_windows_follow_contiguity():
    rng = np.random.default_rng(7)
    def genome(contig_len, n=400):
        chroms = {f"c{i}": contig_len for i in range(n)}
        s = rng.integers(0, contig_len - 1000, n)
        g = pd.DataFrame({"chr": [f"c{i}" for i in range(n)], "start": s, "end": s + 1000})
        return g, chroms
    g, ch = genome(200_000)                                     # contiguous: all windows kept
    kept, t = tables.choose_windows(g, ch, 1500)
    assert kept == [1500, 2500, 5000, 10000, 20000]
    g, ch = genome(12_000)                                      # fragmented: large windows dropped
    kept, t = tables.choose_windows(g, ch, 1500)
    assert 1500 in kept and 20000 not in kept and 10000 not in kept
    assert t.set_index("window").loc[20000, "genes_full_flanks"] == 0
