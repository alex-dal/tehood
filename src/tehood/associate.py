"""tehood test - are the genes of a domain (or domain combination) TE-associated or
TE-depleted, compared with random genes of the same genome?

Statistic (--stat):
  count    distinct TEs within W bp of the set's genes        (obs_gene_TE.tsv)
  density  mean share of the genes' flanks covered by merged TE sequence
                                                              (gene_TE_density.tsv)
Gene sets (--sets):
  domain        one set per domain ID [default]
  pair          genes with BOTH A and B (strict subset of each; identical-gene pairs collapsed);
                significant pairs get a conditional test against genes with A or B
  architecture  genes with exactly the same set of domains
TE classes (--by-class): the test is repeated per TE class + a domain x class summary.

Null: random sets of the same number of genes from the universe (annotated genes by
default; TE-ORF genes = genes with only TE/viral domains (Table S1A) removed; genes
>= --max-body-te TE sequence removed if given).
p-values never 0: p = (1 + #null as extreme) / (N + 1); extreme tails from a negative
binomial (count) or a saddlepoint approximation / 1e6 sets (density); BH-FDR.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from importlib import resources

import numpy as np
import pandas as pd

from . import stats as S

TE_REGEX = re.compile("transpos|reverse transcriptase|integrase|retrotrans|retrovir|ribonuclease h|rnase ?h|"
                      "gag[- ]|helitron|mariner|tc1|piggybac|mutator|mule|dde |\\bhat\\b|ty1|ty3|copia|gypsy|"
                      "line-1|non-ltr|tigger|cenp-b|jerky|zswim|swim-type|duf4219|zinc knuckle|retropepsin|ldoc1",
                      re.I)


def log(msg):
    print(f"[tehood test] {msg}", file=sys.stderr)


def data_file(name):
    return str(resources.files("tehood").joinpath("data", name))


def read_ids(path):
    if not path or not os.path.exists(path):
        return set()
    out = set()
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                out.add(line.split("\t")[0].strip())
    return out


# ================================================================ gene sets ==
def build_sets(dom, mode="domain", min_genes=3, merge=False):
    d = dom[["gene", "domain"]].drop_duplicates()
    desc = dom.groupby("domain")["description"].agg(lambda x: next((v for v in x if v), "")).to_dict()
    if merge:   # IDs annotating exactly the same genes -> one group "ID1=ID2"
        key = d.groupby("domain")["gene"].agg(lambda g: ",".join(sorted(set(g))))
        gid = {}
        for _, ids in key.groupby(key):
            name = "=".join(sorted(ids.index)); gid.update({i: name for i in ids.index})
        d = d.assign(group=d["domain"].map(gid))
        gdesc = {}
        for i, g in gid.items():
            if desc.get(i) and desc[i] not in gdesc.get(g, []):
                gdesc.setdefault(g, []).append(desc[i])
        gdesc = {g: " | ".join(v) for g, v in gdesc.items()}
    else:
        d = d.assign(group=d["domain"]); gdesc = desc
    d = d[["gene", "group"]].drop_duplicates()
    ggenes = d.groupby("group")["gene"].agg(list).to_dict()
    meta = None
    if mode == "domain":
        tab = pd.DataFrame({"gene": d["gene"], "set": d["group"]})
    elif mode == "architecture":
        arch = d.groupby("gene")["group"].agg(lambda x: ";".join(sorted(set(x))))
        tab = pd.DataFrame({"gene": arch.index, "set": arch.values})
        n = tab["set"].value_counts(); tab = tab[tab["set"].isin(n.index[n >= min_genes])]
    else:   # pair
        rows = []
        for g, grp in d.groupby("gene")["group"]:
            x = sorted(set(grp))
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    rows.append((g, x[i], x[j]))
        if not rows:
            raise SystemExit("ERROR: no gene carries two domains")
        pr = pd.DataFrame(rows, columns=["gene", "A", "B"]); pr["set"] = pr["A"] + "&" + pr["B"]
        n = pr["set"].value_counts(); ng = {k: len(v) for k, v in ggenes.items()}
        pr = pr[pr["set"].isin(n.index[n >= min_genes])]
        nab = pr["set"].map(n)
        pr = pr[(nab < pr["A"].map(ng)) & (nab < pr["B"].map(ng))]
        # collapse pairs with identical gene sets; representative = smallest components
        key = pr.groupby("set")["gene"].agg(lambda g: ",".join(sorted(g)))
        info = pr[["set", "A", "B"]].drop_duplicates().assign(key=lambda x: x["set"].map(key))
        info["size"] = info["A"].map(ng) + info["B"].map(ng)
        info = info.sort_values(["key", "size", "set"])
        grp = info.groupby("key")["set"]
        rep = grp.first(); eqv = grp.agg(lambda x: ",".join(list(x)[1:]) or "-"); neq = grp.size() - 1
        log(f"pairs: {len(info)} distinct -> {len(rep)} after collapsing pairs with identical gene sets")
        pr = pr[pr["set"].isin(set(rep))]
        tab = pr[["gene", "set"]]
        meta = pr[["set", "A", "B"]].drop_duplicates().rename(columns={"A": "comp_A", "B": "comp_B"})
        meta["n_genes_A"] = meta["comp_A"].map(ng); meta["n_genes_B"] = meta["comp_B"].map(ng)
        k = meta["set"].map(key)
        meta["n_equivalent_pairs"] = k.map(neq).values; meta["equivalent_pairs"] = k.map(eqv).values
        meta = meta.set_index("set")

    def describe(sid):
        parts = re.split(r"[&;]", sid)
        return " + ".join(gdesc.get(p, "") for p in parts)
    tab = tab.drop_duplicates().reset_index(drop=True)
    sdesc = {s: describe(s) for s in tab["set"].unique()}
    log(f"gene sets: {mode}, {len(sdesc)} sets ({dom['domain'].nunique()} domain IDs -> {len(ggenes)} groups"
        f"{' after merging identical' if merge else ''})")
    return dict(tab=tab, desc=sdesc, meta=meta, group_genes=ggenes)


# ===================================================== per-gene statistics ==
class CountData:
    """distinct TEs near a gene set; vectorised over many random sets."""

    def __init__(self, pairs, universe):
        gi = {g: i for i, g in enumerate(universe)}
        p = pairs[pairs["gene"].isin(gi)]
        tes = pd.factorize(p["te"])[0]
        g = p["gene"].map(gi).to_numpy()
        o = np.lexsort((tes, g)); g, tes = g[o], tes[o]
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(g, minlength=len(universe)))])
        self.indices = tes.astype(np.int64); self.nte = int(tes.max()) + 1 if len(tes) else 1

    def stat_many(self, idx):
        """idx: (B, n) gene indices -> distinct TE count per row."""
        B, n = idx.shape; out = np.zeros(B, dtype=np.int64)
        lens_all = self.indptr[idx + 1] - self.indptr[idx]
        per_row = lens_all.sum(1); ch_rows = max(1, int(2e7 // max(1, per_row.mean() if B else 1)))
        for a in range(0, B, ch_rows):
            b = min(B, a + ch_rows); st = self.indptr[idx[a:b]].ravel(); ln = lens_all[a:b].ravel()
            tot = int(ln.sum())
            if tot == 0:
                continue
            rows = np.repeat(np.repeat(np.arange(b - a), n), ln)
            offs = np.repeat(st - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(tot)
            key = rows * self.nte + self.indices[offs]
            u = np.unique(key)
            out[a:b] = np.bincount(u // self.nte, minlength=b - a)
        return out

    def stat(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        if not len(idx):
            return 0
        parts = [self.indices[self.indptr[i]:self.indptr[i + 1]] for i in idx]
        return int(len(np.unique(np.concatenate(parts)))) if parts else 0


# ===================================================================== tests ==
def set_index(tab, universe):
    gi = {g: i for i, g in enumerate(universe)}
    t = tab[tab["gene"].isin(gi)]
    return {s: np.array(sorted({gi[g] for g in grp}), dtype=np.int64) for s, grp in t.groupby("set")["gene"]}


def test_count(sets, cd, G, rng, B0, pmethod, sampler=None):
    by_n = {}
    for s, idx in sets.items():
        by_n.setdefault(len(idx), []).append(s)
    rows = []
    for n in sorted(by_n):
        ids = by_n[n]; obs = np.array([cd.stat(sets[s]) for s in ids], float)
        draws = sampler.sets(n, B0) if sampler is not None else S.sample_sets(rng, G, n, B0)
        null = cd.stat_many(draws).astype(float)
        r = S.count_pvals(obs, null, pmethod)
        for k, s in enumerate(ids):
            rows.append(dict(set=s, observed=obs[k], null_mean=r["null_mean"], null_sd=r["null_sd"],
                             min_p_dep=r["min_p_dep"], n_null_ge=r["n_null_ge"][k], n_null_le=r["n_null_le"][k],
                             N=r["N"], p_enr=r["p_enr"][k], p_dep=r["p_dep"][k], p_method=r["p_method"][k]))
    log(f"count: {G} genes, {len(sets)} sets, {len(by_n)} set sizes, {B0} random gene sets each")
    return pd.DataFrame(rows)


def test_density(sets, dv, rng, B0, Bmax, sampler=None):
    G = len(dv); f_min = np.mean(dv == dv.min())
    by_n = {}
    for s, idx in sets.items():
        by_n.setdefault(len(idx), []).append(s)
    rows = []; n_sp = n_mc = 0
    for n in sorted(by_n):
        ids = by_n[n]; obs = np.array([dv[sets[s]].mean() for s in ids])
        if n == 1:
            srt = np.sort(dv); ge, le = S.counts_vs(obs, srt); N = G
            pe, pd_ = ge / N, le / N; meth = np.full(len(ids), "exact", dtype=object); mind = f_min; sdn = srt.std(ddof=1)
        else:
            draws = sampler.sets(n, B0) if sampler is not None else S.sample_sets(rng, G, n, B0)
            nv = dv[draws].mean(1); srt = np.sort(nv); ge, le = S.counts_vs(obs, srt); N = B0
            pe = (1 + ge) / (N + 1); pd_ = (1 + le) / (N + 1); meth = np.full(len(ids), "empirical", dtype=object)
            sdn = srt.std(ddof=1)
            ee, ed = ge < 10, le < 10
            if (ee | ed).any():
                if n >= 5:
                    if ee.any(): pe[ee] = S.sp_tail(obs[ee], n, dv, True)
                    if ed.any(): pd_[ed] = S.sp_tail(obs[ed], n, dv, False)
                    meth[ee | ed] = "saddlepoint"; n_sp += int((ee | ed).sum())
                else:
                    extra = int(Bmax) - B0
                    nv2 = np.concatenate([nv, dv[S.sample_sets(rng, G, n, extra)].mean(1)])
                    s2 = np.sort(nv2); g2, l2 = S.counts_vs(obs, s2); N2 = len(s2)
                    pe[ee] = (1 + g2[ee]) / (N2 + 1); pd_[ed] = (1 + l2[ed]) / (N2 + 1)
                    meth[ee | ed] = "empirical_1e6"; n_mc += int((ee | ed).sum())
            pe = np.maximum(pe, 1e-300); pd_ = np.maximum(pd_, 1e-300); mind = f_min ** n
        for k, s in enumerate(ids):
            rows.append(dict(set=s, observed=obs[k], null_mean=dv.mean(), null_sd=sdn, min_p_dep=mind,
                             n_null_ge=ge[k], n_null_le=le[k], N=N, p_enr=pe[k], p_dep=pd_[k], p_method=meth[k]))
    res = pd.DataFrame(rows)
    thr = np.quantile(dv, 0.9)
    res["frac_genes_TE_rich"] = [float(np.mean(dv[sets[s]] >= thr)) for s in res["set"]]
    log(f"density: {G} genes, {len(sets)} sets, {len(by_n)} set sizes, {B0} random sets each; "
        f"tail p for {n_sp} sets (saddlepoint) + {n_mc} (1e6 sets)")
    return res


def conditional(res, SETS, sets_idx, gi, stat_idx, rng, B, alpha):
    """significant pairs vs random sets drawn from genes with A or B."""
    meta = SETS["meta"]
    for c in ["stat_all_A", "stat_all_B", "cond_null_mean", "cond_p_enr", "cond_p_dep"]:
        res[c] = np.nan
    hits = np.flatnonzero(res["status"].isin(["TE-associated", "TE-depleted"]).to_numpy())
    for i in hits:
        s = res.at[i, "domain"]; m = meta.loc[s]
        ia = np.array([gi[g] for g in SETS["group_genes"][m["comp_A"]] if g in gi], dtype=np.int64)
        ib = np.array([gi[g] for g in SETS["group_genes"][m["comp_B"]] if g in gi], dtype=np.int64)
        res.at[i, "stat_all_A"] = stat_idx(ia); res.at[i, "stat_all_B"] = stat_idx(ib)
        U = np.union1d(ia, ib); ab = sets_idx[s]; n = len(ab)
        if n < 1 or len(U) <= n:
            continue
        o = stat_idx(ab)
        nv = np.array([stat_idx(U[k]) for k in S.sample_sets(rng, len(U), n, B)], float)
        res.at[i, "cond_null_mean"] = nv.mean()
        res.at[i, "cond_p_enr"] = (1 + (nv >= o).sum()) / (B + 1)
        res.at[i, "cond_p_dep"] = (1 + (nv <= o).sum()) / (B + 1)
    for c in ["comp_A", "comp_B", "n_genes_A", "n_genes_B", "n_equivalent_pairs", "equivalent_pairs"]:
        res[c] = res["domain"].map(meta[c])
    return res


# ==================================================================== finish ==
def finish(res, ctx, prefix, label, stat):
    a = ctx["args"]; SETS = ctx["SETS"]; dom = ctx["dom"]
    eps = 0.01 if stat == "density" else 0.5
    res = res.rename(columns={"set": "domain"}).copy()
    res.insert(1, "species", a.name)
    tab = SETS["tab"]; sg = tab.groupby("set")["gene"].agg(lambda g: sorted(set(g))).to_dict()
    res["n_genes"] = [len(sg.get(s, [])) for s in res["domain"]]
    res["description"] = res["domain"].map(SETS["desc"]).fillna("")
    res["log2FE"] = np.log2((res["observed"] + eps) / (res["null_mean"] + eps))
    res["z"] = np.where(res["null_sd"] > 0, (res["observed"] - res["null_mean"]) / res["null_sd"].replace(0, np.nan), np.nan)
    comps = [re.split(r"[&;=]", s) for s in res["domain"]]
    res["te_viral_domain"] = [any(c in ctx["te_ids"] for c in cs) for cs in comps]
    res["multicopy_domain"] = [any(c in ctx["mc_ids"] for c in cs) for cs in comps]
    g_te = set(dom.loc[dom["domain"].isin(ctx["te_ids"]), "gene"])
    res["frac_genes_with_TE_domain"] = [round(float(np.mean([g in g_te for g in sg.get(s, [])])) if sg.get(s) else np.nan, 3)
                                        for s in res["domain"]]
    res["te_machinery"] = res["te_viral_domain"] | res["description"].str.replace("_", " ").str.contains(TE_REGEX)
    if ctx["body"] is not None:
        b = ctx["body"]
        res["mean_body_TE_frac"] = [round(float(np.nanmean([b.get(g, np.nan) for g in sg.get(s, [])])), 4)
                                    if sg.get(s) else np.nan for s in res["domain"]]
    res["likely_TE_gene"] = res["te_machinery"] | (res["frac_genes_with_TE_domain"].fillna(0) >= 0.5)
    res["tested"] = (res["n_genes"] >= a.min_genes) if stat == "density" else (res["null_mean"] >= a.min_expected)
    tt = res["tested"].to_numpy()
    res["p"] = np.minimum(1, 2 * np.minimum(res["p_enr"], res["p_dep"]))
    for c, src in [("q", "p"), ("q_enr", "p_enr"), ("q_dep", "p_dep")]:
        q = np.full(len(res), np.nan); q[tt] = S.bh(res.loc[tt, src].to_numpy()); res[c] = q
    res["direction"] = np.where(res["observed"] > res["null_mean"], "more_TEs",
                                np.where(res["observed"] < res["null_mean"], "fewer_TEs", "as_expected"))
    res["depletion_detectable"] = tt & (res["min_p_dep"] <= a.alpha / max(1, tt.sum()))
    sig = res["q"] < a.alpha
    res["status"] = np.where(~tt, "not_tested", np.where(sig & (res["direction"] == "more_TEs"), "TE-associated",
                             np.where(sig & (res["direction"] == "fewer_TEs"), "TE-depleted", "NS")))
    res["statistic"] = label; res["null_model"] = "random_genes"; res["set_type"] = a.sets
    extra = []
    if a.sets == "pair" and ctx.get("cond"):
        c = ctx["cond"]
        res = conditional(res, SETS, c["sets_idx"], c["gi"], c["stat_idx"], ctx["rng"], a.cond_nperm, a.alpha)
        res["cond_log2FE"] = np.log2((res["observed"] + eps) / (res["cond_null_mean"] + eps))
        res["cond_p"] = np.minimum(1, 2 * np.minimum(res["cond_p_enr"], res["cond_p_dep"]))
        res["cond_q"] = S.bh(res["cond_p"].to_numpy())
        res["combination_specific"] = (res["cond_q"] < a.alpha) & (np.sign(res["cond_log2FE"]) == np.sign(res["log2FE"]))
        extra = ["comp_A", "comp_B", "n_genes_A", "n_genes_B", "n_equivalent_pairs", "equivalent_pairs",
                 "stat_all_A", "stat_all_B", "cond_null_mean", "cond_log2FE", "cond_p", "cond_q", "combination_specific"]
        log(f"[{label}] pairs: {int(res['cond_p'].notna().sum())} significant, "
            f"{int(res['combination_specific'].sum())} combination-specific (conditional q < {a.alpha})")
    cols = ["domain", "species", "description", "n_genes", "statistic", "observed", "null_mean", "null_sd",
            "log2FE", "z", "direction", "status", "q", "p", "p_enr", "p_dep", "q_enr", "q_dep",
            "likely_TE_gene", "te_viral_domain", "frac_genes_with_TE_domain", "te_machinery", "multicopy_domain"]
    cols += [c for c in ["mean_body_TE_frac"] if c in res] + ["tested", "depletion_detectable", "min_p_dep"]
    cols += [c for c in ["frac_genes_TE_rich"] if c in res] + extra
    cols += ["N", "n_null_ge", "n_null_le", "p_method", "null_model", "set_type"]
    res = res.assign(_a=-res["log2FE"].abs()).sort_values(["p", "_a"]).drop(columns="_a")[cols].reset_index(drop=True)
    t = res[res["tested"]]
    mid = 0.5 + (t["n_null_le"] - t["n_null_ge"]) / (2 * t["N"])
    log(f"[{label}] {len(t)}/{len(res)} sets tested; mean mid-rank in null {mid.mean():.2f} (0.5 = typical)")
    log(f"[{label}] TE-associated: {(res.status == 'TE-associated').sum()} "
        f"({((res.status == 'TE-associated') & res.likely_TE_gene).sum()} likely TE-encoded)   "
        f"TE-depleted: {(res.status == 'TE-depleted').sum()}   (q < {a.alpha}); "
        f"depletion detectable for {int(res.depletion_detectable.sum())}")
    w = lambda x, f: x.to_csv(prefix + f, sep="\t", index=False, na_rep="NA")
    w(res, "_results.tsv"); w(res[res.status == "TE-associated"], "_TE-associated.tsv")
    w(res[res.status == "TE-depleted"], "_TE-depleted.tsv")
    if a.plot:
        plot_diagnostics(t, prefix, f"{a.name} {label}")
    return res


def plot_diagnostics(t, prefix, title):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].hist(t["p_enr"], bins=50, color="0.7"); ax[0].set_title("p (enrichment), tested")
    ax[1].hist(t["p_dep"], bins=50, color="0.7"); ax[1].set_title("p (depletion), tested")
    col = t["status"].map({"TE-associated": "darkred", "TE-depleted": "seagreen"}).fillna("0.6")
    ax[2].scatter(np.log2(t["n_genes"]), t["log2FE"], s=4, c=col); ax[2].axhline(0, ls="--", c="k", lw=0.5)
    ax[2].set_xlabel("log2(genes in set)"); ax[2].set_ylabel("log2 fold enrichment"); ax[2].set_title(title, fontsize=8)
    fig.tight_layout(); fig.savefig(prefix + "_diagnostics.pdf"); plt.close(fig)


def class_summary(overall, by, prefix, alpha):
    keep = set(overall.loc[overall.status.isin(["TE-associated", "TE-depleted"]), "domain"])
    for x in by.values():
        keep |= set(x.loc[x.status.isin(["TE-associated", "TE-depleted"]), "domain"])
    if not keep:
        log("by-class: no significant set in any class"); return
    s = overall[overall.domain.isin(keep)][["domain", "description", "n_genes", "log2FE", "q", "status"]].copy()
    s.columns = ["domain", "description", "n_genes", "log2FE_allTE", "q_allTE", "status_allTE"]
    for k, x in by.items():
        x = x.set_index("domain")
        s[f"log2FE_{k}"] = s.domain.map(x["log2FE"]).round(2); s[f"q_{k}"] = s.domain.map(x["q"])
        s[f"status_{k}"] = s.domain.map(x["status"])
    st = s[[f"status_{k}" for k in by]].to_numpy(); qq = s[[f"q_{k}" for k in by]].fillna(1).to_numpy()
    sigm = np.isin(st, ["TE-associated", "TE-depleted"])
    names = np.array(list(by))
    s["classes_significant"] = [",".join(names[r]) or "-" for r in sigm]
    s["n_classes_significant"] = sigm.sum(1)
    s["dominant_class"] = names[qq.argmin(1)]
    s["class_specific"] = s["n_classes_significant"] == 1
    s = s.assign(_o=s.status_allTE == "NS").sort_values(["_o", "q_allTE"]).drop(columns="_o")
    s.to_csv(prefix + "_by_class_summary.tsv", sep="\t", index=False, na_rep="NA")
    log(f"by-class: {len(s)} sets significant overall or in >= 1 class; {int(s.class_specific.sum())} in exactly one "
        f"class -> {prefix}_by_class_summary.tsv")


# ====================================================================== main ==
def safe(x):
    return re.sub(r"[^A-Za-z0-9._-]", "_", x)


def run(a):
    rng = np.random.default_rng(a.seed)
    dom = pd.read_csv(a.domains, sep="\t", header=None, dtype=str, quoting=3, comment=None).fillna("")
    dom = dom.iloc[:, :3] if dom.shape[1] >= 3 else dom.assign(d="")
    dom.columns = ["gene", "domain", "description"]
    dom["gene"] = dom["gene"].str.strip(); dom["domain"] = dom["domain"].str.strip()
    dom = dom[(dom.gene != "") & (dom.domain != "")].drop_duplicates()
    te_ids = read_ids(a.te_domains or data_file("te_viral_domains_S1A.tsv"))
    mc_ids = read_ids(a.multicopy_domains or data_file("multicopy_domains_S1B.tsv"))
    body = None; drop = set()
    if a.body_te:
        bt = pd.read_csv(a.body_te, sep="\t", dtype={"gene": str}); body = dict(zip(bt.gene.str.strip(), bt.body_te_frac))
    if a.max_body_te is not None:
        if body is None: raise SystemExit("--max-body-te needs --body-te gene_body_TE.tsv")
        drop = {g for g, f in body.items() if f >= a.max_body_te}
        log(f"excluding {len(drop)} genes whose body is >= {a.max_body_te} TE sequence")
    te_orf = set()
    if te_ids and not a.keep_te_orfs:
        per = dom.groupby("gene")["domain"].agg(lambda d: all(x in te_ids for x in d))
        te_orf = set(per.index[per]); n_fus = len(set(dom.gene[dom.domain.isin(te_ids)]) - te_orf)
        log(f"removing {len(te_orf)} TE-ORF genes (only TE/viral domains, Table S1A); keeping {n_fus} TE+host fusion candidates")
    dom_t = dom[~dom.gene.isin(drop | te_orf)]
    merge = a.merge_identical if a.merge_identical is not None else (a.sets != "domain")
    SETS = build_sets(dom_t, a.sets, a.min_set_genes, merge)
    ctx = dict(args=a, SETS=SETS, dom=dom_t, te_ids=te_ids, mc_ids=mc_ids, body=body, rng=rng)

    genes_all = None
    if a.genes:
        genes_all = [l.split("\t")[3].strip() for l in open(a.genes) if l.strip() and not l.startswith("#")]

    def universe_from(candidates):
        u = [g for g in candidates if g not in drop and g not in te_orf]
        if a.universe == "annotated":
            ann = set(dom_t.gene); u = [g for g in u if g in ann]
        return list(dict.fromkeys(u))

    if a.stat == "count":
        if not (a.pairs and a.genes): raise SystemExit("--stat count needs --pairs and --genes")
        pr = pd.read_csv(a.pairs, sep="\t", header=None, dtype=str, quoting=3)
        pr.columns = ["gene", "te", "te_class"][:pr.shape[1]]
        if "te_class" not in pr: pr["te_class"] = "."
        universe = universe_from(genes_all); gi = {g: i for i, g in enumerate(universe)}
        sets_idx = set_index(SETS["tab"], universe)
        sampler = S.PrefixSampler(rng, len(universe), max(map(len, sets_idx.values()), default=1), a.nperm)

        def one(prs, prefix, label):
            cd = CountData(prs, universe)
            ctx["cond"] = dict(sets_idx=sets_idx, gi=gi, stat_idx=cd.stat)
            return finish(test_count(sets_idx, cd, len(universe), rng, a.nperm, a.pmethod, sampler),
                          ctx, prefix, label, "count")
        overall = one(pr, a.out, "count")
        if a.by_class:
            u = pr[["te", "te_class"]].drop_duplicates()["te_class"].value_counts()
            cls = [c for c in u.index if c != "." and u[c] >= a.min_class_te]
            if not cls: raise SystemExit("--by-class: no TE classes in the pairs file (tehood tables --level)")
            by = {}
            for k in sorted(cls):
                log(f"class {k}: {u[k]} TEs near genes")
                by[safe(k)] = one(pr[pr.te_class == k], f"{a.out}_class-{safe(k)}", f"count_{k}")
            class_summary(overall, by, a.out, a.alpha)
    else:
        if not a.density: raise SystemExit("--stat density needs --density gene_TE_density.tsv")
        d = pd.read_csv(a.density, sep="\t", dtype={"gene": str}, na_values="NA")
        d["gene"] = d["gene"].str.strip(); d = d[d.density.notna()]
        if genes_all is not None: d = d[d.gene.isin(set(genes_all))]
        wins = sorted(d.window.unique())
        if a.window: wins = [w for w in wins if w == a.window]
        if not wins: raise SystemExit(f"window {a.window} not in {a.density}")
        dc = None
        if a.by_class:
            fc = a.density_by_class or os.path.join(os.path.dirname(a.density) or ".", "gene_TE_density_by_class.tsv")
            if not os.path.exists(fc): raise SystemExit(f"--by-class needs {fc} (tehood tables --level)")
            dc = pd.read_csv(fc, sep="\t", dtype={"gene": str}, na_values="NA")
        for W in wins:
            x = d[d.window == W]; universe = universe_from(x.gene.tolist())
            dmap = dict(zip(x.gene, x.density)); dv = np.array([dmap[g] for g in universe])
            gi = {g: i for i, g in enumerate(universe)}; sets_idx = set_index(SETS["tab"], universe)
            pf = f"{a.out}_w{W}" if len(wins) > 1 else a.out
            log(f"density window {W} bp: mean gene-flank TE density {dv.mean():.3f}")

            sampler = S.PrefixSampler(rng, len(universe), max(map(len, sets_idx.values()), default=1), a.nperm)

            def one(vec, prefix, label):
                ctx["cond"] = dict(sets_idx=sets_idx, gi=gi, stat_idx=lambda i, v=vec: float(v[i].mean()) if len(i) else np.nan)
                return finish(test_density(sets_idx, vec, rng, a.nperm, a.max_draws, sampler),
                              ctx, prefix, label, "density")
            overall = one(dv, pf, f"density_w{W}")
            if dc is not None:
                by = {}
                for k in sorted(dc.te_class.unique()):
                    y = dc[(dc.window == W) & (dc.te_class == k)]; m = dict(zip(y.gene, y.density))
                    vk = np.array([m.get(g, 0.0) if pd.notna(m.get(g, 0.0)) else 0.0 for g in universe])
                    if (vk > 0).sum() < a.min_class_te: continue
                    log(f"class {k}: mean flank density {vk.mean():.4f}")
                    by[safe(k)] = one(vk, f"{pf}_class-{safe(k)}", f"density_w{W}_{k}")
                class_summary(overall, by, pf, a.alpha)


def parser():
    ap = argparse.ArgumentParser(prog="tehood test", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stat", choices=["count", "density"], default="count")
    ap.add_argument("--pairs", help="obs_gene_TE.tsv (count)")
    ap.add_argument("--genes", help="genes BED (universe; required for count)")
    ap.add_argument("--density", help="gene_TE_density.tsv (density)")
    ap.add_argument("--density-by-class", help="gene_TE_density_by_class.tsv [next to --density]")
    ap.add_argument("--domains", required=True)
    ap.add_argument("--name", default="sample"); ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--window", type=int, help="density: only this window")
    ap.add_argument("--sets", choices=["domain", "pair", "architecture"], default="domain")
    ap.add_argument("--min-set-genes", type=int, default=3)
    ap.add_argument("--merge-identical", type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
    ap.add_argument("--cond-nperm", type=int, default=2000)
    ap.add_argument("--by-class", action="store_true")
    ap.add_argument("--min-class-te", type=int, default=20)
    ap.add_argument("--universe", choices=["annotated", "all"], default="annotated")
    ap.add_argument("--body-te"); ap.add_argument("--max-body-te", type=float)
    ap.add_argument("--keep-te-orfs", action="store_true")
    ap.add_argument("--te-domains"); ap.add_argument("--multicopy-domains")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-expected", type=float, default=0.25, help="count: min null mean to be tested")
    ap.add_argument("--min-genes", type=int, default=2, help="density: min genes to be tested")
    ap.add_argument("--nperm", type=int, default=10000); ap.add_argument("--max-draws", type=float, default=1e6)
    ap.add_argument("--pmethod", choices=["auto", "empirical", "negbin"], default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    return ap


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
