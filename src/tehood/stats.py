"""p-value machinery for the random-gene null (used by tehood.associate)."""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def bh(p):
    """Benjamini-Hochberg q-values (NaN-safe: NaNs stay NaN)."""
    p = np.asarray(p, dtype=float); q = np.full(len(p), np.nan); ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return q
    pv = p[ok]; o = np.argsort(pv)
    r = pv[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(r, 1)
    q[ok] = out
    return q


def counts_vs(obs, s):
    """# null values >= obs and <= obs (s sorted)."""
    lt = np.searchsorted(s, obs, side="left"); le = np.searchsorted(s, obs, side="right")
    return len(s) - lt, le


# ------------------------------------------------------------- random sets --
def sample_sets(rng, G, n, B):
    """B random sets of n distinct indices from range(G) -> (B, n) int array."""
    if n >= G:
        return np.tile(np.arange(G), (B, 1))[:, :n]
    if n * n < G:                         # few collisions: draw with replacement, redraw bad rows
        idx = rng.integers(0, G, size=(B, n))
        if n > 1:
            srt = np.sort(idx, axis=1)
            bad = np.flatnonzero((srt[:, 1:] == srt[:, :-1]).any(axis=1))
            for i in bad:
                idx[i] = rng.choice(G, n, replace=False)
        return idx
    out = np.empty((B, n), dtype=np.int64)  # many collisions: partial argsort of random keys
    ch = max(1, int(2e7 // G))
    for a in range(0, B, ch):
        b = min(B, a + ch)
        out[a:b] = np.argpartition(rng.random((b - a, G)), n - 1, axis=1)[:, :n]
    return out


class PrefixSampler:
    """Random gene sets of any size from ONE set of B random orderings of range(G).

    The first n entries of a uniformly random ordering are a uniformly random n-subset,
    so every set size uses prefixes of the same B orderings (exact null for each size;
    nulls of different sizes are correlated, which does not affect any single p-value).
    Orderings are only materialised up to nmax; larger sets fall back to sample_sets.
    """

    def __init__(self, rng, G, nmax, B, cap=3000):
        self.rng, self.G, self.B = rng, G, B
        self.nmax = int(min(max(nmax, 1), G, cap))
        P = np.empty((B, self.nmax), dtype=np.int32)
        ch = max(1, int(2e7 // max(G, 1)))
        for a in range(0, B, ch):
            b = min(B, a + ch)
            keys = rng.random((b - a, G))
            if self.nmax < G:
                part = np.argpartition(keys, self.nmax - 1, axis=1)[:, :self.nmax]
            else:
                part = np.tile(np.arange(G), (b - a, 1))
            k = np.take_along_axis(keys, part, axis=1)
            P[a:b] = np.take_along_axis(part, np.argsort(k, axis=1), axis=1)
        self.P = P

    def sets(self, n, B=None):
        B = self.B if B is None else B
        if n <= self.nmax and B <= self.B:
            return self.P[:B, :n].astype(np.int64)
        return sample_sets(self.rng, self.G, n, B)


# ------------------------------------------------------- count statistics --
def nb_tails(obs, mu, var):
    """Negative-binomial (Poisson if var <= mu) tails by moments: P(X >= obs), P(X <= obs), P(X = 0)."""
    obs = np.asarray(obs, float)
    if mu <= 0:
        return np.ones_like(obs), np.ones_like(obs), 1.0
    if var > mu:
        size = mu * mu / (var - mu); pr = size / (size + mu)
        pe = stats.nbinom.sf(obs - 1, size, pr); pd = stats.nbinom.cdf(obs, size, pr)
        p0 = stats.nbinom.pmf(0, size, pr)
    else:
        pe = stats.poisson.sf(obs - 1, mu); pd = stats.poisson.cdf(obs, mu); p0 = stats.poisson.pmf(0, mu)
    return np.maximum(pe, 1e-300), np.maximum(pd, 1e-300), p0


def count_pvals(obs, null, pmethod="auto"):
    """p-values for count statistics against a shared null vector."""
    s = np.sort(null); N = len(s); mu = s.mean(); var = s.var(ddof=1) if N > 1 else 0.0
    ge, le = counts_vs(obs, s); n0 = int((s == 0).sum())
    pe_emp = (1 + ge) / (N + 1); pd_emp = (1 + le) / (N + 1)
    pe_nb, pd_nb, p0_nb = nb_tails(obs, mu, var)
    if pmethod == "empirical":
        pe, pd, meth = pe_emp, pd_emp, np.full(len(obs), "empirical", dtype=object)
        mind = (1 + n0) / (N + 1)
    elif pmethod == "negbin":
        pe, pd, meth = pe_nb, pd_nb, np.full(len(obs), "negbin", dtype=object); mind = p0_nb
    else:
        ext = np.minimum(ge, le) < 10
        pe = np.where(ge < 10, pe_nb, pe_emp); pd = np.where(le < 10, pd_nb, pd_emp)
        meth = np.where(ext, "negbin_tail", "empirical").astype(object)
        mind = p0_nb if n0 < 10 else (1 + n0) / (N + 1)
    return dict(null_mean=mu, null_sd=np.sqrt(var), min_p_dep=mind, n_null_ge=ge, n_null_le=le, N=N,
                p_enr=pe, p_dep=pd, p_method=meth)


# ----------------------------------------------------- density statistics --
def sp_tail(x, n, dv, upper=True):
    """Saddlepoint (Lugannani-Rice) tail of the mean of n iid draws from dv."""
    dv = np.asarray(dv, float); mu = dv.mean(); mx, mn = dv.max(), dv.min()
    f_mx, f_mn = np.mean(dv == mx), np.mean(dv == mn); sd = max(dv.std(ddof=1), 1e-12)
    out = []
    for xi in np.atleast_1d(x):
        if upper and xi >= mx:
            out.append(0.0 if xi > mx else f_mx ** n); continue
        if not upper and xi <= mn:
            out.append(0.0 if xi < mn else f_mn ** n); continue
        if abs(xi - mu) < 1e-12:
            out.append(0.5); continue

        def kd(t):
            z = t * dv; w = np.exp(z - z.max()); return (w * dv).sum() / w.sum()
        s0 = 1 / sd
        if xi > mu:
            lo, hi = 0.0, s0
            while kd(hi) < xi: hi *= 2
        else:
            lo, hi = -s0, 0.0
            while kd(lo) > xi: lo *= 2
        t = optimize.brentq(lambda t: kd(t) - xi, lo, hi, xtol=1e-12)
        z = t * dv; m = z.max(); w = np.exp(z - m); W = w.sum()
        K = m + np.log(W / len(dv)); K1 = (w * dv).sum() / W; K2 = (w * dv * dv).sum() / W - K1 * K1
        r = np.sign(t) * np.sqrt(max(0.0, 2 * n * (t * xi - K))); u = t * np.sqrt(n * K2)
        pu = stats.norm.sf(r) + stats.norm.pdf(r) * (1 / u - 1 / r)
        p = pu if upper else 1 - pu
        out.append(min(1.0, max(p, f_mx ** n if upper else f_mn ** n)))
    return np.array(out)
