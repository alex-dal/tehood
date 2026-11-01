"""Interval arithmetic on sorted numpy arrays (no external tools needed).

All coordinates are 0-based, half-open [start, end), as in BED.
Intervals are handled per sequence; a "track" is a dict  seq -> (starts, ends)
of sorted, non-overlapping intervals (the result of `merge`).
"""
from __future__ import annotations

import numpy as np


def merge(starts, ends):
    """Merge overlapping / book-ended intervals. Returns sorted disjoint (starts, ends)."""
    starts = np.asarray(starts, dtype=np.int64); ends = np.asarray(ends, dtype=np.int64)
    if len(starts) == 0:
        return starts, ends
    o = np.lexsort((ends, starts)); s, e = starts[o], ends[o]
    run_end = np.maximum.accumulate(e)
    new = np.empty(len(s), dtype=bool); new[0] = True
    new[1:] = s[1:] > run_end[:-1]                     # gap -> new block (touching intervals are merged)
    idx = np.flatnonzero(new)
    return s[idx], np.maximum.reduceat(e, idx)


def build_track(df, seq="chr", start="start", end="end"):
    """dict seq -> merged (starts, ends) from a table of intervals."""
    out = {}
    for c, g in df.groupby(seq, sort=False):
        out[c] = merge(g[start].to_numpy(), g[end].to_numpy())
    return out


class Coverage:
    """Fast 'how many bp of [a, b) are covered by the track' for many queries."""

    def __init__(self, track):
        self.t = {}
        for c, (s, e) in track.items():
            cum = np.concatenate([[0], np.cumsum(e - s)])
            self.t[c] = (s, e, cum)

    def _upto(self, c, x):
        """covered bp in [0, x) on sequence c (x: array)."""
        if c not in self.t:
            return np.zeros(len(x), dtype=np.int64)
        s, e, cum = self.t[c]
        i = np.searchsorted(s, x, side="right")          # intervals starting before x
        full = cum[np.maximum(i - 1, 0)]                 # bp of intervals entirely before interval i-1
        part = np.where(i > 0, np.clip(x - s[np.maximum(i - 1, 0)], 0,
                                       e[np.maximum(i - 1, 0)] - s[np.maximum(i - 1, 0)]), 0)
        return np.where(i > 0, full + part, 0)

    def covered(self, seqs, a, b):
        """bp covered within [a, b) for arrays of queries (seqs: array of names)."""
        seqs = np.asarray(seqs); a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        out = np.zeros(len(a), dtype=np.int64)
        for c in np.unique(seqs):
            m = seqs == c
            out[m] = self._upto(c, b[m]) - self._upto(c, a[m])
        return out


def window_pairs(genes, tes, w):
    """All (gene_index, te_index) with the TE overlapping [gene_start - w, gene_end + w).

    genes, tes: DataFrames with chr/start/end. Equivalent to `bedtools window -w w`.
    """
    gi_all, ti_all = [], []
    tes_by = {c: g for c, g in tes.groupby("chr", sort=False)}
    for c, g in genes.groupby("chr", sort=False):
        if c not in tes_by:
            continue
        t = tes_by[c]
        o = np.argsort(t["start"].to_numpy(), kind="stable")
        ts = t["start"].to_numpy()[o]; te = t["end"].to_numpy()[o]; tidx = t.index.to_numpy()[o]
        maxlen = int((te - ts).max()) if len(ts) else 0
        lo = g["start"].to_numpy() - w; hi = g["end"].to_numpy() + w
        i0 = np.searchsorted(ts, lo - maxlen, side="left")
        i1 = np.searchsorted(ts, hi, side="left")        # ts < hi
        gidx = g.index.to_numpy()
        for k in range(len(gidx)):
            if i1[k] <= i0[k]:
                continue
            sl = slice(i0[k], i1[k])
            hit = te[sl] > lo[k]
            if hit.any():
                ti = tidx[sl][hit]
                gi_all.append(np.full(len(ti), gidx[k])); ti_all.append(ti)
    if not gi_all:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(gi_all), np.concatenate(ti_all)


def overlaps_any(query, track):
    """Boolean: does each query interval overlap the track by >= 1 bp?"""
    cov = Coverage(track)
    return cov.covered(query["chr"].to_numpy(), query["start"].to_numpy(), query["end"].to_numpy()) > 0
