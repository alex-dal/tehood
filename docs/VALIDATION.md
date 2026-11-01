# Validation

## 1. Synthetic genomes (`pytest tests/`)

`tehood simulate` builds 4 Mb genomes with TEs only between genes, TE islands,
and planted truth:

| Domain | Planted | Expected |
|---|---|---|
| IPR999999 | 60 genes without TEs nearby | TE-depleted |
| IPR999997 | 25 genes next to TE islands | TE-associated (density) |
| IPR999996 | the largest TE near each of 30 genes split into 6 differently named fragments | **not** called by density |
| IPR999995 | 2 extra Helitrons per flank | Helitron-specific |
| IPR999990 & IPR999991 | A on 150 genes, B on 150; the 30 A+B genes next to TE islands | pair TE-associated and combination-specific |
| IPR1000xx | 120 random domains + 10 random co-occurring pairs | NS |

Pass criteria, per synthetic genome:

* ≤ 3/120 false positives per analysis;
* |median log2FE| < 0.25;
* planted depletion and islands found;
* fragmentation not called by density;
* the pair called and combination-specific, with ≤ 1 random pair significant;
* Helitron-only signal found, with ≤ 3 false positives per class.

Unit tests compare:

* interval arithmetic with brute force;
* BH with `scipy.stats.false_discovery_control`;
* the saddlepoint tail with Monte Carlo;
* random gene sets for uniformity;
* the window choice on contiguous and fragmented synthetic assemblies;
* the meta-analysis on 30 simulated result sets (null domains, a weak signal
  present in every genome, a strong signal in 6 of 30 genomes).

## 2. Meta-analysis on synthetic genomes

15 synthetic genomes, each with an extra domain on 5 TE-flanked genes (never
significant in any single genome):

| Analysis | Random domains significant | Weak planted domain |
|---|---|---|
| count | 1/100 | q = 0.005, `rescued` |
| density | 0/100 | NS (+1 small TE per flank is below island scale) |

## 3. Speed

A full `tehood run` (count, density at up to 5 windows, per-class tests,
pairs, architectures, compare, explain) takes about 1–2 min per fungal genome
(30–60 Mb, 10,000–15,000 genes) on one laptop core; `tehood meta` on 60 genomes
takes seconds. One set of 10,000 random gene orderings is shared by all set
sizes and TE classes (prefixes of a random ordering are uniform random
subsets), so the cost hardly grows with the number of domains.
