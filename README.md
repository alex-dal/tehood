# TEhood

**Which protein domains live in transposable-element neighbourhoods?**

For every protein domain (InterPro, Pfam, or custom gene sets such as candidate
effectors), domain pair or domain architecture, TEhood asks whether the genes
carrying it have more TE sequence nearby (**TE-associated**) or less
(**TE-depleted**) than random genes of the same genome. It can ask this per TE
class (LTR/Gypsy, DNA/DTM, Helitron, …) and combine the evidence across many
genome assemblies.

* Pure Python (numpy, scipy, pandas); runs on a laptop.
* Starts from standard annotation files: GFF3/GTF, InterProScan TSV,
  RepeatMasker `.out` or EDTA GFF3, genome FASTA.
* Two statistics: TE **count** near genes and TE **density** in gene flanks.
* Exact random-gene tests in both directions, with p-values.
* Triage of TE copies that sit inside genes (annotation artefact, gene capture, TE–gene fusion).
* Meta-analysis across multiple genomes, to recover consistent but weak signals. Weak signals often arise in TE-poor genomes and fragmented genome assemblies; meta-analysing multiple assemblies can recover weak signals shared across genomes.

## Contents

1. [Installation](#1-installation)
2. [Quick start (5 minutes, no data needed)](#2-quick-start)
3. [Inputs](#3-inputs)
4. [Usage](#4-usage)
5. [Method](#5-method)
6. [Domain combinations and TE classes](#6-domain-combinations-and-te-classes)
7. [TE copies inside genes](#7-te-copies-inside-genes-artefact-capture-or-fusion)
8. [Many genomes: `tehood meta`](#8-many-genomes-tehood-meta)
9. [Outputs](#9-outputs)
10. [Validation](#10-validation)
11. [Citation and licence](#11-citation-and-licence)

## 1. Installation

Python ≥ 3.9.

```bash
git clone https://github.com/alex-dal/tehood.git
cd tehood
python3 -m venv .venv && source .venv/bin/activate
pip install ".[plot]"          # matplotlib is optional: diagnostic PDFs
tehood --version
```

## 2. Quick start

Simulate a small genome with known answers and analyse it (about 1–2 minutes on a laptop):

```bash
tehood simulate --seed 1 demo/raw
tehood run --gff demo/raw/genomic.gff --ipr demo/raw/protein.faa.tsv \
           --fasta demo/raw/genome.fna --te demo/raw/genome.fna.out \
           --name demo --outdir demo/results
head demo/results/demo_TE-associated.tsv demo/results/demo_TE-depleted.tsv
```

The simulated genome contains planted domains; `supported_by` shows which
analyses call each one:

| Domain | Planted | Expected call |
|---|---|---|
| `IPR999997` | genes in TE islands | TE-associated, count and all density windows |
| `IPR999999` | genes far from TEs | TE-depleted, all analyses |
| `IPR999990`, `IPR999991` | domain pair whose shared genes sit in TE islands | TE-associated by density; the pair itself in `demo_density_pair_results.tsv` |
| `IPR999998` | one extra small TE in each flank | TE-associated by count only |
| `IPR999995` | extra Helitrons | TE-associated by count; Helitron-specific in `demo_density_classes_by_class_summary.tsv` |
| `IPR999996` | a nearby TE annotated as 6 fragments | called by count only: counting is fooled by fragmented annotations, density is not |
| `IPR1000xx` | random | NS (an occasional call in a single analysis is expected at 5% FDR) |

## 3. Inputs

| Input | Format | Typical source |
|---|---|---|
| Genome | FASTA (gzip OK) | assembly |
| Gene models | GFF3 (NCBI, Ensembl, JGI, funannotate, BRAKER) or GTF (JGI, AUGUSTUS) | annotation pipeline |
| Protein domains | InterProScan TSV (`-f TSV -iprlookup`) on the proteins of the **same** annotation | InterProScan ≥ 5 |
| TEs | RepeatMasker `.out`, or EDTA `*.EDTA.TEanno.gff3` | EDTA ≥ 2, or RepeatModeler2 + RepeatMasker |
| Custom gene sets (optional) | `gene<TAB>set_id[<TAB>description]` (`--extra`) | e.g. effector predictions |

TE annotation, either way:

* **EDTA:** `EDTA.pl --genome g.fna --sensitive 1 --anno 1`, then use the
  RepeatMasker `.out` in `*.EDTA.anno/` (or run
  `RepeatMasker -lib g.fna.mod.EDTA.TElib.fa g.fna`).
* **RepeatModeler2:** `BuildDatabase -name g g.fna`,
  `RepeatModeler -database g -LTRStruct`, `RepeatMasker -lib g-families.fa g.fna`.

The RepeatMasker `.out` is preferred: it records which part of the TE consensus
each copy matches, which `tehood classify` uses. Simple repeats, low complexity,
satellites, RNAs, ARTEFACT, Unspecified and Unknown are excluded
(`tehood prepare tes --exclude-classes`).

**Prepared inputs** can be given directly instead (see `tehood run -h`):

| File | Columns (tab-separated, no header) |
|---|---|
| genes BED | `chr start end gene_id . strand` |
| TE BED | `chr start end te_id class/family strand` (or class in column 4 and id in column 5: `--id-col 5`) |
| domains | `gene_id domain_id [description]` |
| chromosome sizes | `chr length` |

## 4. Usage

```bash
# one genome, raw annotations
tehood run --gff g.gff --ipr g.tsv --fasta g.fna --te g.out --name sp1 --outdir results/sp1

# one genome, prepared inputs
tehood run --genes-bed genes.bed --tes-bed tes.bed --domains domains.tsv --chroms chrom.sizes \
           --rm-out g.out --name sp1 --outdir results/sp1

# many genomes: samples.tsv = sample, gff, ipr, fasta, te [, te_format [, window]]
tehood run --samples samples.tsv --outdir results --jobs 4

# combine genomes
tehood meta --dirs results/* --empirical-null --out meta/all
```

On a cluster, `examples/run_slurm.sh` runs one genome per array task.

Each step is also a command of its own (`tehood <command> -h`):

| Command | Does |
|---|---|
| `tehood prepare` | GFF/GTF, InterProScan, RepeatMasker/EDTA, FASTA → `genes.bed`, `domains.tsv`, `tes.bed`, `chrom.sizes`; `tehood prepare window` gives the mean/median intergenic distance of one or several genomes |
| `tehood classify` | TE copies inside genes: removes annotation artefacts, reports gene-capture candidates (section 7) |
| `tehood tables` | Per-gene TE counts, flank TE density, per-class density, gene-body TE, TE family QC |
| `tehood test` | Random-gene tests per domain, pair or architecture, overall or per TE class |
| `tehood compare` | Several analyses of one genome side by side: TE-associated and TE-depleted sets with their support |
| `tehood explain` | The TE families in each hit's flanks, and the self-family share |
| `tehood meta` | Combine results across genomes (section 8) |
| `tehood simulate` | Synthetic genome with planted truth |

Main `tehood run` options:

| Option | Default | Meaning |
|---|---|---|
| `--window` | mean | count window: bp, or `mean` / `median` intergenic distance of the genome |
| `--density-windows` | w,auto | density flank sizes: bp, `w` (= count window), `auto` (section 5) |
| `--auto-min-full` | 0.6 | `auto`: keep a window if ≥ 60% of genes have both flanks complete |
| `--extra-window` | auto | density window for class and set analyses (`auto` = largest density window ≤ 10 kb) |
| `--level` | superfamily | TE class level: `order`, `superfamily`, or `''` to skip |
| `--sets` | pair,architecture | extra gene-set analyses (`''` to skip) |
| `--max-body-te` | 0.5 | do not test genes whose body is ≥ 50% TE sequence |
| `--nperm` | 10000 | random gene sets per set size |
| `--seed` | 1 | random seed |

To compare genomes of one group, use the same count window for all of them, e.g. the
mean intergenic distance pooled over the group: `tehood prepare window results/*/inputs/genes.bed`.

## 5. Method

**Statistics.**

| `--stat` | Measures | Best for | Weak spot |
|---|---|---|---|
| `count` | distinct TEs within *W* bp of the set's genes (TEs overlapping genes ignored) | several small TEs near genes | inflated by fragmented TE annotations |
| `density` | mean share of the genes' flanks (*W* bp each side, gene body excluded) covered by **merged** TE sequence | gene families in TE-rich islands; robust to fragmentation | needs island-scale windows |

**Null model.** Random sets of the same number of genes are drawn from the
same genome. TE positions stay as observed, so nothing is assumed about where
TEs "should" insert, and assembly structure is shared by observed and random
genes. TE positions are never shuffled: shuffled TEs ignore insertion
preferences and the gaps TEs create themselves, and such nulls were
miscalibrated in simulations.

**Background (universe).** Domain-annotated genes, minus TE-ORF genes (only
TE/viral domains; bundled list `data/te_viral_domains_S1A.tsv`) and genes whose
body is ≥ `--max-body-te` TE sequence. Unannotated gene models are often
TE-derived and would make every domain look depleted.

**p-values (both directions, never 0).**

* Empirical p = (1 + number of random sets as extreme) / (N + 1), with 10,000 sets per set size.
* Extreme tails: negative-binomial tail for count; saddlepoint
  (Lugannani–Rice) tail of the mean for density with ≥ 5 genes (agrees with
  10⁶-draw Monte Carlo to p ≈ 10⁻⁵); 10⁶ random sets for density with < 5 genes.
* Two-sided p; Benjamini–Hochberg FDR over the tested sets.
* Not tested: count sets with null mean < 0.25 TEs; density sets with < 2 genes.
* `min_p_dep` is the smallest depletion p a set can reach and
  `depletion_detectable` whether it can pass FDR: for other sets "NS" means
  "no information" rather than "not depleted".

**Density windows adapt to the assembly (`auto`).** Flanks are clipped at
contig ends, so in a fragmented assembly a "20 kb" flank is mostly missing
sequence: noisy, and not comparable with 20 kb in a contiguous genome. `auto`
tries 2.5, 5, 10 and 20 kb and keeps a window only if ≥ 60% of genes have both
flanks complete (`window_choice.tsv`). For example, a contig N50 of ~400 kb keeps
all four windows, ~100 kb keeps up to 10 kb and ~25 kb keeps only 2.5 kb.
Analyses are named by window (`density_w5000`), so `tehood meta` combines each
window over the genomes that support it.

## 6. Domain combinations and TE classes

* **`--sets pair`:** genes carrying both A and B. A pair is tested only if its
  genes are a strict subset of A's and of B's; pairs with identical genes are
  collapsed (`n_equivalent_pairs`). Each significant pair gets a conditional
  test against genes with A or B: `combination_specific` = TRUE means the
  combination is more extreme than its component families.
* **`--sets architecture`:** genes with exactly the same set of domains.
* For both, InterPro IDs that annotate identical genes are merged (`ID1=ID2`).
* **`--by-class`:** the test is repeated for each TE class with ≥ 20 copies;
  `*_by_class_summary.tsv` lists the significant classes, the `dominant_class`
  and whether the signal is `class_specific`. A class-specific signal can be
  invisible when all TEs are pooled.

## 7. TE copies inside genes: artefact, capture or fusion?

| Explanation | Signature |
|---|---|
| Host gene in the TE library (artefact) | old (≥ 15% divergence) partial RepeatMasker hits inside paralogs |
| Gene capture by a TE (Pack-MULE, Helitron, Starship) or segmental duplication | near-identical (< 5%) full-length copies, each containing a gene |
| TE–gene fusion | genes with TE/viral **and** host domains |
| TE ORF annotated as a gene | genes with only TE/viral domains |

* `tehood classify` (run by `tehood run` when a RepeatMasker `.out` is given)
  removes only the artefact copies and reports capture candidates
  (`gene_in_mobile_unit`) in `genic_TE_families.tsv`.
* `tehood explain` gives each hit's **self-family share**: the share of the
  flanking TE sequence from families that also sit inside the hit's own genes.
  Above ~0.5 the association is circular and needs sequence checks (TIRs/TSDs
  at copy ends, transposase in the consensus, EDTA intact annotation, synteny in
  related genomes).

## 8. Many genomes: `tehood meta`

TE-poor genomes and fragmented genome assemblies can give fewer FDR-significant domains on their own.
`tehood meta` combines each analysis across multiple genomes:

```bash
tehood meta --dirs results/* --empirical-null --out meta/all
tehood meta --dirs results/* --clades clades.tsv --empirical-null --out meta/all_clades   # genome<TAB>clade
```

* **Consistency (Stouffer):** each genome contributes z from its one-sided
  mid-p, (1 + p_enr − p_dep)/2. Genomes without information give z ≈ 0;
  genomes lacking the domain are skipped.
* **Recurrence:** number of genomes with p ≤ 0.05 in one direction against
  Binomial(n, 0.05): signals restricted to some genomes (e.g. a clade).
* `p_meta` = Bonferroni over both tests and both directions; BH → `q_meta`,
  `status`. `rescued` = significant across genomes but in no single genome.
  `p_stouffer_drop1` leaves out the strongest genome.
* **Related genomes are not independent:** orthologous genes can inherit their
  flanking TEs. Check `lambda` in the log and in `PREFIX_summary.tsv` (≈ 1
  expected). Above ~1.3, use `--empirical-null` (Z rescaled by the spread of
  all domains) and/or `--clades` (genomes combined within each clade first;
  clades are the units).

## 9. Outputs

Per genome (`--outdir`):

| File | Content |
|---|---|
| `NAME_TE-associated.tsv`, `NAME_TE-depleted.tsv` | Hits from count and each density window, with `n_support` and `supported_by` |
| `NAME_<analysis>_results.tsv` | All sets: observed, null mean, log2FE, z, p, q, status, flags (`likely_TE_gene`, `te_viral_domain`, `frac_genes_with_TE_domain`, `multicopy_domain`, `mean_body_TE_frac`), `depletion_detectable` |
| `NAME_*_hit_TE_families.tsv` | TE families behind each hit, self-family share |
| `NAME_density_classes_by_class_summary.tsv`, `NAME_count_classes_by_class_summary.tsv` | Domain × TE class |
| `NAME_density_pair_results.tsv`, `NAME_count_pair_results.tsv`, `NAME_density_architecture_results.tsv` | Combinations |
| `genic_TE_families.tsv`, `te_family_QC.tsv` | TE copies in genes, TE family QC |
| `window_choice.tsv`, `params.txt` | Windows and parameters, for methods sections |

Across genomes (`tehood meta`): `PREFIX_<analysis>_meta.tsv`,
`PREFIX_<analysis>_genome_z.tsv` (domain × genome z, for heatmaps on a
phylogeny), `PREFIX_TE-associated.tsv`, `PREFIX_TE-depleted.tsv`,
`PREFIX_summary.tsv`, `PREFIX_meta_qq.pdf`.

## 10. Validation

```bash
pip install ".[test]"
pytest tests/                              # ~2 min
TEHOOD_TEST_GENOMES=5 pytest tests/        # more synthetic genomes
```

Unit tests check the interval arithmetic against brute force, BH against
scipy, the saddlepoint tail against Monte Carlo, random gene sets for
uniformity, the meta-analysis calibration and the window choice. Synthetic
genomes with planted protein domain truth check, for every analysis: ≤ 3/120 false positives
and median log2FE ≈ 0; planted TE-depleted and TE-island domains found;
fragmented TE annotation not called by density; a planted domain pair called
and combination-specific; a Helitron-specific domain called for Helitron only.
Details in [docs/VALIDATION.md](docs/VALIDATION.md).

## 11. Citation and licence

MIT licence (see `LICENSE`).

The bundled `data/te_viral_domains_S1A.tsv` and `data/multicopy_domains_S1B.tsv`
are Tables S1A/S1B of: Almeida MV et al., 2025
Transposable elements drive regulatory and functional innovation of F-box
genes. *Mol Biol Evol* 42(5):msaf097. doi:10.1093/molbev/msaf097
