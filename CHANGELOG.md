# Changelog

## 0.1.1

* `tehood run` no longer stops when an optional analysis cannot run (per-TE-class tests in
  TE-poor genomes, domain pairs when no gene carries two domains); it logs the step as skipped
  and keeps going.
* `tehood test --by-class` skips the class tests with a message when no TE class has enough
  TEs near genes, instead of exiting with an error.

## 0.1.0

First public release.
