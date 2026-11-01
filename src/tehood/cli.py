"""tehood - which protein domains are linked to transposable elements?

usage: tehood <command> [options]      (tehood <command> -h for help)

  run        whole analysis for one genome or a samples table
  prepare    raw annotations -> standard inputs (genes, domains, tes, chroms, window, all)
  classify   TE copies inside genes: artefact vs gene capture vs fusion
  tables     per-gene TE counts, densities, TE classes, QC
  test       domain / pair / architecture tests (random-gene null), per TE class
  compare    combine analyses: TE-associated / TE-depleted sets and their support
  explain    which TE families make each hit look TE-associated
  meta       combine results across genomes (rescue weak but consistent / recurrent signals)
  simulate   synthetic genome with known truth (for testing)
"""
from __future__ import annotations

import sys

from . import __version__

COMMANDS = {
    "run": "tehood.pipeline", "prepare": "tehood.prepare", "classify": "tehood.classify",
    "tables": "tehood.tables", "test": "tehood.associate", "compare": "tehood.compare",
    "explain": "tehood.explain", "meta": "tehood.meta", "simulate": "tehood.synthetic",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__); return 0
    if argv[0] in ("-V", "--version"):
        print(f"tehood {__version__}"); return 0
    if argv[0] not in COMMANDS:
        print(f"tehood: unknown command '{argv[0]}'\n{__doc__}", file=sys.stderr); return 2
    import importlib
    importlib.import_module(COMMANDS[argv[0]]).main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
