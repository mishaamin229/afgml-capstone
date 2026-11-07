"""Precomputed fallback tables used when exact inference exceeds the op budget.

A fallback maps the observed evidence to P(y=1) by LOOKUP in a table that was built
offline from the exact BN (offline cost is reported as preprocessing, not charged to
the per-query budget). `subset` lists the variables the fallback may condition on;
all other evidence is ignored (marginalised out of the table)."""
from __future__ import annotations
import numpy as np
from src.inference import variable_elimination


class TableFallback:
    def __init__(self, factors, states, subset, name, uses_protected=False):
        self.factors, self.name, self.uses_protected = factors, name, uses_protected
        self.subset = {v for v in subset if v in states}
        self.cache: dict = {}

    def __call__(self, factors, query, evidence):
        ev = {v: s for v, s in evidence.items() if v in self.subset}
        key = tuple(sorted(ev.items()))
        if key not in self.cache:                      # offline table construction
            self.cache[key] = variable_elimination(self.factors, ["y"], ev, "min_fill")[0]
        return self.cache[key]


def candidate_fallbacks(factors, states, cfg_fb: dict) -> dict:
    """cfg_fb: name -> {subset: [...], uses_protected: bool} (from configs/main.yaml)."""
    return {n: TableFallback(factors, states, c["subset"], n, c.get("uses_protected", False))
            for n, c in cfg_fb.items()}
