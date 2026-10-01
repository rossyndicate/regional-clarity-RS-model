"""SDD-weighted training: a linear per-fold sample-weighting scheme that
pushes squared-error loss to fit the sparse high-SDD tail harder, at some
cost to bulk (low/mid-range) accuracy.

Re-tested against the adopted unweighted v3 config (see project memory
"workflow-v3-findings"): weighting nets a ~0.6% overall regression, so it
is NOT the production default, but it is kept here as a documented
tail-priority alternative - the one configuration in that comparison that
beat the prior published model at both the bulk and the sparse tail
simultaneously.
"""
import numpy as np


def make_sdd_weight_fn(k: float = 2.0):
    """At weight_strength=k, the highest-SDD training observation in a fold
    is weighted (1+k) times as heavily as the lowest. Range is computed
    PER FOLD from that fold's own training data only - no leakage from
    validation/holdout."""
    def weight_fn(y: np.ndarray) -> np.ndarray:
        lo, hi = np.min(y), np.max(y)
        if hi == lo:
            return np.ones_like(y)
        return 1 + k * (y - lo) / (hi - lo)
    return weight_fn
