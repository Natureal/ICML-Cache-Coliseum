from __future__ import annotations

from functools import partial

from cache.evict.algorithms import PredictiveOnlineMinAlgorithm
from cache.evict.evictor import MaxEvictor
from cache.evict.predictor import ReuseDistancePredictor


class ToyReuseDistancePredictor(ReuseDistancePredictor):
    """A tiny predictor for sanity checks.

    Predicts a simple "next use" score based on the address value so the
    PredictiveOnlineMin path is exercised deterministically.
    """

    def predict_score(self, ts, pc, address, cache_state):
        # Larger score => more evictable with MaxEvictor.
        return float(address)


def run() -> None:
    alg = PredictiveOnlineMinAlgorithm(
        associativity=4,
        max_support_factor=3,
        evictor_type=MaxEvictor,
        predictor_type=partial(ToyReuseDistancePredictor),
    )

    seq = [1, 2, 3, 4, 5, 1, 6, 2, 7, 3, 8, 4, 1, 2, 3, 4]
    hits = 0
    for t, addr in enumerate(seq):
        if alg.access(pc=t, address=addr):
            hits += 1
        assert len(alg.cache) == 4

    print("hits", hits, "miss", len(seq) - hits)


if __name__ == "__main__":
    run()
