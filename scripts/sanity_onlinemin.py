from __future__ import annotations

from cache.evict.algorithms import OnlineMinAlgorithm


def run() -> None:
    # Change max_support_factor to test 3*k, 5*k, 10*k, ...
    alg = OnlineMinAlgorithm(associativity=4, max_support_factor=3)

    # A small request sequence with repeats and churn.
    seq = [1, 2, 3, 4, 1, 5, 2, 6, 3, 7, 4, 1, 2, 3, 4]
    hits = 0
    for t, addr in enumerate(seq):
        if alg.access(pc=t, address=addr):
            hits += 1
        # basic invariants
        assert len(alg.cache) == 4
        assert sum(x is None for x in alg.cache) >= 0

    print("hits", hits, "miss", len(seq) - hits)


if __name__ == "__main__":
    run()
