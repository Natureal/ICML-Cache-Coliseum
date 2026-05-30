from __future__ import annotations

from cache.evict.algorithms import RDMAlgorithm


def run() -> None:
    # Change max_support_factor to test 3*k, 5*k, 10*k, ...
    alg = RDMAlgorithm(associativity=4, max_support_factor=3)

    # Mix: repeats (revealed-heavy) + churn (unrevealed/L0 pressure)
    seq = [1, 2, 3, 4, 1, 2, 5, 1, 2, 6, 3, 7, 4, 1, 8, 2, 3, 4]
    hits = 0
    for t, addr in enumerate(seq):
        if alg.access(pc=t, address=addr):
            hits += 1
        assert len(alg.cache) == 4

    print("hits", hits, "miss", len(seq) - hits)


if __name__ == "__main__":
    run()
