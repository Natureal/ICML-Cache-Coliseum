# ICML-Cache-Coliseum

[![Paper](https://img.shields.io/badge/arXiv-2606.01342-b31b1b)](https://arxiv.org/abs/2606.01342)
[![ICML](https://img.shields.io/badge/ICML%202026-Spotlight-blueviolet)](https://arxiv.org/abs/2606.01342)

Built on top of [Cache-Coliseum](https://github.com/OptiSys-ZJU/Cache-Coliseum) (OptiSys-ZJU), this repository extends the benchmark with the **OnlineMin** family of randomized paging algorithms and a predictor-augmented variant, **RPB-OM**. Implementation of the ICML'26 paper [Towards Optimal Robustness in Learning-Augmented Paging](https://arxiv.org/abs/2606.01342)].

## What's New

This fork adds the **OnlineMin** family of algorithms in [`cache/evict/algorithms.py`](cache/evict/algorithms.py):

- **`OnlineMinAlgorithm`** — prediction-free `H_k`-competitive randomized paging baseline.
- **`PredictiveOnlineMinAlgorithm`** — OnlineMin with predictor-driven eviction on L0 misses.
- **`PredictiveRPBOnlineMinAlgorithm`** — budget-gated predictor override (RPB-OM). Budget τ is reset on L0 misses and consumed on non-L0 misses.
- **`PredictiveRPBOnlineMinHitCreditAlgorithm`** — RPB-OM-HC variant. Accumulates fractional credit from cache hits; credit is not reset on L0 misses and converts to budget when it reaches 1.


## Usage

See [Usage](https://github.com/OptiSys-ZJU/Cache-Coliseum#usage) in the README of [Cache-Coliseum](https://github.com/OptiSys-ZJU/Cache-Coliseum).

For example, a quick start with the PLECO predictor on the `xalanc` dataset:

```bash
python -m benchmark --dataset xalanc --real --pred pleco --boost --boost_fr
```
