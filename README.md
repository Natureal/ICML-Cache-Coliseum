# ICML-Cache-Coliseum

[![Paper](https://img.shields.io/badge/arXiv-2606.01342-b31b1b)](https://arxiv.org/abs/2606.01342)
[![ICML](https://img.shields.io/badge/ICML-Accepted-blueviolet)](https://arxiv.org/abs/2606.01342)

Built on top of [Cache-Coliseum](https://github.com/OptiSys-ZJU/Cache-Coliseum) (OptiSys-ZJU), this repository extends the benchmark with the **OnlineMin** family of randomized paging algorithms and a predictor-augmented variant, **RPB-OM**. Implementation of the ICML'26 paper Towards Optimal Robustness in Learning-Augmented Paging [[arXiv:2606.01342](https://arxiv.org/abs/2606.01342)].

## What's New in This Fork

Three algorithm classes are added in [`cache/evict/algorithms.py`](cache/evict/algorithms.py):

| Class | Pretty name | Role |
|:------|:------------|:-----|
| `OnlineMinAlgorithm` | `OnlineMin` | Prediction-free randomized paging baseline (`H_k`-competitive). Maintains support layers L1..Lk with the Equitable2 update rule plus per-page random priorities; on a miss it evicts the minimum-priority page from a layer-determined prefix. |
| `PredictiveOnlineMinAlgorithm` | `PredictiveOnlineMin[<pred>]-msf-<f>` | OnlineMin whose eviction on **L0 misses** (a requested page not currently in the support) is delegated to a predictor + evictor; all other evictions fall back to OnlineMin's rule. |
| `PredictiveRPBOnlineMinAlgorithm` | `RPB-OnlineMin[<pred>]-msf-<f>-pb-<τ>` | OnlineMin with a **budget-gated** predictor override. A true L0 miss always consults the predictor and resets the budget to τ; non-L0 misses may spend (and occasionally replenish) the budget to invoke the predictor, otherwise defer to OnlineMin. This bounds how often the predictor is queried while preserving OnlineMin's robustness. |

Both predictive variants implement the same predictor interface as `PredictAlgorithm`, so they are built through `PredictAlgorithmFactory.generate_predictive_algorithm` and work with **every predictor** the benchmark supports (PLECO, POPU, LRB, Parrot, Oracle-Dis, Oracle-Bin).

### Algorithm Parameters

Constructor arguments (not CLI flags), set at registration in [`benchmark/__main__.py`](benchmark/__main__.py):

- **`max_support_factor`** (shown as `-msf-<f>`): the support is capped at `max_support_factor × k` pages, where `k` is the associativity.
- **`pred_budget`** (RPB-OM only, shown as `-pb-<τ>`): the initial prediction budget τ. `pb-0` consults the predictor only on true L0 misses; larger τ lets non-L0 misses spend more budget on predictor-driven evictions. The benchmark sweeps several τ values per predictor (e.g. `0, 1, 2, 4` for real predictors, up to `1000` for `oracle_dis`).

## Usage

See [Usage](https://github.com/OptiSys-ZJU/Cache-Coliseum#usage) in the README of [Cache-Coliseum](https://github.com/OptiSys-ZJU/Cache-Coliseum).

For example, a quick start with the PLECO predictor on the `xalanc` dataset:

```bash
python -m benchmark --dataset xalanc --real --pred pleco --boost --boost_fr
```