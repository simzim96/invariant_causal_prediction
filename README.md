# invariant-causal-prediction

[![PyPI version](https://img.shields.io/pypi/v/invariant-causal-prediction.svg)](https://pypi.org/project/invariant-causal-prediction/)
[![Python Versions](https://img.shields.io/pypi/pyversions/invariant-causal-prediction.svg)](https://pypi.org/project/invariant-causal-prediction/)
[![CI](https://github.com/simzim96/invariant_causal_prediction/actions/workflows/test.yml/badge.svg)](https://github.com/simzim96/invariant_causal_prediction/actions/workflows/test.yml)
[![Docs](https://readthedocs.org/projects/invariant-causal-prediction/badge/?version=latest)](https://invariant-causal-prediction.readthedocs.io/en/latest/)
[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2%2B-blue.svg)](./LICENSE)

Invariant Causal Prediction (ICP) in Python. Identify causal predictors that are invariant across environments, with confidence intervals and parity tests against the original R implementation.

## Features

- ICP for regression and binary classification targets
- Invariance tests: `normal`, `ks`, `ranks`, `correlation`, `exact` (permutation), or a custom callable
- Variable preselection: `all`, `lasso`, `stability`, `boosting`
- Confidence intervals and maximin coefficients aggregated across accepted sets
- Options: `alpha`, `max_set_size`, `gof`, `stop_if_empty`, `max_no_obs`, `random_state`
- Plotting (`plot_conf_intervals`, `plot_accepted_sets`) and summary (`summarize_icp`)
- Parity tests against R's `InvariantCausalPrediction::ICP`

## Install

```bash
pip install invariant-causal-prediction
```

## Quick start

```python
import numpy as np
from invariant_causal_prediction import icp, summarize_icp, plot_conf_intervals

rng = np.random.default_rng(0)
n = 800
p = 5
X = rng.standard_normal((n, p))
ExpInd = np.r_[np.zeros(n//2), np.ones(n - n//2)]
X[ExpInd == 1] *= rng.normal(1.5, 0.3, size=(p,))
beta = np.array([1.0, 1.0] + [0.0]*(p-2))
Y = X @ beta + rng.standard_normal(n)

res = icp(X, Y, ExpInd, alpha=0.01, test="normal", selection="lasso", max_no_variables=5, max_set_size=3)
print(summarize_icp(res))
plot_conf_intervals(res)
```

Binary classification:

```python
rng = np.random.default_rng(1)
X = rng.standard_normal((600, 4))
ExpInd = np.r_[np.zeros(300), np.ones(300)]
X[ExpInd == 1] *= 1.2
logit_beta = np.array([1.0, 0.7, 0.0, 0.0])
lin = X @ logit_beta
p = 1/(1+np.exp(-lin))
Y = (rng.random(600) < p).astype(int)
res = icp(X, Y, ExpInd, alpha=0.05, test="ks", selection="stability", max_set_size=2)
```

Hidden ICP (simplified):

```python
from invariant_causal_prediction import hidden_icp
res_hidden = hidden_icp(X, Y, ExpInd, alpha=0.05, test="normal", max_set_size=2)
print(res_hidden["accepted_sets"])  # accepted sets allowing hidden-parent shifts
```

## API

- `ICP(alpha=0.01, test=..., selection=..., max_no_variables=None, max_set_size=3, show_accepted_sets=False, show_completion=False, stop_if_empty=False, gof=0.0, max_no_obs=None, random_state=0)`
  - `fit(X, y, exp_ind)` → dict with keys: `conf_int` (lower, upper), `maximin_coefficients`, `accepted_sets`, `used_variables`, `pvalues`, `model_reject`, `best_model_pvalue`, `noEnv`, `factor`.
- `icp(...)` convenience wrapper returning the same dict.
- `hidden_icp(...)` simplified hidden-variable ICP under shift/additive interventions; returns accepted sets and diagnostics.
- Plotting: `plot_conf_intervals(result, feature_names=None)`, `plot_accepted_sets(result, feature_names=None)`
- Summary: `summarize_icp(result, feature_names=None)`

## R vs Python

This implementation mirrors the spirit and options of the R package `InvariantCausalPrediction` while not being a line-by-line port. Parity tests validate accepted sets across tests/selections and environments. Confidence intervals are aggregated empirically from accepted sets.

- R docs: [ICP on RDocumentation](https://www.rdocumentation.org/packages/InvariantCausalPrediction/versions/0.8/topics/ICP)

## Development

- Setup (editable install):
  ```bash
  pip install -e .
  pip install pytest rpy2  # optional, for R parity via Rscript fallback
  ```
- Run tests:
  ```bash
  pytest -q
  ```
- Release
  - Bump version in `pyproject.toml` and `invariant_causal_prediction/_version.py`
  - Tag: `git tag vX.Y.Z && git push origin vX.Y.Z` (publishes via GitHub Actions)

## Citation

If you use this package, please cite the original ICP paper:

Peters J., Bühlmann P., Meinshausen N. (2015): Causal inference using invariant prediction: identification and confidence intervals. arXiv:1501.01332.

## License

GPL-2.0-or-later. See [LICENSE](./LICENSE). 