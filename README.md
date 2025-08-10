# invariant-causal-prediction

Invariant Causal Prediction (ICP) in Python.

- Identify causal predictors of a target across environments by testing invariance of residuals.
- Returns confidence intervals, accepted sets, and diagnostics.

## Install

```bash
pip install -e .
```

## Quickstart (regression)

```python
import numpy as np
from invariant_causal_prediction import icp, plot_conf_intervals, summarize_icp

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

## Quickstart (binary classification)

```python
import numpy as np
from invariant_causal_prediction import icp

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

## Hidden ICP

```python
from invariant_causal_prediction import hidden_icp
res_hidden = hidden_icp(X, Y, ExpInd, alpha=0.05, test="normal", max_set_size=2)
print(res_hidden["accepted_sets"])  # accepted sets allowing hidden-parent shifts
```

## API

- `ICP(alpha=0.01, test="normal"|"ks"|"ranks"|"correlation"|"exact"|callable, selection="all"|"lasso"|"stability"|"boosting", max_no_variables=None, max_set_size=3, show_accepted_sets=False, show_completion=False, stop_if_empty=False, gof=0.0, max_no_obs=None, random_state=0)`
  - `fit(X, y, exp_ind)` → dict with keys: `conf_int`, `maximin_coefficients`, `accepted_sets`, `used_variables`, `pvalues`, `model_reject`, `best_model_pvalue`, `noEnv`, `factor`.
- `icp(...)` convenience function with same parameters.
- `hidden_icp(...)`: simplified hidden-variable ICP under shift/additive interventions; returns accepted sets and diagnostics.
- Plotting helpers: `plot_conf_intervals(result, feature_names=None)`, `plot_accepted_sets(result, feature_names=None)`.
- Summary: `summarize_icp(result, feature_names=None)`.

## Notes

- The implementation follows the ICP principle from the R package documentation but is not a line-by-line port. Confidence intervals are aggregated empirically from coefficients across accepted sets.
- The `exact` test is implemented via permutation with optional subsampling by `max_no_obs`.
- For classification targets, invariance is applied to logistic residuals. 