from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import statsmodels.api as sm


TestName = Literal["normal", "ks", "ranks", "correlation", "exact"]
SelectionName = Literal["all", "lasso", "stability", "boosting"]
TestCallable = Callable[[np.ndarray, np.ndarray], float]


def _prepare_inputs(
    X: ArrayLike,
    y: ArrayLike,
    exp_ind: Union[ArrayLike, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], bool]:
    X = np.asarray(X)
    y_arr = np.asarray(y)
    n = len(y_arr)
    if X.ndim != 2:
        raise ValueError("X must be 2D array-like")
    if y_arr.ndim != 1 or len(y_arr) != X.shape[0]:
        raise ValueError("y must be 1D with same length as X rows")

    # classification if binary factor-like or bool
    is_factor = False
    y = y_arr
    # detect binary factor
    uniq_y = np.unique(y_arr)
    if y_arr.dtype.kind in ("b",) or (uniq_y.size == 2):
        # map to {0,1}
        y = (y_arr == uniq_y.max()).astype(float)
        is_factor = True

    if isinstance(exp_ind, list):
        envs = [np.asarray(idx, dtype=int) for idx in exp_ind]
    else:
        g = np.asarray(exp_ind)
        uniq = pd.Series(g).astype("category").cat.codes.to_numpy()
        envs = [np.where(uniq == k)[0] for k in np.unique(uniq)]
    return X, np.asarray(y).astype(float), envs, is_factor


def _glm_residuals(
    X: np.ndarray, y: np.ndarray, cols: Sequence[int], is_factor: bool
) -> np.ndarray:
    n = len(y)
    if len(cols) == 0:
        Xs = np.ones((n, 1))
    else:
        Xs = np.column_stack([np.ones(n)] + [X[:, j] for j in cols])

    if not is_factor:
        beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        resid = y - Xs @ beta
        return resid
    else:
        # logistic regression residuals: y - p_hat
        try:
            model = sm.Logit(y, Xs).fit(disp=0)
            phat = model.predict(Xs)
        except Exception:
            # fallback to sklearn
            clf = LogisticRegressionCV(cv=5, max_iter=1000)
            clf.fit(Xs, y)
            phat = clf.predict_proba(Xs)[:, 1]
        return y - phat


def _coef_for_set(X: np.ndarray, y: np.ndarray, cols: Sequence[int], is_factor: bool) -> np.ndarray:
    if len(cols) == 0:
        return np.array([])
    Xs = np.column_stack([X[:, j] for j in cols])
    if not is_factor:
        beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        return beta
    else:
        # logistic coefficients
        Xs_const = sm.add_constant(Xs)
        try:
            model = sm.Logit(y, Xs_const).fit(disp=0)
            params = model.params[1:].to_numpy()
            return params
        except Exception:
            clf = LogisticRegressionCV(cv=5, max_iter=1000)
            clf.fit(Xs, y)
            # return coefficients from best C; ensure 1D array shape (k,)
            coef = np.ravel(clf.coef_)
            return coef


def _fisher_combine(pvals: List[float]) -> float:
    pvals = [max(min(float(p), 1.0), np.finfo(float).tiny) for p in pvals]
    if not pvals:
        return 1.0
    stat = -2 * np.sum(np.log(pvals))
    df = 2 * len(pvals)
    return 1 - stats.chi2.cdf(stat, df)


def _pairwise_groups(values: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            pairs.append((values[i], values[j]))
    return pairs


def _exact_permutation_pvalue(groups: List[np.ndarray], n_perm: int = 1000, random_state: int = 0) -> float:
    rng = np.random.default_rng(random_state)
    # test statistic: sum of absolute mean differences across all pairs
    pairs = _pairwise_groups(groups)
    def stat_of(groups_local: List[np.ndarray]) -> float:
        return float(sum(abs(g1.mean() - g2.mean()) for g1, g2 in _pairwise_groups(groups_local)))
    obs = stat_of(groups)
    # pool and permute group labels
    pooled = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        idx = 0
        perm_groups = []
        for s in sizes:
            perm_groups.append(pooled[idx: idx + s])
            idx += s
        if stat_of(perm_groups) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def _invariance_pvalue(
    residuals: np.ndarray,
    envs: List[np.ndarray],
    test: Union[TestName, TestCallable],
    X: Optional[np.ndarray] = None,
    cols: Optional[Sequence[int]] = None,
    max_no_obs: Optional[int] = None,
    random_state: int = 0,
) -> float:
    groups = [residuals[ix] for ix in envs]
    # optional subsample for exact test
    if isinstance(test, str):
        name = test
    else:
        name = "callable"

    if name == "normal":
        pvals: List[float] = []
        for g1, g2 in _pairwise_groups(groups):
            pvals.append(stats.ttest_ind(g1, g2, equal_var=False).pvalue)
            pvals.append(stats.levene(g1, g2, center="median").pvalue)
        return _fisher_combine(pvals)
    elif name == "ks":
        pvals = [stats.ks_2samp(g1, g2).pvalue for g1, g2 in _pairwise_groups(groups)]
        return _fisher_combine(pvals)
    elif name == "ranks":
        pvals = [stats.mannwhitneyu(g1, g2, alternative="two-sided").pvalue for g1, g2 in _pairwise_groups(groups)]
        return _fisher_combine(pvals)
    elif name == "correlation":
        if X is None or cols is None:
            raise ValueError("correlation test requires X and cols")
        pvals: List[float] = []
        for env_ix in envs:
            r = residuals[env_ix]
            for j in cols:
                xj = X[env_ix, j]
                if np.std(xj) == 0 or np.std(r) == 0:
                    continue
                pvals.append(stats.pearsonr(xj, r).pvalue)
        return _fisher_combine(pvals or [1.0])
    elif name == "exact":
        # subsample to max_no_obs if provided
        if max_no_obs is not None:
            # uniform subsample within each env
            rng = np.random.default_rng(random_state)
            new_groups = []
            per_env = max(1, max_no_obs // max(1, len(envs)))
            for g in groups:
                if len(g) > per_env:
                    idx = rng.choice(len(g), size=per_env, replace=False)
                    new_groups.append(g[idx])
                else:
                    new_groups.append(g)
            groups = new_groups
        return _exact_permutation_pvalue(groups, n_perm=1000, random_state=random_state)
    elif name == "callable":
        pvals: List[float] = []
        for g1, g2 in _pairwise_groups(groups):
            p = float(test(g1, g2))  # type: ignore[arg-type]
            pvals.append(p)
        return _fisher_combine(pvals)
    else:
        raise ValueError(f"Unknown test: {test}")


@dataclass
class ICP:
    alpha: float = 0.01
    test: Union[TestName, TestCallable] = "normal"
    selection: SelectionName = "all"
    max_no_variables: Optional[int] = None
    max_set_size: int = 3
    show_accepted_sets: bool = False
    show_completion: bool = False
    stop_if_empty: bool = False
    gof: float = 0.0
    max_no_obs: Optional[int] = None
    random_state: int = 0

    def _preselect(self, X: np.ndarray, y: np.ndarray, is_factor: bool) -> List[int]:
        p = X.shape[1]
        if self.selection == "all":
            idx = list(range(p))
        elif self.selection == "lasso":
            model = make_pipeline(StandardScaler(with_mean=True, with_std=True), LassoCV(cv=5, random_state=self.random_state))
            model.fit(X, y)
            lasso: LassoCV = model.named_steps["lassocv"]
            coef = lasso.coef_
            idx = [j for j, c in enumerate(coef) if abs(c) > 1e-12]
            if not idx:
                corr = np.abs(np.corrcoef(X, y, rowvar=False)[-1, :-1])
                idx = list(np.argsort(corr)[::-1][: min(self.max_no_variables or p, p)])
        elif self.selection == "boosting":
            if is_factor:
                est = GradientBoostingClassifier(random_state=self.random_state)
            else:
                est = GradientBoostingRegressor(random_state=self.random_state)
            est.fit(X, y)
            importances = np.asarray(getattr(est, "feature_importances_", np.zeros(p)))
            idx = list(np.argsort(importances)[::-1])
        elif self.selection == "stability":
            # subsample + lasso across B replicates; select by frequency
            B = 50
            freq = np.zeros(p)
            rng = np.random.default_rng(self.random_state)
            for _ in range(B):
                ix = rng.choice(X.shape[0], size=int(0.8 * X.shape[0]), replace=True)
                model = make_pipeline(StandardScaler(with_mean=True, with_std=True), LassoCV(cv=3, random_state=self.random_state))
                model.fit(X[ix], y[ix])
                coef = model.named_steps["lassocv"].coef_
                freq += (np.abs(coef) > 1e-12).astype(float)
            idx = list(np.argsort(freq)[::-1])
        else:
            raise ValueError(f"Unknown selection: {self.selection}")
        if self.max_no_variables is not None:
            idx = idx[: self.max_no_variables]
        return idx

    def _accepted_sets(
        self,
        X: np.ndarray,
        y: np.ndarray,
        envs: List[np.ndarray],
        candidates: List[int],
        is_factor: bool,
    ) -> Tuple[List[Tuple[int, ...]], float]:
        accepted: List[Tuple[int, ...]] = []
        best_p = 0.0
        max_k = min(self.max_set_size, len(candidates))
        total = sum(len(list(combinations(candidates, k))) for k in range(0, max_k + 1))
        done = 0
        for k in range(0, max_k + 1):
            for subset in combinations(candidates, k):
                resid = _glm_residuals(X, y, subset, is_factor=is_factor)
                pval = _invariance_pvalue(
                    resid, envs, self.test, X=X, cols=subset, max_no_obs=self.max_no_obs, random_state=self.random_state
                )
                best_p = max(best_p, pval)
                if pval >= self.alpha:
                    accepted.append(tuple(subset))
                    if self.show_accepted_sets:
                        print(f"Accepted set {subset} with p-value {pval:.4g}")
                    if self.stop_if_empty and len(subset) == 0:
                        return accepted, best_p
                done += 1
                if self.show_completion and total > 0 and done % max(1, total // 10) == 0:
                    print(f"Progress: {done}/{total} subsets tested")
        return accepted, best_p

    def _conf_intervals(
        self, X: np.ndarray, y: np.ndarray, accepted_sets: List[Tuple[int, ...]], p: int, is_factor: bool
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[float]]]:
        coef_lists: Dict[int, List[float]] = {j: [] for j in range(p)}
        for subset in accepted_sets:
            if len(subset) == 0:
                continue
            beta = _coef_for_set(X, y, subset, is_factor=is_factor)
            for col_pos, j in enumerate(subset):
                coef_lists[j].append(float(beta[col_pos]))
        conf_int = np.full((p, 2), np.nan, dtype=float)
        maximin = np.full(p, np.nan, dtype=float)
        for j in range(p):
            coefs = np.array(coef_lists[j])
            if coefs.size == 0:
                continue
            lower = np.quantile(coefs, self.alpha / 2)
            upper = np.quantile(coefs, 1 - self.alpha / 2)
            conf_int[j, 0] = lower
            conf_int[j, 1] = upper
            if lower <= 0.0 <= upper:
                maximin[j] = 0.0
            else:
                maximin[j] = lower if abs(lower) < abs(upper) else upper
        return conf_int, maximin, coef_lists

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        exp_ind: Union[ArrayLike, List[np.ndarray]],
    ) -> Dict[str, object]:
        X, y, envs, is_factor = _prepare_inputs(X, y, exp_ind)
        p = X.shape[1]
        candidates = self._preselect(X, y, is_factor=is_factor)
        accepted_sets, best_p = self._accepted_sets(X, y, envs, candidates, is_factor=is_factor)
        # goodness-of-fit: require at least one set (including empty) with p >= gof
        model_reject = False
        if self.gof > 0.0:
            has_good = False
            for subset in accepted_sets:
                resid = _glm_residuals(X, y, subset, is_factor=is_factor)
                pval = _invariance_pvalue(
                    resid, envs, self.test, X=X, cols=subset, max_no_obs=self.max_no_obs, random_state=self.random_state
                )
                if pval >= self.gof:
                    has_good = True
                    break
            model_reject = not has_good
        if len(accepted_sets) == 0:
            model_reject = True

        conf_int, maximin, coef_lists = self._conf_intervals(X, y, accepted_sets, p, is_factor=is_factor)
        # simple per-variable p-values from OLS/Logit full model
        try:
            if not is_factor:
                Xs = sm.add_constant(X)
                ols = sm.OLS(y, Xs).fit()
                pvals = ols.pvalues[1:].to_numpy()
            else:
                Xs = sm.add_constant(X)
                logit = sm.Logit(y, Xs).fit(disp=0)
                pvals = logit.pvalues[1:].to_numpy()
        except Exception:
            pvals = np.full(p, np.nan)

        result = {
            "conf_int": conf_int,  # (p, 2)
            "maximin_coefficients": maximin,
            "accepted_sets": accepted_sets,
            "used_variables": candidates,
            "pvalues": pvals,
            "model_reject": bool(model_reject),
            "best_model_pvalue": float(best_p),
            "noEnv": len(envs),
            "factor": bool(is_factor),
        }
        return result


def icp(
    X: ArrayLike,
    y: ArrayLike,
    exp_ind: Union[ArrayLike, List[np.ndarray]],
    alpha: float = 0.01,
    test: Union[TestName, TestCallable] = "normal",
    selection: SelectionName = "all",
    max_no_variables: Optional[int] = None,
    max_set_size: int = 3,
    show_accepted_sets: bool = False,
    show_completion: bool = False,
    stop_if_empty: bool = False,
    gof: float = 0.0,
    max_no_obs: Optional[int] = None,
    random_state: int = 0,
) -> Dict[str, object]:
    return ICP(
        alpha=alpha,
        test=test,
        selection=selection,
        max_no_variables=max_no_variables,
        max_set_size=max_set_size,
        show_accepted_sets=show_accepted_sets,
        show_completion=show_completion,
        stop_if_empty=stop_if_empty,
        gof=gof,
        max_no_obs=max_no_obs,
        random_state=random_state,
    ).fit(X, y, exp_ind)


from .hidden_icp import hidden_icp 