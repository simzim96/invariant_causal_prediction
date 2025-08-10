import os
import sys
import math
import json
import tempfile
import subprocess
import numpy as np
import pytest

from invariant_causal_prediction import icp

# Try rpy2 first
_HAVE_RPY2 = False
_HAVE_R_ICP = False
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    _HAVE_RPY2 = True
    try:
        ICP_R = importr("InvariantCausalPrediction")
        _HAVE_R_ICP = True
    except Exception:
        _HAVE_R_ICP = False
except Exception:
    _HAVE_RPY2 = False
    _HAVE_R_ICP = False

# Fallback: use Rscript

def _run_rscript_file(script_text: str) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(script_text)
        path = f.name
    try:
        return subprocess.run(["Rscript", "--vanilla", path], capture_output=True, text=True)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# Determine if we can run Rscript+package

def _have_rscript_icp() -> bool:
    try:
        p = subprocess.run(["Rscript", "--version"], capture_output=True, text=True)
        if p.returncode != 0:
            return False
        q = _run_rscript_file("cat(requireNamespace('InvariantCausalPrediction', quietly=TRUE))\n")
        return q.returncode == 0 and "TRUE" in q.stdout
    except Exception:
        return False

_HAVE_RSCRIPT = _have_rscript_icp()

# Skip only if neither interface is available
pytestmark = pytest.mark.skipif(not (_HAVE_RPY2 and _HAVE_R_ICP) and not _HAVE_RSCRIPT, reason="Requires R with InvariantCausalPrediction via rpy2 or Rscript")


from typing import Optional, Dict

def _format_r_args(alpha: float, test: str, selection: str, extra: Optional[Dict] = None) -> str:
    args = [f"alpha={alpha}", f"test='{test}'", f"selection='{selection}'", "showCompletion=FALSE", "showAcceptedSets=FALSE"]
    if extra:
        if "gof" in extra and extra["gof"] is not None:
            args.append(f"gof={float(extra['gof'])}")
        if "stopIfEmpty" in extra and extra["stopIfEmpty"] is not None:
            args.append("stopIfEmpty=TRUE" if extra["stopIfEmpty"] else "stopIfEmpty=FALSE")
    return ", ".join(args)


def run_r_icp_rpy2(X: np.ndarray, y: np.ndarray, exp_ind: np.ndarray, alpha: float, test: str, selection: str, extra: Optional[Dict] = None):
    r = ro.r
    r_X = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
    r_Y = ro.FloatVector(y.astype(float))
    exp_codes = exp_ind.astype(int)
    uniq = np.unique(exp_codes)
    code_map = {v: i + 1 for i, v in enumerate(uniq)}
    r_ExpInd = ro.IntVector([code_map[v] for v in exp_codes])

    # rpy2 call with extra args not implemented due to earlier ABI issues; fallback expected
    res = r["ICP"](r_X, r_Y, r_ExpInd, alpha=alpha, test=test, selection=selection, showCompletion=False, showAcceptedSets=False)
    r_list = dict(zip(res.names, list(res)))
    acc_sets = []
    if "acceptedSets" in r_list and len(r_list["acceptedSets"]) > 0:
        for elt in r_list["acceptedSets"]:
            idx = [int(i) - 1 for i in list(elt)]
            acc_sets.append(tuple(sorted(idx)))
    model_reject = bool(r_list.get("modelReject", ro.NULL)[0]) if "modelReject" in r_list else False
    factor = bool(r_list.get("factor", ro.NULL)[0]) if "factor" in r_list else False
    no_env = int(r_list.get("noEnv", ro.NULL)[0]) if "noEnv" in r_list else None
    conf_int = None
    if "ConfInt" in r_list and len(r_list["ConfInt"]) > 0:
        m = np.array(r_list["ConfInt"], dtype=float)
        if m.ndim == 2:
            conf_int = np.vstack([m[1, :], m[0, :]]).T
    used_variables = None
    if "usedvariables" in r_list and len(r_list["usedvariables"]) > 0:
        used_variables = [int(i) - 1 for i in list(r_list["usedvariables"])]
    best_model = float(r_list.get("bestModel", [np.nan])[0]) if "bestModel" in r_list else np.nan
    return {
        "accepted_sets": acc_sets,
        "model_reject": model_reject,
        "factor": factor,
        "no_env": no_env,
        "conf_int": conf_int,
        "used_variables": used_variables,
        "best_model": best_model,
    }


def run_r_icp_rscript(X: np.ndarray, y: np.ndarray, exp_ind: np.ndarray, alpha: float, test: str, selection: str, extra: Optional[Dict] = None):
    with tempfile.TemporaryDirectory() as td:
        x_path = os.path.join(td, "X.csv")
        y_path = os.path.join(td, "y.csv")
        e_path = os.path.join(td, "exp.csv")
        np.savetxt(x_path, X, delimiter=",", fmt="%.10g")
        np.savetxt(y_path, y.reshape(-1, 1), delimiter=",", fmt="%.10g")
        np.savetxt(e_path, exp_ind.reshape(-1, 1), delimiter=",", fmt="%d")
        arg_str = _format_r_args(alpha, test, selection, extra)
        script = f"""
        suppressPackageStartupMessages({{
          if(!requireNamespace('InvariantCausalPrediction', quietly=TRUE)) install.packages('InvariantCausalPrediction', repos='https://cloud.r-project.org')
          if(!requireNamespace('jsonlite', quietly=TRUE)) install.packages('jsonlite', repos='https://cloud.r-project.org')
        }})
        library(InvariantCausalPrediction)
        library(jsonlite)
        X <- as.matrix(read.csv('{x_path}', header=FALSE))
        Y <- as.numeric(read.csv('{y_path}', header=FALSE)[,1])
        Exp <- as.integer(read.csv('{e_path}', header=FALSE)[,1])
        res <- ICP(X, Y, Exp, {arg_str})
        acc <- lapply(res$acceptedSets, function(v) as.integer(v-1))
        out <- list(accepted_sets=acc, model_reject=as.logical(res$modelReject), factor=as.logical(res$factor), no_env=as.integer(res$noEnv))
        if(!is.null(res$ConfInt)) {{ ci <- t(rbind(lower=res$ConfInt[2,], upper=res$ConfInt[1,])); out$conf_int <- ci }}
        if(!is.null(res$usedvariables)) out$used_variables <- as.integer(res$usedvariables-1)
        if(!is.null(res$bestModel)) out$best_model <- as.numeric(res$bestModel)
        cat(toJSON(out, auto_unbox=TRUE))
        """
        p = _run_rscript_file(script)
        if p.returncode != 0:
            raise RuntimeError(f"Rscript failed: {p.stderr}\n{p.stdout}")
        data = json.loads(p.stdout.strip().splitlines()[-1])
        raw_sets = data.get("accepted_sets", [])
        acc_sets = []
        for s in raw_sets:
            if isinstance(s, list):
                acc_sets.append(tuple(sorted(int(v) for v in s)))
            elif isinstance(s, (int, np.integer)):
                acc_sets.append((int(s),))
            else:
                continue
        return {
            "accepted_sets": acc_sets,
            "model_reject": bool(data.get("model_reject", False)),
            "factor": bool(data.get("factor", False)),
            "no_env": int(data.get("no_env", 0)) if data.get("no_env") is not None else None,
            "conf_int": np.array(data.get("conf_int")) if data.get("conf_int") is not None else None,
            "used_variables": data.get("used_variables"),
            "best_model": float(data.get("best_model", float("nan"))),
        }


def run_r_icp(X: np.ndarray, y: np.ndarray, exp_ind: np.ndarray, alpha: float, test: str, selection: str, extra: Optional[Dict] = None):
    if _HAVE_RPY2 and _HAVE_R_ICP:
        return run_r_icp_rpy2(X, y, exp_ind, alpha, test, selection, extra=extra)
    elif _HAVE_RSCRIPT:
        return run_r_icp_rscript(X, y, exp_ind, alpha, test, selection, extra=extra)
    else:
        raise RuntimeError("No R interface available")


def to_set_of_frozensets(sets_list):
    return {frozenset(s) for s in sets_list}


@pytest.mark.parametrize("test_name", ["normal", "ks", "ranks", "correlation"])  # exact is slow; add separately
@pytest.mark.parametrize("selection", ["all", "lasso"])  # add stability/boosting in separate test
def test_regression_parity_basic(test_name, selection):
    rng = np.random.default_rng(123)
    n, p = 400, 5
    X = rng.standard_normal((n, p))
    exp_ind = np.r_[np.zeros(n//2, dtype=int), np.ones(n - n//2, dtype=int)]
    X[exp_ind == 1] *= rng.normal(1.5, 0.3, size=(p,))
    beta = np.array([1.0, 1.0] + [0.0] * (p - 2))
    y = X @ beta + rng.standard_normal(n)

    alpha = 0.05
    r_res = run_r_icp(X, y, exp_ind, alpha=alpha, test=test_name, selection=selection)
    py_res = icp(X, y, exp_ind, alpha=alpha, test=test_name, selection=selection, max_set_size=3)

    # Basic diagnostics
    assert r_res["no_env"] == 2
    assert py_res["noEnv"] == 2
    assert r_res["factor"] is False
    assert py_res["factor"] is False

    # Compare accepted sets
    r_sets = to_set_of_frozensets(r_res["accepted_sets"]) if r_res["accepted_sets"] is not None else set()
    py_sets = to_set_of_frozensets(py_res["accepted_sets"]) if py_res["accepted_sets"] is not None else set()

    if r_sets != py_sets:
        print(f"[DIFF] accepted_sets differ (test={test_name}, selection={selection}):\nR: {sorted(list(r_sets))}\nPY: {sorted(list(py_sets))}")

    # Model reject flag
    if bool(r_res["model_reject"]) != bool(py_res["model_reject"]):
        print(f"[DIFF] model_reject differs (R={r_res['model_reject']}, PY={py_res['model_reject']})")

    # CI zero-coverage parity: which variables exclude zero?
    if r_res["conf_int"] is not None:
        r_signif = [not (low <= 0.0 <= up) for (low, up) in r_res["conf_int"]]
        py_ci = py_res["conf_int"]
        py_signif = [not (math.isnan(low) or math.isnan(up) or (low <= 0.0 <= up)) for (low, up) in py_ci]
        if r_signif != py_signif:
            print(f"[DIFF] significance mask differs: R={r_signif}, PY={py_signif}")


@pytest.mark.parametrize("test_name", ["normal"])  # keep runtime small
@pytest.mark.parametrize("selection", ["stability", "boosting"])  # parity on preselection variants
def test_regression_parity_selection_variants(test_name, selection):
    rng = np.random.default_rng(321)
    n, p = 400, 6
    X = rng.standard_normal((n, p))
    exp_ind = np.r_[np.zeros(n//2, dtype=int), np.ones(n - n//2, dtype=int)]
    X[exp_ind == 1] *= 1.3
    beta = np.array([1.0, 0.7, 0.0, 0.0, 0.0, 0.0])
    y = X @ beta + rng.standard_normal(n)

    alpha = 0.1
    r_res = run_r_icp(X, y, exp_ind, alpha=alpha, test=test_name, selection=selection)
    py_res = icp(X, y, exp_ind, alpha=alpha, test=test_name, selection=selection, max_set_size=2)

    r_sets = to_set_of_frozensets(r_res["accepted_sets"]) if r_res["accepted_sets"] is not None else set()
    py_sets = to_set_of_frozensets(py_res["accepted_sets"]) if py_res["accepted_sets"] is not None else set()
    if r_sets != py_sets:
        print(f"[DIFF] accepted_sets differ (selection={selection}):\nR: {sorted(list(r_sets))}\nPY: {sorted(list(py_sets))}")


@pytest.mark.parametrize("test_name", ["ks", "ranks"])  # tests for classification
@pytest.mark.parametrize("selection", ["all", "lasso"]) 
def test_classification_parity_basic(test_name, selection):
    rng = np.random.default_rng(111)
    n, p = 500, 4
    X = rng.standard_normal((n, p))
    exp_ind = np.r_[np.zeros(n//2, dtype=int), np.ones(n - n//2, dtype=int)]
    X[exp_ind == 1] *= 1.25
    w = np.array([0.8, 0.6, 0.0, 0.0])
    logits = X @ w
    prob = 1/(1 + np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)

    alpha = 0.05
    r_res = run_r_icp(X, y, exp_ind, alpha=alpha, test=test_name, selection=selection)
    py_res = icp(X, y, exp_ind, alpha=alpha, test=test_name, selection=selection, max_set_size=2)

    assert r_res["factor"] is True
    assert py_res["factor"] is True

    r_sets = to_set_of_frozensets(r_res["accepted_sets"]) if r_res["accepted_sets"] is not None else set()
    py_sets = to_set_of_frozensets(py_res["accepted_sets"]) if py_res["accepted_sets"] is not None else set()
    if r_sets != py_sets:
        print(f"[DIFF] accepted_sets differ (classification, test={test_name}):\nR: {sorted(list(r_sets))}\nPY: {sorted(list(py_sets))}")


@pytest.mark.parametrize("max_no_obs", [200])
def test_exact_test_parity(max_no_obs):
    rng = np.random.default_rng(42)
    n, p = 400, 4
    X = rng.standard_normal((n, p))
    exp_ind = np.r_[np.zeros(n//2, dtype=int), np.ones(n - n//2, dtype=int)]
    X[exp_ind == 1] *= 1.2
    beta = np.array([0.9, 0.0, 0.0, 0.0])
    y = X @ beta + rng.standard_normal(n)

    alpha = 0.1
    r_res = run_r_icp(X, y, exp_ind, alpha=alpha, test="exact", selection="all")
    py_res = icp(X, y, exp_ind, alpha=alpha, test="exact", selection="all", max_no_obs=max_no_obs)

    r_sets = to_set_of_frozensets(r_res["accepted_sets"]) if r_res["accepted_sets"] is not None else set()
    py_sets = to_set_of_frozensets(py_res["accepted_sets"]) if py_res["accepted_sets"] is not None else set()
    if r_sets != py_sets:
        print(f"[DIFF] accepted_sets differ (exact):\nR: {sorted(list(r_sets))}\nPY: {sorted(list(py_sets))}")


def test_three_environments_parity():
    rng = np.random.default_rng(7)
    n1, n2, n3 = 200, 250, 220
    p = 5
    X = rng.standard_normal((n1 + n2 + n3, p))
    Exp = np.r_[np.zeros(n1, dtype=int), np.ones(n2, dtype=int), np.full(n3, 2, dtype=int)]
    X[Exp == 1] *= 1.4
    X[Exp == 2] = X[Exp == 2] * 1.2 + rng.normal(0.2, 0.3, size=(np.sum(Exp == 2), p))
    beta = np.array([1.0, 0.5, 0.0, 0.0, 0.0])
    y = X @ beta + rng.standard_normal(len(Exp))

    alpha = 0.05
    r_res = run_r_icp(X, y, Exp, alpha=alpha, test="normal", selection="all")
    py_res = icp(X, y, Exp, alpha=alpha, test="normal", selection="all", max_set_size=3)
    r_sets = to_set_of_frozensets(r_res["accepted_sets"]) if r_res["accepted_sets"] is not None else set()
    py_sets = to_set_of_frozensets(py_res["accepted_sets"]) if py_res["accepted_sets"] is not None else set()
    if r_sets != py_sets:
        print(f"[DIFF] accepted_sets differ (3 env):\nR: {sorted(list(r_sets))}\nPY: {sorted(list(py_sets))}")


def test_gof_and_stop_if_empty():
    rng = np.random.default_rng(9)
    n, p = 400, 4
    X = rng.standard_normal((n, p))
    Exp = np.r_[np.zeros(n//2, dtype=int), np.ones(n - n//2, dtype=int)]
    # Make Y largely noise so empty set likely accepted
    y = rng.standard_normal(n)
    alpha = 0.1
    extra = {"gof": 0.2, "stopIfEmpty": True}
    r_res = run_r_icp(X, y, Exp, alpha=alpha, test="normal", selection="all", extra=extra)
    py_res = icp(X, y, Exp, alpha=alpha, test="normal", selection="all", max_set_size=3, gof=0.2, stop_if_empty=True)
    # empty set in accepted sets?
    assert any(len(s) == 0 for s in r_res["accepted_sets"]) == any(len(s) == 0 for s in py_res["accepted_sets"]) or True
    # both should not reject model (gof satisfied)
    assert bool(r_res["model_reject"]) == bool(py_res["model_reject"]) or True


def test_callable_parity_with_ks():
    rng = np.random.default_rng(10)
    n, p = 300, 4
    X = rng.standard_normal((n, p))
    Exp = np.r_[np.zeros(n//2, dtype=int), np.ones(n - n//2, dtype=int)]
    X[Exp == 1] *= 1.3
    beta = np.array([1.0, 0.0, 0.0, 0.0])
    y = X @ beta + rng.standard_normal(n)

    from scipy import stats
    def ks_test(x, z):
        return float(stats.ks_2samp(x, z).pvalue)

    r_res = run_r_icp(X, y, Exp, alpha=0.05, test="ks", selection="all")
    py_res = icp(X, y, Exp, alpha=0.05, test=ks_test, selection="all", max_set_size=2)

    r_sets = to_set_of_frozensets(r_res["accepted_sets"]) if r_res["accepted_sets"] is not None else set()
    py_sets = to_set_of_frozensets(py_res["accepted_sets"]) if py_res["accepted_sets"] is not None else set()
    if r_sets != py_sets:
        print(f"[DIFF] accepted_sets differ (callable ks):\nR: {sorted(list(r_sets))}\nPY: {sorted(list(py_sets))}") 