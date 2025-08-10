import numpy as np
from invariant_causal_prediction import ICP, icp


def test_smoke():
    rng = np.random.default_rng(1)
    n = 300
    p = 4
    X = rng.standard_normal((n, p))
    ExpInd = np.r_[np.zeros(n//2), np.ones(n - n//2)]
    X[ExpInd == 1] *= rng.normal(1.5, 0.5, size=(p,))
    beta = np.array([1.0, 0.5, 0.0, 0.0])
    y = X @ beta + rng.standard_normal(n)

    model = ICP(alpha=0.05, test="normal", selection="lasso", max_no_variables=4, max_set_size=2)
    res = model.fit(X, y, ExpInd)
    assert "conf_int" in res
    assert res["conf_int"].shape == (p, 2)
    assert isinstance(res["accepted_sets"], list)

    res2 = icp(X, y, ExpInd, alpha=0.05, test="ks", selection="all", max_set_size=2)
    assert "conf_int" in res2 