from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .icp import _prepare_inputs, _glm_residuals, _invariance_pvalue, TestName, TestCallable


@dataclass
class HiddenICP:
    alpha: float = 0.01
    test: Union[TestName, TestCallable] = "normal"
    max_set_size: int = 3
    max_no_obs: Optional[int] = None
    random_state: int = 0

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        exp_ind: Union[ArrayLike, List[np.ndarray]],
    ) -> Dict[str, object]:
        # This simplified version assumes hidden variables act as shift/additive interventions to features.
        X, y, envs, is_factor = _prepare_inputs(X, y, exp_ind)
        p = X.shape[1]

        accepted: List[Tuple[int, ...]] = []
        best_p = 0.0
        from itertools import combinations
        for k in range(0, min(self.max_set_size, p) + 1):
            for subset in combinations(range(p), k):
                resid = _glm_residuals(X, y, subset, is_factor=is_factor)
                pval = _invariance_pvalue(
                    resid, envs, self.test, X=X, cols=subset, max_no_obs=self.max_no_obs, random_state=self.random_state
                )
                best_p = max(best_p, pval)
                if pval >= self.alpha:
                    accepted.append(tuple(subset))
        result = {
            "accepted_sets": accepted,
            "best_model_pvalue": float(best_p),
            "noEnv": len(envs),
            "factor": bool(is_factor),
        }
        return result


def hidden_icp(
    X: ArrayLike,
    y: ArrayLike,
    exp_ind: Union[ArrayLike, List[np.ndarray]],
    alpha: float = 0.01,
    test: Union[TestName, TestCallable] = "normal",
    max_set_size: int = 3,
    max_no_obs: Optional[int] = None,
    random_state: int = 0,
) -> Dict[str, object]:
    return HiddenICP(
        alpha=alpha,
        test=test,
        max_set_size=max_set_size,
        max_no_obs=max_no_obs,
        random_state=random_state,
    ).fit(X, y, exp_ind) 