from __future__ import annotations

from typing import Any, Dict, Optional

from numpy.typing import ArrayLike

from .icp import ICP


class ICPRegressor:
    def __init__(
        self,
        alpha: float = 0.01,
        test: Any = "normal",
        selection: str = "all",
        max_no_variables: Optional[int] = None,
        max_set_size: int = 3,
        show_accepted_sets: bool = False,
        show_completion: bool = False,
        stop_if_empty: bool = False,
        gof: float = 0.0,
        max_no_obs: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        self.alpha = alpha
        self.test = test
        self.selection = selection
        self.max_no_variables = max_no_variables
        self.max_set_size = max_set_size
        self.show_accepted_sets = show_accepted_sets
        self.show_completion = show_completion
        self.stop_if_empty = stop_if_empty
        self.gof = gof
        self.max_no_obs = max_no_obs
        self.random_state = random_state
        self._result: Optional[Dict[str, Any]] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.__dict__.copy()

    def set_params(self, **params: Any) -> ICPRegressor:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: ArrayLike, y: ArrayLike, exp_ind: ArrayLike) -> ICPRegressor:
        self._result = ICP(
            alpha=self.alpha,
            test=self.test,
            selection=self.selection,
            max_no_variables=self.max_no_variables,
            max_set_size=self.max_set_size,
            show_accepted_sets=self.show_accepted_sets,
            show_completion=self.show_completion,
            stop_if_empty=self.stop_if_empty,
            gof=self.gof,
            max_no_obs=self.max_no_obs,
            random_state=self.random_state,
        ).fit(X, y, exp_ind)
        return self

    @property
    def result_(self) -> Dict[str, Any]:
        if self._result is None:
            raise RuntimeError("Call fit first")
        return self._result


class ICPClassifier(ICPRegressor):
    pass
