from __future__ import annotations

from typing import Dict, Any, List, Sequence
import numpy as np
import matplotlib.pyplot as plt


def plot_conf_intervals(result: Dict[str, Any], feature_names: List[str] | None = None, ax: plt.Axes | None = None):
    conf_int = result.get("conf_int")
    if conf_int is None:
        raise ValueError("result['conf_int'] is None")
    p = conf_int.shape[0]
    names = feature_names if feature_names is not None else [f"X{j+1}" for j in range(p)]
    lower = conf_int[:, 0]
    upper = conf_int[:, 1]
    mid = (lower + upper) / 2.0
    err = np.vstack([mid - lower, upper - mid])

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, p * 0.5), 4))
    ax.errorbar(range(p), mid, yerr=err, fmt="o", capsize=4)
    ax.axhline(0.0, color="gray", lw=1, ls="--")
    ax.set_xticks(range(p))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Coefficient")
    ax.set_title("ICP Confidence Intervals")
    plt.tight_layout()
    return ax


def plot_accepted_sets(result: Dict[str, Any], feature_names: List[str] | None = None, ax: plt.Axes | None = None):
    sets: List[Sequence[int]] = result.get("accepted_sets", [])
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, len(sets) * 0.3)))
    if not sets:
        ax.text(0.5, 0.5, "No accepted sets", ha="center", va="center")
        return ax
    p = len(result.get("maximin_coefficients", []))
    names = feature_names if feature_names is not None else [f"X{j+1}" for j in range(p)]
    mat = np.zeros((len(sets), p))
    for i, s in enumerate(sets):
        for j in s:
            if 0 <= j < p:
                mat[i, j] = 1
    ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels(["{" + ",".join(str(j+1) for j in s) + "}" for s in sets])
    ax.set_xticks(range(p))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Accepted Sets")
    plt.tight_layout()
    return ax 