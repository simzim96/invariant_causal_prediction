from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np


def summarize_icp(result: Dict[str, Any], feature_names: List[str] | None = None) -> str:
    p = len(result.get("maximin_coefficients", []))
    names = feature_names if feature_names is not None else [f"X{j+1}" for j in range(p)]
    conf_int = result.get("conf_int")
    maximin = result.get("maximin_coefficients")
    accepted = result.get("accepted_sets", [])
    used = result.get("used_variables", [])
    model_reject = result.get("model_reject", False)

    lines: List[str] = []
    lines.append(f"Factor target: {result.get('factor', False)}, environments: {result.get('noEnv', 'NA')}")
    lines.append(f"Model rejected: {model_reject}")
    lines.append(f"Used variables: {used}")
    lines.append("Confidence intervals and maximin coefficients:")
    for j, name in enumerate(names):
        ci = conf_int[j] if conf_int is not None and j < len(conf_int) else (np.nan, np.nan)
        mm = maximin[j] if maximin is not None and j < len(maximin) else np.nan
        lines.append(f"  {name}: [{ci[0]:.4g}, {ci[1]:.4g}], maximin={mm:.4g}")
    lines.append(f"Accepted sets ({len(accepted)}): {accepted}")
    return "\n".join(lines) 