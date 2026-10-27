#!/usr/bin/env python3
"""Apply two deterministic corrections to benchmark.py before execution.

1. The unrestricted KRR reference is fitted on K stratum sufficient statistics,
   not an n-by-n sample kernel matrix. This keeps the reference computationally
   feasible without changing its data budget.
2. The AIPW target empirical-mean variance uses the actual target sample size,
   not the source confirmation sample size.

The workflow archives the patched benchmark.py with its results.
"""
from pathlib import Path
import re

path = Path(__file__).with_name("benchmark.py")
text = path.read_text(encoding="utf-8")

old = re.compile(
    r"def fit_full_krr\(data: dict\[str, np\.ndarray\], dgp: DGP, nuisance: dict\[str, np\.ndarray\],\n"
    r"                 qhat_dev: np\.ndarray, p_hat: np\.ndarray, ratio_cap: float\) -> np\.ndarray:\n"
    r"    # Reference unrestricted learner: KRR on pseudo outcomes using all labels\.\n"
    r"    ix = np\.arange\(len\(data\[\"k\"\]\)\)\n"
    r"    phi = pseudo_outcome\(data, ix, nuisance\)\n"
    r"    xobs = dgp\.x\[data\[\"k\"\]\]\[\.\.\., None\]\n"
    r"    xgrid = dgp\.x\[\.\.\., None\]\n"
    r"    best = None\n"
    r"    for gamma in \[20\.0, 60\.0, 160\.0\]:\n"
    r"        for lam in \[0\.05, 0\.2, 1\.0\]:\n"
    r"            model = KernelRidge\(alpha=lam, kernel=\"rbf\", gamma=gamma\)\.fit\(xobs, phi\)\n"
    r"            pred = np\.clip\(model\.predict\(xgrid\), -0\.8, 0\.8\)\n"
    r"            # deterministic source-target plugin criterion, no target labels\n"
    r"            score = float\(np\.dot\(qhat_dev, \(pred - nuisance\[\"tau\"\]\) \*\* 2\)\)\n"
    r"            if best is None or score < best\[0\]:\n"
    r"                best = \(score, pred\)\n"
    r"    return np\.asarray\(best\[1\]\)\n"
)
new = '''def fit_full_krr(data: dict[str, np.ndarray], dgp: DGP, nuisance: dict[str, np.ndarray],
                 qhat_dev: np.ndarray, p_hat: np.ndarray, ratio_cap: float) -> np.ndarray:
    # Unrestricted reference refitted on all labels, using K stratum sufficient
    # statistics. Hyperparameters are selected without target outcomes.
    tau_all = nuisance["tau"]
    arm_info = 1.0 / (1.0 / np.maximum(nuisance["n0"], 0.5)
                      + 1.0 / np.maximum(nuisance["n1"], 0.5))
    best = None
    for bandwidth, lam in [(0.06, 0.02), (0.10, 0.04), (0.18, 0.08), (0.30, 0.15)]:
        pred = np.clip(krr_fit_from_strata(dgp.x, tau_all, arm_info,
                                           bandwidth, lam), -0.8, 0.8)
        roughness = float(np.mean(np.diff(pred, n=2) ** 2))
        score = float(np.dot(p_hat, (pred - tau_all) ** 2) + 0.05 * roughness)
        if best is None or score < best[0]:
            best = (score, pred)
    return np.asarray(best[1])
'''
text, n = old.subn(new, text)
if n != 1:
    raise RuntimeError(f"fit_full_krr replacement count={n}")

text = text.replace(
    "nuisance: dict[str, np.ndarray], p_hat: np.ndarray, qhat: np.ndarray,\n                    ratio_cap: float) -> tuple[float, float]:",
    "nuisance: dict[str, np.ndarray], p_hat: np.ndarray, qhat: np.ndarray,\n                    ratio_cap: float, target_m: int) -> tuple[float, float]:",
)
text = text.replace(
    "se_tar = float(np.dot(qhat, (target_values - mean_t) ** 2)) / max(1, int(round(len(ix))))",
    "se_tar = float(np.dot(qhat, (target_values - mean_t) ** 2)) / max(1, target_m)",
)
text = text.replace(
    "alpha: float, ratio_cap: float, safe: bool = True) -> tuple[np.ndarray, float, bool]:\n    Bhat, se = estimate_aipw_B(v, dgp, data, ix, nuisance, p_hat, qhat, ratio_cap)",
    "alpha: float, ratio_cap: float, target_m: int, safe: bool = True) -> tuple[np.ndarray, float, bool]:\n    Bhat, se = estimate_aipw_B(v, dgp, data, ix, nuisance, p_hat, qhat, ratio_cap, target_m)",
)
text = text.replace(
    "nuisance: dict[str, np.ndarray], p_hat: np.ndarray,\n                          qhat: np.ndarray, ratio_cap: float,",
    "nuisance: dict[str, np.ndarray], p_hat: np.ndarray,\n                          qhat: np.ndarray, ratio_cap: float, target_m: int,",
)
text = text.replace(
    "estimate_aipw_B(V[:, j], dgp, data, ix, nuisance, p_hat, qhat, ratio_cap)",
    "estimate_aipw_B(V[:, j], dgp, data, ix, nuisance, p_hat, qhat, ratio_cap, target_m)",
)
text = text.replace(
    "config.ratio_cap)\n\n    # 3. transport AIPW",
    "config.ratio_cap, len(td))\n\n    # 3. transport AIPW",
)
text = text.replace(
    "config.alpha, config.ratio_cap, safe=True)",
    "config.alpha, config.ratio_cap, len(tc), safe=True)",
)
text = text.replace(
    "config.ratio_cap, subset=krr_ids)",
    "config.ratio_cap, len(td), subset=krr_ids)",
)

path.write_text(text, encoding="utf-8")
print("benchmark.py patched successfully")
