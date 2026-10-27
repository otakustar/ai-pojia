#!/usr/bin/env python3
"""Six-hospital ELAIA-1 audit using the final finite-sieve MCSA certificate.

The analysis uses the public processed RCT Bench ELAIA-1 file. For each held-out
hospital, the other hospitals are divided into historical training, candidate
training, and source confirmation. Half of the target hospital contributes only
covariates to the MCSA target moments; the other half is kept for outcome audit.

The primary analysis controls the familywise probability budget at .05 by using
alpha=.05/6 for each hospital. A clearly labelled unadjusted .05 sensitivity is
also reported. The function-class radius is a declared sensitivity parameter;
results for R in {0.5,1,2} are returned and R=1 is the primary specification.
This is a retrospective geographic holdout, not historical deployment evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import chi2, norm
from sklearn.linear_model import Ridge

DATA_URL = (
    "https://raw.githubusercontent.com/syl051088/RCT_Bench/"
    "refs/heads/main/cleaned_data/Non_Clustered_RCT/trial3.csv"
)
EXPECTED_SHA256 = "7c9911d1db66ffc6cad3b654b3dece2394d67d14f56a8a35eabeafe8885a20eb"
FEATURES = [
    "X_age_0m", "X_bun_0m", "X_creat_0m", "X_diabetes_0m",
    "X_chf_0m", "X_icu_0m", "X_sofa_0m", "X_systolic_0m",
    "X_hypertension_0m",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def get_data(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(DATA_URL, path)
    observed = sha256(path)
    if observed != EXPECTED_SHA256:
        raise RuntimeError(f"SHA mismatch: {observed}")
    return path


def raw_matrix(frame: pd.DataFrame, medians: np.ndarray | None = None):
    x = frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if medians is None:
        medians = np.nanmedian(x, axis=0)
    return np.where(np.isnan(x), medians, x), medians


class HistoryModel:
    def __init__(self, alpha: float = 120.0):
        self.alpha = alpha

    def fit(self, frame: pd.DataFrame):
        x, self.medians = raw_matrix(frame)
        self.mean = x.mean(0)
        self.sd = np.maximum(x.std(0), 1e-6)
        z = np.clip((x - self.mean) / self.sd, -5, 5)
        a = frame.a.to_numpy(float)
        design = np.column_stack([z, a, z * a[:, None]])
        self.model = Ridge(alpha=self.alpha).fit(design, frame.y.to_numpy(float))
        self.p = z.shape[1]
        return self

    def standardized(self, frame: pd.DataFrame) -> np.ndarray:
        x, _ = raw_matrix(frame, self.medians)
        return np.clip((x - self.mean) / self.sd, -5, 5)

    def psi(self, frame: pd.DataFrame) -> np.ndarray:
        z = self.standardized(frame)
        return np.column_stack([np.ones(len(z)), z, z**2])

    def cate(self, frame: pd.DataFrame) -> np.ndarray:
        z = self.standardized(frame)
        return np.clip(self.model.coef_[self.p] + z @ self.model.coef_[self.p + 1:], -1, 1)


def candidate_dictionary(history: HistoryModel, candidate: HistoryModel,
                         source_train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    v = np.clip(candidate.cate(frame) - history.cate(frame), -0.35, 0.35)
    # A source-support direction: suppress corrections outside the rectangle that
    # contains 99% of standardized candidate-training covariates.
    zs = history.standardized(source_train)
    z = history.standardized(frame)
    lower = np.quantile(zs, 0.005, axis=0)
    upper = np.quantile(zs, 0.995, axis=0)
    supported = np.all((z >= lower) & (z <= upper), axis=1)
    return np.column_stack([v, v * supported])


def matrix_invsqrt(G: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    val, vec = eigh((G + G.T) / 2 + ridge * np.eye(G.shape[0]))
    val = np.maximum(val, ridge)
    return (vec / np.sqrt(val)) @ vec.T


def joint_weights(psi: np.ndarray, A: np.ndarray, sigma_bound: float,
                  G: np.ndarray, R: float, alpha_source: float):
    n, p = psi.shape
    d = A.shape[1]
    Ginv = matrix_invsqrt(G)
    zrad = math.sqrt(chi2.ppf(1 - alpha_source, df=d))
    gram = psi.T @ psi / n

    def solve(loglam: float):
        lam = math.exp(loglam)
        coef = np.linalg.solve(gram + lam * np.eye(p), A)
        W = psi @ coef
        D = A - psi.T @ W / n
        bias = R * np.linalg.svd(Ginv @ D.T, compute_uv=False)[0]
        Cov = sigma_bound**2 * (W.T @ W) / n**2
        NCov = Ginv @ Cov @ Ginv.T
        noise = zrad * math.sqrt(max(np.linalg.eigvalsh((NCov + NCov.T)/2).max(), 0))
        return bias + noise, W, bias, noise

    opt = minimize_scalar(lambda x: solve(x)[0], bounds=(-15, 15), method="bounded")
    return solve(float(opt.x))


def robust_qp(Bhat: np.ndarray, Ghat: np.ndarray, radius: float, eta_g: float):
    d = len(Bhat)
    Gup = (Ghat + Ghat.T) / 2 + (eta_g + 1e-9) * np.eye(d)
    def lower(theta):
        nrm = math.sqrt(max(float(theta @ Gup @ theta), 0))
        return float(2 * Bhat @ theta - theta @ Gup @ theta - 2 * radius * nrm)
    res = minimize(lambda th: -lower(th), np.zeros(d), method="SLSQP",
                   bounds=[(0, 1)] * d,
                   constraints=[{"type": "ineq", "fun": lambda th: 1 - th.sum()}],
                   options={"maxiter": 1000, "ftol": 1e-11})
    th = np.asarray(res.x if res.success else np.zeros(d))
    val = lower(th)
    if val <= 0:
        th[:] = 0
        val = 0.0
    return th, val


def certify(history: HistoryModel, candidate: HistoryModel,
            candidate_train: pd.DataFrame, confirmation: pd.DataFrame,
            target_dev: pd.DataFrame, R: float, alpha: float):
    Vt = candidate_dictionary(history, candidate, candidate_train, target_dev)
    Vs = candidate_dictionary(history, candidate, candidate_train, confirmation)
    psi_s = history.psi(confirmation)
    psi_t = history.psi(target_dev)
    m, n = len(target_dev), len(confirmation)
    A = (psi_t[:, :, None] * Vt[:, None, :]).mean(0)
    Ghat = Vt.T @ Vt / m + 1e-8 * np.eye(Vt.shape[1])

    # With randomized 1:1 assignment, this is an unbiased conditional CATE
    # pseudo-outcome. Binary Y implies phi in {-2,0,2}; subtracting clipped g
    # gives an absolute range no larger than 3, used as a conservative bound.
    a = confirmation.a.to_numpy(float)
    y = confirmation.y.to_numpy(float)
    phi = 2 * a * y - 2 * (1 - a) * y
    residual = phi - history.cate(confirmation)
    sigma_bound = 3.0

    alpha_s, alpha_b, alpha_g = alpha/2, alpha/4, alpha/4
    source_radius, W, bias, noise = joint_weights(
        psi_s, A, sigma_bound, Ghat, R, alpha_s
    )
    Bhat = W.T @ residual / n
    max_v = max(np.linalg.norm(Vt, axis=1).max(), 1e-8)
    H = 2.0  # valid for binary-outcome CATE and clipped historical model
    target_b = H * max_v * math.sqrt(2 * (Vt.shape[1] + math.log(2/alpha_b)) / m)
    eta_g = max_v**2 * math.sqrt(2 * math.log(2*Vt.shape[1]**2/alpha_g) / m)
    theta, lower = robust_qp(Bhat, Ghat, source_radius + target_b, eta_g)
    return {
        "theta": theta, "certified_lower_gain": lower,
        "source_radius": source_radius, "bias_radius": bias,
        "noise_radius": noise, "target_B_radius": target_b,
        "gram_radius": eta_g,
    }


def audit_difference(history: HistoryModel, candidate: HistoryModel,
                     candidate_train: pd.DataFrame, target_audit: pd.DataFrame,
                     theta: np.ndarray):
    g = history.cate(target_audit)
    V = candidate_dictionary(history, candidate, candidate_train, target_audit)
    update = V @ theta
    f = g + update
    a = target_audit.a.to_numpy(float)
    y = target_audit.y.to_numpy(float)
    phi = 2 * a * y - 2 * (1 - a) * y
    difference = (f - phi)**2 - (g - phi)**2
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0
    return mean, se, difference


def run(data_path: Path, output: Path):
    df = pd.read_csv(data_path)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Treatment", "YP_composite_event", "X_hospital_0m"]).copy()
    df["a"] = (df.Treatment == "Alert").astype(int)
    df["y"] = df.YP_composite_event.astype(float)
    hospitals = sorted(df.X_hospital_0m.unique())
    rows, audit_rows = [], []

    for hi, hospital in enumerate(hospitals):
        source = df[df.X_hospital_0m != hospital].reset_index(drop=True)
        target = df[df.X_hospital_0m == hospital].reset_index(drop=True)
        rng = np.random.default_rng(20261027 + hi)
        sidx = rng.permutation(len(source))
        n1 = len(source)//2
        n2 = (len(source)-n1)//2
        history_data = source.iloc[sidx[:n1]].copy()
        candidate_data = source.iloc[sidx[n1:n1+n2]].copy()
        confirmation = source.iloc[sidx[n1+n2:]].copy()
        tidx = rng.permutation(len(target))
        target_dev = target.iloc[tidx[:len(target)//2]].copy()
        target_audit = target.iloc[tidx[len(target)//2:]].copy()
        history = HistoryModel(120).fit(history_data)
        candidate = HistoryModel(30).fit(candidate_data)

        for R in [0.5, 1.0, 2.0]:
            for label, alpha in [("familywise", 0.05/len(hospitals)), ("unadjusted", 0.05)]:
                cert = certify(history, candidate, candidate_data, confirmation,
                               target_dev, R=R, alpha=alpha)
                mean, se, diff = audit_difference(history, candidate, candidate_data,
                                                   target_audit, cert["theta"])
                rows.append({
                    "hospital": str(hospital), "R": R, "multiplicity": label,
                    "alpha": alpha, "n_history": len(history_data),
                    "n_candidate": len(candidate_data), "n_confirmation": len(confirmation),
                    "n_target_development": len(target_dev), "n_target_audit": len(target_audit),
                    "theta_prediction": cert["theta"][0],
                    "theta_supported": cert["theta"][1],
                    "certified_lower_gain": cert["certified_lower_gain"],
                    "source_radius": cert["source_radius"],
                    "bias_radius": cert["bias_radius"],
                    "noise_radius": cert["noise_radius"],
                    "target_B_radius": cert["target_B_radius"],
                    "gram_radius": cert["gram_radius"],
                    "audit_risk_change": mean, "audit_se": se,
                    "audit_ci_low": mean - 1.96*se, "audit_ci_high": mean + 1.96*se,
                    "updated": float(np.linalg.norm(cert["theta"]) > 1e-10),
                    "status": "final_MCSA_retrospective_geographic_audit",
                })
                if R == 1.0 and label == "familywise":
                    audit_rows.extend({"hospital": str(hospital), "audit_rank": i,
                                       "risk_score_difference": float(v)}
                                      for i, v in enumerate(diff))

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "elaia_mcsa_results.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output / "elaia_mcsa_audit_scores.csv", index=False)
    (output / "provenance.json").write_text(json.dumps({
        "url": DATA_URL, "sha256": sha256(data_path), "bytes": data_path.stat().st_size,
        "primary_specification": "R=1; familywise alpha=.05/6",
        "limitation": "retrospective geographic holdout using processed RCT Bench data"
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/trial3.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/elaia_mcsa"))
    args = ap.parse_args()
    data = get_data(args.data)
    run(data, args.out)


if __name__ == "__main__":
    main()
