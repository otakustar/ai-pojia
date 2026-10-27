#!/usr/bin/env python3
"""Equal-budget eight-method benchmark for safe transported CATE updating.

The experiment is deliberately finite-stratum and fully inspectable. Every method
receives the same labeled source sample, unlabeled target sample, historical model,
candidate dictionary, random seed, and target-risk definition. The source data are
split once into candidate/nuisance training (40%), selection (30%), and final
confirmation (30%). Target covariates are split into selection and confirmation
halves. Target labels are never supplied to any method and are unnecessary in the
simulation because the true target risk is analytically available.

Methods
-------
1 no_update
2 unrestricted_retrain        (performance ceiling; not alpha-safe)
3 transport_aipw_safe
4 aml_point                    (AML/Riesz point update; no safety gate)
5 best_single_safe
6 qaggregation_safe            (official Q-aggregation objective, transported)
7 krr_transfer_safe            (COKE-style KRR transfer learner; not exact COKE)
8 mcsa                         (joint AML confidence ball + robust decision)

All deployable methods 3, 5, 6, 7 and 8 use alpha=.05 final confirmation.
Methods 2 and 4 are intentionally unsafe references, as required by the benchmark.

Outputs include replicate-level risk, harm, harm magnitude, acceptance, certificate
coverage, beneficial-opportunity detection, runtime, and paired differences versus
MCSA. Python >=3.10; dependencies listed in requirements.txt.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, chi2
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge


METHODS = [
    "no_update",
    "unrestricted_retrain",
    "transport_aipw_safe",
    "aml_point",
    "best_single_safe",
    "qaggregation_safe",
    "krr_transfer_safe",
    "mcsa",
]
SAFE_METHODS = {
    "no_update",
    "transport_aipw_safe",
    "best_single_safe",
    "qaggregation_safe",
    "krr_transfer_safe",
    "mcsa",
}


@dataclass(frozen=True)
class Config:
    K: int = 24
    alpha: float = 0.05
    target_ratio: float = 2.0
    reps: int = 300
    seed: int = 20261027
    class_radius: float = 1.30
    residual_bound: float = 0.75
    max_direction: float = 0.70
    top_d: int = 3
    ratio_cap: float = 40.0


@dataclass
class DGP:
    name: str
    x: np.ndarray
    p: np.ndarray
    q: np.ndarray
    e: np.ndarray
    mu0: np.ndarray
    tau: np.ndarray
    g: np.ndarray
    sigma: np.ndarray
    class_radius: float
    target_only_bound: float
    theory_condition: bool


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, float), 1e-14)
    return x / x.sum()


def make_dgp(name: str, K: int, class_radius: float) -> DGP:
    x = (np.arange(K) + 0.5) / K
    mu0 = 0.25 * np.sin(2 * np.pi * x) + 0.15 * x
    tau = 0.15 + 0.18 * np.sin(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x)
    sigma = 0.70 + 0.25 * x

    if name == "direction_favorable":
        p = normalize(np.exp(-3.4 * x) + 0.10)
        q = normalize(0.58 * np.exp(-((x - 0.43) / 0.15) ** 2) + 0.42 * np.exp(4.5 * (x - 1)))
        e = np.clip(0.50 - 0.42 * (x > 0.72) + 0.10 * np.sin(2 * np.pi * x), 0.06, 0.94)
        delta = 0.34 * np.exp(-((x - 0.42) / 0.12) ** 2)
        g = tau - delta
        theory = True
        radius = class_radius
        H = 0.75
    elif name == "regular_overlap":
        p = normalize(0.85 + 0.15 * np.cos(2 * np.pi * x))
        q = normalize(0.90 + 0.10 * np.sin(2 * np.pi * x))
        e = np.full(K, 0.50)
        delta = 0.16 + 0.10 * np.sin(2 * np.pi * x)
        g = tau - delta
        theory = True
        radius = class_radius
        H = 0.75
    elif name == "null_update":
        p = normalize(np.exp(-3.4 * x) + 0.10)
        q = normalize(0.58 * np.exp(-((x - 0.43) / 0.15) ** 2) + 0.42 * np.exp(4.5 * (x - 1)))
        e = np.clip(0.50 - 0.42 * (x > 0.72), 0.06, 0.94)
        g = tau.copy()
        theory = True
        radius = class_radius
        H = 0.75
    elif name == "radius_misspecified":
        p = normalize(np.exp(-4.3 * x) + 0.025)
        q = normalize(np.exp(5.0 * (x - 1)) + 0.10 * np.exp(-((x - 0.45) / 0.12) ** 2))
        e = np.clip(0.48 - 0.43 * (x > 0.68), 0.035, 0.965)
        delta = 0.45 * (x > 0.73) + 0.08 * np.sin(4 * np.pi * x)
        g = tau - delta
        theory = False
        radius = 0.45  # deliberately too small relative to the actual residual
        H = 0.20       # deliberately too small: stress test, never theory validation
    else:
        raise ValueError(name)

    return DGP(
        name=name,
        x=x,
        p=normalize(p),
        q=normalize(q),
        e=np.asarray(e, float),
        mu0=mu0,
        tau=tau,
        g=g,
        sigma=sigma,
        class_radius=radius,
        target_only_bound=H,
        theory_condition=theory,
    )


def draw_source(rng: np.random.Generator, dgp: DGP, n: int) -> dict[str, np.ndarray]:
    k = rng.choice(len(dgp.x), size=n, p=dgp.p)
    a = rng.binomial(1, dgp.e[k])
    y = dgp.mu0[k] + a * dgp.tau[k] + rng.normal(0.0, dgp.sigma[k], size=n)
    return {"k": k, "a": a, "y": y}


def draw_target(rng: np.random.Generator, dgp: DGP, m: int) -> np.ndarray:
    return rng.choice(len(dgp.x), size=m, p=dgp.q)


def split_indices(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = rng.permutation(n)
    n1 = int(0.40 * n)
    n2 = int(0.30 * n)
    return ix[:n1], ix[n1:n1+n2], ix[n1+n2:]


def counts_prob(k: np.ndarray, K: int, smoothing: float = 0.5) -> np.ndarray:
    c = np.bincount(k, minlength=K).astype(float) + smoothing
    return c / c.sum()


def fit_nuisance(data: dict[str, np.ndarray], ix: np.ndarray, K: int) -> dict[str, np.ndarray]:
    k, a, y = data["k"][ix], data["a"][ix], data["y"][ix]
    global_e = (a.sum() + 1.0) / (len(a) + 2.0)
    global_m0 = y[a == 0].mean() if np.any(a == 0) else y.mean()
    global_m1 = y[a == 1].mean() if np.any(a == 1) else y.mean()
    ehat = np.full(K, global_e)
    m0 = np.full(K, global_m0)
    m1 = np.full(K, global_m1)
    n0 = np.zeros(K); n1 = np.zeros(K)
    for j in range(K):
        mask = k == j
        nj = mask.sum()
        if nj:
            aj = a[mask]; yj = y[mask]
            ehat[j] = (aj.sum() + 1.0) / (nj + 2.0)
            n0[j] = np.sum(aj == 0); n1[j] = np.sum(aj == 1)
            # empirical-Bayes shrinkage toward global arm mean
            if n0[j]:
                m0[j] = (yj[aj == 0].sum() + 4.0 * global_m0) / (n0[j] + 4.0)
            if n1[j]:
                m1[j] = (yj[aj == 1].sum() + 4.0 * global_m1) / (n1[j] + 4.0)
    ehat = np.clip(ehat, 0.04, 0.96)
    return {"e": ehat, "m0": m0, "m1": m1, "tau": m1 - m0, "n0": n0, "n1": n1}


def pseudo_outcome(data: dict[str, np.ndarray], ix: np.ndarray, nuisance: dict[str, np.ndarray]) -> np.ndarray:
    k, a, y = data["k"][ix], data["a"][ix], data["y"][ix]
    e = nuisance["e"][k]
    m0 = nuisance["m0"][k]
    m1 = nuisance["m1"][k]
    return (m1 - m0) + a / e * (y - m1) - (1 - a) / (1 - e) * (y - m0)


def gaussian_kernel(x: np.ndarray, bandwidth: float) -> np.ndarray:
    d = x[:, None] - x[None, :]
    return np.exp(-0.5 * (d / bandwidth) ** 2)


def krr_fit_from_strata(x: np.ndarray, tau_hat: np.ndarray, weights: np.ndarray,
                        bandwidth: float, lam: float) -> np.ndarray:
    Kmat = gaussian_kernel(x, bandwidth)
    W = np.diag(np.maximum(weights, 1e-6))
    coef = np.linalg.solve(Kmat @ W @ Kmat + lam * np.eye(len(x)), Kmat @ W @ tau_hat)
    return Kmat @ coef


def make_candidates(dgp: DGP, nuisance: dict[str, np.ndarray], source_p: np.ndarray) -> tuple[np.ndarray, list[str], list[int]]:
    x, g = dgp.x, dgp.g
    tau0 = nuisance["tau"]
    arm_info = 1.0 / (1.0 / np.maximum(nuisance["n0"], 0.5) + 1.0 / np.maximum(nuisance["n1"], 0.5))
    candidates: list[np.ndarray] = []
    names: list[str] = []
    krr_indices: list[int] = []

    # full empirical-Bayes direction
    full = np.clip(tau0 - g, -0.70, 0.70)
    candidates.append(full); names.append("eb_full")

    # KRR transfer candidates (the COKE-style learner selects among these)
    for bw, lam in [(0.08, 0.03), (0.15, 0.05), (0.28, 0.08)]:
        tau_krr = krr_fit_from_strata(x, tau0, arm_info, bw, lam)
        krr_indices.append(len(candidates))
        candidates.append(np.clip(tau_krr - g, -0.70, 0.70))
        names.append(f"krr_b{bw}_l{lam}")

    # fixed low-complexity and local directions
    X = np.column_stack([np.ones_like(x), x, x**2, np.sin(2*np.pi*x), np.cos(2*np.pi*x)])
    model = Ridge(alpha=2.0, fit_intercept=False).fit(X, tau0, sample_weight=np.maximum(arm_info, 1e-4))
    smooth = np.clip(model.predict(X) - g, -0.70, 0.70)
    candidates.append(smooth); names.append("smooth_basis")
    candidates.append(full * ((x >= 0.18) & (x <= 0.62))); names.append("local_mid")
    candidates.append(full * (x <= 0.66)); names.append("local_low")
    candidates.append(full * (x >= 0.68)); names.append("local_tail")
    candidates.append(0.50 * full); names.append("half_full")

    V = np.column_stack(candidates)
    # remove numerically duplicate or zero columns
    keep = []
    for j in range(V.shape[1]):
        if np.linalg.norm(V[:, j]) < 1e-10:
            continue
        if any(np.linalg.norm(V[:, j] - V[:, h]) < 1e-8 for h in keep):
            continue
        keep.append(j)
    remap = {old: new for new, old in enumerate(keep)}
    return V[:, keep], [names[j] for j in keep], [remap[j] for j in krr_indices if j in remap]


def stratum_contrasts(data: dict[str, np.ndarray], ix: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    k, a, y = data["k"][ix], data["a"][ix], data["y"][ix]
    tau = np.zeros(K); se = np.full(K, np.inf)
    global_var = max(np.var(y, ddof=1), 1e-6)
    for j in range(K):
        yy0 = y[(k == j) & (a == 0)]
        yy1 = y[(k == j) & (a == 1)]
        if len(yy0) and len(yy1):
            tau[j] = yy1.mean() - yy0.mean()
            v0 = np.var(yy0, ddof=1) if len(yy0) > 1 else global_var
            v1 = np.var(yy1, ddof=1) if len(yy1) > 1 else global_var
            se[j] = math.sqrt(v0 / len(yy0) + v1 / len(yy1))
    return tau, se


def safe_inverse_sqrt(G: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    vals, vecs = eigh((G + G.T) / 2 + ridge * np.eye(G.shape[0]))
    vals = np.maximum(vals, ridge)
    return (vecs / np.sqrt(vals)) @ vecs.T


def class_radius_true(dgp: DGP) -> float:
    return float(np.linalg.norm(dgp.tau - dgp.g))


def aml_weights_vector(V: np.ndarray, qhat: np.ndarray, se: np.ndarray,
                       G: np.ndarray, R: float, alpha_source: float) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Joint ridge AML weights for a Kxd dictionary.

    W has shape Kxd and estimates B_j=sum_k q_k V_kj f_k. The common scalar
    ridge is selected to minimize worst-case normalized imbalance plus Gaussian
    radius. This is a computable conservative specialization of the joint AML
    program used in the theorem.
    """
    K, d = V.shape
    A = qhat[:, None] * V
    finite = np.isfinite(se)
    se2 = np.where(finite, se**2, 1e12)
    Ginvhalf = safe_inverse_sqrt(G)
    zrad = math.sqrt(chi2.ppf(1 - alpha_source, df=d))

    def pieces(loglam: float) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        lam = math.exp(loglam)
        W = A / (1.0 + lam * se2[:, None])
        W[~finite, :] = 0.0
        D = (A - W).T  # d x K
        bias = R * np.linalg.svd(Ginvhalf @ D, compute_uv=False)[0]
        Cov = W.T @ (se2[:, None] * W)
        NCov = Ginvhalf @ Cov @ Ginvhalf.T
        noise = zrad * math.sqrt(max(np.linalg.eigvalsh((NCov + NCov.T) / 2).max(), 0.0))
        return bias + noise, bias, noise, W, Cov

    opt = minimize_scalar(lambda z: pieces(z)[0], bounds=(-14, 14), method="bounded")
    radius, bias, noise, W, Cov = pieces(float(opt.x))
    return W, radius, bias, Cov


def target_error_terms(V: np.ndarray, qhat: np.ndarray, m: int,
                       H: float, alpha_b: float, alpha_g: float) -> tuple[float, float, np.ndarray]:
    Ghat = V.T @ (qhat[:, None] * V)
    row_norm = np.linalg.norm(V, axis=1)
    max_v = max(float(row_norm.max()), 1e-8)
    # bounded-vector and matrix Hoeffding-style conservative radii
    b = H * max_v * math.sqrt(2.0 * (V.shape[1] + math.log(2 / alpha_b)) / m)
    g = max_v**2 * math.sqrt(2.0 * math.log(2 * V.shape[1]**2 / alpha_g) / m)
    return b, g, Ghat


def aml_vector_estimate(V: np.ndarray, qhat: np.ndarray, tauhat: np.ndarray,
                        se: np.ndarray, g: np.ndarray, R: float, H: float,
                        m: int, alpha: float) -> dict:
    d = V.shape[1]
    _, _, Ghat0 = target_error_terms(V, qhat, m, H, alpha/4, alpha/4)
    Ghat0 = (Ghat0 + Ghat0.T) / 2 + 1e-8 * np.eye(d)
    W, source_radius, bias, Cov = aml_weights_vector(V, qhat, se, Ghat0, R, alpha/2)
    Bhat = W.T @ (tauhat - g)
    target_b, eta_g, Ghat = target_error_terms(V, qhat, m, H, alpha/4, alpha/4)
    radius = source_radius + target_b
    return {"Bhat": Bhat, "Ghat": Ghat, "eta_g": eta_g, "radius": radius,
            "W": W, "source_radius": source_radius, "bias_radius": bias, "Cov": Cov}


def robust_qp(Bhat: np.ndarray, Ghat: np.ndarray, radius: float,
              eta_g: float, simplex: bool = True) -> tuple[np.ndarray, float]:
    d = len(Bhat)
    Gup = (Ghat + Ghat.T) / 2 + eta_g * np.eye(d) + 1e-9 * np.eye(d)
    def lower(theta: np.ndarray) -> float:
        qnorm = math.sqrt(max(float(theta @ Gup @ theta), 0.0))
        return float(2 * Bhat @ theta - theta @ Gup @ theta - 2 * radius * qnorm)
    constraints = []
    if simplex:
        constraints.append({"type": "ineq", "fun": lambda th: 1.0 - np.sum(th)})
    res = minimize(lambda th: -lower(th), np.zeros(d), method="SLSQP",
                   bounds=[(0.0, 1.0)] * d, constraints=constraints,
                   options={"ftol": 1e-11, "maxiter": 1000})
    theta = np.asarray(res.x if res.success else np.zeros(d))
    value = lower(theta)
    if value <= 0:
        theta[:] = 0.0
        value = 0.0
    return theta, value


def single_aml_gate(v: np.ndarray, qhat: np.ndarray, tauhat: np.ndarray,
                    se: np.ndarray, g: np.ndarray, R: float, H: float,
                    m: int, alpha: float, safe: bool = True) -> tuple[np.ndarray, float, bool]:
    est = aml_vector_estimate(v[:, None], qhat, tauhat, se, g, R, H, m, alpha)
    B = float(est["Bhat"][0]); A = float(est["Ghat"][0, 0] + est["eta_g"])
    if safe:
        BL = B - est["radius"]
        t = max(0.0, min(1.0, BL / max(A, 1e-12)))
        lower = max(0.0, 2 * t * BL - t * t * A)
    else:
        t = max(0.0, min(1.0, B / max(float(est["Ghat"][0, 0]), 1e-12)))
        lower = float("nan")
    return t * v, lower, bool(t > 0)


def estimate_aipw_B(v: np.ndarray, dgp: DGP, data: dict[str, np.ndarray], ix: np.ndarray,
                    nuisance: dict[str, np.ndarray], p_hat: np.ndarray, qhat: np.ndarray,
                    ratio_cap: float) -> tuple[float, float]:
    K = len(dgp.x)
    k = data["k"][ix]
    phi = pseudo_outcome(data, ix, nuisance)
    tau_plugin = nuisance["tau"]
    ratio = np.minimum(qhat / np.maximum(p_hat, 1e-8), ratio_cap)
    src = ratio[k] * v[k] * (phi - tau_plugin[k])
    target_values = v * (tau_plugin - dgp.g)
    Bhat = float(np.dot(qhat, target_values) + src.mean())
    se_src = np.var(src, ddof=1) / len(src) if len(src) > 1 else 0.0
    # target empirical variance approximation using qhat categories
    mean_t = float(np.dot(qhat, target_values))
    se_tar = float(np.dot(qhat, (target_values - mean_t) ** 2)) / max(1, int(round(len(ix))))
    return Bhat, math.sqrt(max(se_src + se_tar, 1e-14))


def aipw_gate(v: np.ndarray, dgp: DGP, data: dict[str, np.ndarray], ix: np.ndarray,
              nuisance: dict[str, np.ndarray], p_hat: np.ndarray, qhat: np.ndarray,
              alpha: float, ratio_cap: float, safe: bool = True) -> tuple[np.ndarray, float, bool]:
    Bhat, se = estimate_aipw_B(v, dgp, data, ix, nuisance, p_hat, qhat, ratio_cap)
    Ahat = float(np.dot(qhat, v**2))
    if safe:
        BL = Bhat - norm.ppf(1 - alpha) * se
        t = max(0.0, min(1.0, BL / max(Ahat, 1e-12)))
        lower = max(0.0, 2 * t * BL - t*t*Ahat)
    else:
        t = max(0.0, min(1.0, Bhat / max(Ahat, 1e-12)))
        lower = float("nan")
    return t * v, lower, bool(t > 0)


def weighted_qaggregation(F: np.ndarray, y: np.ndarray, weights: np.ndarray,
                          nu: float = 0.5) -> np.ndarray:
    n, M = F.shape
    w = np.maximum(weights, 0.0)
    w = w / max(w.mean(), 1e-12)
    ploss = np.mean(w[:, None] * (y[:, None] - F) ** 2, axis=0)
    def qfun(a: np.ndarray) -> float:
        pred = F @ a
        loss = np.mean(w * (y - pred) ** 2)
        return float((1 - nu) * loss + nu * np.dot(a, ploss))
    res = minimize(qfun, np.ones(M) / M, method="SLSQP",
                   bounds=[(0.0, 1.0)] * M,
                   constraints=[{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}],
                   options={"ftol": 1e-12, "maxiter": 1500})
    return np.asarray(res.x if res.success else np.ones(M) / M)


def select_candidate_aipw(V: np.ndarray, dgp: DGP, data: dict[str, np.ndarray], ix: np.ndarray,
                          nuisance: dict[str, np.ndarray], p_hat: np.ndarray,
                          qhat: np.ndarray, ratio_cap: float,
                          subset: Iterable[int] | None = None) -> tuple[int, np.ndarray]:
    ids = list(range(V.shape[1])) if subset is None else list(subset)
    gains = []
    for j in ids:
        B, _ = estimate_aipw_B(V[:, j], dgp, data, ix, nuisance, p_hat, qhat, ratio_cap)
        A = float(np.dot(qhat, V[:, j] ** 2))
        gains.append(2 * B - A)
    loc = int(np.argmax(gains))
    return ids[loc], np.asarray(gains)


def fit_full_krr(data: dict[str, np.ndarray], dgp: DGP, nuisance: dict[str, np.ndarray],
                 qhat_dev: np.ndarray, p_hat: np.ndarray, ratio_cap: float) -> np.ndarray:
    # Reference unrestricted learner: KRR on pseudo outcomes using all labels.
    ix = np.arange(len(data["k"]))
    phi = pseudo_outcome(data, ix, nuisance)
    xobs = dgp.x[data["k"]][..., None]
    xgrid = dgp.x[..., None]
    best = None
    for gamma in [20.0, 60.0, 160.0]:
        for lam in [0.05, 0.2, 1.0]:
            model = KernelRidge(alpha=lam, kernel="rbf", gamma=gamma).fit(xobs, phi)
            pred = np.clip(model.predict(xgrid), -0.8, 0.8)
            # deterministic source-target plugin criterion, no target labels
            score = float(np.dot(qhat_dev, (pred - nuisance["tau"]) ** 2))
            if best is None or score < best[0]:
                best = (score, pred)
    return np.asarray(best[1])


def risk_change(update: np.ndarray, dgp: DGP) -> float:
    return float(np.dot(dgp.q, (dgp.g + update - dgp.tau) ** 2 - (dgp.g - dgp.tau) ** 2))


def oracle_candidate_gain(V: np.ndarray, dgp: DGP) -> float:
    best = 0.0
    for j in range(V.shape[1]):
        A = float(np.dot(dgp.q, V[:, j] ** 2))
        B = float(np.dot(dgp.q, V[:, j] * (dgp.tau - dgp.g)))
        if A > 1e-12:
            t = max(0.0, min(1.0, B / A))
            best = max(best, 2*t*B - t*t*A)
    return best


def evaluate_method(method: str, update: np.ndarray, dgp: DGP, cert_lower: float,
                    accepted: bool, opportunity: bool, elapsed: float,
                    theory_condition: bool) -> dict:
    rc = risk_change(update, dgp)
    gain = -rc
    if method in SAFE_METHODS:
        coverage = float(gain + 1e-10 >= cert_lower)
    else:
        coverage = np.nan
    return {
        "method": method,
        "risk_change": rc,
        "target_risk": float(np.dot(dgp.q, (dgp.g + update - dgp.tau) ** 2)),
        "harm": float(rc > 1e-10),
        "harm_size": max(rc, 0.0),
        "accepted": float(accepted),
        "coverage": coverage,
        "detected_benefit": float(opportunity and accepted),
        "opportunity": float(opportunity),
        "certified_lower_gain": cert_lower if np.isfinite(cert_lower) else np.nan,
        "runtime_sec": elapsed,
        "theory_condition": float(theory_condition),
    }


def one_replicate(config: Config, scenario: str, n: int, rep: int) -> list[dict]:
    seed = config.seed + 100000 * ["direction_favorable", "regular_overlap", "null_update", "radius_misspecified"].index(scenario) + 1000 * n + rep
    rng = np.random.default_rng(seed)
    dgp = make_dgp(scenario, config.K, config.class_radius)
    data = draw_source(rng, dgp, n)
    m = int(round(config.target_ratio * n))
    target = draw_target(rng, dgp, m)
    i1, i2, i3 = split_indices(rng, n)
    tperm = rng.permutation(m)
    td = target[tperm[:m//2]]
    tc = target[tperm[m//2:]]
    qdev = counts_prob(td, config.K)
    qcert = counts_prob(tc, config.K)
    nuisance = fit_nuisance(data, i1, config.K)
    p_hat = counts_prob(data["k"][i1], config.K)
    V, candidate_names, krr_ids = make_candidates(dgp, nuisance, p_hat)
    opportunity = oracle_candidate_gain(V, dgp) > 5e-4
    rows: list[dict] = []

    def record(name: str, update: np.ndarray, lower: float, accepted: bool, start: float) -> None:
        row = evaluate_method(name, update, dgp, lower, accepted, opportunity,
                              time.perf_counter() - start, dgp.theory_condition)
        row.update({"scenario": scenario, "n": n, "rep": rep, "seed": seed,
                    "M": V.shape[1], "candidate_names": "|".join(candidate_names),
                    "true_residual_norm": class_radius_true(dgp),
                    "declared_radius": dgp.class_radius})
        rows.append(row)

    # 1. no update
    st = time.perf_counter()
    record("no_update", np.zeros(config.K), 0.0, False, st)

    # 2. unrestricted retraining reference
    st = time.perf_counter()
    nuisance_all = fit_nuisance(data, np.arange(n), config.K)
    retrained = np.clip(fit_full_krr(data, dgp, nuisance_all, qdev, p_hat, config.ratio_cap) - dgp.g,
                        -config.max_direction, config.max_direction)
    record("unrestricted_retrain", retrained, np.nan, bool(np.linalg.norm(retrained) > 0), st)

    # Selection on D2 for AIPW/single/KRR
    j_aipw, gains = select_candidate_aipw(V, dgp, data, i2, nuisance, p_hat, qdev, config.ratio_cap)

    # 3. transport AIPW + common alpha gate on D3
    st = time.perf_counter()
    upd, lower, acc = aipw_gate(V[:, j_aipw], dgp, data, i3, nuisance, p_hat, qcert,
                                config.alpha, config.ratio_cap, safe=True)
    record("transport_aipw_safe", upd, lower, acc, st)

    # D3 sufficient statistics for AML methods
    tau3, se3 = stratum_contrasts(data, i3, config.K)

    # 4. AML/Riesz point update without gate
    st = time.perf_counter()
    upd, lower, acc = single_aml_gate(V[:, j_aipw], qcert, tau3, se3, dgp.g,
                                      dgp.class_radius, dgp.target_only_bound,
                                      len(tc), config.alpha, safe=False)
    record("aml_point", upd, lower, acc, st)

    # 5. best single safe ray (D2 selection, one D3 gate)
    st = time.perf_counter()
    j_single = int(np.argmax(gains))
    upd, lower, acc = single_aml_gate(V[:, j_single], qcert, tau3, se3, dgp.g,
                                      dgp.class_radius, dgp.target_only_bound,
                                      len(tc), config.alpha, safe=True)
    record("best_single_safe", upd, lower, acc, st)

    # 6. official Q-aggregation objective, transported with D2 weights, then common D3 gate
    st = time.perf_counter()
    k2 = data["k"][i2]
    phi2 = pseudo_outcome(data, i2, nuisance)
    r2 = np.minimum(qdev / np.maximum(p_hat, 1e-8), config.ratio_cap)[k2]
    F2 = dgp.g[k2, None] + V[k2, :]
    qa = weighted_qaggregation(F2, phi2, r2, nu=0.5)
    vqa = V @ qa
    upd, lower, acc = single_aml_gate(vqa, qcert, tau3, se3, dgp.g,
                                      dgp.class_radius, dgp.target_only_bound,
                                      len(tc), config.alpha, safe=True)
    record("qaggregation_safe", upd, lower, acc, st)

    # 7. COKE-style KRR transfer: select only KRR candidates on D2, then identical gate
    st = time.perf_counter()
    if not krr_ids:
        krr_ids = [j_aipw]
    j_krr, _ = select_candidate_aipw(V, dgp, data, i2, nuisance, p_hat, qdev,
                                      config.ratio_cap, subset=krr_ids)
    upd, lower, acc = single_aml_gate(V[:, j_krr], qcert, tau3, se3, dgp.g,
                                      dgp.class_radius, dgp.target_only_bound,
                                      len(tc), config.alpha, safe=True)
    record("krr_transfer_safe", upd, lower, acc, st)

    # 8. MCSA: top d D2 directions, joint D3 confidence ball and robust QP
    st = time.perf_counter()
    top = np.argsort(gains)[::-1][:min(config.top_d, V.shape[1])]
    Vtop = V[:, top]
    est = aml_vector_estimate(Vtop, qcert, tau3, se3, dgp.g, dgp.class_radius,
                              dgp.target_only_bound, len(tc), config.alpha)
    theta, lower = robust_qp(est["Bhat"], est["Ghat"], est["radius"], est["eta_g"], simplex=True)
    upd = Vtop @ theta
    record("mcsa", upd, lower, bool(np.linalg.norm(theta) > 1e-10), st)

    return rows


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["target_risk", "risk_change", "harm", "harm_size", "coverage",
               "accepted", "detected_benefit", "runtime_sec"]
    summary_rows = []
    pair_rows = []
    for (scenario, n, method), g in raw.groupby(["scenario", "n", "method"], sort=True):
        row = {"scenario": scenario, "n": n, "method": method, "reps": len(g)}
        for metric in metrics:
            vals = g[metric].dropna().to_numpy(float)
            row[f"mean_{metric}"] = float(vals.mean()) if len(vals) else np.nan
            row[f"mcse_{metric}"] = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else np.nan
        harms = g.loc[g["harm"] > 0, "harm_size"].to_numpy(float)
        row["mean_positive_harm"] = float(harms.mean()) if len(harms) else 0.0
        row["max_positive_harm"] = float(harms.max()) if len(harms) else 0.0
        row["opportunity_rate"] = float(g["opportunity"].mean())
        summary_rows.append(row)

    pivot = raw.pivot_table(index=["scenario", "n", "rep"], columns="method", values="risk_change")
    for scenario in sorted(raw.scenario.unique()):
        for n in sorted(raw.n.unique()):
            sub = pivot.loc[(scenario, n)]
            if "mcsa" not in sub:
                continue
            for method in METHODS:
                if method not in sub:
                    continue
                diff = (sub[method] - sub["mcsa"]).dropna().to_numpy(float)
                mean = float(diff.mean())
                se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else np.nan
                pair_rows.append({"scenario": scenario, "n": n, "method_minus_mcsa": method,
                                  "reps": len(diff), "mean_difference": mean,
                                  "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se})
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/eight_method"))
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--sizes", type=int, nargs="+", default=[800, 1600, 3200])
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    config = Config(reps=args.reps, seed=args.seed)
    scenarios = ["direction_favorable", "regular_overlap", "null_update", "radius_misspecified"]
    all_rows = []
    started = time.perf_counter()
    for scenario in scenarios:
        for n in args.sizes:
            for rep in range(args.reps):
                all_rows.extend(one_replicate(config, scenario, n, rep))
            print(f"completed scenario={scenario} n={n}", flush=True)
    raw = pd.DataFrame(all_rows)
    summary, paired = summarize(raw)
    args.out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out / "replications.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    paired.to_csv(args.out / "paired_vs_mcsa.csv", index=False)
    metadata = asdict(config) | {
        "sizes": args.sizes,
        "scenarios": scenarios,
        "methods": METHODS,
        "elapsed_sec": time.perf_counter() - started,
        "important_note": (
            "Causal Q-aggregation uses the official Q-aggregation objective adapted with target weights. "
            "krr_transfer_safe is a COKE-style KRR transfer comparator, not an exact reproduction of COKE. "
            "unrestricted_retrain and aml_point are intentionally unsafe performance references."
        ),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
