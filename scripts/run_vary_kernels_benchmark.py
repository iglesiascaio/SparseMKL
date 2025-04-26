# benchmarks_vary_kernels.py
#   Runs EasyMKL, AverageMKL, and CKA for q = 20…200 (step 20) on
#   the four datasets used in the Julia experiment.  The kernel list
#   grows cumulatively with q, exactly like in the Julia code.

import os, sys, time, math, json, random
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# 1)  PyJulia to fetch datasets                                      #
# ------------------------------------------------------------------ #
from julia.api import Julia

Julia(compiled_modules=False)
from julia import Main, Base

Main.include("../data/get_data.jl")  # adjust path as needed


def get_julia_dataset(name, frac=1.0, train_ratio=0.8, seed=123):
    Main.eval(f"using Random; Random.seed!({seed})")
    X_tr, y_tr, X_te, y_te = Main.GetData.get_dataset(
        Base.Symbol(name), force_download=False, frac=frac, train_ratio=train_ratio
    )
    return (np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te))


# ------------------------------------------------------------------ #
# 2)  Benchmarks + kernels                                           #
# ------------------------------------------------------------------ #
from MKLpy.algorithms import EasyMKL, AverageMKL, CKA
from MKLpy.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.metrics.pairwise import sigmoid_kernel, laplacian_kernel
import torch

BENCHMARKS = {
    "EasyMKL": EasyMKL,
    "AverageMKL": AverageMKL,
    "CKA": CKA,
}

DATASETS = ["haberman", "mammographic", "ionosphere", "parkinsons"]
Q_LIST = list(range(20, 201, 20))
TOL_BETA = 1e-6
RNG_SEED = 42
OUT_CSV = "../results/results_benchmarks_vary_kernels.csv"
# EasyMKL tuned λ values
BEST_PARAM_CSV = "../results/final_benchmark_results_mklpy.csv"
best_param_df = pd.read_csv(BEST_PARAM_CSV).set_index("Dataset")


# ------------------------------------------------------------------ #
# 3)  Sampling & cumulative kernel logic (mirrors Julia)             #
# ------------------------------------------------------------------ #
def sample_kernel(rng):
    kind = rng.choice(["linear", "poly", "rbf", "sigmoid", "laplacian"])
    if kind == "linear":
        return dict(type="linear", params={})
    if kind == "poly":
        return dict(
            type="poly", params=dict(degree=rng.integers(2, 7), coef0=rng.random() * 2)
        )
    if kind == "rbf":
        return dict(type="rbf", params=dict(gamma=10 ** (rng.random() * 3 - 2)))
    if kind == "sigmoid":
        return dict(
            type="sigmoid",
            params=dict(gamma=rng.random() * 0.9 + 0.1, coef0=rng.random() * 4 - 2),
        )
    # laplacian
    return dict(type="laplacian", params=dict(gamma=10 ** (rng.random() * 3 - 2)))


def extend_kernel_bank(bank, q_target, rng):
    has_linear = any(k["type"] == "linear" for k in bank)
    while len(bank) < q_target:
        kdict = sample_kernel(rng)
        if kdict["type"] == "linear" and has_linear:
            continue
        if kdict not in bank:
            bank.append(kdict)
            has_linear |= kdict["type"] == "linear"
    return bank


def compute_kernels(X_tr, X_te, spec, jitter=1e-6):
    Ktr, Kte = [], []
    for d in spec:
        tp, p = d["type"], d["params"]
        if tp == "linear":
            K1 = linear_kernel(X_tr, X_tr)
            K2 = linear_kernel(X_tr, X_te)
        elif tp == "poly":
            K1 = polynomial_kernel(X_tr, X_tr, **p)
            K2 = polynomial_kernel(X_tr, X_te, **p)
        elif tp == "rbf":
            K1 = rbf_kernel(X_tr, X_tr, **p)
            K2 = rbf_kernel(X_tr, X_te, **p)
        elif tp == "sigmoid":
            K1 = sigmoid_kernel(X_tr, X_tr, **p)
            K2 = sigmoid_kernel(X_tr, X_te, **p)
        else:  # laplacian
            K1 = laplacian_kernel(X_tr, X_tr, **p)
            K2 = laplacian_kernel(X_tr, X_te, **p)

        K1 = torch.tensor(K1, dtype=torch.float64).clone()
        K2 = torch.tensor(K2, dtype=torch.float64).clone()
        K1 += jitter * torch.eye(K1.shape[0])
        Ktr.append(K1)
        Kte.append(K2)
    return Ktr, Kte


# ------------------------------------------------------------------ #
# 4)  Metrics & baseline                                             #
# ------------------------------------------------------------------ #
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def metrics(y, yhat):
    return (
        accuracy_score(y, yhat),
        precision_score(y, yhat, pos_label=1, zero_division=0),
        recall_score(y, yhat, pos_label=1, zero_division=0),
        f1_score(y, yhat, pos_label=1, zero_division=0),
    )


def majority_baseline(y_tr, y_te):
    maj = 1 if (y_tr == 1).sum() >= (y_tr == -1).sum() else -1
    return np.full_like(y_te, maj)


# ------------------------------------------------------------------ #
# 5)  Result frame identical to Julia + Method column                #
# ------------------------------------------------------------------ #
COLS = [
    "Dataset",
    "TrainSize",
    "TestSize",
    "q",
    "k",
    "NumSelectedKernels",
    "MKL_TrainAccuracy",
    "MKL_TestAccuracy",
    "MKL_Precision",
    "MKL_Recall",
    "MKL_F1_Score",
    "MKL_Objective",
    "Baseline_Accuracy",
    "Baseline_Precision",
    "Baseline_Recall",
    "Baseline_F1_Score",
    "Betas",
    "MKL_BestC",
    "MKL_BestLambda",
    "MKL_FitTime",
    "Status",
    "Method",
]
results = []

# ------------------------------------------------------------------ #
# 6)  Main loop                                                      #
# ------------------------------------------------------------------ #
rng_global = np.random.default_rng(RNG_SEED)

for ds in DATASETS:
    print(f"\n================ Dataset: {ds} ================")
    X_tr, y_tr, X_te, y_te = get_julia_dataset(ds, seed=RNG_SEED)
    base_acc, base_prec, base_rec, base_f1 = metrics(
        y_te, majority_baseline(y_tr, y_te)
    )

    rng_ds = np.random.default_rng(hash((RNG_SEED, ds)) & 0xFFFFFFFF)
    kernels_cum = []  # grows with q

    for q in Q_LIST:
        extend_kernel_bank(kernels_cum, q, rng_ds)
        K_train, K_test = compute_kernels(X_tr, X_te, kernels_cum)

        for meth_name, MKLClass in BENCHMARKS.items():

            # parameter for EasyMKL
            lam = None
            if meth_name == "EasyMKL":
                try:
                    lam = best_param_df.loc[ds, "MKL1_Param"]
                except KeyError:
                    lam = 0.1
            start = time.time()
            model = (
                MKLClass(lam=lam, max_iter=500, tolerance=1e-5)
                if lam is not None
                else MKLClass(max_iter=500, tolerance=1e-5)
            )
            model.fit(K_train, y_tr.ravel())
            fit_time = time.time() - start

            yhat_tr = model.predict([K.T for K in K_train])
            yhat_te = model.predict([K.T for K in K_test])

            acc_tr, *_ = metrics(y_tr, yhat_tr)
            acc_te, prec, rec, f1 = metrics(y_te, yhat_te)

            try:
                betas = model.solution.weights
            except Exception:
                betas = np.array([])
            nz = (np.abs(betas) > TOL_BETA).sum()

            results.append(
                [
                    ds,
                    len(y_tr),
                    len(y_te),
                    q,
                    np.nan,
                    int(nz),
                    acc_tr,
                    acc_te,
                    prec,
                    rec,
                    f1,
                    np.nan,
                    base_acc,
                    base_prec,
                    base_rec,
                    base_f1,
                    ", ".join(f"{b:.4f}" for b in betas),
                    np.nan,
                    np.nan,
                    fit_time,
                    "Success",
                    meth_name,
                ]
            )
            print(
                f"  [{meth_name:9s}] q={q:3d}  test={acc_te:.3f}  nz={nz}  t={fit_time:.2f}s"
            )

# ------------------------------------------------------------------ #
# 7)  Save                                                           #
# ------------------------------------------------------------------ #
df = pd.DataFrame(results, columns=COLS)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nAll experiments finished – results saved to {OUT_CSV}\n")
