###############################################################################
# benchmarks_vary_kernels.py
#   Runs EasyMKL, AverageMKL, and CKA with the **exact same kernel
#   dictionaries produced by the Julia experiment**.  Those kernels were
#   saved in  ../results/kernels_used_vary_q.csv  by the updated Julia script.
###############################################################################

import os, sys, time, json, numpy as np, pandas as pd

sys.stdout.reconfigure(line_buffering=True)

# ------------------------------------------------------------------ #
# 1)  Load kernels chosen by Julia                                   #
# ------------------------------------------------------------------ #
KERNEL_LOG_CSV = "../results/kernels_used_vary_q.csv"  # <-- created by Julia
kern_df = pd.read_csv(KERNEL_LOG_CSV)

# Map (dataset, q) → list[dict]
kern_bank = {
    (row.Dataset, int(row.q)): json.loads(row.KernelsJSON)
    for _, row in kern_df.iterrows()
}

# ------------------------------------------------------------------ #
# 2)  PyJulia to fetch datasets                                      #
# ------------------------------------------------------------------ #
from julia.api import Julia

Julia(compiled_modules=False)
from julia import Main, Base

Main.include("../data/get_data.jl")


def get_julia_dataset(name, frac=1.0, train_ratio=0.8, seed=123):
    Main.eval(f"using Random; Random.seed!({seed})")
    Xtr, ytr, Xte, yte = Main.GetData.get_dataset(
        Base.Symbol(name), force_download=False, frac=frac, train_ratio=train_ratio
    )
    return map(np.asarray, (Xtr, ytr, Xte, yte))


# ------------------------------------------------------------------ #
# 3)  Benchmarks & kernel helpers                                    #
# ------------------------------------------------------------------ #
from MKLpy.algorithms import EasyMKL, AverageMKL, CKA
from MKLpy.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.metrics.pairwise import sigmoid_kernel, laplacian_kernel
import torch

BENCHMARKS = {"EasyMKL": EasyMKL, "AverageMKL": AverageMKL, "CKA": CKA}
DATASETS = ["haberman", "mammographic", "ionosphere", "parkinsons"]
Q_LIST = list(range(20, 201, 20))
TOL_BETA = 1e-6
OUT_CSV = "../results/results_benchmarks_vary_kernels.csv"
BEST_PARAM_CSV = "../results/final_benchmark_results_mklpy.csv"
best_param_df = pd.read_csv(BEST_PARAM_CSV).set_index("Dataset")
RNG_SEED = 42


def _poly_params(params):
    """Julia saved :c (or :c0) – map to sklearn's coef0."""
    p = params.copy()
    if "c" in p:
        p["coef0"] = p.pop("c")
    if "c0" in p:
        p["coef0"] = p.pop("c0")
    return p


def compute_kernels(X_tr, X_te, spec, jitter=1e-6):
    Ktr, Kte = [], []
    for d in spec:
        tp, p = d["type"], d["params"]
        p = {str(k): v for k, v in p.items()}  # ensure str keys
        if tp == "linear":
            K1 = linear_kernel(X_tr, X_tr)
            K2 = linear_kernel(X_tr, X_te)
        elif tp == "polynomial":
            K1 = polynomial_kernel(X_tr, X_tr, **_poly_params(p))
            K2 = polynomial_kernel(X_tr, X_te, **_poly_params(p))
        elif tp == "poly":  # should not occur, but keep for safety
            K1 = polynomial_kernel(X_tr, X_tr, **_poly_params(p))
            K2 = polynomial_kernel(X_tr, X_te, **_poly_params(p))
        elif tp == "rbf":
            K1 = rbf_kernel(X_tr, X_tr, **p)
            K2 = rbf_kernel(X_tr, X_te, **p)
        elif tp == "sigmoid":
            K1 = sigmoid_kernel(X_tr, X_tr, **_poly_params(p))
            K2 = sigmoid_kernel(X_tr, X_te, **_poly_params(p))
        elif tp == "laplacian":
            K1 = laplacian_kernel(X_tr, X_tr, **p)
            K2 = laplacian_kernel(X_tr, X_te, **p)
        else:
            raise ValueError(f"Unknown kernel type {tp}")

        K1 = torch.tensor(K1, dtype=torch.float64)
        K1 += jitter * torch.eye(K1.shape[0])
        K2 = torch.tensor(K2, dtype=torch.float64)
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
# 5)  Result frame                                                   #
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
rows = []

# ------------------------------------------------------------------ #
# 6)  Main loop                                                      #
# ------------------------------------------------------------------ #
for ds in DATASETS:
    print(f"\n================ Dataset: {ds} ================")
    X_tr, y_tr, X_te, y_te = get_julia_dataset(ds, seed=RNG_SEED)
    base_acc, base_prec, base_rec, base_f1 = metrics(
        y_te, majority_baseline(y_tr, y_te)
    )

    for q in Q_LIST:
        kernels = kern_bank[(ds, q)]  # ← use exact kernels picked by Julia
        K_train, K_test = compute_kernels(X_tr, X_te, kernels)

        for method, MKLClass in BENCHMARKS.items():
            lam = (method == "EasyMKL" and best_param_df.loc[ds, "MKL1_Param"]) or None
            model = (
                MKLClass(lam=lam, max_iter=500, tolerance=1e-5)
                if lam is not None
                else MKLClass(max_iter=500, tolerance=1e-5)
            )

            t0 = time.time()
            model.fit(K_train, y_tr.ravel())
            fit_t = time.time() - t0
            yhat_tr = model.predict([K.T for K in K_train])
            yhat_te = model.predict([K.T for K in K_test])

            acc_tr, *_ = metrics(y_tr, yhat_tr)
            acc_te, prec, rec, f1 = metrics(y_te, yhat_te)
            betas = getattr(getattr(model, "solution", None), "weights", np.array([]))
            nz = (np.abs(betas) > TOL_BETA).sum()

            rows.append(
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
                    fit_t,
                    "Success",
                    method,
                ]
            )
            print(f"  [{method:9s}] q={q:3d} test={acc_te:.3f} nz={nz} t={fit_t:.2f}s")

# ------------------------------------------------------------------ #
# 7)  Save                                                           #
# ------------------------------------------------------------------ #
df = pd.DataFrame(rows, columns=COLS)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nAll experiments finished – results saved to {OUT_CSV}\n")
