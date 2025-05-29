import numpy as np
import pandas as pd

# 1) PyJulia / Julia imports
from julia.api import Julia

Julia(compiled_modules=False)  # ensures a “safe” load
from julia import Main, Base

import sys

sys.stdout.reconfigure(line_buffering=True)

# 2) Include your get_data.jl file (which defines the module GetData)
Main.include("../data/get_data.jl")  # adjust path as needed


###############################################################################
# 3) Wrapping your Julia get_dataset function in a Python function
###############################################################################
def get_julia_dataset(ds_name, frac=1.0, train_ratio=0.8, seed=123):
    """
    Calls the Julia function GetData.get_dataset(ds_name; frac=..., train_ratio=..., etc.)
    and converts the resulting Julia arrays into NumPy arrays.

    Returns (X_train, y_train, X_test, y_test) as NumPy arrays with shape/length as needed.
    """
    # We can set the Julia random seed if your get_data code depends on that.
    Main.eval(f"using Random; Random.seed!({seed})")

    ds_symbol = Base.Symbol(ds_name)  # Convert Python string -> Julia Symbol

    julia_tuple = Main.GetData.get_dataset(
        ds_symbol, force_download=False, frac=frac, train_ratio=train_ratio
    )

    X_train_j, y_train_j, X_test_j, y_test_j = julia_tuple
    X_train = np.array(X_train_j)
    y_train = np.array(y_train_j)
    X_test = np.array(X_test_j)
    y_test = np.array(y_test_j)

    return X_train, y_train, X_test, y_test


###############################################################################
# 4) Other necessary imports (MKLpy, scikit-learn, etc.)
###############################################################################
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

# MKLpy
from MKLpy.algorithms import EasyMKL, AverageMKL, CKA
from MKLpy.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

# If sigmoid_kernel, laplacian_kernel are missing in MKLpy, we can use scikit-learn’s versions:
from sklearn.metrics.pairwise import sigmoid_kernel, laplacian_kernel

import torch
import time

###############################################################################
# 5) Kernel specification and computation
###############################################################################
kernels_spec = [
    {"type": "linear", "params": {}},
    {"type": "poly", "params": {"degree": 2, "coef0": 1.0}},
    {"type": "poly", "params": {"degree": 3, "coef0": 1.0}},
    {"type": "poly", "params": {"degree": 5, "coef0": 1.0}},
    {"type": "rbf", "params": {"gamma": 0.5}},
    {"type": "rbf", "params": {"gamma": 0.3}},
    {"type": "rbf", "params": {"gamma": 0.1}},
    {"type": "sigmoid", "params": {"gamma": 0.5, "coef0": 1.0}},
    {"type": "sigmoid", "params": {"gamma": 0.7, "coef0": 1.0}},
    {"type": "laplacian", "params": {"gamma": 0.3}},
]


def compute_kernels(X_train, X_test, kernels_spec, jitter=1e-6):
    """
    Compute kernel matrices and add jitter for numerical stability.
    """
    K_train_list = []
    K_test_list = []

    for kdict in kernels_spec:
        ktype = kdict["type"]
        params = kdict["params"]

        if ktype == "linear":
            K_tr = linear_kernel(X_train, X_train)
            K_te = linear_kernel(X_train, X_test)
        elif ktype == "poly":
            K_tr = polynomial_kernel(X_train, X_train, **params)
            K_te = polynomial_kernel(X_train, X_test, **params)
        elif ktype == "rbf":
            K_tr = rbf_kernel(X_train, X_train, **params)
            K_te = rbf_kernel(X_train, X_test, **params)
        elif ktype == "sigmoid":
            K_tr = sigmoid_kernel(X_train, X_train, **params)
            K_te = sigmoid_kernel(X_train, X_test, **params)
        elif ktype == "laplacian":
            K_tr = laplacian_kernel(X_train, X_train, **params)
            K_te = laplacian_kernel(X_train, X_test, **params)
        else:
            raise ValueError(f"Unknown kernel type: {ktype}")

        # Convert to PyTorch tensors
        K_tr = torch.tensor(K_tr, dtype=torch.float64, device="cpu").clone().detach()
        K_te = torch.tensor(K_te, dtype=torch.float64, device="cpu").clone().detach()

        #  Add small jitter to diagonal for numerical stability
        K_tr += jitter * torch.eye(K_tr.shape[0])

        print(
            f"Kernel Type: {ktype} | K_train Shape: {K_tr.shape} | K_test Shape: {K_te.shape}"
        )

        K_train_list.append(K_tr)
        K_test_list.append(K_te)

    return K_train_list, K_test_list


###############################################################################
# 6) Utilities: metrics, cross-validation, etc.
###############################################################################
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    return acc, prec, rec, f1


def kfold_indices(n, k=5, shuffle=True, seed=123):
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
    folds = []
    for _, val_idx in kf.split(range(n)):
        folds.append(val_idx)
    return folds


Cs_range = [5.0, 10.0, 50.0, 100.0]
lambdas_range = [
    0.0001,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    1.0,
]


def cross_validate_mkl(
    X, y, kernel_specs, MKLClass, param_name=None, param_values=[None], n_folds=10
):
    """
    For algorithms with a single hyperparam 'lam' or 'C', or none at all.
    If a fold fails (raises), it’s skipped; if all folds for a given val fail,
    we ignore that val entirely.
    """
    folds = kfold_indices(len(y), k=n_folds, shuffle=True, seed=123)
    best_acc = -999
    best_param = None

    for val in param_values:
        fold_accuracies = []

        for fold_idx in range(n_folds):
            try:
                # split
                val_idx = folds[fold_idx]
                train_idx = np.hstack(
                    [folds[i] for i in range(n_folds) if i != fold_idx]
                )

                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                # compute kernels
                K_tr_list, K_val_list = compute_kernels(X_tr, X_val, kernel_specs)

                # instantiate
                if param_name is None or val is None:
                    mkl_model = MKLClass(max_iter=500, tolerance=1e-5)
                else:
                    if param_name == "lam":
                        mkl_model = MKLClass(lam=val, max_iter=500, tolerance=1e-5)
                    elif param_name == "C":
                        mkl_model = MKLClass(C=val, max_iter=500, tolerance=1e-5)
                    else:
                        raise ValueError(f"Unsupported param={param_name}")

                # fit & eval
                mkl_model.fit(K_tr_list, y_tr.ravel())
                y_val_pred = mkl_model.predict([K.T for K in K_val_list])
                acc, _, _, _ = compute_metrics(y_val, y_val_pred)
                fold_accuracies.append(acc)

            except Exception:
                # skip this fold if anything goes wrong
                continue

        # if no folds succeeded for this val, skip it
        if not fold_accuracies:
            continue

        mean_acc = np.mean(fold_accuracies)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_param = val

    return best_param, best_acc


def cross_validate_svm(X, y, Cs=Cs_range, n_folds=10):
    folds = kfold_indices(len(y), k=n_folds, shuffle=True, seed=123)
    best_acc = -999
    best_C = None

    for c_val in Cs:
        accs = []
        for fold_idx in range(n_folds):
            val_idx = folds[fold_idx]
            train_idx = np.hstack([folds[i] for i in range(n_folds) if i != fold_idx])

            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            clf = SVC(kernel="linear", C=c_val)
            clf.fit(X_tr, y_tr)
            y_val_pred = clf.predict(X_val)
            acc, _, _, _ = compute_metrics(y_val, y_val_pred)
            accs.append(acc)

        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_C = c_val

    return best_C, best_acc


def majority_vote_baseline(y_train, y_test):
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == -1)
    maj_label = 1 if pos_count >= neg_count else -1
    return np.full_like(y_test, maj_label)


###############################################################################
# 7) Main experiment code
###############################################################################
DATASETS = [
    "iris",
    "wine",
    "breastcancer",
    "ionosphere",
    "spambase",
    "banknote",
    "heart",
    "haberman",
    "mammographic",
    "parkinsons",
]

RESULT_COLUMNS = [
    "Dataset",
    "TrainSize",
    "TestSize",
    # EasyMKL
    "MKL1_TrainAccuracy",
    "MKL1_TestAccuracy",
    "MKL1_Precision",
    "MKL1_Recall",
    "MKL1_F1_Score",
    "MKL1_Param",
    # AverageMKL
    "MKL2_TrainAccuracy",
    "MKL2_TestAccuracy",
    "MKL2_Precision",
    "MKL2_Recall",
    "MKL2_F1_Score",
    # CKA
    "MKL3_TrainAccuracy",
    "MKL3_TestAccuracy",
    "MKL3_Precision",
    "MKL3_Recall",
    "MKL3_F1_Score",
    # Baseline
    "Baseline_Accuracy",
    "Baseline_Precision",
    "Baseline_Recall",
    "Baseline_F1_Score",
    # SVM
    "SVM_TrainAccuracy",
    "SVM_TestAccuracy",
    "SVM_Precision",
    "SVM_Recall",
    "SVM_F1_Score",
    # Additional columns
    "MKL1_Param",  # Chosen lam for EasyMKL
    "MKL1_TrainTime",  # Final fit time for EasyMKL
    "MKL2_TrainTime",  # Final fit time for AverageMKL
    "MKL3_TrainTime",  # Final fit time for CKA
    "SVM_Param",  # Chosen C for SVM
    "SVM_TrainTime",  # Final fit time for final SVM training
    "Betas_Algo1",
    "Betas_Algo2",
    "Betas_Algo3",
    "Status",
]

results_df = pd.DataFrame(columns=RESULT_COLUMNS)


def run_experiment_for_dataset(ds_name):
    print(f"\n=== Processing dataset: {ds_name} ===")
    frac = 1.0

    # -- Call the Julia code to get the dataset
    X_train, y_train, X_test, y_test = get_julia_dataset(ds_name, frac=frac, seed=123)

    train_size = len(y_train)
    test_size = len(y_test)

    # Precompute kernel lists for entire train set
    K_train_list, K_test_list = compute_kernels(X_train, X_test, kernels_spec)

    # 1) EasyMKL
    best_lam, best_cv_acc_easymkl = cross_validate_mkl(
        X_train,
        y_train,
        kernels_spec,
        MKLClass=EasyMKL,
        param_name="lam",
        param_values=lambdas_range,
        n_folds=10,
    )
    print(f"[EasyMKL] best lam={best_lam}, CV acc={best_cv_acc_easymkl:.4f}")

    # Measure final fit time
    start_time = time.time()
    mkl1 = EasyMKL(lam=best_lam, max_iter=500, tolerance=1e-5)
    mkl1.fit(K_train_list, y_train.ravel())
    end_time = time.time()
    mkl1_train_time = end_time - start_time

    y_train_pred1 = mkl1.predict([K.T for K in K_train_list])
    y_test_pred1 = mkl1.predict([K.T for K in K_test_list])

    mkl1_train_acc, mkl1_train_prec, mkl1_train_rec, mkl1_train_f1 = compute_metrics(
        y_train, y_train_pred1
    )
    mkl1_test_acc, mkl1_test_prec, mkl1_test_rec, mkl1_test_f1 = compute_metrics(
        y_test, y_test_pred1
    )

    try:
        betas_mkl1 = mkl1.solution.weights
    except:
        betas_mkl1 = np.array([])

    # 2) AverageMKL (no param)
    _, best_cv_acc_avg = cross_validate_mkl(
        X_train,
        y_train,
        kernels_spec,
        MKLClass=AverageMKL,
        param_name=None,
        param_values=[None],
        n_folds=10,
    )
    print(f"[AverageMKL] CV acc={best_cv_acc_avg:.4f}")

    # Measure final fit time
    start_time = time.time()
    mkl2 = AverageMKL(max_iter=500, tolerance=1e-5)
    mkl2.fit(K_train_list, y_train.ravel())
    end_time = time.time()
    mkl2_train_time = end_time - start_time

    y_train_pred2 = mkl2.predict([K.T for K in K_train_list])
    y_test_pred2 = mkl2.predict([K.T for K in K_test_list])

    mkl2_train_acc, mkl2_train_prec, mkl2_train_rec, mkl2_train_f1 = compute_metrics(
        y_train, y_train_pred2
    )
    mkl2_test_acc, mkl2_test_prec, mkl2_test_rec, mkl2_test_f1 = compute_metrics(
        y_test, y_test_pred2
    )

    try:
        betas_mkl2 = mkl2.solution.weights
    except:
        betas_mkl2 = np.array([])

    # 3) CKA (no param)
    _, best_cv_acc_cka = cross_validate_mkl(
        X_train,
        y_train,
        kernels_spec,
        MKLClass=CKA,
        param_name=None,
        param_values=[None],
        n_folds=10,
    )
    print(f"[CKA] CV acc={best_cv_acc_cka:.4f}")

    # Measure final fit time
    start_time = time.time()
    mkl3 = CKA(max_iter=500, tolerance=1e-5)
    mkl3.fit(K_train_list, y_train)
    end_time = time.time()
    mkl3_train_time = end_time - start_time

    y_train_pred3 = mkl3.predict([K.T for K in K_train_list])
    y_test_pred3 = mkl3.predict([K.T for K in K_test_list])

    mkl3_train_acc, mkl3_train_prec, mkl3_train_rec, mkl3_train_f1 = compute_metrics(
        y_train, y_train_pred3
    )
    mkl3_test_acc, mkl3_test_prec, mkl3_test_rec, mkl3_test_f1 = compute_metrics(
        y_test, y_test_pred3
    )

    try:
        betas_mkl3 = mkl3.solution.weights
    except:
        betas_mkl3 = np.array([])

    # 4) Baseline
    y_test_pred_baseline = majority_vote_baseline(y_train, y_test)
    base_acc, base_prec, base_rec, base_f1 = compute_metrics(
        y_test, y_test_pred_baseline
    )

    # 5) SVM cross-validation
    best_C_svm, best_acc_svm = cross_validate_svm(
        X_train, y_train, Cs=Cs_range, n_folds=10
    )
    print(f"[SVM] best C={best_C_svm}, CV acc={best_acc_svm:.4f}")

    # Measure final fit time
    start_time = time.time()
    svm_clf = SVC(kernel="linear", C=best_C_svm)
    svm_clf.fit(X_train, y_train)
    end_time = time.time()
    svm_train_time = end_time - start_time

    y_train_pred_svm = svm_clf.predict(X_train)
    y_test_pred_svm = svm_clf.predict(X_test)

    svm_train_acc, _, _, _ = compute_metrics(y_train, y_train_pred_svm)
    svm_test_acc, svm_test_prec, svm_test_rec, svm_test_f1 = compute_metrics(
        y_test, y_test_pred_svm
    )

    # 6) Prepare the result dict
    def betas_to_str(betas):
        if betas.size == 0:
            return ""
        return ", ".join(f"{b:.4f}" for b in betas)

    row = {
        "Dataset": ds_name,
        "TrainSize": train_size,
        "TestSize": test_size,
        # EasyMKL
        "MKL1_TrainAccuracy": mkl1_train_acc,
        "MKL1_TestAccuracy": mkl1_test_acc,
        "MKL1_Precision": mkl1_test_prec,
        "MKL1_Recall": mkl1_test_rec,
        "MKL1_F1_Score": mkl1_test_f1,
        "MKL1_Param": best_lam,
        # AverageMKL
        "MKL2_TrainAccuracy": mkl2_train_acc,
        "MKL2_TestAccuracy": mkl2_test_acc,
        "MKL2_Precision": mkl2_test_prec,
        "MKL2_Recall": mkl2_test_rec,
        "MKL2_F1_Score": mkl2_test_f1,
        # CKA
        "MKL3_TrainAccuracy": mkl3_train_acc,
        "MKL3_TestAccuracy": mkl3_test_acc,
        "MKL3_Precision": mkl3_test_prec,
        "MKL3_Recall": mkl3_test_rec,
        "MKL3_F1_Score": mkl3_test_f1,
        # Baseline
        "Baseline_Accuracy": base_acc,
        "Baseline_Precision": base_prec,
        "Baseline_Recall": base_rec,
        "Baseline_F1_Score": base_f1,
        # SVM
        "SVM_TrainAccuracy": svm_train_acc,
        "SVM_TestAccuracy": svm_test_acc,
        "SVM_Precision": svm_test_prec,
        "SVM_Recall": svm_test_rec,
        "SVM_F1_Score": svm_test_f1,
        "MKL1_Param": best_lam,  # Param selected by CV for EasyMKL
        "MKL1_TrainTime": mkl1_train_time,
        "MKL2_TrainTime": mkl2_train_time,
        "MKL3_TrainTime": mkl3_train_time,
        "SVM_Param": best_C_svm,  # Param selected by CV for SVM
        "SVM_TrainTime": svm_train_time,
        "Betas_Algo1": betas_to_str(betas_mkl1),
        "Betas_Algo2": betas_to_str(betas_mkl2),
        "Betas_Algo3": betas_to_str(betas_mkl3),
        "Status": "Success",
    }
    return row


def main():
    global results_df

    DATASETS = [
        "iris",
        "wine",
        "breastcancer",
        "ionosphere",
        "spambase",
        "banknote",
        "heart",
        "haberman",
        "mammographic",
        "parkinsons",
    ]

    all_rows = []
    for ds_name in DATASETS:
        try:
            row = run_experiment_for_dataset(ds_name)
            all_rows.append(row)
        except Exception as e:
            print(f"Error processing dataset {ds_name}: {e}")
            error_row = {
                "Dataset": ds_name,
                "TrainSize": 0,
                "TestSize": 0,
                "Status": "Error",
            }
            # Fill missing columns with None
            for c in RESULT_COLUMNS:
                if c not in error_row:
                    error_row[c] = None
            all_rows.append(error_row)

    results_df = pd.DataFrame(all_rows, columns=RESULT_COLUMNS)

    out_csv = "results_mklpy.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n=== Results saved to {out_csv} ===\n")

    # Summaries
    success_df = results_df[results_df["Status"] == "Success"]
    error_df = results_df[results_df["Status"] == "Error"]
    print("=== Successful Datasets ===")
    print(success_df)
    print("\n=== Datasets with Errors ===")
    print(error_df["Dataset"].tolist())


if __name__ == "__main__":
    main()
