using Profile
using Printf
using Infiltrator
using DataFrames, CSV
using LIBSVM            # For the vanilla SVM baseline
using Random
using Statistics
using Base: time        # For measuring final MKL fit runtime

# Toggle profiling on/off:
const ENABLE_PROFILING = false

macro maybe_profile(expr)
    if ENABLE_PROFILING
        return :( @profile $expr )
    else
        return expr
    end
end

include("../data/get_data.jl")
using .GetData: get_dataset

include("../src/MKL/multi_kernel.jl")
include("../src/MKL/plot_mkl.jl")
using .MKL: compute_kernels, train_mkl, compute_bias, predict_mkl, compute_combined_kernel

include("../src/Interpretable_MKL/interpretable_multi_kernel.jl")
using .InterpretableMKL: train_interpretable_mkl

################################################################################
# Data / Datasets
################################################################################
DATASETS = [
    :iris, 
    # :adult,
    :wine, 
    :breastcancer,
    :ionosphere,
    :spambase,
    :banknote,
    :heart,
    :haberman,
    :mammographic,
    :parkinsons,
]

################################################################################
# Kernel specifications for MKL
################################################################################
kernels = [
    Dict(:type => "linear", :params => Dict()),
    Dict(:type => "polynomial", :params => Dict(:degree => 2, :c => 1.0)),
    Dict(:type => "polynomial", :params => Dict(:degree => 3, :c => 1.0)),
    Dict(:type => "polynomial", :params => Dict(:degree => 5, :c => 1.0)),
    Dict(:type => "rbf",       :params => Dict(:gamma => 0.5)),
    Dict(:type => "rbf",       :params => Dict(:gamma => 0.3)),
    Dict(:type => "rbf",       :params => Dict(:gamma => 0.1)),
    Dict(:type => "sigmoid",   :params => Dict(:gamma => 0.5, :c0 => 1.0)),
    Dict(:type => "sigmoid",   :params => Dict(:gamma => 0.7, :c0 => 1.0)),
    Dict(:type => "laplacian", :params => Dict(:gamma => 0.3)),
]

################################################################################
# Hyperparameters to search
# For MKL, we do a grid over (C, λ).
# For SVM, we do a grid over (C).
################################################################################
Cs_range = [5.0, 10.0, 50.0, 100.0]
lambdas_range = [0.01, 0.1, 1.0, 10.0, 100.0]

# Additional MKL hyperparameters
k0 = 3
max_iter = 50
sum_beta_val = 1.0
tolerance = 1e-2

# Cross-validation folds
N_FOLDS = 5

################################################################################
# DataFrame to store results
# We add columns for best C/λ and the final MKL fit time
################################################################################
results = DataFrame(
    Dataset              = String[],
    TrainSize            = Int[],
    TestSize             = Int[],

    MKL_TrainAccuracy    = Float64[],
    MKL_TestAccuracy     = Float64[],
    MKL_Precision        = Float64[],
    MKL_Recall           = Float64[],
    MKL_F1_Score         = Float64[],

    Baseline_Accuracy    = Float64[],
    Baseline_Precision   = Float64[],
    Baseline_Recall      = Float64[],
    Baseline_F1_Score    = Float64[],

    SVM_TrainAccuracy    = Float64[],
    SVM_TestAccuracy     = Float64[],
    SVM_Precision        = Float64[],
    SVM_Recall           = Float64[],
    SVM_F1_Score         = Float64[],

    Betas                = String[],  # Already existing
    MKL_BestC            = Float64[], # New
    MKL_BestLambda       = Float64[], # New
    MKL_FitTime          = Float64[], # New

    Status               = String[]
)

################################################################################
# Utility: Compute classification metrics
################################################################################
function compute_metrics(y_actual, y_pred)
    TP = sum((y_actual .== 1) .& (y_pred .== 1))
    TN = sum((y_actual .== -1) .& (y_pred .== -1))
    FP = sum((y_actual .== -1) .& (y_pred .== 1))
    FN = sum((y_actual .== 1) .& (y_pred .== -1))

    accuracy  = (TP + TN) / length(y_actual)
    precision = TP / (TP + FP + 1e-9)  # Avoid division by zero
    recall    = TP / (TP + FN + 1e-9)
    f1_score  = 2 * (precision * recall) / (precision + recall + 1e-9)
    return accuracy, precision, recall, f1_score
end

################################################################################
# Utility: Create k-fold indices for cross-validation
################################################################################
function kfold_indices(n::Int, k::Int=5; shuffle::Bool=true)
    indices = collect(1:n)
    if shuffle
        shuffle!(indices)
    end
    # Partition into k roughly equal folds
    folds = [Int[] for _ in 1:k]
    for (i, idx) in enumerate(indices)
        push!(folds[mod1(i, k)], idx)
    end
    return folds
end

################################################################################
# Cross-validation for the MKL approach
################################################################################
function cross_validate_mkl(X, y, kernels;
                            Cs = Cs_range,
                            lambdas = lambdas_range,
                            k0=3, max_iter=50, sum_beta_val=1.0, tolerance=1e-2,
                            nfolds=5)

    n = size(X, 1)
    folds = kfold_indices(n, nfolds)
    best_acc = -Inf
    best_C   = nothing
    best_lam = nothing

    for c_val in Cs
        for lam_val in lambdas
            accs = Float64[]
            for fold_idx in 1:nfolds
                val_indices = folds[fold_idx]
                train_indices = vcat(folds[setdiff(1:nfolds, fold_idx)]...)

                X_tr = X[train_indices, :]
                y_tr = y[train_indices]
                X_val = X[val_indices, :]
                y_val = y[val_indices]

                # Compute kernels for train/val
                K_list_train = compute_kernels(X_tr, X_tr, kernels)
                K_list_val   = compute_kernels(X_tr, X_val, kernels)

                α_tmp, β_tmp, K_comb_tmp, _, _ = train_interpretable_mkl(
                    X_tr, y_tr, c_val, K_list_train, lam_val;
                    max_iter=max_iter,
                    tolerance=tolerance,
                    k0=k0,
                    sum_beta_val=sum_beta_val,
                    solver_type=:LIBSVM,
                    beta_method=:gssp
                )
                b_tmp = compute_bias(α_tmp, y_tr, K_comb_tmp, c_val)

                y_pred_val = predict_mkl(
                    α_tmp, y_tr, X_tr, X_val, β_tmp, b_tmp,
                    K_list_val, kernel_type="precomputed";
                    tolerance=tolerance
                )

                fold_acc, _, _, _ = compute_metrics(y_val, y_pred_val)
                push!(accs, fold_acc)
            end
            mean_acc = mean(accs)
            if mean_acc > best_acc
                best_acc = mean_acc
                best_C   = c_val
                best_lam = lam_val
            end
        end
    end
    return best_C, best_lam, best_acc
end

################################################################################
# Cross-validation for vanilla SVM (Polynomial kernel)
################################################################################
function cross_validate_svm(X, y; Cs=Cs_range, kernel=Kernel.Linear, nfolds=5)
    n = size(X, 1)
    folds = kfold_indices(n, nfolds)
    best_acc   = -Inf
    best_C_svm = nothing

    for c_val in Cs
        accs = Float64[]
        for fold_idx in 1:nfolds
            val_indices = folds[fold_idx]
            train_indices = vcat(folds[setdiff(1:nfolds, fold_idx)]...)

            X_tr = X[train_indices, :]
            y_tr = y[train_indices]
            X_val = X[val_indices, :]
            y_val = y[val_indices]

            println("X_train shape: ", size(X_tr))
            println("y_train length: ", length(y_tr))
            println("X_test shape: ", size(X_val))
            println("y_test length: ", length(y_val))

            svm_model = svmtrain(
                X_tr', Float64.(y_tr);
                svmtype = LIBSVM.SVC,
                kernel  = kernel,
                cost    = c_val
            )

            println(size(X_val'))
            println(size(X_tr'))
            y_pred_val, _ = svmpredict(svm_model, X_val')
            println((y_pred_val))
            println((y_val))
            fold_acc, _, _, _ = compute_metrics(y_val, y_pred_val)
            push!(accs, fold_acc)
        end

        mean_acc = mean(accs)
        if mean_acc > best_acc
            best_acc   = mean_acc
            best_C_svm = c_val
        end
    end
    return best_C_svm, best_acc
end

################################################################################
# Main loop over datasets
################################################################################
for dataset in DATASETS
    println("\n=== Processing dataset: $dataset ===")
    # Download/Load data
    frac = dataset == :adult ? 0.33 : 1.00
    X_train, y_train, X_test, y_test = get_dataset(dataset; force_download=false, frac=frac, train_ratio=0.8)

    # Possibly X comes as (#features, #samples). LIBSVM wants (#samples, #features).
    if size(X_train,1) != length(y_train) && size(X_train,2) == length(y_train)
        X_train = X_train'
    end
    if size(X_test,1) != length(y_test) && size(X_test,2) == length(y_test)
        X_test = X_test'
    end

    # Ensure y in {-1, +1}, stored as Float64
    y_train = Float64.(y_train)
    y_test  = Float64.(y_test)

    train_size = length(y_train)
    test_size  = length(y_test)

    # 1) Tune MKL hyperparams
    best_C_mkl, best_lam_mkl, best_cv_acc_mkl = cross_validate_mkl(
        X_train, y_train, kernels;
        Cs=Cs_range,
        lambdas=lambdas_range,
        k0=k0,
        max_iter=max_iter,
        sum_beta_val=sum_beta_val,
        tolerance=tolerance,
        nfolds=N_FOLDS
    )
    println("  [MKL] Best (C, λ) = ($best_C_mkl, $best_lam_mkl); avg val acc = $(round(best_cv_acc_mkl, digits=4))")

    # 2) Retrain MKL on entire training set, measuring runtime
    K_list_train = compute_kernels(X_train, X_train, kernels)
    K_list_test  = compute_kernels(X_train, X_test,  kernels)

    t0 = time()  # record time before final MKL fit
    α, β, K_combined, _, _ = train_interpretable_mkl(
        X_train, y_train, best_C_mkl, K_list_train, best_lam_mkl;
        max_iter=max_iter,
        tolerance=tolerance,
        k0=k0,
        sum_beta_val=sum_beta_val,
        solver_type=:LIBSVM,
        beta_method=:gssp
    )
    fit_time_mkl = time() - t0  # measure final fit time

    b = compute_bias(α, y_train, K_combined, best_C_mkl)

    # Evaluate MKL on train
    y_pred_train_mkl = predict_mkl(
        α, y_train, X_train, X_train, β, b,
        K_list_train, kernel_type="precomputed";
        tolerance=tolerance
    )
    MKL_TrainAccuracy, _, _, _ = compute_metrics(y_train, y_pred_train_mkl)

    # Evaluate MKL on test
    y_pred_test_mkl = predict_mkl(
        α, y_train, X_train, X_test, β, b,
        K_list_test, kernel_type="precomputed";
        tolerance=tolerance
    )
    MKL_TestAccuracy, MKL_Precision, MKL_Recall, MKL_F1_Score = compute_metrics(y_test, y_pred_test_mkl)

    # 3) Baseline: majority vote
    majority_label = (sum(y_train .== 1) >= sum(y_train .== -1)) ? 1.0 : -1.0
    y_pred_baseline_test = fill(majority_label, length(y_test))
    Baseline_Accuracy, Baseline_Precision, Baseline_Recall, Baseline_F1_Score = compute_metrics(y_test, y_pred_baseline_test)

    # 4) Vanilla SVM baseline cross-validation
    best_C_svm, best_cv_acc_svm = cross_validate_svm(
        X_train, y_train; 
        Cs=Cs_range,
        kernel=Kernel.Linear,
        nfolds=N_FOLDS
    )
    println("  [SVM] Best C = $best_C_svm; avg val acc = $(round(best_cv_acc_svm, digits=4))")

    # Retrain SVM on entire training set with the polynomial kernel
    svm_model = svmtrain(
        X_train', Float64.(y_train); 
        svmtype = LIBSVM.SVC, 
        kernel  = Kernel.Linear, 
        cost    = best_C_svm
    )

    # Evaluate SVM on training
    y_pred_train_svm, _ = svmpredict(svm_model, X_train')
    SVM_TrainAccuracy, _, _, _ = compute_metrics(y_train, y_pred_train_svm)

    # Evaluate SVM on test
    y_pred_test_svm, _ = svmpredict(svm_model, X_test')
    SVM_TestAccuracy, SVM_Precision, SVM_Recall, SVM_F1_Score = compute_metrics(y_test, y_pred_test_svm)

    # 5) Store results (adding best C/λ + final fit time for MKL)
    betas_str = join(round.(β, digits=4), ", ")

    push!(results, (
        string(dataset),
        train_size,
        test_size,

        MKL_TrainAccuracy,
        MKL_TestAccuracy,
        MKL_Precision,
        MKL_Recall,
        MKL_F1_Score,

        Baseline_Accuracy,
        Baseline_Precision,
        Baseline_Recall,
        Baseline_F1_Score,

        SVM_TrainAccuracy,
        SVM_TestAccuracy,
        SVM_Precision,
        SVM_Recall,
        SVM_F1_Score,

        betas_str,
        best_C_mkl,        # new column: best C for MKL
        best_lam_mkl,      # new column: best λ for MKL
        fit_time_mkl,      # new column: final MKL fit time

        "Success"
    ))
end

################################################################################
# Save and display results
################################################################################
println("Start writing results to CSV file")
CSV.write("results.csv", results)  # Save results to a CSV file
println("Results written to results.csv")

# Filter successful datasets
successful_results = filter(row -> row.Status == "Success", results)
println("\n=== Successful Datasets ===")
println(successful_results)

# Print datasets that encountered errors
failed_datasets = filter(row -> row.Status == "Error", results)
println("\n=== Datasets with Errors ===")
println(failed_datasets.Dataset)  # Print only the dataset names
