################################################################################
# run_synthetic_experiments.jl — Support-recovery on synthetic data (2D demo)
#   * unique kernel specs (no duplicates)
#   * two training regimes:  fixed / cv
#   * sparse α, small DIM_X, small reps
################################################################################

using Random
using LinearAlgebra
using Statistics
using Distributions
using Dates
using Printf
using DataFrames
using CSV
using JLD2
using NPZ

using Base: time

include("../src/MKL/multi_kernel.jl")
using .MKL: compute_kernels, compute_bias, predict_mkl

include("../src/Interpretable_MKL/interpretable_multi_kernel.jl")
using .InterpretableMKL: train_interpretable_mkl

# ------------------------ MASTER SEED & CONSTANTS -----------------------------
const GLOBAL_SEED         = 2025
const RUN_CV              = false   # set true to enable cross-validation

const TRAIN_SIZES         = [50, 100]
const TEST_SIZE           = 100
const DIM_X               = 2
const TOTAL_KERNELS_FIXED = 4
const TRUE_KERNELS        = 2
const N_KERNELS_LIST      = [3, 4]
const N_TRAIN_FIXED       = 100
const NOISE_RATE          = 0.1
const N_REPS              = 5

const Cs_range            = [1.0]    # dummy for demo
const lambdas_range       = [0.1]
const k0_range            = [TRUE_KERNELS]
const N_FOLDS             = 3
const MAX_ITER            = 20
const TOL                 = 1e-2

# ----------------------- UNIQUE KERNEL CATALOGUE ------------------------------
const BASE_KERNEL_SPECS = [
    Dict(:type=>"rbf",        :params=>Dict(:gamma=>0.5)),
    Dict(:type=>"rbf",        :params=>Dict(:gamma=>1.0)),
    Dict(:type=>"polynomial", :params=>Dict(:degree=>2, :c=>1.0)),
    Dict(:type=>"polynomial", :params=>Dict(:degree=>3, :c=>0.0)),
]

function make_unique_kernel_specs(rng::AbstractRNG, n_k::Int)
    if n_k > length(BASE_KERNEL_SPECS)
        error("Need $n_k kernels but only $(length(BASE_KERNEL_SPECS)) defined.")
    end
    return sample(rng, BASE_KERNEL_SPECS, n_k; replace=false)
end

dirichlet_vec(rng, k) = rand(rng, Dirichlet(fill(5.0, k)))

# ------------------------ METRIC HELPERS --------------------------------------
function compute_metrics(y, ŷ)
    TP = sum((y .== 1) .& (ŷ .== 1))
    TN = sum((y .== -1) .& (ŷ .== -1))
    FP = sum((y .== -1) .& (ŷ .== 1))
    FN = sum((y .== 1) .& (ŷ .== -1))
    accuracy  = (TP + TN) / length(y)
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1_score  = 2 * (precision * recall) / (precision + recall + 1e-9)
    return accuracy, precision, recall, f1_score
end

function support_metrics(β̂::Vector{Float64}, true_idx::Vector{Int}; tol=1e-3)
    pred_idx = findall(abs.(β̂) .> tol)
    S_true   = Set(true_idx)
    S_pred   = Set(pred_idx)
    TP = length(intersect(S_true, S_pred))
    FN = length(setdiff(S_true, S_pred))
    FP = length(setdiff(S_pred, S_true))
    TN = length(β̂) - TP - FN - FP
    accuracy = (TP + TN) / length(β̂)
    tpr      = TP / max(length(S_true), 1)
    tnr      = TN / max(length(β̂) - length(S_true), 1)
    return accuracy, tpr, tnr, length(pred_idx)
end

# ----------------------- SYNTHETIC DATA GENERATION ----------------------------
function synthetic_dataset(
    rng::AbstractRNG,
    n_train::Int,
    n_test::Int,
    dim::Int,
    kernels,
    true_idx::Vector{Int},
    β_true::Vector{Float64};
    alpha_sparsity::Float64 = 0.8,
    noise_rate::Float64 = NOISE_RATE
)
    N = n_train + n_test
    X_all   = randn(rng, N, dim)
    # return concrete Arrays, not SubArrays
    X_train = Array(view(X_all, 1:n_train, :))
    X_test  = Array(view(X_all, n_train+1:N, :))

    # compute all base kernels on X_all
    Ks_all = compute_kernels(X_all, X_all, kernels)

    # combined true Gram
    Kc = zeros(N, N)
    for k in true_idx
        Kc .+= β_true[k] .* Ks_all[k]
    end

    # sparse α
    α = zeros(N)
    n_nz = round(Int, (1 - alpha_sparsity) * N)
    nz_idx = sample(rng, 1:N, n_nz; replace=false)
    α[nz_idx] = randn(rng, n_nz)
    b = randn(rng)

    # latent & noisy labels
    f = Kc * α .+ b
    y_all = map(x-> x ≥ 0 ? 1.0 : -1.0, f)
    flip_idx = sample(rng, 1:n_train, round(Int, noise_rate*n_train); replace=false)
    y_all[flip_idx] .*= -1

    y_train = y_all[1:n_train]
    y_test  = y_all[n_train+1:end]

    return X_train, y_train, X_test, y_test, X_all, α
end

# ---------------------- CROSS-VALIDATION UTILITIES ----------------------------
function kfold_indices(n::Int, k::Int; rng=Random.GLOBAL_RNG)
    idxs = collect(1:n)
    shuffle!(rng, idxs)
    [idxs[i:k:n] for i in 1:k]
end

function cross_validate_mkl(X, y, kernels)
    folds = kfold_indices(size(X,1), N_FOLDS; rng=MersenneTwister(GLOBAL_SEED))
    best_acc, bestC, bestλ, bestk0 = -Inf, NaN, NaN, NaN

    for k0 in k0_range, C in Cs_range, λ in lambdas_range
        accs = Float64[]
        for fold in folds
            trn = setdiff(1:size(X,1), fold)
            Xtr, ytr = X[trn, :], y[trn]
            Xv,  yv  = X[fold, :],    y[fold]
            Xtr_mat = Array(Xtr)
            Xv_mat  = Array(Xv)
            Ktr = compute_kernels(Xtr_mat, Xtr_mat, kernels)
            Kv  = compute_kernels(Xtr_mat, Xv_mat, kernels)

            α̂, β̂, Kĉ, _, _, _ = train_interpretable_mkl(
                Xtr_mat, ytr, C, Ktr, λ;
                max_iter=MAX_ITER,
                tolerance=TOL,
                k0=k0,
                sum_beta_val=1.0,
                solver_type=:LIBSVM,
                beta_method=:gssp,
                verbose=false
            )

            b̂ = compute_bias(α̂, ytr, Kĉ, C)
            ŷ = predict_mkl(α̂, ytr, Xtr_mat, Xv_mat, β̂, b̂, Kv; kernel_type="precomputed")
            push!(accs, compute_metrics(yv, ŷ)[1])
        end
        m = mean(accs)
        if m > best_acc
            best_acc, bestC, bestλ, bestk0 = m, C, λ, k0
        end
    end

    return bestC, bestλ, bestk0, best_acc
end

# ------------------------- FITTING ROUTINE -----------------------------------
function fit_mkl(Xtr, ytr, Xte, yte, kernels; C, λ, k0)
    # convert to dense matrices
    Xtr_mat = Array(Xtr)
    Xte_mat = Array(Xte)
    ytr_vec = ytr

    Ktr = compute_kernels(Xtr_mat, Xtr_mat, kernels)
    Kte = compute_kernels(Xtr_mat, Xte_mat, kernels)

    t0 = time()
    α̂, β̂, Kĉ, _, _, _ = train_interpretable_mkl(
        Xtr_mat, ytr_vec, C, Ktr, λ;
        max_iter=MAX_ITER,
        tolerance=TOL,
        k0=k0,
        sum_beta_val=1.0,
        solver_type=:LIBSVM,
        beta_method=:gssp,
        verbose=false
    )
    fit_time = time() - t0

    b̂    = compute_bias(α̂, ytr_vec, Kĉ, C)
    ŷ_tr = predict_mkl(α̂, ytr_vec, Xtr_mat, Xtr_mat, β̂, b̂, Ktr; kernel_type="precomputed")
    ŷ_te = predict_mkl(α̂, ytr_vec, Xtr_mat, Xte_mat, β̂, b̂, Kte; kernel_type="precomputed")

    tr_acc = compute_metrics(ytr_vec, ŷ_tr)[1]
    te_acc = compute_metrics(yte, ŷ_te)[1]

    return α̂, β̂, tr_acc, te_acc, fit_time
end

# ------------------------------ RESULT TABLE ----------------------------------
results = DataFrame(
    setting        = String[],
    method         = String[],
    TrainSize      = Int[],
    TotalKernels   = Int[],
    TrueKernels    = Int[],
    Rep            = Int[],
    SupportAcc     = Float64[],
    SupportTPR     = Float64[],
    SupportTNR     = Float64[],
    PredSupSize    = Int[],
    TrainAcc       = Float64[],
    TestAcc        = Float64[],
    FitTime        = Float64[],
    File           = String[],
)

# -------------------------------- MAIN LOOP ------------------------------------
rng_main = MersenneTwister(GLOBAL_SEED)
mkpath("mkl_sim_results")
println("Starting experiments at ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

for setting in (:vary_k, :vary_n)
    params = setting == :vary_n ? TRAIN_SIZES : N_KERNELS_LIST

    for p in params, rep in 1:N_REPS
        rng      = MersenneTwister(rand(rng_main, UInt))
        total_k  = setting == :vary_k ? p : TOTAL_KERNELS_FIXED
        n_tr     = setting == :vary_n ? p : N_TRAIN_FIXED

        kernels  = make_unique_kernel_specs(rng, total_k)
        true_idx = sample(rng, 1:total_k, TRUE_KERNELS; replace=false)
        smallβ   = dirichlet_vec(rng, TRUE_KERNELS)
        β_true   = zeros(total_k); β_true[true_idx] .= smallβ

        @info "[$(Dates.format(now(), "HH:MM:SS"))] $setting p=$p rep=$rep"

        Xtr, ytr, Xte, yte, Xall, α_true = synthetic_dataset(
            rng, n_tr, TEST_SIZE, DIM_X,
            kernels, true_idx, β_true;
            alpha_sparsity=0.8,
            noise_rate=NOISE_RATE
        )

        # --- fixed hyperparams ---
        α_est, β_est, tr_acc, te_acc, t_f = fit_mkl(
            Xtr, ytr, Xte, yte, kernels;
            C=1.0, λ=10.0, k0=TRUE_KERNELS
        )
        sup_acc, tpr, tnr, ps = support_metrics(β_est, true_idx)
        fname = "rep_$(setting)_p$(p)_r$(lpad(rep,2,'0')).jld2"
        push!(results, (
            string(setting), "fixed", n_tr, total_k, TRUE_KERNELS, rep,
            sup_acc, tpr, tnr, ps, tr_acc, te_acc, t_f, fname
        ))
        # @save joinpath("mkl_sim_results", fname) Xall α_true β_true α_est β_est
         # save replicate data in NumPy .npz format, so Python can load directly
        npz_path = joinpath("mkl_sim_results", replace(fname, r"\.jld2$"=>".npz"))
        npzwrite(
          npz_path,
          Dict(
            "X_all"      => Xall,
            "alpha_true" => α_true,
            "beta_true"  => β_true,
            "alpha_est"  => α_est,
            "beta_est"   => β_est,
            "y_train"    => ytr,
            "y_test"     => yte,
            "X_train"   => Xtr,
            "X_test"    => Xte,
          )
        )


    end
end

CSV.write("mkl_sim_results/summary.csv", results)
println("All experiments complete. Results saved to mkl_sim_results/")
