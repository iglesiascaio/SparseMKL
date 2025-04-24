################################################################################
# run_synthetic_experiments.jl — Support‑recovery on synthetic data
#   * unique kernel specs (no duplicates)
#   * two training regimes:  fixed / cv
################################################################################

using Infiltrator
using Random
using LinearAlgebra
using Statistics
using Distributions
using Dates
using Printf
using DataFrames
using CSV
using Base: time
using NPZ

include("../src/MKL/multi_kernel.jl")
using .MKL: compute_kernels, compute_bias, predict_mkl

include("../src/Interpretable_MKL/interpretable_multi_kernel.jl")
using .InterpretableMKL: train_interpretable_mkl

# ------------------------ MASTER SEED & CONSTANTS -----------------------------
const GLOBAL_SEED         = 42
const RUN_CV = false          # ⇐ set to false to skip cross-validation
Random.seed!(GLOBAL_SEED)

const TRAIN_SIZES         = [50]
const TEST_SIZE           = 500
const DIM_X               = 2
const TOTAL_KERNELS_FIXED = 10
const TRUE_KERNELS        = 3

const N_KERNELS_LIST      = [5]
const N_TRAIN_FIXED       = 400
const NOISE_RATE          = 0.07
const N_REPS              = 5

const Cs_range            = [5.0, 10.0, 50.0, 100.0]
const lambdas_range       = [0.01, 0.1, 1.0, 10.0, 100.0]
const k0_range            = [3]
const N_FOLDS             = 5
const MAX_ITER            = 50
const TOL                 = 1e-2

# ----------------------- UNIQUE KERNEL CATALOGUE ------------------------------
const BASE_KERNEL_SPECS = [
    # Linear
    # Dict(:type => "linear",     :params => Dict()),

    # Polynomial kernels (degrees and offsets varied reasonably)
    Dict(:type => "polynomial", :params => Dict(:degree => 2, :c => 0.0)),
    Dict(:type => "polynomial", :params => Dict(:degree => 3, :c => 1.0)),
    Dict(:type => "polynomial", :params => Dict(:degree => 4, :c => 5.0)),
    Dict(:type => "polynomial", :params => Dict(:degree => 5, :c => 10.0)),
    Dict(:type => "polynomial", :params => Dict(:degree => 5, :c => 20.0)),


    # RBF kernels (gamma spanning local ↔ global)
    Dict(:type => "rbf",        :params => Dict(:gamma  => 0.01)),
    Dict(:type => "rbf",        :params => Dict(:gamma  => 0.1)),
    Dict(:type => "rbf",        :params => Dict(:gamma  => 1.0)),
    Dict(:type => "rbf",        :params => Dict(:gamma  => 10.0)),
    Dict(:type => "rbf",        :params => Dict(:gamma  => 20.0)),


    # Laplacian kernels (similar span, L1 vs. L2)
    # Dict(:type => "laplacian",  :params => Dict(:gamma  => 0.01)),
    # Dict(:type => "laplacian",  :params => Dict(:gamma  => 0.5)),
    # Dict(:type => "laplacian",  :params => Dict(:gamma  => 5.0)),

    # Sigmoid kernels (vary both slope and bias moderately)
    # Dict(:type => "sigmoid",    :params => Dict(:gamma  => 0.01, :c0 => 0.0)),
    # Dict(:type => "sigmoid",    :params => Dict(:gamma  => 1.0,  :c0 => 1.0)),
]


function make_unique_kernel_specs(rng::AbstractRNG, n_k::Int)
    if n_k > length(BASE_KERNEL_SPECS)
        error("Need $n_k kernels but only $(length(BASE_KERNEL_SPECS)) unique specs defined.")
    end
    return sample(rng, BASE_KERNEL_SPECS, n_k; replace = false)
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
    
    println("β̂: ", β̂)
    println("true_idx: ", true_idx)

    # Compute support recovery metrics
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
        noise_rate::Float64 = 0.05)


    # ------------------------------------------------------------------
    # 1. Draw ALL inputs first and split into plain Matrices
    # ------------------------------------------------------------------
    N       = n_train + n_test
    X_all   = randn(rng, N, dim)
    X_train = copy(view(X_all, 1:n_train,     :))
    X_test  = copy(view(X_all, n_train+1:N,   :))

    # ------------------------------------------------------------------
    # 2. Build the combined Gram matrix for ALL samples
    # ------------------------------------------------------------------
    Kc     = zeros(N, N)
    K_tt   = compute_kernels(X_train, X_train, kernels[true_idx])
    K_tr   = compute_kernels(X_train, X_test,  kernels[true_idx])
    K_rr   = compute_kernels(X_test,  X_test,  kernels[true_idx])

    for (β, Ktt, Ktr, Krr) in zip(β_true, K_tt, K_tr, K_rr)
        # train–train
        Kc[1:n_train,     1:n_train]       .+= β .* Ktt
        # train–test
        Kc[1:n_train,     n_train+1:end]   .+= β .* Ktr
        # test–train (transpose of train–test)
        Kc[n_train+1:end, 1:n_train]       .+= β .* Ktr'
        # test–test
        Kc[n_train+1:end, n_train+1:end]   .+= β .* Krr
    end

    # ------------------------------------------------------------------
    # 3. Draw α for *all* N samples and a bias term b
    # ------------------------------------------------------------------
    α      = randn(rng, N)
    b      = randn(rng)

    # ------------------------------------------------------------------
    # 4. Generate latent scores and labels
    # ------------------------------------------------------------------
    f      = Kc * α .+ b
    y_all  = map(x -> x ≥ 0 ? 1.0 : -1.0, f)

    # flip labels inside the TRAIN set only
    n_flip   = round(Int, noise_rate * n_train)
    flip_idx = randperm(rng, n_train)[1:n_flip]
    y_all[flip_idx] .*= -1

    # ------------------------------------------------------------------
    # 5. Split outputs into plain Vectors
    # ------------------------------------------------------------------
    y_train = copy(view(y_all, 1:n_train))
    y_test  = copy(view(y_all, n_train+1:N))

    return X_train, y_train, X_test, y_test, α, b
    end



# ---------------------- CROSS‑VALIDATION UTILITIES ----------------------------
function kfold_indices(n::Int, k::Int; rng = Random.GLOBAL_RNG)
    idxs  = collect(1:n)
    shuffle!(rng, idxs)
    folds = [Int[] for _ in 1:k]
    for (i, ix) in enumerate(idxs)
        push!(folds[mod1(i, k)], ix)
    end
    return folds
end

function cross_validate_mkl(
    X::Matrix{Float64},
    y::Vector{Float64},
    kernels;
    Cs        = Cs_range,
    lambdas   = lambdas_range,
    k0_range  = k0_range,
    nfolds    = N_FOLDS,
    max_iter  = MAX_ITER,
    tolerance = TOL
)
    folds    = kfold_indices(size(X,1), nfolds)
    best_acc = -Inf
    bestC, bestλ, bestk0 = NaN, NaN, NaN

    for k0 in k0_range, C in Cs, λ in lambdas
        accs = Float64[]
        for f in 1:nfolds
            val_idxs = folds[f]
            trn_idxs = vcat(folds[setdiff(1:nfolds, f)]...)
            Xtr, ytr = X[trn_idxs, :], y[trn_idxs]
            Xv,  yv  = X[val_idxs, :], y[val_idxs]
            Ktr  = compute_kernels(Xtr, Xtr, kernels)
            Kv   = compute_kernels(Xtr, Xv,  kernels)
            α, β, Kc, _, _, _ = train_interpretable_mkl(
                Xtr, ytr, C, Ktr, λ;
                k0          = k0,
                max_iter    = max_iter,
                tolerance   = tolerance,
                sum_beta_val= 1.0,
                solver_type = :LIBSVM,
                beta_method = :gssp,
                verbose     = true
            )
            b   = compute_bias(α, ytr, Kc, C)
            ŷ   = predict_mkl(α, ytr, Xtr, Xv, β, b, Kv; kernel_type="precomputed")
            acc, _, _, _ = compute_metrics(yv, ŷ)
            push!(accs, acc)
        end
        m = mean(accs)
        if m > best_acc
            best_acc  = m
            bestC     = C
            bestλ     = λ
            bestk0    = k0
        end
    end

    return bestC, bestλ, bestk0, best_acc
end

# ------------------------- FITTING ROUTINE -----------------------------------
function fit_mkl(
    Xtr::Matrix{Float64},
    ytr::Vector{Float64},
    Xte::Matrix{Float64},
    yte::Vector{Float64},
    kernels;
    C::Float64,
    λ::Float64,
    k0::Int
)
    Ktr = compute_kernels(Xtr, Xtr, kernels)
    Kte = compute_kernels(Xtr, Xte, kernels)

    t0 = time()
    α, β, Kc, _, _, _ = train_interpretable_mkl(
        Xtr, ytr, C, Ktr, λ;
        k0          = k0,
        max_iter    = MAX_ITER,
        tolerance   = TOL,
        sum_beta_val= 1.0,
        solver_type = :LIBSVM,
        beta_method = :gssp,
        verbose     = false
    )
    fit_time = time() - t0

    b      = compute_bias(α, ytr, Kc, C)
    ŷ_tr   = predict_mkl(α, ytr, Xtr, Xtr, β, b, Ktr; kernel_type="precomputed")
    ŷ_te   = predict_mkl(α, ytr, Xtr, Xte, β, b, Kte; kernel_type="precomputed")
    tr_acc = compute_metrics(ytr, ŷ_tr)[1]
    te_acc = compute_metrics(yte, ŷ_te)[1]

    return β, α, tr_acc, te_acc, fit_time, b
end

# ------------------------------ RESULT TABLE ----------------------------------
results = DataFrame(
    ExpType         = String[],
    Method          = String[],
    TrainSize       = Int[],
    TotalKernels    = Int[],
    TrueKernels     = Int[],
    Rep             = Int[],
    SupportAcc      = Float64[],
    SupportTPR      = Float64[],
    SupportTNR      = Float64[],
    PredSupportSize = Int[],
    TrainAcc        = Float64[],
    TestAcc         = Float64[],
    FitTime         = Float64[],
)

# -------------------------------- MAIN LOOP ------------------------------------
seed_base = 2025
println("Starting synthetic MKL experiments at ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

for setting in (:vary_k, :vary_n)
    params = setting == :vary_n ? TRAIN_SIZES : N_KERNELS_LIST

    for p in params, rep in 1:N_REPS
        rng      = MersenneTwister(hash((seed_base, setting, p, rep)))
        total_k  = setting == :vary_n ? TOTAL_KERNELS_FIXED : p
        n_train  = setting == :vary_n ? p : N_TRAIN_FIXED
        kernels  = make_unique_kernel_specs(rng, total_k)
        true_k   = TRUE_KERNELS

        true_idx = sample(rng, 1:total_k, true_k; replace = false)
        β_true   = dirichlet_vec(rng, true_k)


        println("\n[", Dates.format(now(), "HH:MM:SS"), "] ",
                setting == :vary_n ? "Vary n_samples" : "Vary n_kernels",
                "  p=", p, "  rep=", rep)

        Xtr, ytr, Xte, yte, α_true, b_true = synthetic_dataset(
            rng, n_train, TEST_SIZE, DIM_X, kernels, true_idx, β_true;
            noise_rate = NOISE_RATE
        )


        # --- fixed hyperparams ------------------------------------------------
        β, α, tr_acc, te_acc, fit_t, b = fit_mkl(
            Xtr, ytr, Xte, yte, kernels;
            C  = 1.0,
            λ  = 10.0,
            k0 = true_k
        )
        sup_acc, tpr, tnr, ps = support_metrics(β, true_idx)
        println("  [fixed] support_acc=", round(sup_acc, digits=3),
                ", TPR=", round(tpr, digits=3),
                ", TNR=", round(tnr, digits=3),
                ", train_acc=", round(tr_acc, digits=3),
                ", test_acc=", round(te_acc, digits=3))
        push!(results, (
            string(setting), "fixed", n_train, total_k, true_k, rep,
            sup_acc, tpr, tnr, ps, tr_acc, te_acc, fit_t
        ))

        # --- cross‑validation ------------------------------------------------
        # if !RUN_CV
        #     println("  [cv]    SKIPPED")
        #     continue
        # end
        # Cstar, λstar, k0star, _ = cross_validate_mkl(Xtr, ytr, kernels)
        # βcv, tr_acc, te_acc, fit_t = fit_mkl(
        #     Xtr, ytr, Xte, yte, kernels;
        #     C  = Cstar,
        #     λ  = λstar,
        #     k0 = k0star
        # )
        # sup_acc, tpr, tnr, ps = support_metrics(βcv, true_idx)
        # println("  [cv]    support_acc=", round(sup_acc, digits=3),
        #         ", TPR=", round(tpr, digits=3),
        #         ", TNR=", round(tnr, digits=3),
        #         ", train_acc=", round(tr_acc, digits=3),
        #         ", test_acc=", round(te_acc, digits=3))
        # push!(results, (
        #     string(setting), "cv", n_train, total_k, true_k, rep,
        #     sup_acc, tpr, tnr, ps, tr_acc, te_acc, fit_t
        # ))

        fname = "rep_$(setting)_p$(p)_r$(lpad(rep,2,'0')).jld2"
        mkpath("mkl_sim_results")
        npz_path = joinpath("mkl_sim_results", replace(fname, r"\.jld2$"=>".npz"))
        npzwrite(
          npz_path,
          Dict(
            # "X_all"      => Xall,
            "alpha_true" => α_true,
            "b_true"     => b_true,
            "beta_true"  => β_true,
            "b_est"     => b,
            "alpha_est"  => α,
            "beta_est"   => β,
            "y_train"    => ytr,
            "y_test"     => yte,
            "X_train"   => Xtr,
            "X_test"    => Xte,
          )
        )
    end
end

# ---------------------------------- SAVE --------------------------------------
out_file = "run_synthetic_experiments_results_$(Dates.format(now(), "yyyyMMdd_HHMMSS")).csv"
CSV.write(out_file, results)
println("\nAll experiments complete. Results saved to ", out_file)
