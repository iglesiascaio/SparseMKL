###############################################################################
# new_mkl_experiments.jl
#   Author : Caio de Prospero Iglesias
#   Date   : 2025-04-25
#
#   Same as before, but kernels now grow cumulatively: the q=40 bank contains
#   every kernel drawn at q=20 plus 20 new ones, and so on.
###############################################################################

using Printf
using Random, Statistics, Dates
using Base: time
using DataFrames, CSV
using Distributions                    # for log-uniform draws
using LIBSVM                           # still used inside InterpretableMKL
include("../data/get_data.jl")
using .GetData: get_dataset
include("../src/MKL/multi_kernel.jl")
using .MKL: compute_kernels, compute_bias, predict_mkl
include("../src/Interpretable_MKL/interpretable_multi_kernel.jl")
using .InterpretableMKL: train_interpretable_mkl

# ------------------------------------------------------------------ #
#                     0.  Global hyper-parameters                    #
# ------------------------------------------------------------------ #
const DATASETS       = [:haberman, :mammographic, :ionosphere, :parkinsons]
const Q_LIST         = 20:20:200
const K_LIST         = [3, 5, 10, 15]               # passed as k0
const MAX_ITER       = 50
const SUM_BETA_VAL   = 1.0
const TOLERANCE      = 1e-2
const PARAM_CSV_PATH = "../results/final_results_with_k_and_objectives.csv"
const OUT_CSV_PATH   = "../results/results_vary_kernels.csv"
const RNG_SEED       = 42

# ------------------------------------------------------------------ #
#        1.  Read pre-tuned (C, λ) for every dataset once            #
# ------------------------------------------------------------------ #
param_df = CSV.read(PARAM_CSV_PATH, DataFrame)
function get_C_lambda(dset::Symbol)
    sub = filter(r -> r.Dataset == string(dset), param_df)
    nrow(sub) == 0 && error("No (C,λ) for dataset=$dset in $PARAM_CSV_PATH")
    return sub.MKL_BestC[1], sub.MKL_BestLambda[1]
end

# ------------------------------------------------------------------ #
#        2.  Utility: metrics                                        #
# ------------------------------------------------------------------ #
function metrics(y, ŷ)
    TP = sum((y .== 1) .& (ŷ .== 1))
    TN = sum((y .== -1) .& (ŷ .== -1))
    FP = sum((y .== -1) .& (ŷ .== 1))
    FN = sum((y .== 1) .& (ŷ .== -1))
    acc  = (TP + TN) / length(y)
    prec = TP / (TP + FP + 1e-9)
    rec  = TP / (TP + FN + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return acc, prec, rec, f1
end

# ------------------------------------------------------------------ #
#        3.  Kernel sampling helpers                                 #
# ------------------------------------------------------------------ #
"""
    sample_kernel_dict(rng) -> Dict
Randomly returns a kernel spec from {linear, polynomial, rbf, sigmoid, laplacian}.
"""
function sample_kernel_dict(rng::AbstractRNG)
    kt = rand(rng, ["linear","polynomial","rbf","sigmoid","laplacian"])
    kt == "linear"     && return Dict(:type=>"linear",     :params=>Dict())
    kt == "polynomial" && return Dict(:type=>"polynomial", :params=>Dict(:degree=>rand(rng,2:6), :c=>rand(rng)*2))
    kt == "rbf"        && return Dict(:type=>"rbf",        :params=>Dict(:gamma=>10^(rand(rng)*3-2)))
    kt == "sigmoid"    && return Dict(:type=>"sigmoid",    :params=>Dict(:gamma=>rand(rng)*0.9+0.1, :c0=>rand(rng)*4-2))
    return Dict(:type=>"laplacian", :params=>Dict(:gamma=>10^(rand(rng)*3-2)))
end

"""
    extend_kernel_bank!(bank, q_target, rng)
Grows `bank` in-place until `length(bank)==q_target`, observing:
  • at most one linear kernel
  • no duplicate kernel specs
"""
function extend_kernel_bank!(bank::Vector{Dict}, q_target::Int, rng::AbstractRNG)
    has_linear = any(k->k[:type]=="linear", bank)
    while length(bank) < q_target
        kdict = sample_kernel_dict(rng)
        if kdict[:type]=="linear" && has_linear
            continue
        end
        # avoid identical duplicates
        if !any(existing -> existing == kdict, bank)
            push!(bank, kdict)
            has_linear |= (kdict[:type]=="linear")
        end
    end
    return bank
end

# ------------------------------------------------------------------ #
#        4.  Prepare results DataFrame  (unchanged)                  #
# ------------------------------------------------------------------ #
results = DataFrame(
    Dataset            = String[],
    TrainSize          = Int[],
    TestSize           = Int[],
    q                  = Int[],
    k                  = Int[],
    NumSelectedKernels = Int[],
    MKL_TrainAccuracy  = Float64[],
    MKL_TestAccuracy   = Float64[],
    MKL_Precision      = Float64[],
    MKL_Recall         = Float64[],
    MKL_F1_Score       = Float64[],
    MKL_Objective      = Float64[],
    Baseline_Accuracy  = Float64[],
    Baseline_Precision = Float64[],
    Baseline_Recall    = Float64[],
    Baseline_F1_Score  = Float64[],
    Betas              = String[],
    MKL_BestC          = Float64[],
    MKL_BestLambda     = Float64[],
    MKL_FitTime        = Float64[],
    Status             = String[]
)

# ------------------------------------------------------------------ #
#        5.  Main experiment loop (only kernel logic changed)        #
# ------------------------------------------------------------------ #
Random.seed!(RNG_SEED)
for dset in DATASETS
    @printf "\n================ Dataset: %s ================\n" dset
    C_best, λ_best = get_C_lambda(dset)
    λ_best = λ_best * 10
    C_best = C_best * 0.01
    X_tr, y_tr, X_te, y_te = get_dataset(dset; force_download=false, frac=1.0, train_ratio=0.8)
    size(X_tr,1) != length(y_tr) && (X_tr = X_tr')
    size(X_te,1) != length(y_te) && (X_te = X_te')
    y_tr = Float64.(y_tr);  y_te = Float64.(y_te)

    # majority baseline
    majority = (sum(y_tr .== 1) ≥ sum(y_tr .== -1)) ? 1.0 : -1.0
    y_base = fill(majority, length(y_te))
    base_acc, base_prec, base_rec, base_f1 = metrics(y_te, y_base)

    # cumulative kernel list & RNG for this dataset
    rng_ds = MersenneTwister(hash((RNG_SEED, dset)))
    kernels_cum = Dict[]                  # starts empty, grows with each q

    for q in Q_LIST
        extend_kernel_bank!(kernels_cum, q, rng_ds)
        kernels = kernels_cum             # alias for clarity

        # pre-compute kernel matrices
        K_train = compute_kernels(X_tr, X_tr, kernels)
        K_test  = compute_kernels(X_tr, X_te, kernels)

        for k0 in K_LIST
            @printf "    [q=%3d, k=%2d] ... " q k0; flush(stdout)
            status = "Success";  t0 = time()
            try
                α, β, K_comb, obj, _, _ = train_interpretable_mkl(
                    X_tr, y_tr, C_best, K_train, λ_best;
                    k0=k0, max_iter=MAX_ITER, tolerance=TOLERANCE,
                    sum_beta_val=SUM_BETA_VAL, solver_type=:LIBSVM,
                    beta_method=:gssp, warm_start=false)
                fit_time = time() - t0
                b = compute_bias(α, y_tr, K_comb, C_best)

                ŷ_tr = predict_mkl(α, y_tr, X_tr, X_tr, β, b, K_train, kernel_type="precomputed"; tolerance=TOLERANCE)
                ŷ_te = predict_mkl(α, y_tr, X_tr, X_te, β, b, K_test,  kernel_type="precomputed"; tolerance=TOLERANCE)

                acc_tr, _, _, _   = metrics(y_tr, ŷ_tr)
                acc_te, prec, rec, f1 = metrics(y_te, ŷ_te)
                nz = sum(abs.(β) .> 1e-6)

                @printf "test=%.3f train=%.3f F1=%.3f nz=%d t=%.2fs\n" acc_te acc_tr f1 nz fit_time

                push!(results, (string(dset), length(y_tr), length(y_te), q, k0, nz,
                                acc_tr, acc_te, prec, rec, f1, obj,
                                base_acc, base_prec, base_rec, base_f1,
                                join(round.(β,digits=4),", "),
                                C_best, λ_best, fit_time, status))
            catch e
                @printf "ERROR: %s\n" e.msg
                push!(results, (string(dset), length(y_tr), length(y_te), q, k0, 0,
                                0.0,0.0,0.0,0.0,0.0,0.0,
                                base_acc, base_prec, base_rec, base_f1,
                                "", C_best, λ_best, 0.0, "Error"))
            end
        end
    end
end

# ------------------------------------------------------------------ #
#        6.  Persist results                                         #
# ------------------------------------------------------------------ #
mkpath(dirname(OUT_CSV_PATH))
CSV.write(OUT_CSV_PATH, results)
@printf "\nAll experiments finished – results saved to %s\n" OUT_CSV_PATH
