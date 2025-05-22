###############################################################################
# new_mkl_experiments.jl  – now also logs the kernels used at each q
###############################################################################

using Printf
using Random, Statistics, Dates
using Base: time
using DataFrames, CSV, JSON                    # ← JSON added
using Distributions
using LIBSVM
include("../data/get_data.jl")
using .GetData: get_dataset
include("../src/MKL/multi_kernel.jl")
using .MKL: compute_kernels, compute_bias, predict_mkl
include("../src/Interpretable_MKL/interpretable_multi_kernel.jl")
using .InterpretableMKL: train_interpretable_mkl

# -------------------  hyper-parameters  ------------------------------------- #
const DATASETS    = [:haberman, :mammographic, :ionosphere, :parkinsons]
const Q_LIST      = 20:20:200
const K_LIST      = [3, 5, 10, 15]
const MAX_ITER    = 50
const SUM_BETA_VAL = 1.0
const TOLERANCE   = 1e-2
const PARAM_CSV_PATH = "../results/final_results_with_k_and_objectives.csv"
const OUT_RES_CSV   = "../results/results_vary_kernels.csv"
const OUT_KER_CSV   = "../results/kernels_used_vary_q.csv"
const RNG_SEED      = 42

# -------------------  helper: best (C,λ) per dataset ------------------------ #
param_df = CSV.read(PARAM_CSV_PATH, DataFrame)
function get_C_lambda(dset::Symbol)
    sub = filter(r -> r.Dataset == string(dset), param_df)
    nrow(sub)==0 && error("No (C,λ) for $dset")
    return sub.MKL_BestC[1], sub.MKL_BestLambda[1]
end

# -------------------  helper: metrics  -------------------------------------- #
metrics(y, ŷ) = let
    TP = sum((y .== 1) .& (ŷ .== 1))
    TN = sum((y .== -1) .& (ŷ .== -1))
    FP = sum((y .== -1) .& (ŷ .== 1))
    FN = sum((y .== 1) .& (ŷ .== -1))
    acc  = (TP + TN) / length(y)
    prec = TP / (TP + FP + 1e-9)
    rec  = TP / (TP + FN + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    (acc,prec,rec,f1)
end

# -------------------  kernel sampling helpers  ----------------------------- #
function sample_kernel_dict(rng::AbstractRNG)
    kt = rand(rng, ["linear","polynomial","rbf","sigmoid","laplacian"])
    kt=="linear"     && return Dict(:type=>"linear",     :params=>Dict())
    kt=="polynomial" && return Dict(:type=>"polynomial", :params=>Dict(:degree=>rand(rng,2:6), :c=>rand(rng)*2))
    kt=="rbf"        && return Dict(:type=>"rbf",        :params=>Dict(:gamma=>10^(rand(rng)*3-2)))
    kt=="sigmoid"    && return Dict(:type=>"sigmoid",    :params=>Dict(:gamma=>rand(rng)*0.9+0.1, :c0=>rand(rng)*4-2))
    Dict(:type=>"laplacian", :params=>Dict(:gamma=>10^(rand(rng)*3-2)))
end

function extend_kernel_bank!(bank::Vector{Dict}, q_target::Int, rng::AbstractRNG)
    has_linear = any(k->k[:type]=="linear", bank)
    while length(bank) < q_target
        k = sample_kernel_dict(rng)
        if k[:type]=="linear" && has_linear; continue; end
        (k ∈ bank) || (push!(bank,k); has_linear |= k[:type]=="linear")
    end
    bank
end

# -------------------  result & kernel logs  -------------------------------- #
results_df = DataFrame(
    Dataset=String[], TrainSize=Int[], TestSize=Int[],
    q=Int[], k=Int[], NumSelectedKernels=Int[],
    MKL_TrainAccuracy=Float64[], MKL_TestAccuracy=Float64[],
    MKL_Precision=Float64[], MKL_Recall=Float64[], MKL_F1_Score=Float64[],
    MKL_Objective=Float64[],
    Baseline_Accuracy=Float64[], Baseline_Precision=Float64[],
    Baseline_Recall=Float64[], Baseline_F1_Score=Float64[],
    Betas=String[], MKL_BestC=Float64[], MKL_BestLambda=Float64[],
    MKL_FitTime=Float64[], Status=String[]
)

kernlog_df = DataFrame(Dataset=String[], q=Int[], KernelsJSON=String[])

# -------------------  main loop  ------------------------------------------- #
Random.seed!(RNG_SEED)
for dset in DATASETS
    @printf "\n================ Dataset: %s ================\n" dset
    C_best, λ_best = get_C_lambda(dset)
    λ_best *= 10;   C_best *= 0.01    # ← your scaling trick

    X_tr,y_tr,X_te,y_te = get_dataset(dset; train_ratio=0.8)
    size(X_tr,1)!=length(y_tr) && (X_tr=X_tr');  size(X_te,1)!=length(y_te) && (X_te=X_te')
    y_tr = Float64.(y_tr);  y_te = Float64.(y_te)

    base_acc,base_prec,base_rec,base_f1 = metrics(y_te,
        fill((sum(y_tr.==1)≥sum(y_tr.==-1)) ? 1.0 : -1.0, length(y_te)))

    rng_ds = MersenneTwister(hash((RNG_SEED,dset)))
    kernels_cum = Dict[]

    for q in Q_LIST
        extend_kernel_bank!(kernels_cum, q, rng_ds)
        # --------- save the kernel list for this (dataset,q) -------------
        push!(kernlog_df, (string(dset), q, JSON.json(kernels_cum)))

        K_train = compute_kernels(X_tr, X_tr, kernels_cum)
        K_test  = compute_kernels(X_tr, X_te, kernels_cum)

        for k0 in K_LIST
            @printf "    [q=%3d, k=%2d] ... " q k0; flush(stdout)
            status="Success"; t0=time()
            try
                α,β,Kcomb,obj,_,_ = train_interpretable_mkl(
                    X_tr,y_tr,C_best,K_train,λ_best;
                    k0=k0,max_iter=MAX_ITER,tolerance=TOLERANCE,
                    sum_beta_val=SUM_BETA_VAL,solver_type=:LIBSVM,
                    beta_method=:gssp,warm_start=false)
                fit_time=time()-t0
                b = compute_bias(α,y_tr,Kcomb,C_best)
                ŷ_tr = predict_mkl(α,y_tr,X_tr,X_tr,β,b,K_train,kernel_type="precomputed";tolerance=TOLERANCE)
                ŷ_te = predict_mkl(α,y_tr,X_tr,X_te,β,b,K_test,kernel_type="precomputed";tolerance=TOLERANCE)
                acc_tr,_,_,_ = metrics(y_tr,ŷ_tr)
                acc_te,prec,rec,f1 = metrics(y_te,ŷ_te)
                nz = sum(abs.(β).>1e-6)
                @printf "test=%.3f  nz=%d  t=%.2fs\n" acc_te nz fit_time

                push!(results_df,(string(dset),length(y_tr),length(y_te),
                    q,k0,nz,acc_tr,acc_te,prec,rec,f1,obj,
                    base_acc,base_prec,base_rec,base_f1,
                    join(round.(β,digits=4),", "),C_best,λ_best,fit_time,status))
            catch e
                @printf "ERROR: %s\n" e.msg
                push!(results_df,(string(dset),length(y_tr),length(y_te),
                    q,k0,0,zeros(8)...,"",C_best,λ_best,0.0,"Error"))
            end
        end
    end
end

# -------------------  save both tables  ------------------------------------ #
mkpath(dirname(OUT_RES_CSV))
CSV.write(OUT_RES_CSV, results_df)
CSV.write(OUT_KER_CSV, kernlog_df)
@printf "\nSaved results to %s\nSaved kernel specs to %s\n" OUT_RES_CSV OUT_KER_CSV
