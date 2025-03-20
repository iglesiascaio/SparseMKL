############### 1) Imports ###############
include("../data/get_data.jl")
using .GetData: get_dataset

include("../src/MKL/multi_kernel.jl")
using .MKL: compute_kernels, compute_combined_kernel

using JuMP
using MosekTools
using LinearAlgebra

# For an SVM sub-problem (LIBSVM):
using LIBSVM
using Statistics  # mean, etc.

############### 2) Perspective-based MISDP solver ###############
"""
psd_2x2_mkl_sdp_solver(K_list, y, C, λ, k; sum_beta_val=true)

Implements the MKL problem:

  minimize    C * sum(σᵢ) + 0.5*θ + λ * sum(ωⱼ)
  subject to
    1 - σᵢ ≤ yᵢ(η + γᵢ),   σᵢ ≥ 0,
    [ θ    γᵀ ]
    [ γ   Σⱼ βⱼ Kⱼ ]   ⪰ 0,
    βⱼ ≥ 0,  ωⱼ ≥ 0,
    Σⱼ zⱼ ≤ k,  zⱼ ∈ {0,1},
    (optional) Σⱼ βⱼ = 1,
    and  [ zⱼ   βⱼ
           βⱼ   ωⱼ ]  ⪰ 0   =>  βⱼ² ≤ zⱼ ωⱼ.

This is a **mixed-integer semidefinite** model (MISDP).
"""
function psd_2x2_mkl_sdp_solver(
    K_list::Vector{Matrix{Float64}},
    y::Vector{Float64},
    C::Float64,
    λ::Float64,
    k::Int;
    sum_beta_val::Bool=true
)
    n = size(K_list[1], 1)
    q = length(K_list)
    @assert length(y) == n "Mismatch: length(y) != kernel dimension"
    @assert all(size(K) == (n,n) for K in K_list) "All K_list[i] must be n×n."

    # Build JuMP model with Mosek
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "QUIET", true)
    #
    # Variables
    #
    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)

    # If you want to do a continuous relaxation, keep 0 <= z[j] <= 1
    # If you want them strictly integer, do: @variable(model, z[1:q], Bin)
    @variable(model, 0 <= z[1:q] <= 1)

    # Big (n+1)×(n+1) PSD matrix
    @variable(model, M[1:(n+1), 1:(n+1)], PSD)

    #
    # Constraints
    #
    # 1) 1 - σᵢ ≤ yᵢ(η + γᵢ)
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    # 2) Link the big PSD block:
    @constraint(model, M[1,1] == θ)
    for i in 1:n
        @constraint(model, M[i+1,1] == γ[i])
        @constraint(model, M[1,i+1] == γ[i])
    end
    for i in 1:n, j_ in 1:n
        @constraint(model, M[i+1, j_+1] == sum(β[r]*K_list[r][i,j_] for r in 1:q))
    end

    # 3) (Optional) sum(β) = 1
    if sum_beta_val
        @constraint(model, sum(β[j] for j in 1:q) == 1)
    end

    # 4) Cardinality constraint
    @constraint(model, sum(z[j] for j in 1:q) <= k)

    # 5) The 2×2 PSD blocks for each j =>  βⱼ² ≤ zⱼ ωⱼ
    #    [ zⱼ   βⱼ ]
    #    [ βⱼ   ωⱼ ]   ⪰ 0
    for j in 1:q
        @constraint(model, [ [z[j]  β[j]];
                             [β[j]  ω[j]] ] in PSDCone())
    end

    #
    # Objective
    #
    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))

    optimize!(model)

    # Print the objective value if optimal
    if termination_status(model) == MOI.OPTIMAL
        println("Objective value = ", objective_value(model))
    else
        @warn "psd_2x2_mkl_sdp_solver: solver returned status $(termination_status(model))"
    end

    return value.(β)
end


function symmetrize!(K::Matrix{Float64}; tol::Float64=1e-10)
    @assert size(K,1) == size(K,2) "Matrix must be square"
    n = size(K,1)
    
    # Compute the maximum asymmetry
    max_diff = 0.0
    @inbounds for i in 1:n
        @inbounds for j in i+1:n
            diff = abs(K[i,j] - K[j,i])
            if diff > max_diff
                max_diff = diff
            end
        end
    end

    # If asymmetry exceeds tolerance, raise an error
    if max_diff > tol
        error("Matrix is not symmetric: max asymmetry = $max_diff, exceeds tolerance $tol")
    end

    # Otherwise, symmetrize the matrix in-place
    @inbounds for i in 1:n
        @inbounds for j in i+1:n
            val = 0.5 * (K[i,j] + K[j,i])
            K[i,j] = val
            K[j,i] = val
        end
    end

    return K
end

############### 3) Compute α from β ###############
function compute_dual_coefs_for_beta(
    K_list::Vector{Matrix{Float64}},
    β::Vector{Float64},
    y::Vector{Float64},
    C::Float64
)
    # Combine kernels
    K_comb = compute_combined_kernel(K_list, β)

    # LIBSVM precomputed format
    n = size(K_comb,1)
    K_pre = zeros(n, n+1)
    for i in 1:n
        K_pre[i,1] = i
        @inbounds @simd for j in 1:n
            K_pre[i, j+1] = K_comb[i,j]
        end
    end

    symmetrize!(K_pre)

    model = svmtrain(K_pre, Float64.(y);
                     svmtype = LIBSVM.SVC,
                     kernel  = LIBSVM.Kernel.Precomputed,
                     cost    = C)
    α_res = zeros(n)
    sv_inds = model.SVs.indices
    sv_coef = model.coefs[:,1]
    for (local_i, global_i) in enumerate(sv_inds)
        α_res[global_i] = abs(sv_coef[local_i])
    end
    return α_res
end


function predict(α, y, K, b::Float64)
    ntest = size(K, 2)
    ypred = Vector{Float64}(undef, ntest)
    for j in 1:ntest
        s = b
        for i in 1:length(α)
            s += α[i]*y[i]*K[i,j]
        end
        ypred[j] = s>=0 ? 1.0 : -1.0
    end
    return ypred
end

function compute_metrics(y_true, y_pred)
    tp = sum((y_true .== 1) .& (y_pred .== 1))
    tn = sum((y_true .== -1) .& (y_pred .== -1))
    fp = sum((y_true .== -1) .& (y_pred .== 1))
    fn = sum((y_true .== 1) .& (y_pred .== -1))
    acc = (tp+tn)/length(y_true)
    prec = tp/(tp+fp+1e-9)
    rec  = tp/(tp+fn+1e-9)
    f1   = 2*(prec*rec)/(prec+rec+1e-9)
    return acc,prec,rec,f1
end

############### 5) Demo: run on one dataset ###############
function run_sdp_for_dataset(dataset::Symbol=:iris; C=1.0, λ=0.1, k=2)
    println("=== Loading dataset: $dataset ===")
    X_train, y_train, X_test, y_test = get_dataset(dataset; force_download=false, frac=1.0, train_ratio=0.8)

    # Ensure shape (n, d)
    if size(X_train,1) != length(y_train)
        X_train = X_train'
    end
    if size(X_test,1) != length(y_test)
        X_test = X_test'
    end

    y_train = Float64.(y_train)
    y_test  = Float64.(y_test)

    println("Train set: ", size(X_train), ", Test set: ", size(X_test))

    # Some kernel specs
    kernel_specs = [
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
    K_list_train = compute_kernels(X_train, X_train, kernel_specs)
    K_list_test  = compute_kernels(X_train, X_test,  kernel_specs)

    # 1) Solve the perspective MISDP
    println("\nSolving 2x2 PSD perspective MKL with C=$C, λ=$λ, k=$k...")
    β_star = psd_2x2_mkl_sdp_solver(K_list_train, y_train, C, λ, k; sum_beta_val=true)
    println("  Found β = ", β_star)

end

##################### Run an example #####################
run_sdp_for_dataset(:iris; C=5.0, λ=100.0, k=3)
