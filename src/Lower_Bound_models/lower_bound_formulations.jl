module LowerBoundFormulations

############### 1) Imports ###############
include("../../data/get_data.jl")
using .GetData: get_dataset

include("../MKL/multi_kernel.jl")
using .MKL: compute_kernels, compute_combined_kernel

using JuMP
using MosekTools     # For the perspective-based MISDP and our new 3×3, 4×4 PSD constraints
using Gurobi         # (We keep this for the existing two SOC relaxations, if desired.)
using LinearAlgebra
using LIBSVM
using Statistics
using CSV
using DataFrames

########################
# Utility: Symmetrize a square matrix in-place
########################
function symmetrize!(K::Matrix{Float64}; tol::Float64=1e-10)
    @assert size(K,1) == size(K,2) "Matrix must be square"
    n = size(K,1)

    # Check maximum asymmetry
    max_diff = 0.0
    @inbounds for i in 1:n
        @inbounds for j in i+1:n
            diff = abs(K[i,j] - K[j,i])
            max_diff = max(max_diff, diff)
        end
    end
    if max_diff > tol
        error("Matrix is not symmetric (max asymmetry = $max_diff, tolerance = $tol).")
    end

    # Symmetrize
    @inbounds for i in 1:n
        @inbounds for j in i+1:n
            val = 0.5 * (K[i,j] + K[j,i])
            K[i,j] = val
            K[j,i] = val
        end
    end
    return K
end

############### Method A: Perspective‐based MISDP solver (Mosek) ###############
function mkl_psd_solver(
    K_list::Vector{Matrix{Float64}},
    y::Vector{Float64},
    C::Float64,
    λ::Float64,
    k::Int;
    sum_beta_val::Bool=true
)
    n = size(K_list[1], 1)
    q = length(K_list)

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "QUIET", true)

    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)
    @variable(model, 0 <= z[1:q] <= 1)

    # Big (n+1)×(n+1) PSD matrix
    @variable(model, M[1:(n+1), 1:(n+1)], PSD)

    # 1) margin constraints
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    # 2) link M to (θ,γ, ∑ β[i]K_list[i])
    @constraint(model, M[1,1] == θ)
    for i in 1:n
        @constraint(model, M[i+1,1] == γ[i])
        @constraint(model, M[1,i+1] == γ[i])
    end
    for i in 1:n, j_ in 1:n
        @constraint(model, M[i+1, j_+1] == sum(β[r]*K_list[r][i,j_] for r in 1:q))
    end

    # 3) sum(β)=1 if desired
    if sum_beta_val
        @constraint(model, sum(β) == 1)
    end

    # 4) cardinality constraint
    @constraint(model, sum(z) <= k)

    # 5) for each i, β[i]^2 <= z[i]* ω[i] => 2x2 PSD
    for i in 1:q
        @constraint(model, [
            [z[i]   β[i]];
            [β[i]   ω[i]]
        ] in PSDCone())
    end

    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))
    optimize!(model)

    return objective_value(model), value.(β)
end

################ Method B: SOC Relaxation (Using e_j) ################
function soc_relaxation_1_solver(
    K_list::Vector{Matrix{Float64}},
    y::Vector{Float64},
    C::Float64,
    λ::Float64,
    k::Int;
    sum_beta_val::Bool=true
)
    n = size(K_list[1], 1)
    q = length(K_list)

    model = Model(Mosek.Optimizer)
    # set_optimizer_attribute(model, "QUIET", true)

    #
    # === Variables ===
    #
    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)
    @variable(model, 0 <= z[1:q] <= 1)

    #
    # === 1) Margin constraints ===
    #
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    #
    # === 2) 2×2 minors: θ * (∑β[i]K_i[j,j]) ≥ (γ[j])^2
    #
    @variable(model, s[1:n] >= 0)
    for j in 1:n
        @constraint(model, s[j] == sum( β[i]*K_list[i][j,j] for i=1:q ))
        # Rotated SOC: [ s[j], θ, sqrt(2)*γ[j] ] in RotatedSecondOrderCone()
        @constraint(model, [s[j], θ, sqrt(2)*γ[j]] in RotatedSecondOrderCone())
    end

    #
    # === 3) sum(β) = 1 if desired ===
    #
    if sum_beta_val
        @constraint(model, sum(β) == 1)
    end

    #
    # === 4) sum(z) <= k ===
    #
    @constraint(model, sum(z) <= k)

    #
    # === 5) β[i]^2 <= z[i]*ω[i] => RSoC
    #
    for i in 1:q
        @constraint(model, [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone())
    end

    #
    # === Objective ===
    #
    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))

    optimize!(model)

    return objective_value(model), value.(β)
end

"""
An adaptation of the SOC relaxation that also samples L random vectors x
and adds constraints of the form (∑β[i] xᵀK_i x)*θ ≥ (xᵀγ)².
"""
function soc_relaxation_2_solver_random(
    K_list::Vector{Matrix{Float64}},
    y::Vector{Float64},
    C::Float64,
    λ::Float64,
    k::Int;
    sum_beta_val::Bool = true,
    L::Int = 10
)
    n = size(K_list[1], 1)
    q = length(K_list)

    model = Model(Mosek.Optimizer)

    # === Variables ===
    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)
    @variable(model, 0 <= z[1:q] <= 1)

    # === 1) Margin constraints ===
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    # === 2a) Standard-basis 2×2 constraints (like soc1)
    @variable(model, diag_s[1:n] >= 0)
    for j in 1:n
        @constraint(model, diag_s[j] == sum(β[i]*K_list[i][j,j] for i=1:q))
        @constraint(model, [diag_s[j], θ, sqrt(2)*γ[j]] in RotatedSecondOrderCone())
    end

    # === 2b) L random unit vectors for extra constraints
    random_vectors = Vector{Vector{Float64}}(undef, L)
    for l in 1:L
        v = randn(n)
        v /= norm(v)  # normalize
        random_vectors[l] = v
    end

    @variable(model, rand_s[1:L] >= 0)
    for l in 1:L
        @constraint(model,
            rand_s[l] == sum(β[i]*(random_vectors[l]' * K_list[i] * random_vectors[l]) for i in 1:q)
        )
        @constraint(model,
            [rand_s[l], θ, sqrt(2)*(sum(random_vectors[l][j]*γ[j] for j in 1:n))]
            in RotatedSecondOrderCone()
        )
    end

    # === 3) sum(β) = 1 if desired
    if sum_beta_val
        @constraint(model, sum(β) == 1)
    end

    # === 4) sum(z) <= k
    @constraint(model, sum(z) <= k)

    # === 5) β[i]^2 <= z[i]*ω[i] => RSoC
    for i in 1:q
        @constraint(model, [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone())
    end

    # === 6) Objective
    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))

    optimize!(model)

    return objective_value(model), value.(β)
end


#################### soc_relaxation_3_solver ####################
function soc_relaxation_3_solver(
    K_list::Vector{Matrix{Float64}},
    y::Vector{Float64},
    C::Float64,
    λ::Float64,
    k::Int;
    sum_beta_val::Bool=true
)
    # Match the style of mkl_psd_solver but only enforce 3×3 sub-blocks PSD.
    n = size(K_list[1],1)
    q = length(K_list)

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "QUIET", true)

    # Variables
    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)
    @variable(model, 0 <= z[1:q] <= 1)

    # Big (n+1)×(n+1) matrix M, not declared fully PSD
    @variable(model, M[1:(n+1), 1:(n+1)])

    # 1) Margin constraints
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    # 2) Link M to (θ, γ, ∑β[i]K_list[i])
    @constraint(model, M[1,1] == θ)
    for i in 1:n
        @constraint(model, M[i+1,1] == γ[i])
        @constraint(model, M[1,i+1] == γ[i])
    end
    for i in 1:n, j_ in 1:n
        @constraint(model, M[i+1, j_+1] == sum(β[r]*K_list[r][i,j_] for r in 1:q))
    end

    # 3) sum(β) = 1 if desired
    if sum_beta_val
        @constraint(model, sum(β) == 1)
    end

    # 4) sum(z) <= k
    @constraint(model, sum(z) <= k)

    # 5) PSD constraints: 2x2 PSD for β
    for i in 1:q
        @constraint(model, [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone())
    end

    # 6) 3×3 sub-blocks that include θ (row/col 1)
    for j in 1:(n-1)
        for k_ in (j+1):n
            @constraint(model,
                M[[1, j+1, k_+1], [1, j+1, k_+1]] in PSDCone()
            )
        end
    end

    # 7) Objective
    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))

    optimize!(model)
    return objective_value(model), value.(β)
end

#################### soc_relaxation_4_solver ####################
function soc_relaxation_4_solver(
    K_list::Vector{Matrix{Float64}},
    y::Vector{Float64},
    C::Float64,
    λ::Float64,
    k::Int;
    sum_beta_val::Bool=true
)
    # Match the style of mkl_psd_solver but only enforce 4×4 sub-blocks PSD.
    n = size(K_list[1],1)
    q = length(K_list)

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "QUIET", true)

    # Variables
    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)
    @variable(model, 0 <= z[1:q] <= 1)

    # Big (n+1)×(n+1) matrix M, not declared fully PSD
    @variable(model, M[1:(n+1), 1:(n+1)])

    # 1) Margin constraints
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    # 2) Link M to (θ, γ, ∑β[i]K_list[i])
    @constraint(model, M[1,1] == θ)
    for i in 1:n
        @constraint(model, M[i+1,1] == γ[i])
        @constraint(model, M[1,i+1] == γ[i])
    end
    for i in 1:n, j_ in 1:n
        @constraint(model, M[i+1, j_+1] == sum(β[r]*K_list[r][i,j_] for r in 1:q))
    end

    # 3) sum(β) = 1 if desired
    if sum_beta_val
        @constraint(model, sum(β) == 1)
    end

    # 4) sum(z) <= k
    @constraint(model, sum(z) <= k)

    # 5) Perspective constraints: 2x2 PSD for β
    for i in 1:q
        @constraint(model, [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone())
    end

    # 6) 4×4 sub-blocks that include θ (row/col 1)
    for j in 1:(n-2)
        for k_ in (j+1):(n-1)
            for l in (k_+1):n
                @constraint(model,
                    M[[1, j+1, k_+1, l+1], [1, j+1, k_+1, l+1]] in PSDCone()
                )
            end
        end
    end

    # 7) Objective
    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))

    optimize!(model)
    return objective_value(model), value.(β)
end

############### 5) A helper to dispatch any of the six methods ###############
function solve_mkl_lower_bound(
    method::Symbol,
    K_list::Vector{Matrix{Float64}},
    y::Vector{Float64},
    C::Float64,
    λ::Float64,
    k::Int;
    sum_beta_val::Bool=true,
    L::Int=10
)
    if method == :perspective
        return mkl_psd_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    elseif method == :soc1
        return soc_relaxation_1_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    elseif method == :soc2random
        return soc_relaxation_2_solver_random(
            K_list, y, C, λ, k;
            sum_beta_val=sum_beta_val,
            L=L
        )
    elseif method == :soc3
        return soc_relaxation_3_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    elseif method == :soc4
        return soc_relaxation_4_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    else
        error("Unknown method symbol: $method")
    end
end

############### 6) Main driver ###############
function main()
    # Just test one dataset for demonstration
    datasets = [
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

    methods = [
        :perspective,
        :soc1,
        :soc2random,
        :soc3,
        :soc4
    ]

    C = 5.0
    λ = 100.0
    k = 3
    L = 1000   # number of random vectors for soc2random

    results = DataFrame(
        dataset = String[],
        method = String[],
        objective_value = Float64[],
        runtime = Float64[]
    )

    for dset in datasets
        println("\n=== DATASET: $dset ===")
        X_train, y_train, X_test, y_test = get_dataset(dset; force_download=false, frac=1.0, train_ratio=0.8)
        if size(X_train,1) != length(y_train)
            X_train = X_train'
        end
        if size(X_test,1) != length(y_test)
            X_test = X_test'
        end
        y_train = Float64.(y_train)
        y_test  = Float64.(y_test)
    
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
    
        for meth in methods
            println("  >> Solving with method: $meth")
    
            t0 = time()
            obj_val = nothing
            β_star = nothing
            try
                obj_val, β_star = solve_mkl_lower_bound(meth, K_list_train, y_train, C, λ, k;
                                                    sum_beta_val=true, L=L)
                runtime = time() - t0
                println("     objective = ", obj_val)
                println("     β* = ", β_star)
                println("     runtime = ", runtime)
    
                push!(results, (
                    string(dset),
                    string(meth),
                    obj_val,
                    runtime
                ))
            catch e
                runtime = time() - t0
                @warn "Method $meth failed on dataset $dset: $(sprint(showerror, e))"
                push!(results, (
                    string(dset),
                    string(meth),
                    NaN,
                    runtime
                ))
            end
        end
    end
    

    CSV.write("lower_bound_results.csv", results)
    println("\nAll done. Results stored in `mkl_results.csv`.")
end

end # module
