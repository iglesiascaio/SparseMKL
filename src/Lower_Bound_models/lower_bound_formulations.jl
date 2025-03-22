module LowerBoundFormulations

############### 1) Imports ###############
include("../../data/get_data.jl")
using .GetData: get_dataset

include("../MKL/multi_kernel.jl")
using .MKL: compute_kernels, compute_combined_kernel

using JuMP
using MosekTools     # For the perspective-based MISDP
using Gurobi         # For the two SOC relaxations
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


############### Method B: First SOC Relaxation (Gurobi) ###############
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
    # set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)
    @variable(model, 0 <= z[1:q] <= 1)

    # 1) margin constraints
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    # 2) for each j, s[j] = θ * ∑ β[i]*K_list[i][j,j], and s[j] >= γ[j]^2
    # @variable(model, s[1:n] >= 0)
    # for j in 1:n
    #     @constraint(model, s[j] == θ * sum(β[i]*K_list[i][j,j] for i=1:q))

    #     # Rotated SOC: s[j]*1 >= (γ[j])^2
    #     # => [s[j], 1, sqrt(2)*γ[j]] in RotatedSecondOrderCone()
    #     @constraint(model, [s[j], 1, sqrt(2)*γ[j]] in RotatedSecondOrderCone())
    # end
    @variable(model, s[1:n] >= 0)
    for j in 1:n
        @constraint(model, s[j] == sum(β[i]*K_list[i][j,j] for i=1:q))
        @constraint(model, [0.5 * θ; s[j]; γ[j]] in RotatedSecondOrderCone())
    end
    # 3) sum(β)=1 if desired
    if sum_beta_val
        @constraint(model, sum(β) == 1)
    end

    # 4) sum(z) <= k
    @constraint(model, sum(z) <= k)

    # 5) β[i]^2 <= z[i]* ω[i], use RSoC form
    for i in 1:q
        @constraint(model, z[i] >= 0)
        @constraint(model, ω[i] >= 0)
        @constraint(model, [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone())
    end

    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))
    optimize!(model)

    return objective_value(model), value.(β)
end



################ Method C: Second, Tighter SOC Relaxation (Using e_j) ################

function soc_relaxation_2_solver(
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
    # You can set optimizer attributes as needed, e.g.:
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
    # === 1) Margin constraints: 1 - σ[i] <= y[i]*(η + γ[i]) ===
    #
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    #
    # === 2) For each j, enforce  ( sum_i β[i] K_i[j,j ] ) * θ  >=  (γ[j])^2
    #     i.e. s[j] * θ >= γ[j]^2
    #     via rotated SOC: [ s[j], θ, sqrt(2)*γ[j] ] in RotatedSecondOrderCone()
    #
    @variable(model, s[1:n] >= 0)  # s[j] = sum( β[i]*K_i[j,j] )
    for j in 1:n
        @constraint(model, s[j] == sum( β[i]*K_list[i][j,j] for i=1:q ))
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
    # === 5) β[i]^2 <= z[i]*ω[i] => [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone()
    #
    for i in 1:q
        @constraint(model, [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone())
    end

    #
    # === Objective: C * ∑σ[i] + 0.5*θ + λ * ∑ω[i] ===
    #
    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))

    optimize!(model)

    return objective_value(model), value.(β)
end


using JuMP, Gurobi, LinearAlgebra, Random

"""
    soc_relaxation_2_solver_D(K_list, y, C, λ, k; sum_beta_val=true, L=10)

An adaptation of Method C that, in addition to the standard basis vector
constraints, also imposes, for L randomly generated unit vectors x:

    [ r(x), θ, √2·(xᵀγ) ] ∈ RotatedSecondOrderCone()

where r(x) = ∑₍i₌1₎^q β[i]*(xᵀ K_i x). This further tightens the SOC
relaxation by sampling more directions from the unit sphere.
"""
function soc_relaxation_2_solver_D(
    K_list::Vector{Matrix{Float64}},
    y::Vector{Float64},
    C::Float64,
    λ::Float64,
    k::Int;
    sum_beta_val::Bool = true,
    L::Int = 10
)
    # Dimensions and number of kernels.
    n = size(K_list[1], 1)
    q = length(K_list)

    # Create the model using Gurobi.
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    # === Decision Variables ===
    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)
    @variable(model, 0 <= z[1:q] <= 1)

    # === 1) Margin Constraints ===
    # For each data point, 1 - σ[i] <= y[i]*(η + γ[i])
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    # === 2a) SOC Constraints for the Standard Basis Directions ===
    # For each j, define diag_s[j] = ∑₍i₌1₎^q β[i]*K_list[i][j,j]
    # and impose [ diag_s[j], θ, √2·γ[j] ] ∈ RSoC, i.e.
    # diag_s[j] * θ ≥ γ[j]^2.
    @variable(model, diag_s[1:n] >= 0)
    for j in 1:n
        @constraint(model, diag_s[j] == sum(β[i] * K_list[i][j,j] for i in 1:q))
        @constraint(model, [diag_s[j], θ, sqrt(2)*γ[j]] in RotatedSecondOrderCone())
    end

    # === 2b) SOC Constraints for L Random Unit Vectors ===
    # Generate L random unit vectors in ℝⁿ.
    random_vectors = Vector{Vector{Float64}}(undef, L)
    for l in 1:L
        v = randn(n)
        v /= norm(v)  # Normalize to unit length.
        random_vectors[l] = v
    end

    # For each random vector x (indexed by l), define:
    #   r[l] = ∑₍i₌1₎^q β[i]*(xᵀ*K_list[i]*x)
    # and impose [ r[l], θ, √2·(xᵀγ) ] ∈ RSoC.
    @variable(model, rand_s[1:L] >= 0)
    for l in 1:L
        @constraint(model,
            rand_s[l] == sum(β[i] * (random_vectors[l]' * K_list[i] * random_vectors[l])
                              for i in 1:q)
        )
        # Compute the affine expression xᵀγ.
        @constraint(model,
            [rand_s[l], θ, sqrt(2) * (sum(random_vectors[l][j]*γ[j] for j in 1:n))]
            in RotatedSecondOrderCone()
        )
    end

    # === 3) Enforce sum(β) = 1 if desired ===
    if sum_beta_val
        @constraint(model, sum(β) == 1)
    end

    # === 4) Cardinality Constraint on z: sum(z) <= k ===
    @constraint(model, sum(z) <= k)

    # === 5) Linking β and z via a rotated SOC constraint ===
    # For each i, enforce β[i]² <= z[i]*ω[i] by
    # requiring [ z[i], ω[i], √2·β[i] ] ∈ RSoC.
    for i in 1:q
        @constraint(model, [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone())
    end

    # === 6) Objective ===
    # Minimize: C * ∑σ[i] + 0.5 * θ + λ * ∑ω[i]
    @objective(model, Min, C * sum(σ) + 0.5 * θ + λ * sum(ω))

    optimize!(model)

    return objective_value(model), value.(β)
end



############### 5) A helper to dispatch any of the four methods ###############
function solve_mkl_problem(
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
        return psd_2x2_mkl_sdp_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    elseif method == :soc1
        return soc_relaxation_1_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    elseif method == :soc2
        return soc_relaxation_2_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    elseif method == :soc2random
        # Our new "Method D"
        return soc_relaxation_2_solver_D(
            K_list, y, C, λ, k;
            sum_beta_val=sum_beta_val,
            L=L
        )
    else
        error("Unknown method symbol: $method")
    end
end


############### 6) Main driver ###############
function main()
    # Just test one dataset for demonstration
    datasets = [:parkinsons]

    # We now also add :soc2random to show the new method
    methods = [
        # :perspective, 
        # :soc1, 
        :soc2, 
        :soc2random
        ]

    C = 5.0
    λ = 100.0
    k = 3
    L = 1000   # number of random vectors

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
                obj_val, β_star = solve_mkl_problem(meth, K_list_train, y_train, C, λ, k;
                                                    sum_beta_val=true, L=L)
                runtime = time() - t0
                println("     objective = ", obj_val)
                println("     β* = ", β_star)

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

    CSV.write("mkl_results.csv", results)
    println("\nAll done. Results stored in `mkl_results.csv`.")
end

end # module
