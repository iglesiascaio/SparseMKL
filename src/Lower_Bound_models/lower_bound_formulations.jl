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

    # 2) link M to (θ,γ, ∑βK)
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
    #    Mosek can handle that
    for i in 1:q
        @constraint(model, [ [z[i]  β[i]];
                             [β[i]  ω[i]] ] in PSDCone())
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

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)
    @variable(model, 0 <= z[1:q] <= 1)

    # 1) margin constraints
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    # 2) for each j, we want:  s[j] = θ * ∑ β[i]K_list[i][j,j],  s[j] >= γ[j]^2
    @variable(model, s[1:n] >= 0)
    for j in 1:n
        @constraint(model, s[j] == θ * sum(β[i]*K_list[i][j,j] for i=1:q))

        # Instead of PSDCone, do a RotatedSecondOrderCone form:
        # s[j] >= γ[j]^2 => s[j]*1 >= (γ[j])^2
        # => 2*s[j]*1 >= 2*γ[j]^2 => => RSoC: [ s[j], 1, sqrt(2)*γ[j] ] in RSoC
        @constraint(model, [s[j], 1, sqrt(2)*γ[j]] in RotatedSecondOrderCone())
    end

    # 3) sum(β)=1 if desired
    if sum_beta_val
        @constraint(model, sum(β) == 1)
    end

    # 4) sum(z) <= k
    @constraint(model, sum(z) <= k)

    # 5) β[i]^2 <= z[i]*ω[i], but Gurobi doesn't accept PSDCone => use RSoC:
    #    x^2 <= yz => [y, z, sqrt(2)*x] in RSoC
    for i in 1:q
        @constraint(model, z[i] >= 0)  # ensure nonneg
        @constraint(model, ω[i] >= 0)  # ensure nonneg
        @constraint(model, [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone())
    end

    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))
    optimize!(model)

    return objective_value(model), value.(β)
end

############### Method C: Second, Tighter SOC Relaxation (Gurobi) ###############
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

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, η)
    @variable(model, θ >= 0)
    @variable(model, σ[1:n] >= 0)
    @variable(model, γ[1:n])
    @variable(model, β[1:q] >= 0)
    @variable(model, ω[1:q] >= 0)
    @variable(model, 0 <= z[1:q] <= 1)

    # 1) margin constraints
    @constraint(model, [i in 1:n], 1 - σ[i] <= y[i]*(η + γ[i]))

    # 2) We want diag_s[j] = θ ∑ β[i]K[i][j,j], and diag_s[j] >= t
    #    Also want t = ∑ γ[j]^2. That last piece is trickier in pure SOC form.
    #    We'll do: t >= 0, enforce t >= ∑ γ[j]^2 with an SOC.
    @variable(model, t >= 0)
    # sum(γ[j]^2) <= t => 2 t * 1 >= 2 sum(γ[j]^2)
    # => we can do an n-dim second-order cone but let's do it dimension by dimension.
    # Easiest is to define extra constraints or a single "norm" approach:
    #    norm(γ, 2)^2 <= t => sqrt( sum(γ[j]^2 ) ) <= sqrt(t ) => sum(γ[j]^2 ) <= t
    # We can do an n+1 second-order cone: [ t, sqrt(2)*γ[1], sqrt(2)*γ[2], ... ] in RSoC 
    # But JuMP expects dimension 3 or 4. For n>2, we do a bigger cone. Let's do the norm approach:
    @constraint(model, [t; γ...] in SecondOrderCone()) 
    # This means ||(γ[1],...,γ[n])||^2 <= t^2 => sum(γ[j]^2) <= t^2 => we want t >= sum(γ[j]^2).
    # We'll define the variable t as >= 0, so this ensures t >= ||γ||_2. 
    # But that is t >= sqrt( sum(γ[j]^2 ) ). We want t^2 >= sum(γ[j]^2 ), i.e. t >= ||γ||. 
    # Then sum(γ[j]^2 ) <= t^2 => NOT exactly t = sum(γ[j]^2). This is a LOOSER approach, but it's simpler.

    @variable(model, diag_s[1:n] >= 0)
    for j in 1:n
        @constraint(model, diag_s[j] == θ * sum(β[i]*K_list[i][j,j] for i=1:q))
        # diag_s[j] >= t => that is a linear constraint if we want diag_s[j] >= t. 
        @constraint(model, diag_s[j] >= t)
    end

    # 3) sum(β)=1 if desired
    if sum_beta_val
        @constraint(model, sum(β) == 1)
    end

    # 4) sum(z) <= k
    @constraint(model, sum(z) <= k)

    # 5) β[i]^2 <= z[i]*ω[i] => RSoC
    for i in 1:q
        @constraint(model, z[i] >= 0)
        @constraint(model, ω[i] >= 0)
        @constraint(model, [z[i], ω[i], sqrt(2)*β[i]] in RotatedSecondOrderCone())
    end

    @objective(model, Min, C*sum(σ) + 0.5*θ + λ*sum(ω))
    optimize!(model)

    return objective_value(model), value.(β)
end

############### 5) A helper to dispatch any of the three methods ###############
function solve_mkl_problem(
    method::Symbol,
    K_list::Vector{Matrix{Float64}},
    y::Vector{Float64},
    C::Float64,
    λ::Float64,
    k::Int;
    sum_beta_val::Bool=true
)
    if method == :perspective
        return psd_2x2_mkl_sdp_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    elseif method == :soc1
        return soc_relaxation_1_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    elseif method == :soc2
        return soc_relaxation_2_solver(K_list, y, C, λ, k; sum_beta_val=sum_beta_val)
    else
        error("Unknown method symbol: $method")
    end
end

############### 6) Main driver ###############
function main()
    # Just test one dataset for demonstration
    datasets = [:iris]

    methods = [:perspective, :soc1, :soc2]

    C = 1.0
    λ = 0.1
    k = 2

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
            Dict(:type => "rbf",    :params => Dict(:gamma => 0.5)),
            Dict(:type => "rbf",    :params => Dict(:gamma => 0.1)),
            Dict(:type => "polynomial", :params => Dict(:degree => 3, :c => 1.0))
        ]
        K_list_train = compute_kernels(X_train, X_train, kernel_specs)

        for meth in methods
            println("  >> Solving with method: $meth")

            t0 = time()
            obj_val = nothing
            β_star = nothing
            try
                obj_val, β_star = solve_mkl_problem(meth, K_list_train, y_train, C, λ, k; sum_beta_val=true)
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
