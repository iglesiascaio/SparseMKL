module InterpretableMKL

using CSV,
      DataFrames,
      LinearAlgebra,
      Statistics,
      Random,
      StatsBase,
      Plots,
      Infiltrator,
      Debugger,
      Revise,
      JuMP,
      Gurobi,
      LIBSVM

include("../MKL/multi_kernel.jl")  # compute_combined_kernel, etc.
include("./gssp.jl")              # GSSP algorithm for β
using .MKL: compute_combined_kernel
using .GSSPAlgorithm: GSSP


"""
    symmetrize!(K; tol=1e-10)

Ensures matrix K is symmetric within a given tolerance.
- If `max(|K[i,j] - K[j,i]|) > tol`, raises an error.
- Otherwise, symmetrizes K in-place.

### Arguments:
- `K::Matrix{Float64}`: The matrix to check and symmetrize.
- `tol::Float64=1e-10`: The allowed asymmetry tolerance.

### Returns:
- The symmetrized matrix (or original if already symmetric).
"""
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





################################################################################
#                   B E T A - U P D A T E   F U N C T I O N S
################################################################################

"""
    sparse_optimize_beta(K_list, α, y, λ, k0)

A hard-thresholding approach to produce a β with at most k0 non-zero entries.
"""
function sparse_optimize_beta(K_list, α, y, λ, k0)
    if k0 == 0
        @warn "k0=0 ⇒ all β=0 (may be meaningless)."
        return zeros(length(K_list))
    end

    q  = length(K_list)
    yα = y .* α
    # u[i] = (yα)' Kᵢ (yα)
    u  = [yα' * K_list[i] * yα for i in 1:q]

    β_unconstrained = u ./(4λ)

    # Keep top k0
    if k0 < q
        top_inds = partialsortperm(β_unconstrained, 1:k0; rev=true)
    else
        top_inds = 1:q
    end

    β = zeros(q)
    β[top_inds] = β_unconstrained[top_inds]

    # Normalize
    s = sum(β)
    if s > 0
        β ./= s
    else
        β .= 1.0 / q
    end

    return β
end


function sparse_optimize_beta_proximal(K_list, α, y, β_old, η::Float64, k0::Int, λ::Float64)
        """
        Implements a proximal-type β update for interpretable MKL
        """
        q   = length(K_list)
        yα  = y .* α

        # 1) Compute νᵢ = 0.5 * (yα' * Kᵢ * yα)
        ν = [0.5 * (yα' * K_list[i] * yα) for i in 1:q]

        β_star = similar(β_old)
        Δ      = similar(β_old)
        for i in 1:q
            # Unconstrained prox solution
            num      = ν[i] + 2*η*β_old[i]
            den      = 2*(λ + η)
            β_star[i] = num/den

            # Score that decides which Beta indices to keep (the more negative, the better to keep)
            Δ[i] = - (ν[i] + 2*η*β_old[i])^2 / (4*(λ + η)) + η*β_old[i]^2
        end

        β_new = zeros(q)
        if k0 < q
            # Indices of k0 smallest Δᵢ
            keep_inds = partialsortperm(Δ, 1:k0)
            β_new[keep_inds] = β_star[keep_inds]
        else
            β_new .= β_star
        end

        return β_new
    end


"""
    sparse_optimize_beta_gssp(K_list, α, y, λ, k0, sum_beta_val)

Uses a specialized GSSP method to pick up to k0 large components.
"""
function sparse_optimize_beta_gssp(K_list, α, y, λ, k0, sum_beta_val)
    # 1) Compute w = ν / (4λ), where νᵢ = (yα)' * Kᵢ * (yα)
    q   = length(K_list)
    yα  = y .* α
    ν   = [yα' * K_list[i] * yα for i in 1:q]
    w   = ν ./ (4λ)

    # 2) Run the GSSP algorithm
    β   = GSSP(w, k0, sum_beta_val)
    return β
end



################################################################################
#                    D U A L   O B J E C T I V E
################################################################################

"""
    compute_objective(α, y, K, β)

Compute the SVM dual objective = sum(α) - 0.5 * Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K[i,j].
"""
function compute_objective(α, y, K, β, λ)
    sum(α) - 0.5 * sum(y[i]*y[j]*α[i]*α[j]*K[i,j]
                       for i in eachindex(y), j in eachindex(y)) + λ * sum(β.^2)
end


################################################################################
#                    1) SMO  S O L V E R
################################################################################

"""
    train_svm_smo!(K, y, C; tol=1e-3, eps=1e-3, max_passes=5) -> (α, b)

Sequential Minimal Optimization for the SVM dual. Returns α, b.
"""

function train_svm_smo!(
    K::Matrix{Float64},
    y::Vector{Float64},
    C::Float64;
    tol::Float64 = 1e-3,
    eps::Float64 = 1e-3,
    max_passes::Int = 5
)
    """
    A faster SMO SVM dual solver with minimal overhead.

    K: kernel matrix (n×n)
    y: ±1 label vector of length n
    C: box constraint
    tol: KKT tolerance
    eps: threshold for tiny changes in α
    max_passes: how many passes with no alpha changes before we stop
    """
    @assert size(K,1) == size(K,2) == length(y)
    n = length(y)

    # Initialize alpha and threshold
    α = zeros(n)
    b = 0.0

    # Error cache E[i] = f(x_i) - y_i
    E = -copy(y)  # since f(x_i)=0 initially

    # We update E incrementally.  This avoids big matrix ops each time.
    # i.e. for any i, f(i) = sum( α[j]*y[j]*K[i,j] ) + b.
    # Then E[i] = f(i) - y[i].

    @inline function f(i)
        # We do not call this often, only inside incremental updates if needed
        # because we keep E[] in sync already.
        return sum(α[j]*y[j]*K[i,j] for j in 1:n) + b
    end

    # Incrementally update error cache after alpha changes
    @inline function updateErrorCache(i1, i2, oldAlpha1, oldAlpha2, newAlpha1, newAlpha2, bOld, bNew)
        dAlpha1 = newAlpha1 - oldAlpha1
        dAlpha2 = newAlpha2 - oldAlpha2
        db      = bNew - bOld
        @inbounds @simd for i in 1:n
            # E[i] += (y[i1]*dAlpha1*K[i1,i] + y[i2]*dAlpha2*K[i2,i] + db)
            E[i] = E[i] + y[i1]*dAlpha1*K[i,i1] + y[i2]*dAlpha2*K[i,i2] + db
            # (Switched K[i1,i] to K[i,i1], etc. if that’s more cache‐friendly)
        end
    end

    # The "takeStep" update for alpha1 & alpha2
    function takeStep(i1::Int, i2::Int)
        if i1 == i2
            return 0
        end

        alph1 = α[i1]
        alph2 = α[i2]
        y1    = y[i1]
        y2    = y[i2]
        E1    = E[i1]
        E2    = E[i2]
        s     = y1*y2

        # Bounds for alpha2
        if y1 != y2
            L = max(0.0, alph2 - alph1)
            H = min(C, C + alph2 - alph1)
        else
            L = max(0.0, alph1 + alph2 - C)
            H = min(C, alph1 + alph2)
        end
        if L == H
            return 0
        end

        # Save old
        oldAlpha1 = alph1
        oldAlpha2 = alph2
        bOld      = b

        # Compute eta
        k11 = K[i1,i1]
        k22 = K[i2,i2]
        k12 = K[i1,i2]
        eta = k11 + k22 - 2*k12

        # Unconstrained new alpha2 or fallback
        a2 = alph2
        if eta > 0
            a2 = alph2 + y2*(E1 - E2)/eta
            if a2 < L
                a2 = L
            elseif a2 > H
                a2 = H
            end
        else
            # fallback: evaluate objective at endpoints L, H
            # quick partial approach:
            # We define a small inline function objAt(a2test):
            @inline function objAt(a2test)
                a1test = alph1 + s*(alph2 - a2test)
                d1 = a1test - alph1
                d2 = a2test - alph2
                # approximate partial improvement in W(α)
                # We'll just do a quick linear+quadratic combo:
                return d1*E1 + d2*E2 - 0.5*k11*d1^2 - 0.5*k22*d2^2 - s*k12*d1*d2
            end
            Lobj = objAt(L)
            Hobj = objAt(H)
            if Lobj < (Hobj - eps)
                a2 = L
            elseif Lobj > (Hobj + eps)
                a2 = H
            else
                a2 = alph2
            end
        end

        if abs(a2 - alph2) < eps*(a2 + alph2 + eps)
            return 0
        end

        α[i2] = a2
        α[i1] = alph1 + s*(alph2 - a2)

        # Recompute threshold
        newAlpha1 = α[i1]
        newAlpha2 = α[i2]

        b1 = bOld - E1 - y1*(newAlpha1 - oldAlpha1)*k11 - y2*(newAlpha2 - oldAlpha2)*k12
        b2 = bOld - E2 - y1*(newAlpha1 - oldAlpha1)*k12 - y2*(newAlpha2 - oldAlpha2)*k22

        if 0 < newAlpha1 && newAlpha1 < C
            b = b1
        elseif 0 < newAlpha2 && newAlpha2 < C
            b = b2
        else
            b = 0.5*(b1 + b2)
        end

        # Update error cache
        updateErrorCache(i1, i2, oldAlpha1, oldAlpha2, newAlpha1, newAlpha2, bOld, b)

        return 1
    end

    # examineExample: tries i2, finds i1 that can make progress
    # We'll do a simpler version of Platt's heuristics without random shuffle.
    function examineExample(i2::Int)
        alph2 = α[i2]
        E2    = E[i2]
        r2    = E2*y[i2]

        # KKT check if outside tol
        if (r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0)
            # 1) second choice heuristic: pick i1 with max |E1 - E2|
            # We'll do a single pass to find best i1 among non-bound alphas
            best_i1     = -1
            best_diff   = -1.0
            # Build non-bound indices in a temporary static array:
            nonBoundCount = 0
            @inbounds for i in 1:n
                if α[i] > 0 && α[i] < C
                    nonBoundCount += 1
                end
            end
            if nonBoundCount > 1
                # we do a single pass to find i with largest error difference
                # no need for array: just keep track of best in a loop
                best_i1 = 0
                best_diff = -1.0
                @inbounds for i in 1:n
                    if α[i] > 0 && α[i] < C
                        local diff = abs(E[i] - E2)
                        if diff > best_diff
                            best_i1   = i
                            best_diff = diff
                        end
                    end
                end
                if best_i1 > 0
                    if takeStep(best_i1, i2) == 1
                        return 1
                    end
                end
            end

            # 2) if that fails, loop over non-bound alphas in order
            #    (we skip random shuffles to reduce overhead)
            @inbounds for i1 in 1:n
                if α[i1] > 0 && α[i1] < C
                    if takeStep(i1, i2) == 1
                        return 1
                    end
                end
            end

            # 3) if that fails, loop over entire set in order
            @inbounds for i1 in 1:n
                if takeStep(i1, i2) == 1
                    return 1
                end
            end
        end
        return 0
    end

    # SMO outer loop
    numChanged = 0
    examineAll = true
    passes     = 0

    while passes < max_passes
        numChanged = 0
        if examineAll
            # examine all i2
            @inbounds for i in 1:n
                numChanged += examineExample(i)
            end
        else
            # examine only non-bound
            @inbounds for i in 1:n
                if α[i] > 0 && α[i] < C
                    numChanged += examineExample(i)
                end
            end
        end

        if examineAll
            examineAll = false
        elseif numChanged == 0
            examineAll = true
        end

        if numChanged == 0
            passes += 1
        else
            passes = 0
        end
    end

    return α, b
end

################################################################################
#         G U R O B I    &    LIBSVM    -   HELPER METHODS
################################################################################

"""
    init_svm_model(n, y, C) -> (model, α_vars)

Builds a JuMP model with Gurobi:
 - α[i] in [0, C]
 - sum(α[i]*y[i]) = 0
Doesn't define objective yet (we'll do that each iteration).
"""
function init_svm_model(n::Int, y::Vector{Float64}, C::Float64)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    @variable(model, 0 <= α[1:n] <= C)
    @constraint(model, sum(α[i]*y[i] for i in 1:n) == 0)
    return model, α
end

"""
    update_svm_objective!(model, α_vars, K, y)

Re-define the dual objective:
Maximize sum(α[i]) - 0.5 * ∑ (y[i]y[j] α[i]α[j] K[i,j]).
"""
function update_svm_objective!(model::Model, α_vars::Vector{VariableRef},
                               K::Matrix{Float64}, y::Vector{Float64})
    n = length(y)
    @objective(model, Max,
        sum(α_vars[i] for i in 1:n)
        - 0.5 * sum(y[i]*y[j]*α_vars[i]*α_vars[j]*K[i,j] for i in 1:n, j in 1:n)
    )
end

"""
    train_svm_gurobi!(model, α_vars, K, y) -> (α, b)

Updates the model's objective with kernel K, solves, returns α, b=0.0 for now.
"""
function train_svm_gurobi!(model::Model, α_vars::Vector{VariableRef},
                           K::Matrix{Float64}, y::Vector{Float64})
    update_svm_objective!(model, α_vars, K, y)
    optimize!(model)
    α_opt = value.(α_vars)
    b_approx = 0.0
    return α_opt, b_approx
end


"""
    train_svm_libsvm(K, y, C) -> (α, b)

Trains an SVM in LIBSVM with a precomputed kernel K. The combined kernel is n×n,
and we build n×(n+1) for LIBSVM. Then b = -model.rho[1].
"""
function train_svm_libsvm(K::Matrix{Float64}, y::Vector{Float64}, C::Float64)
    @assert size(K,1) == size(K,2) "K must be n×n"
    println(size(K))
    @assert length(y) == size(K,1) "K, y size mismatch"

    symmetrize!(K)

    n = length(y)

    #    cost is passed as a keyword.
    model = LIBSVM.svmtrain(
        K,
        Float64.(y);
        svmtype = LIBSVM.SVC,
        kernel  = Kernel.Precomputed,
        cost    = C
    )


    # 4) Extract dual coefficients α from the trained model
    α_res = zeros(n)
    sv_inds = model.SVs.indices
    sv_coef = model.coefs[:, 1]   # for 2-class SVM, single column
    for (local_i, global_i) in enumerate(sv_inds)
        # LIBSVM encodes alpha_i as (sv_coef[i] * original_label_of_SV[i]).
        ## however, LIBSVM sometimes flips the sign of the label, so we take abs
        α_res[global_i] = abs(sv_coef[local_i])
    end

    # 5) The intercept in LIBSVM is rho[1] for 2-class SVM
    b_val = model.rho[1]

    return α_res, b_val
end


################################################################################
#        M A I N   I N T E R P R E T A B L E   M K L   T R A I N I N G 
################################################################################
# Helper function to initialize alphas randomly while satisfying the constraints:
#   0 ≤ αᵢ ≤ C   for all i
#   ∑ᵢ αᵢ yᵢ = 0
function random_init_alphas(n::Int, C::Float64, y::Vector{Float64})
    # Generate a random noise vector in [-0.5, 0.5]
    δ = rand(n) .- 0.5
    # Project δ onto the hyperplane {δ: dot(δ, y) = 0} so that the noise does not affect the constraint.
    δ = δ .- (dot(δ, y) / dot(y, y)) * y
    # Start from a neutral value C/2 for all alphas and add the noise.
    α = fill(C/2, n) .+ δ
    # Ensure box constraints hold.
    α = clamp.(α, 0.0, C)
    
    # (Optional) Adjust one element to force the equality constraint exactly.
    # Here we look for at least one positive (y==1) and one negative (y==-1) sample.
    pos_idx = findall(x -> x == 1.0, y)
    neg_idx = findall(x -> x == -1.0, y)
    if !isempty(pos_idx) && !isempty(neg_idx)
        discrepancy = dot(α, y)
        # Adjust the first positive element so that the weighted sum becomes zero.
        i = pos_idx[1]
        α[i] = clamp(α[i] - discrepancy, 0.0, C)
    end
    return α
end

function train_interpretable_mkl(
    X::Matrix{Float64},
    y::Vector{Float64},
    C::Float64,
    K_list::Vector{Matrix{Float64}},
    λ::Float64;
    max_iter::Int=100,
    tolerance::Float64=1e-5,
    k0::Int=3,
    sum_beta_val::Float64=1.0,
    solver_type::Symbol=:SMO,
    beta_method::Symbol=:gssp,
    max_non_decrease::Int=3
)
    @assert !isempty(K_list) "Empty kernel list!"
    n = size(X,1)
    q = length(K_list)

    ###################################################################
    # Step 1: Initialize β as a linear combination of three random kernels
    ###################################################################
    Random.seed!(10)
    random_indices = randperm(q)[1:k0]  # Select 3 random kernel indices
    β = zeros(q)
    β[random_indices] .= 1/k0  # Assign equal weight to 3 kernels
    β_old = copy(β)

    println("Initial random β = ", β)

    ###################################################################
    # Step 2: Compute the initial combined kernel using this β
    ###################################################################
    K_combined = compute_combined_kernel(K_list, β)
    symmetrize!(K_combined)
    K_combined_old = copy(K_combined)

    ###################################################################
    # Step 3: Solve for initial α using this kernel
    ###################################################################
    if solver_type == :SMO
        α, _ = train_svm_smo!(K_combined, y, C; tol=tolerance, eps=1e-3, max_passes=5)
    elseif solver_type == :GUROBI
        model, α_vars = init_svm_model(n, y, C)
        α, _ = train_svm_gurobi!(model, α_vars, K_combined, y)
    elseif solver_type == :LIBSVM
        α, _ = train_svm_libsvm(K_combined, y, C)
    else
        error("Unknown solver_type=$solver_type. Choose :SMO, :GUROBI, or :LIBSVM.")
    end

    α_old = copy(α)
    println("Initial α computed from first SVM step.")

    list_alphas = Vector{Vector{Float64}}()
    list_betas  = Vector{Vector{Float64}}()

    ###################################################################
    # Step 4: Enter the main optimization loop
    ###################################################################
    obj_best = Inf
    non_decrease_count = 0

    for iter in 1:max_iter
        println("Iteration $iter...")

        ###################################################################
        # 1) Optimize β (using the current α)
        ###################################################################
        if beta_method == :hard
            β = sparse_optimize_beta(K_list, α, y, λ, k0)
        elseif beta_method == :proximal
            β = sparse_optimize_beta_proximal(K_list, α, y, β_old, 100.0, k0, λ)
        elseif beta_method == :gssp
            β = sparse_optimize_beta_gssp(K_list, α, y, λ, k0, sum_beta_val)
        else
            error("Unknown beta_method=$beta_method. Choose :hard, :proximal, or :gssp.")
        end

        ###################################################################
        # 2) Recompute combined kernel with the updated β
        ###################################################################
        K_combined = compute_combined_kernel(K_list, β)
        symmetrize!(K_combined)

        ###################################################################
        # 3) Solve for α (SVM subproblem) with the new K_combined
        ###################################################################
        if solver_type == :SMO
            α, _ = train_svm_smo!(K_combined, y, C; tol=tolerance, eps=1e-3, max_passes=5)
        elseif solver_type == :GUROBI
            α, _ = train_svm_gurobi!(model, α_vars, K_combined, y)
        elseif solver_type == :LIBSVM
            α, _ = train_svm_libsvm(K_combined, y, C)
        else
            error("Unknown solver_type=$solver_type. Choose :SMO, :GUROBI, or :LIBSVM.")
        end

        ###################################################################
        # 4) Compute and check objective function
        ###################################################################
        obj = compute_objective(α, y, K_combined, β, λ)
        println("Objective = ", obj)
        println("β = ", β)
        println("sum(α) = ", sum(α))

        # Check if the current iteration yields a sufficient decrease.
        if obj_best - obj < tolerance
            non_decrease_count += 1
        else
            # reset the counter if the objective decreases
            non_decrease_count = 0
            obj_best = obj
            α_old .= α
            β_old .= β
            K_combined_old .= K_combined
        end

        # If the total count of iterations without sufficient decrease exceeds the limit, break.
        if non_decrease_count >= max_non_decrease
            println("Stopping criterion met: objective did not decrease enough in a total of $max_non_decrease iterations in a row.")
            println("Returning the best solution found so far.")
            println("Final β = ", β_old)
            println("Final objective = ", obj_best)
            return α_old, β_old, K_combined_old, list_alphas, list_betas
        end

        push!(list_alphas, copy(α))
        push!(list_betas,  copy(β))
    end

    return α, β, K_combined, list_alphas, list_betas
end
end # module InterpretableMKL
