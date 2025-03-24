module InterpretableMKL

    using CSV,
        DataFrames,
        LinearAlgebra,
        Statistics,
        Random,
        JuMP,
        Gurobi,
        StatsBase,
        Infiltrator,
        Debugger,
        Revise

        include("../MKL/multi_kernel.jl")
        include("./gssp.jl")
        using .MKL: compute_combined_kernel, train_svm
        using .GSSPAlgorithm: GSSP

    # Optimized function to compute sparse β
    function sparse_optimize_beta(K_list, α, y, λ, k0)
        # Check if k0 is 0 and raise a warning
        if k0 == 0
            @warn "k0 is set to 0. This will result in a zero β vector, which may not be meaningful."
            return zeros(length(K_list))  # Return all zeros since k0 = 0 implies no kernels are selected
        end

        q = length(K_list)  # Number of kernels
        yα = y .* α         # Element-wise product of labels and dual variables

        # Efficient computation of u[i] = (yα)' * K_i * (yα)
        u = [yα' * K * yα for K in K_list]

        # Unconstrained minimizer: β_unconstrained = u / (4λ)
        β_unconstrained = u ./ (4λ)

        # Hard-thresholding to enforce sparsity constraint ||β||_0 <= k0
        if k0 < q
            # Get indices of the top k0 elements in β_unconstrained
            top_indices = partialsortperm(β_unconstrained, 1:k0; rev=true)
        else
            # If k0 >= q, retain all indices
            top_indices = 1:q
        end

        # Construct sparse β vector
        β = zeros(q)
        β[top_indices] = β_unconstrained[top_indices]

        # println("β: ", β)
        # println("α: ", α)

        # Normalize β to avoid numerical instability (optional)
        if sum(β) > 0
            β /= sum(β)
        else
            β .= 1.0 / q  # Fallback to equal weights if all are zero
        end

        return β
    end


    function sparse_optimize_beta_proximal(K_list, α, y, β_old, η::Float64, k0::Int, λ::Float64)
        """
        Implements the proximal β-update for the interpretable MKL:
        """
        
        q = length(K_list)
        yα = y .* α  # elementwise product
    
        # 1) Compute νᵢ
        ν = [0.5 * (yα' * K_list[i] * yα) for i in 1:q]
    
        # 2) Compute Beta star and Δ that defines which indices to keep
        β_star = similar(β_old)
        Δ = similar(β_old)
        for i in 1:q
            # Unconstrained prox solution
            num = ν[i] + 2 * η * β_old[i]
            den = 2 * (λ + η)
            β_star[i] = num / den
        
            # Variable that decides which Beta indices to keep
            Δ[i] = - (ν[i] + 2*η*β_old[i])^2 / (4*(λ + η)) + η * β_old[i]^2
        end
        println("Δ: ", Δ)
        # 3) Hard-threshold to enforce at most k0 nonzeros
        β_new = zeros(q)
        if k0 < q
            # Indices of the k0 smallest Δᵢ
            keep_inds = partialsortperm(Δ, 1:k0)
            β_new[keep_inds] = β_star[keep_inds]
            println("keep_inds: ", keep_inds)
            println("β_new: ", β_new)
        else
            # If k0 >= q, keep all
            β_new .= β_star
        end
    
        return β_new
    end


    function sparse_optimize_beta_gssp(K_list, α, y, λ, k0, sum_beta_val)
        # 1) Compute w = ν / (4λ)
        q  = length(K_list)
        yα = y .* α
        ν  = [yα' * K_list[i] * yα for i in 1:q]
        w  = ν ./ (4λ)
    
        # 2) Run Algorithm 1 (GSSP)
        β  = GSSP(w, k0, sum_beta_val)
    
        return β
    end    

    function compute_objective(α, y, K, β)
        """
        Compute the objective function value for the interpretable MKL:
        """
        return sum(α) - 0.5 * sum(y[i]*y[j]*α[i]*α[j]*K[i,j] for i in eachindex(y), j in eachindex(y))
    end



    ###############################################################################
    # 1) Initialize an SVM model once.
    ###############################################################################
    function init_svm_model(n::Int, y::Vector{Float64}, C::Float64)
        """
        Create a JuMP model for the SVM dual problem with:
        - α[i] in [0, C]
        - ∑ α[i]*y[i] = 0
        We'll not specify the objective yet. We'll do that with update_svm_objective! later.
        Returns: (model, α_vars)
        """
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)  # silent mode

        # Create α[i] in [0, C]
        @variable(model, 0 <= α[1:n] <= C)

        # Add linear constraint: sum(α[i]*y[i]) = 0
        @constraint(model, sum(α[i]*y[i] for i in 1:n) == 0)

        # We won't define the objective yet. We'll do that each iteration in update_svm_objective!.
        return model, α
    end

    ###############################################################################
    # 2) Update the objective in place (via MOI or re-@objective)
    ###############################################################################
    function update_svm_objective!(model::Model, α_vars::Vector{VariableRef},
                                K::Matrix{Float64}, y::Vector{Float64})
        """
        In-place update of the SVM dual objective:
        Maximize sum(α) - 0.5 * sum(y[i] * y[j] * α[i] * α[j] * K[i,j])
        """
        # One approach: we can just rebuild the objective with a JuMP macro each time
        # because re-setting the objective is cheaper than building the entire model.
        # This still uses some JuMP overhead, but far less than building a brand-new model/constraints.
        #
        # We clear the old objective by setting a dummy first, then re-add:
        # (Alternatively, we could do lower-level MOI calls. For demonstration, we'll do the macro approach.)

        @objective(model, Max,
        sum(α_vars[i] for i in 1:length(y)) -
        0.5 * sum(y[i]*y[j]*α_vars[i]*α_vars[j]*K[i,j]
                    for i in 1:length(y), j in 1:length(y))
        )
    end

    ###############################################################################
    # 3) Solve the model with the updated kernel, and return α
    ###############################################################################
    function train_svm!(model::Model, α_vars::Vector{VariableRef},
                        K::Matrix{Float64}, y::Vector{Float64})
        """
        Update the SVM objective with the given K, then solve.
        Returns the vector of α values.
        """
        # Update objective:
        update_svm_objective!(model, α_vars, K, y)

        # Solve
        optimize!(model)

        # Extract α
        α_values = value.(α_vars)
        return α_values
    end

    ###############################################################################
    # 4) The main loop with interpretable MKL
    ###############################################################################
    function train_interpretable_mkl(
        X::Matrix{Float64}, y::Vector{Float64}, C::Float64,
        K_list::Vector{Matrix{Float64}}, λ::Float64;
        max_iter::Int=100, tolerance::Float64=1e-5, k0::Int=3, sum_beta_val::Float64=1.0
    )
        q = length(K_list)    # number of kernels
        n = size(X, 1)        # number of data points

        # Initialize β and α
        β = ones(q) ./ q      # start with equal weights
        β_old = copy(β)
        α = zeros(n)          # dummy initial α
        K_combined = compute_combined_kernel(K_list, β)
        list_alphas = []
        list_betas = []

        # ----------------------------------------------------------------------
        # Build the SVM model once, outside the loop:
        # ----------------------------------------------------------------------
        model, α_vars = init_svm_model(n, y, C)

        for iter in 1:max_iter
            println("Iteration $iter...")

            # === Step 1: Optimize α given β ===
            # We already have a single model. We'll just update its kernel matrix and solve.
            α = train_svm!(model, α_vars, K_combined, y)

            # === Step 2: Optimize β given α ===
            # β = sparse_optimize_beta(K_list, α, y, λ, k0)
            # β = sparse_optimize_beta_proximal(K_list, α, y, β_old, 15.0, k0, λ)
            β = sparse_optimize_beta_gssp(K_list, α, y, λ, k0, sum_beta_val)

            # Recompute the combined kernel with the new β
            K_combined = compute_combined_kernel(K_list, β)

            # Convergence check (optional)
            if iter > 1 && norm(β - β_old) < tolerance
                println("Converged after $iter iterations.")
                println("Final β: ", β)
                println("Final objective function: ", compute_objective(α, y, K_combined, β))
                break
            end

            println("β: ", β)
            println("Objective function: ", compute_objective(α, y, K_combined, β))

            β_old .= β
            push!(list_alphas, copy(α))
            push!(list_betas, copy(β))
        end



        return α, β, K_combined, list_alphas, list_betas
    end

end # module InterpretableMKL
