# module InterpretableMKL

#     using CSV,
#         DataFrames,
#         LinearAlgebra,
#         Statistics,
#         Random,
#         JuMP,
#         Gurobi,
#         StatsBase,
#         Plots,
#         Infiltrator,
#         Debugger,
#         Revise

#     include("../MKL/multi_kernel.jl")
#     using .MKL: compute_combined_kernel, train_svm

#     # Optimized function to compute sparse β
#     function sparse_optimize_beta(K_list, α, y, λ, k0)
#         # Check if k0 is 0 and raise a warning
#         if k0 == 0
#             @warn "k0 is set to 0. This will result in a zero β vector, which may not be meaningful."
#             return zeros(length(K_list))  # Return all zeros since k0 = 0 implies no kernels are selected
#         end

#         q = length(K_list)  # Number of kernels
#         yα = y .* α         # Element-wise product of labels and dual variables

#         # Efficient computation of u[i] = (yα)' * K_i * (yα)
#         u = [yα' * K * yα for K in K_list]

#         # Unconstrained minimizer: β_unconstrained = u / (4λ)
#         β_unconstrained = u ./ (4λ)

#         # Hard-thresholding to enforce sparsity constraint ||β||_0 <= k0
#         if k0 < q
#             # Get indices of the top k0 elements in β_unconstrained
#             top_indices = partialsortperm(β_unconstrained, 1:k0; rev=true)
#         else
#             # If k0 >= q, retain all indices
#             top_indices = 1:q
#         end

#         # Construct sparse β vector
#         β = zeros(q)
#         β[top_indices] = β_unconstrained[top_indices]
        
#         # Normalize β to avoid numerical instability (optional)
#         if sum(β) > 0
#             β /= sum(β)
#         else
#             β .= 1.0 / q  # Fallback to equal weights if all are zero
#         end

#         return β
#     end


#     # Extended interpretable MKL-SVM training function
#     function train_interpretable_mkl(X, y, C, K_list, λ; max_iter=100, tolerance=1e-5, k0=3)
#         q = length(K_list)  # Number of kernels
#         n = size(X, 1)  # Number of data points

#         # Initialize β and α
#         β = ones(q) / q  # Start with equal weights
#         β_old = copy(β)  # Initialize β_old for convergence check
#         α = zeros(n)     # Initialize α
#         K_combined = compute_combined_kernel(K_list, β)

#         for iter in 1:max_iter
#             println("Iteration $iter...")

#             # Step 1: Optimize α given β
#             ## CONTINUE USING GUROBI HERE FOR NOW
#             α, K_combined = train_svm(X, y, C, "precomputed"; K_combined=K_combined)

#             # Step 2: Optimize β given α
#             ### CHANGE HERE FOR STEP 1 OF NEXT STEPS - believes the optimal value can be written in closed form
#             β = sparse_optimize_beta(K_list, α, y, λ, k0)

#             # Update the combined kernel
#             K_combined = compute_combined_kernel(K_list, β)

#             # Convergence check (optional)
#             if iter > 1 && norm(β - β_old) < tolerance
#                 println("Converged after $iter iterations.")
#                 break
#             end
#             β_old = copy(β)
#         end

#         return α, β, K_combined
#     end

# end



module InterpretableMKL

    using CSV,
        DataFrames,
        LinearAlgebra,
        Statistics,
        Random,
        JuMP,
        Gurobi,
        StatsBase,
        Plots,
        Infiltrator,
        Debugger,
        Revise

        include("../MKL/multi_kernel.jl")
        using .MKL: compute_combined_kernel, train_svm

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
        
        # Normalize β to avoid numerical instability (optional)
        if sum(β) > 0
            β /= sum(β)
        else
            β .= 1.0 / q  # Fallback to equal weights if all are zero
        end

        return β
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
        max_iter::Int=100, tolerance::Float64=1e-5, k0::Int=3
    )
        q = length(K_list)    # number of kernels
        n = size(X, 1)        # number of data points

        # Initialize β and α
        β = ones(q) ./ q      # start with equal weights
        β_old = copy(β)
        α = zeros(n)          # dummy initial α
        K_combined = compute_combined_kernel(K_list, β)

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
            β = sparse_optimize_beta(K_list, α, y, λ, k0)

            # Recompute the combined kernel with the new β
            K_combined = compute_combined_kernel(K_list, β)

            # Convergence check (optional)
            if iter > 1 && norm(β - β_old) < tolerance
                println("Converged after $iter iterations.")
                break
            end

            β_old .= β
        end

        return α, β, K_combined
    end

end # module InterpretableMKL
