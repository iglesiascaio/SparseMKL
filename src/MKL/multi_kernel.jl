module MKL

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


    # Function to make predictions using the trained SVM model
    function predict_svm(
        α,
        y_train,
        X_train,
        X_new,
        b,
        kernel_type;
        degree = 3,
        gamma = 1.0,
        tolerance = 1e-5,
        K_combined = nothing,
    )
        # Identify support vectors
        support_indices = findall(α .> tolerance)
        α_support = α[support_indices]
        y_support = y_train[support_indices]
        X_support = X_train[support_indices, :]

        n_new = size(X_new, 1)
        predictions = zeros(n_new)

        if isnothing(K_combined)
            # Compute the kernel between support vectors and new data
            if kernel_type == "linear"
                K_new = X_support * X_new'
            elseif kernel_type == "polynomial"
                K_new = (X_support * X_new' .+ 1) .^ degree
            elseif kernel_type == "rbf"
                K_new = zeros(length(support_indices), n_new)
                for i = 1:length(support_indices)
                    for j = 1:n_new
                        K_new[i, j] = exp(-gamma * norm(X_support[i, :] - X_new[j, :])^2)
                    end
                end
            else
                throw(ArgumentError("Unsupported kernel type: $kernel_type"))
            end
        else
            # Use the precomputed kernel matrix
            K_new = K_combined
        end

        # Make predictions
        for i = 1:n_new
            s = sum(α_support .* y_support .* K_new[:, i])
            predictions[i] = s + b
        end

        # Return the sign of the predictions (+1 or -1)
        return sign.(predictions)
    end


    # Function to compute kernel matrix between two datasets
    function compute_kernel(X_train, X_new, kernel_type; degree = 3, gamma = 1.0)
        n_train = size(X_train, 1)
        n_new = size(X_new, 1)
        K = zeros(n_train, n_new)

        if kernel_type == "linear"
            K = X_train * X_new'  # Linear kernel
        elseif kernel_type == "polynomial"
            K = (X_train * X_new' .+ 1) .^ degree  # Polynomial kernel
        elseif kernel_type == "rbf"
            for i = 1:n_train
                for j = 1:n_new
                    K[i, j] = exp(-gamma * norm(X_train[i, :] - X_new[j, :])^2)  # RBF kernel
                end
            end
        else
            throw(ArgumentError("Unsupported kernel type: $kernel_type"))
        end

        return K
    end

    function compute_kernels(X1::AbstractMatrix{<:Real}, X2::AbstractMatrix{<:Real}, kernels::AbstractVector{<:Dict})
        """
        Compute multiple kernels for given data matrices X1 and X2 efficiently.
    
        Parameters:
            X1: Matrix of size (n1, d) - Data on one side of the kernel.
            X2: Matrix of size (n2, d) - Data on the other side of the kernel.
            kernels: Vector of Dicts where each Dict specifies:
                - :type (String): Type of kernel ("linear", "polynomial", "rbf", "sigmoid", "laplacian", "chi_squared").
                - :params (Dict): Hyperparameters for the kernel (e.g., degree, c, gamma, c0).
    
        Returns:
            Vector of kernel matrices corresponding to the specified kernels.
        """
        n1, d = size(X1)
        n2, d2 = size(X2)
        @assert d == d2 "X1 and X2 must have the same number of columns."
    
        # Determine which precomputations are needed.
        need_inner = any(kernel[:type] in ("linear", "polynomial", "sigmoid", "rbf") for kernel in kernels)
        need_rbf     = any(kernel[:type] == "rbf" for kernel in kernels)
        need_laplacian = any(kernel[:type] == "laplacian" for kernel in kernels)
        need_chi_squared = any(kernel[:type] == "chi_squared" for kernel in kernels)
    
        # Precompute X1*X2' if any kernel requires an inner product.
        X1X2t = need_inner ? X1 * X2' : nothing
    
        # Precompute squared norms for RBF.
        X1_sq = nothing
        X2_sq = nothing
        if need_rbf
            X1_sq = sum(abs2, X1; dims=2)  # (n1×1)
            X2_sq = sum(abs2, X2; dims=2)  # (n2×1)
        end
    
        # Precompute Manhattan distances for Laplacian kernel.
        laplacian_dists = nothing
        if need_laplacian
            laplacian_dists = zeros(eltype(X1), n1, n2)
            @inbounds for j in 1:d
                @views laplacian_dists .+= abs.(X1[:, j] .- X2[:, j]')
            end
        end
    
        # Precompute chi-squared distances.
        chi_sq_dists = nothing
        if need_chi_squared
            chi_sq_dists = zeros(eltype(X1), n1, n2)
            @inbounds for j in 1:d
                @views chi_sq_dists .+= ((X1[:, j] .- X2[:, j]').^2) ./ (X1[:, j] .+ X2[:, j]')
            end
        end
    
        # Allocate the vector of kernel matrices.
        kernel_matrices = Vector{Matrix{Float64}}(undef, length(kernels))
        for (i, kernel) in enumerate(kernels)
            kernel_type = kernel[:type]
            params = kernel[:params]
            if kernel_type == "linear"
                # Return a copy to avoid accidental mutation.
                kernel_matrices[i] = copy(X1X2t)
            elseif kernel_type == "polynomial"
                degree = get(params, :degree, 3)
                # gamma  = get(params, :gamma, 1.0)  
                c      = get(params, :c, 1.0)
                # standard polynomial kernel: (gamma * xᵀx' + c)^degree
                kernel_matrices[i] = (0.01 .* X1X2t .+ c) .^ degree
            
            elseif kernel_type == "rbf"
                gamma = get(params, :gamma, 1.0)
                # Compute squared Euclidean distances.
                dists = X1_sq .- (2 .* X1X2t) .+ X2_sq'
                # Optionally, to avoid tiny negative numbers due to floating-point errors, you might add:
                # dists = max.(dists, 0.0)
                kernel_matrices[i] = exp.(-gamma .* dists)
            elseif kernel_type == "sigmoid"
                gamma = get(params, :gamma, 1.0)
                c0 = get(params, :c0, 0.0)
                kernel_matrices[i] = tanh.(gamma .* X1X2t .+ c0)
            elseif kernel_type == "laplacian"
                gamma = get(params, :gamma, 1.0)
                kernel_matrices[i] = exp.(-gamma .* laplacian_dists)
            elseif kernel_type == "chi_squared"
                gamma = get(params, :gamma, 1.0)
                kernel_matrices[i] = exp.(-gamma .* chi_sq_dists)
            else
                error("Unsupported kernel type: $kernel_type")
            end
        end
    
        return kernel_matrices
    end
    

    # Train SVM model with kernel options
    function train_svm(X, y, C, kernel_type; degree=3, gamma=1.0, K_combined=nothing)
        n = size(X, 1)
        
        # Use the provided precomputed kernel matrix, or compute the kernel matrix
        if K_combined === nothing
            K = compute_kernel(X, X, kernel_type; degree=degree, gamma=gamma)
        else
            K = K_combined  # Use the precomputed kernel matrix
        end


        # Start the JuMP model
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)  # Suppress Gurobi output

        # Define the dual variables α
        @variable(model, 0 <= α[i=1:n] <= C)

        # Define the objective function
        @objective(model, Max, sum(α) - 0.5 * sum((y[i] * y[j] * α[i] * α[j] * K[i, j])
                                                for i in 1:n, j in 1:n))

        # Add the constraint: ∑ α_i * y_i = 0
        @constraint(model, sum(α[i] * y[i] for i in 1:n) == 0)

        # Solve the model
        optimize!(model)

        # Retrieve the α values
        α_values = value.(α)

        return α_values, K  # Return both α and K
    end


    # Function to compute the combined kernel matrix K(β)
    function compute_combined_kernel(K_list, β)

        # @infiltrate

        # Ensure that the kernel matrices do not have very large/small values
        for j in 1:length(K_list)
            K_list[j] .= clamp.(K_list[j], -1e9, 1e9)  # Prevent overflow in kernels
        end

        # Verify that K_list is not empty
        if isempty(K_list)
            throw(ArgumentError("K_list is empty."))
        end
        
        # Initialize K_combined based on the size of the first kernel matrix
        K_combined = zeros(size(K_list[1]))
        
        # Iterate over each kernel and its corresponding weight β[j]
        for j in 1:length(K_list)
            # Verify that all kernel matrices have the same size
            if size(K_list[j]) != size(K_combined)
                throw(DimensionMismatch("All kernel matrices must have the same dimensions. Kernel $j has size $(size(K_list[j])), expected size $(size(K_combined))."))
            end
            
            # Accumulate the weighted kernel
            K_combined .+= β[j] * K_list[j]
        end
        
        return K_combined
    end



    function optimize_beta(K_list, α, y, λ)
        q = length(K_list)  # Number of kernels
        c = zeros(q)

        # Compute c_k for each kernel
        for k in 1:q
            c[k] = sum((y .* α) .* (K_list[k] * (y .* α)))
        end

        # Optimize β using the closed-form solution
        β = max.(0, c / (4 * λ))

    #     Normalize β to avoid numerical instability
        if sum(β) > 0
            β /= sum(β)
        else
            β .= 1.0 / q  # Fallback to equal weights
        end

        return β
    end


    function optimize_beta_gurobi(K_list, α, y, λ)
        q = length(K_list)  # Number of kernels
        n = size(K_list[1], 1)  # Number of data points

        # Compute c_k for each kernel
        c = zeros(q)
        for k in 1:q
            c[k] = sum((y .* α) .* (K_list[k] * (y .* α)))  # c_k = (y⊗α)^T K_k (y⊗α)
        end

        # Construct the Gurobi optimization problem
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)  # Suppress Gurobi output

        @variable(model, β[1:q] >= 0)  # Non-negativity constraint for β
        @objective(model, Min, 0.5 * λ * sum(β[i]^2 for i in 1:q) - 0.5 * dot(c, β))  # Objective function

        # Solve the problem
        optimize!(model)

        # Extract the optimized β values
        β_opt = value.(β)

        # Normalize β to avoid numerical instability (optional)
        if sum(β_opt) > 0
            β_opt /= sum(β_opt)
        else
            β_opt .= 1.0 / q  # Fallback to equal weights if all are zero
        end

        return β_opt
    end

    # Function to compute the bias term b
    function compute_bias(α, y, K, C; tolerance = 1e-6)
        n = length(α)
        b_values = Float64[]  # Explicitly define the type of b_values

        for i = 1:n
            if α[i] > tolerance && α[i] < C - tolerance  # Support vectors with 0 < α_i < C
                s = sum(α[j] * y[j] * K[j, i] for j = 1:n)  # Compute Σ α[j] * y[j] * K[j, i]
                push!(b_values, y[i] - s)  # Compute b and store it
            end
        end
        
        # Handle empty b_values
        if isempty(b_values)
            @infiltrate
            error("No valid support vectors found. Check the input parameters.")
        end

        # Take the average of all computed b values
        b = mean(b_values)
        return b
    end


    # Extended MKL-SVM training function
    function train_mkl(X, y, C, K_list, λ; max_iter=100, tolerance=1e-5)
        q = length(K_list)  # Number of kernels
        n = size(X, 1)  # Number of data points

        # Initialize β and α
        β = ones(q) / q  # Start with equal weights
        β_old = copy(β)  # Initialize β_old for convergence check
        α = zeros(n)     # Initialize α
        K_combined = compute_combined_kernel(K_list, β)
        list_alphas = []
        list_betas = []
        

        for iter in 1:max_iter
            println("Iteration $iter...")

            # Step 1: Optimize α given β
            α, K_combined = train_svm(X, y, C, "precomputed"; K_combined=K_combined)

            # Step 2: Optimize β given α
            β = optimize_beta(K_list, α, y, λ)

            # Update the combined kernel
            K_combined = compute_combined_kernel(K_list, β)

            # Convergence check (optional)
            if iter > 1 && norm(β - β_old) < tolerance
                println("Converged after $iter iterations.")
                break
            end
            β_old = copy(β)
            push!(list_alphas, copy(α))
            push!(list_betas, copy(β))

        end

        return α, β, K_combined, list_alphas, list_betas
    end


    function predict_mkl(α, y_train, X_train, X_new, β, b, K_list; kernel_type="precomputed", degree=3, gamma=1.0, tolerance=1e-5)
        # Identify support vectors
        support_indices = findall(α .> tolerance)
        α_support = α[support_indices]
        y_support = y_train[support_indices]

        if kernel_type == "precomputed"
            # For precomputed kernels, we assume K_list are the kernels from X_train to X_new.
            # Just extract the support vector rows and combine.
            K_list_support = [K[support_indices, :] for K in K_list]
            K_combined_new = compute_combined_kernel(K_list_support, β)
        else
            # For non-precomputed kernels, we need to compute them on the fly.
            K_list_dynamic = Vector{Matrix}(undef, length(K_list))
            for j in 1:length(K_list)
                K_new = compute_kernel(X_train[support_indices, :], X_new, kernel_type; degree=degree, gamma=gamma)
                K_list_dynamic[j] = K_new
            end
            K_combined_new = compute_combined_kernel(K_list_dynamic, β)
        end

        # println("K_combined_new dimensions: ", size(K_combined_new))

        # Use predict_svm to get final predictions
        return predict_svm(α_support, y_support, X_train[support_indices, :], X_new, b, kernel_type; K_combined=K_combined_new)
    end



end
