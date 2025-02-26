module SVM
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
            error("No valid support vectors found. Check the input parameters.")
        end

        # Take the average of all computed b values
        b = mean(b_values)
        return b
    end

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
    )
        # Identify support vectors
        support_indices = findall(α .> tolerance)
        α_support = α[support_indices]
        y_support = y_train[support_indices]
        X_support = X_train[support_indices, :]

        n_new = size(X_new, 1)
        predictions = zeros(n_new)

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

        # Make predictions
        for i = 1:n_new
            s = sum(α_support .* y_support .* K_new[:, i])
            predictions[i] = s + b
        end

        # Return the sign of the predictions (+1 or -1)
        return sign.(predictions)
    end

    # Define the kernel functions
    function kernel(X, kernel_type; degree = 3, gamma = 1.0)
        n = size(X, 1)
        K = zeros(n, n)

        if kernel_type == "linear"
            K = X * X'  # Linear kernel
        elseif kernel_type == "polynomial"
            K = (X * X' .+ 1) .^ degree  # Polynomial kernel
        elseif kernel_type == "rbf"
            for i = 1:n
                for j = 1:n
                    K[i, j] = exp(-gamma * norm(X[i, :] - X[j, :])^2)  # RBF kernel
                end
            end
        else
            throw(ArgumentError("Unsupported kernel type: $kernel_type"))
        end

        return K
    end

    # Train SVM model with kernel options
    function train_svm(X, y, C, kernel_type; degree = 3, gamma = 1.0)
        n = size(X, 1)
        K = kernel(X, kernel_type; degree = degree, gamma = gamma)  # Generate kernel matrix

        # Start the JuMP model
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)  # Suppress Gurobi output

        # Define the dual variables α
        @variable(model, 0 <= α[i = 1:n] <= C)

        # Define the objective function
        @objective(
            model,
            Max,
            sum(α) - 0.5 * sum((y[i] * y[j] * α[i] * α[j] * K[i, j]) for i = 1:n, j = 1:n)
        )

        # Add the constraint: ∑ α_i * y_i = 0
        @constraint(model, sum(α[i] * y[i] for i = 1:n) == 0)

        # Solve the model
        optimize!(model)

        # Retrieve the α values
        α_values = value.(α)

        return α_values, K  # Return both α and K
    end


end
