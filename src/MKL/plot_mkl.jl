module PlotMKL
    using Plots
    using Distances

    include("./multi_kernel.jl")
    using .MKL: compute_kernels, predict_mkl

    # Function to create meshgrid
    function meshgrid(x, y)
        X = repeat(x', length(y), 1)
        Y = repeat(y, 1, length(x))
        return X, Y
    end

    function plot_mkl_decision_boundary(α, y_train, X_train, β, b, C, kernels; 
                                        resolution=300, tolerance=1e-5)
        """
        Plot the MKL-SVM decision boundary.

        Parameters:
            α: Coefficients for support vectors.
            y_train: Training labels.
            X_train: Training features.
            β: Kernel weights.
            b: Bias term.
            C: Regularization parameter.
            kernels: Vector of Dicts specifying kernel types and hyperparameters.
            resolution: Number of points along each axis for the grid.
            tolerance: Threshold for determining support vectors.
        """
        # Identify support vectors where 0 < α_i < C
        support_indices = findall((α .> tolerance) .& (α .< (C - tolerance)))
        
        # If no such "strict" support vectors, fallback to any α > tolerance
        if isempty(support_indices)
            support_indices = findall(α .> tolerance)
        end
        
        if isempty(support_indices)
            throw(ArgumentError("No support vectors found. Please check the tolerance and C parameters."))
        end

        # Determine the range for the grid
        x_min = minimum(X_train[:, 1]) - 1
        x_max = maximum(X_train[:, 1]) + 1
        y_min = minimum(X_train[:, 2]) - 1
        y_max = maximum(X_train[:, 2]) + 1

        # Create a grid of points
        x_range = range(x_min, stop=x_max, length=resolution)
        y_range = range(y_min, stop=y_max, length=resolution)
        xx, yy = meshgrid(x_range, y_range)
        grid_points = hcat(vec(xx), vec(yy))  # (resolution^2) x 2 matrix

        # Compute kernels for the grid points with the training data
        K_list_grid = compute_kernels(X_train, grid_points, kernels)

        # Use predict_mkl to get predictions on the grid
        y_pred_grid = predict_mkl(α, y_train, X_train, grid_points, β, b, K_list_grid; 
                                kernel_type="precomputed", tolerance=tolerance)

        Z = reshape(y_pred_grid, size(xx))

        # Initialize the plot
        plt = plot(title="MKL-SVM Decision Boundary",
                xlabel="Feature 1", ylabel="Feature 2",
                xlims=(x_min, x_max), ylims=(y_min, y_max),
                aspect_ratio=:equal, legend=false)

        # Add a colored background based on predictions
        contourf!(plt, x_range, y_range, Z; color=:RdBu, alpha=0.6)

        # Plot the decision boundary (where prediction changes sign)
        contour!(plt, x_range, y_range, Z; levels=[0], linewidth=2, color=:black)

        # Plot the data points
        unique_labels = sort(unique(y_train))
        label_colors = Dict(unique_labels[1] => :blue, unique_labels[2] => :green)

        for lbl in unique_labels
            indices = findall(y_train .== lbl)
            scatter!(plt, X_train[indices, 1], X_train[indices, 2],
                    color=label_colors[lbl], marker=:circle)
        end

        # Highlight support vectors
        if !isempty(support_indices)
            sv_x = X_train[support_indices, 1]
            sv_y = X_train[support_indices, 2]
            sv_labels = y_train[support_indices]
            sv_colors = [label_colors[lbl] for lbl in sv_labels]

            scatter!(plt, sv_x, sv_y,
                    color=sv_colors,
                    marker=:circle, markersize=10,
                    markerstrokecolor=:black, markerstrokewidth=2)
        end

        display(plt)
    end

end