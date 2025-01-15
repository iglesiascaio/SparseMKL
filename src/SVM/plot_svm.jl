module PlotSVM

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

    include("./svm.jl")
    using .SVM: predict_svm

    # Function to create meshgrid
    function meshgrid(x, y)
        X = repeat(x', length(y), 1)
        Y = repeat(y, 1, length(x))
        return X, Y
    end

    # Function to plot the decision boundary
    function plot_decision_boundary(
        α,
        b,
        X_train,
        y_train,
        kernel_type;
        degree = 3,
        gamma = 1.0,
        resolution = 100,
        tolerance = 1e-5,
    )
        # Determine the range of the grid
        x_min = minimum(X_train[:, 1]) - 1
        x_max = maximum(X_train[:, 1]) + 1
        y_min = minimum(X_train[:, 2]) - 1
        y_max = maximum(X_train[:, 2]) + 1

        # Create a grid of points
        x_range = range(x_min, x_max, length = resolution)
        y_range = range(y_min, y_max, length = resolution)
        xx, yy = meshgrid(x_range, y_range)
        grid_points = hcat(vec(xx), vec(yy))

        # Predict over the grid
        Z = predict_svm(
            α,
            y_train,
            X_train,
            grid_points,
            b,
            kernel_type;
            degree = degree,
            gamma = gamma,
            tolerance = tolerance,
        )
        Z = reshape(Z, size(xx))

        # Define colors for each class
        unique_labels = sort(unique(y_train))  # Ensure consistent ordering
        label_colors = Dict(unique_labels[1] => :blue, unique_labels[2] => :green)

        # Initialize the plot
        plt = plot(
            title = "SVM Decision Boundary ($kernel_type kernel)",
            xlabel = "Feature 1",
            ylabel = "Feature 2",
            legend = :topright,
        )

        # Plot the data points for each class
        for lbl in unique_labels
            indices = findall(y_train .== lbl)
            scatter!(
                plt,
                X_train[indices, 1],
                X_train[indices, 2],
                color = label_colors[lbl],
                marker = :circle,
                label = "Class $lbl",
            )
        end

        # Plot the decision boundary
        contour!(
            plt,
            x_range,
            y_range,
            Z;
            levels = [0.0],
            linewidth = 2,
            color = :black,
            label = "Decision Boundary",
        )

        # Highlight support vectors
        support_indices = findall(α .> tolerance)
        if !isempty(support_indices)
            sv_x = X_train[support_indices, 1]
            sv_y = X_train[support_indices, 2]
            sv_labels = y_train[support_indices]
            sv_colors = [label_colors[lbl] for lbl in sv_labels]

            # Plot all support vectors together
            scatter!(
                plt,
                sv_x,
                sv_y,
                color = sv_colors,
                marker = :circle,
                markersize = 10,
                markerstrokecolor = :black,
                markerstrokewidth = 2,
                label = "Support Vectors",
            )
        end

        display(plt)
    end
end
