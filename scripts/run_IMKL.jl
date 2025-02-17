using Profile
using Printf  # Because your code already uses @sprintf
using Infiltrator

# Toggle profiling on/off:
const ENABLE_PROFILING = false  # Set to false to deactivate profiling

macro maybe_profile(expr)
    if ENABLE_PROFILING
        return :( @profile $expr )
    else
        return expr
    end
end

############################
# 1) RUN YOUR EXISTING CODE
############################

include("../data/get_data.jl")
using .GetData: get_adult_data, get_dataset

include("../src/MKL/multi_kernel.jl")
include("../src/MKL/plot_mkl.jl")
using .MKL: compute_kernels, train_mkl, compute_bias, predict_mkl, compute_combined_kernel

include("../src/Interpretable_MKL/interpretable_multi_kernel.jl")
using .InterpretableMKL: sparse_optimize_beta, train_interpretable_mkl

println("Loading data...")
# X_train, y_train, X_test, y_test = get_dataset(:adult; force_download=false, frac=0.33, train_ratio=0.8)
X_train, y_train, X_test, y_test = get_dataset(:banknote; force_download=false, frac=1.00, train_ratio=0.8)
# X_train, y_train, X_test, y_test = get_dataset(:wine; force_download=false, frac=1.00, train_ratio=0.8)


@infiltrate
# #print head of data
# println("X_train: ", X_train[1:5, 1:5])
# println("y_train: ", y_train[1:5])
# println("X_test: ", X_test[1:5, 1:5])



# Define kernel specifications
kernels = [
    Dict(:type => "linear", :params => Dict()),
    Dict(:type => "polynomial", :params => Dict(:degree => 3, :c => 1.0)),
    Dict(:type => "polynomial", :params => Dict(:degree => 2, :c => 1.0)),
    Dict(:type => "rbf",        :params => Dict(:gamma => 0.5)),
    Dict(:type => "rbf",        :params => Dict(:gamma => 0.3)),
    Dict(:type => "rbf",        :params => Dict(:gamma => 0.1)),
    Dict(:type => "sigmoid",    :params => Dict(:gamma => 0.5, :c0 => 1.0)),
    Dict(:type => "laplacian",  :params => Dict(:gamma => 0.3)),
    # Dict(:type => "chi_squared",:params => Dict(:gamma => 0.2))
]



# Hyperparameters
C = 1
k0 = 3
# λ = 200.0
λ = length(X_train) * 1e-1
max_iter = 50
sum_beta_val = 1.0
tolerance = 1e-2

println("Computing kernels...")
@time begin
    global K_list_train = compute_kernels(X_train, X_train, kernels)
    global K_list_test  = compute_kernels(X_train, X_test,  kernels)
end

y_train = Float64.(y_train)
C = float(C)

############################
# 2) PROFILE THE TRAINING
############################

Profile.clear()
@maybe_profile begin
    α, β, K_combined, list_alphas, list_betas = train_interpretable_mkl(
        X_train, y_train, C, K_list_train, λ;
        max_iter=max_iter, tolerance=tolerance, k0=k0, sum_beta_val=sum_beta_val,
        solver_type=:LIBSVM,  # or :SMO, :GUROBI
        beta_method=:gssp      # or :hard, :proximal
    )

    b = compute_bias(α, y_train, K_combined, C)

    y_pred_train = predict_mkl(
        α, y_train, X_train, X_train, β, b, K_list_train,
        kernel_type="precomputed"; tolerance=tolerance
    )
    y_pred_test = predict_mkl(
        α, y_train, X_train, X_test, β, b, K_list_test,
        kernel_type="precomputed"; tolerance=tolerance
    )

    accuracy_train = sum(y_train .== y_pred_train) / length(y_train)
    accuracy_test  = sum(y_test  .== y_pred_test)  / length(y_test)

    println("Training Accuracy: $(round(accuracy_train*100, digits=2))%")
    println("Test Accuracy: $(round(accuracy_test*100, digits=2))%")

    # Confusion metrics
    function print_confusion_metrics(y_actual, y_pred, set_name="Data Set")
        println("---------------- $set_name ----------------")
        TP = 0; TN = 0; FP = 0; FN = 0

        for i in 1:length(y_actual)
            actual = y_actual[i]
            predicted = y_pred[i]
            if      actual == 1  && predicted == 1   TP += 1
            elseif  actual == -1 && predicted == -1  TN += 1
            elseif  actual == -1 && predicted == 1   FP += 1
            elseif  actual == 1  && predicted == -1  FN += 1
            end
        end

        println("Confusion Matrix ($set_name):")
        println("            Predicted")
        println("            -1     +1")
        println(@sprintf("Actual -1   %-6d %-6d", TN, FP))
        println(@sprintf("       +1   %-6d %-6d", FN, TP))

        precision = (TP + FP == 0) ? 0 : TP / (TP + FP)
        recall    = (TP + FN == 0) ? 0 : TP / (TP + FN)
        f1_score  = (precision + recall == 0) ? 0 : 2*(precision*recall)/(precision+recall)

        println("Precision: $precision")
        println("Recall: $recall")
        println("F1-score: $f1_score\n")
    end

    print_confusion_metrics(y_train, y_pred_train, "Train Set")
    print_confusion_metrics(y_test,  y_pred_test,  "Test Set")
end

############################
# 3) CAPTURE THE PROFILE PRINT
############################

if ENABLE_PROFILING
    # Capture profile print output only if profiling was enabled.
    buf = IOBuffer()
    Profile.print(buf, format=:flat, maxdepth=50)  # or higher depth if needed
    seek(buf, 0)
    full_profile_str = read(buf, String)

    # Filter lines containing your code references
    lines = split(full_profile_str, '\n')
    my_code_lines = filter(line ->
        occursin("Interpretable_MKL", line) || occursin("multi_kernel", line),
        lines
    )

    ############################
    # 4) WRITE FILTERED RESULTS
    ############################

    open("my_filtered_profile.txt", "w") do io
        println(io, "=== Filtered lines referencing your code ===\n")
        for line in my_code_lines
            println(io, line)
        end
        println(io, "\n(Showing only lines containing 'Interpretable_MKL' or 'multi_kernel'.)")
    end

    println("\nFiltered profile saved to 'my_filtered_profile.txt'.")
else
    println("Profiling is disabled. Skipping profile capture and printing.")
end

println("Done!")
