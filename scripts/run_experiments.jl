using Profile
using Printf
using Infiltrator
using DataFrames, CSV

# Toggle profiling on/off:
const ENABLE_PROFILING = false

macro maybe_profile(expr)
    if ENABLE_PROFILING
        return :( @profile $expr )
    else
        return expr
    end
end

include("../data/get_data.jl")
using .GetData: get_dataset

include("../src/MKL/multi_kernel.jl")
include("../src/MKL/plot_mkl.jl")
using .MKL: compute_kernels, train_mkl, compute_bias, predict_mkl, compute_combined_kernel

include("../src/Interpretable_MKL/interpretable_multi_kernel.jl")
using .InterpretableMKL: sparse_optimize_beta, train_interpretable_mkl

# Define datasets to process
DATASETS = [:iris, :adult, :wine, :breastcancer, :ionosphere, :spambase, :banknote, 
            :pimaindians, :heart, :german, :haberman, :mammographic, :parkinsons, :sonar]

# Kernel specifications
kernels = [
    Dict(:type => "linear", :params => Dict()),
    Dict(:type => "polynomial", :params => Dict(:degree => 3, :c => 1.0)),
    Dict(:type => "polynomial", :params => Dict(:degree => 2, :c => 1.0)),
    Dict(:type => "rbf", :params => Dict(:gamma => 0.5)),
    Dict(:type => "rbf", :params => Dict(:gamma => 0.3)),
    Dict(:type => "rbf", :params => Dict(:gamma => 0.1)),
    Dict(:type => "sigmoid", :params => Dict(:gamma => 0.5, :c0 => 1.0)),
    Dict(:type => "laplacian", :params => Dict(:gamma => 0.3)),
    # Dict(:type => "chi_squared", :params => Dict(:gamma => 0.2))
]

# Hyperparameters
C = 1.0
k0 = 3
λ_factor = 1e-1
max_iter = 50
sum_beta_val = 1.0
tolerance = 1e-2

# Store results
results = DataFrame(Dataset=String[], Accuracy=Float64[], Precision=Float64[], 
                    Recall=Float64[], F1_Score=Float64[], Baseline_Accuracy=Float64[], 
                    Betas=String[], Status=String[])

# Function to compute metrics
function compute_metrics(y_actual, y_pred)
    TP = sum((y_actual .== 1) .& (y_pred .== 1))
    TN = sum((y_actual .== -1) .& (y_pred .== -1))
    FP = sum((y_actual .== -1) .& (y_pred .== 1))
    FN = sum((y_actual .== 1) .& (y_pred .== -1))

    accuracy = (TP + TN) / length(y_actual)
    precision = TP / (TP + FP + 1e-9)  # Avoid division by zero
    recall = TP / (TP + FN + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    return accuracy, precision, recall, f1_score
end

# Loop over datasets
for dataset in DATASETS
    try
        println("\nProcessing dataset: $dataset")
        frac = dataset == :adult ? 0.33 : 1.00
        X_train, y_train, X_test, y_test = get_dataset(dataset; force_download=false, frac=frac, train_ratio=0.8)

        λ = length(X_train) * λ_factor  # Scale λ dynamically
        y_train = Float64.(y_train)

        println("Computing kernels...")
        K_list_train = compute_kernels(X_train, X_train, kernels)
        K_list_test = compute_kernels(X_train, X_test, kernels)

        println("Training MKL model...")
        α, β, K_combined, _, _ = train_interpretable_mkl(X_train, y_train, C, K_list_train, λ;
            max_iter=max_iter, tolerance=tolerance, k0=k0, sum_beta_val=sum_beta_val,
            solver_type=:LIBSVM, beta_method=:gssp)

        b = compute_bias(α, y_train, K_combined, C)

        # Predictions
        y_pred_test = predict_mkl(α, y_train, X_train, X_test, β, b, K_list_test, kernel_type="precomputed"; tolerance=tolerance)

        # Compute MKL metrics
        accuracy, precision, recall, f1_score = compute_metrics(y_test, y_pred_test)

        # Baseline (Majority vote)
        majority_label = (sum(y_train .== 1) >= sum(y_train .== -1)) ? 1.0 : -1.0
        y_pred_baseline = fill(majority_label, length(y_test))
        baseline_accuracy, _, _, _ = compute_metrics(y_test, y_pred_baseline)

        # Convert best betas to a readable string format
        betas_str = join(round.(β, digits=4), ", ")

        # Store results
        push!(results, (string(dataset), accuracy, precision, recall, f1_score, baseline_accuracy, betas_str, "Success"))

    catch e
        println("Error processing dataset $dataset: $e")
        push!(results, (string(dataset), NaN, NaN, NaN, NaN, NaN, "N/A", "Error"))
    end
end

# Save and display results
CSV.write("results.csv", results)  # Save results to a CSV file

# Filter successful datasets
successful_results = filter(row -> row.Status == "Success", results)
println("\n=== Successful Datasets ===")
println(successful_results)

# Print datasets that encountered errors
failed_datasets = filter(row -> row.Status == "Error", results)
println("\n=== Datasets with Errors ===")
println(failed_datasets.Dataset)  # Print only the dataset names
