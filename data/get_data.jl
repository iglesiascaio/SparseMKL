module GetData

using CSV
using DataFrames
using Random
using Statistics
using StatsBase    
using Downloads
using Printf

# Define the directory for local data storage.
const DATA_DIR = "../data"

# ----------------------------------------------------------------------------
# 1) Dataset Configuration Dictionary
# ----------------------------------------------------------------------------
# We'll keep the same dictionary structure. 
# The only changes are that we no longer rely on StatsBase.dummyvar.
# Instead, we define a custom function below to handle one-hot encoding.

const DATASET_CONFIG = Dict{Symbol, NamedTuple}(
    # IRIS (Originally 3 classes => artificially turned into binary)
    :iris => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        local_filename = "iris.csv",
        has_header     = false,
        rename_map     = [:sepal_length, :sepal_width, :petal_length, :petal_width, :species],
        label_fn       = row -> (row[:species] == "Iris-setosa" ? 1 : -1),
        numeric_cols   = [:sepal_length, :sepal_width],
        categorical_cols = []
    ),


    # WINE (Originally 3 classes => forcibly turned into binary)
    :wine => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        local_filename = "wine.csv",
        has_header     = false,
        rename_map     = [
            :class, :alcohol, :malic_acid, :ash, :ash_alcalinity, :magnesium,
            :total_phenols, :flavanoids, :nonflav_phenols, :proanthocyanins,
            :color_intensity, :hue, :od280_od315, :proline
        ],
        label_fn       = row -> (row[:class] == 1 ? 1 : -1),
        numeric_cols   = [
            :alcohol, :malic_acid, :ash, :ash_alcalinity, :magnesium,
            :total_phenols, :flavanoids, :nonflav_phenols, :proanthocyanins,
            :color_intensity, :hue, :od280_od315, :proline
        ],
        categorical_cols = []
    ),

    # BREASTCANCER
    :breastcancer => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        local_filename = "breastcancer.csv",
        has_header     = false,
        rename_map     = [
            :id, :diagnosis,
            :feat1, :feat2, :feat3, :feat4, :feat5, :feat6, :feat7, :feat8, :feat9,
            :feat10, :feat11, :feat12, :feat13, :feat14, :feat15, :feat16, :feat17,
            :feat18, :feat19, :feat20, :feat21, :feat22, :feat23, :feat24, :feat25,
            :feat26, :feat27, :feat28, :feat29, :feat30
        ],
        label_fn       = row -> (row[:diagnosis] == "M" ? 1 : -1),
        numeric_cols   = [
            :feat1, :feat2, :feat3, :feat4, :feat5, :feat6, :feat7, :feat8, :feat9,
            :feat10, :feat11, :feat12, :feat13, :feat14, :feat15, :feat16, :feat17,
            :feat18, :feat19, :feat20, :feat21, :feat22, :feat23, :feat24, :feat25,
            :feat26, :feat27, :feat28, :feat29, :feat30
        ],
        categorical_cols = []
    ),

    # IONOSPHERE
    :ionosphere => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        local_filename = "ionosphere.csv",
        has_header     = false,
        rename_map     = vcat([Symbol("feat$i") for i in 1:34], [:label]),
        label_fn       = row -> (row[:label] == "g" ? 1 : -1),
        numeric_cols   = [Symbol("feat$i") for i in 1:34],
        categorical_cols = []
    ),

    # SPAMBASE
    :spambase => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        local_filename = "spambase.csv",
        has_header     = false,
        rename_map     = vcat([Symbol("feat$i") for i in 1:57], [:is_spam]),
        label_fn       = row -> (row[:is_spam] == 1 ? 1 : -1),
        numeric_cols   = [Symbol("feat$i") for i in 1:57],
        categorical_cols = []
    ),

    # BANKNOTE AUTHENTICATION
    :banknote => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        local_filename = "banknote.csv",
        has_header     = false,
        rename_map     = [:variance, :skewness, :curtosis, :entropy, :class],
        label_fn       = row -> (row[:class] == 1 ? 1 : -1),
        numeric_cols   = [:variance, :skewness, :curtosis, :entropy],
        categorical_cols = []
    ),

    # HEART DISEASE (Cleveland) - multi-class turned to binary
    :heart => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        local_filename = "heart.csv",
        has_header     = false,
        rename_map     = [
            :age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach,
            :exang, :oldpeak, :slope, :ca, :thal, :num
        ],
        label_fn       = row -> (row[:num] == 0 ? -1 : 1),
        numeric_cols   = [
            :age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach,
            :exang, :oldpeak, :slope, :ca, :thal
        ],
        categorical_cols = []
    ),

    # HABERMAN
    :haberman => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data",
        local_filename = "haberman.csv",
        has_header     = false,
        rename_map     = [:age, :operation_year, :positive_axillary_nodes, :survival_status],
        label_fn       = row -> (row[:survival_status] == 1 ? 1 : -1),
        numeric_cols   = [:age, :operation_year, :positive_axillary_nodes],
        categorical_cols = []
    ),

    # MAMMOGRAPHIC MASSES
    :mammographic => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data",
        local_filename = "mammographic.csv",
        has_header     = false,
        rename_map     = [:BI_RADS, :age, :shape, :margin, :density, :severity],
        label_fn       = row -> (row[:severity] == 1 ? 1 : -1),
        numeric_cols   = [:BI_RADS, :age, :shape, :margin, :density],
        categorical_cols = []
    ),

    # PARKINSON'S
    :parkinsons => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
        local_filename = "parkinsons.csv",
        has_header     = true,
        rename_map     = [
            :name, :mdvp_fo, :mdvp_fhi, :mdvp_flo, :mdvp_jitter_percent,
            :mdvp_jitter_abs, :mdvp_rap, :mdvp_ppq, :jitter_ddp, :mdvp_shimmer,
            :mdvp_shimmer_dB, :shimmer_apq3, :shimmer_apq5, :mdvp_apq, :shimmer_dda,
            :nhr, :hnr, :status, :rpde, :dfa, :spread1, :spread2, :d2, :ppe
        ],
        label_fn       = row -> (row[:status] == 1 ? 1 : -1),
        numeric_cols   = [
            :mdvp_fo, :mdvp_fhi, :mdvp_flo, :mdvp_jitter_percent,
            :mdvp_jitter_abs, :mdvp_rap, :mdvp_ppq, :jitter_ddp, :mdvp_shimmer,
            :mdvp_shimmer_dB, :shimmer_apq3, :shimmer_apq5, :mdvp_apq,
            :shimmer_dda, :nhr, :hnr, :rpde, :dfa, :spread1, :spread2,
            :d2, :ppe
        ],
        categorical_cols = []
    ),
)

# ----------------------------------------------------------------------------
# 2) Utility Function: Download and Read CSV
# ----------------------------------------------------------------------------
function load_dataset_csv(ds_name::Symbol; force_download::Bool=false)::DataFrame
    ds_info = DATASET_CONFIG[ds_name]
    local_path = joinpath(DATA_DIR, ds_info.local_filename)

    if force_download || !isfile(local_path)
        println("Downloading $(ds_name) dataset...")
        url = ds_info.url
        temp_file = Downloads.download(url)
        if !isdir(DATA_DIR)
            mkpath(DATA_DIR)
        end
        cp(temp_file, local_path; force=true)
    else
        println("Loading $(ds_name) dataset from local storage...")
    end

    df = CSV.read(local_path, DataFrame; header = ds_info.has_header ? 1 : 0)

    rename_map = ds_info.rename_map
    if length(rename_map) == ncol(df)
        rename!(df, rename_map)
    else
        @warn "For dataset $(ds_name), rename_map size ($(length(rename_map))) does not match number of columns ($(ncol(df)))."
    end

    # Convert numeric columns to Float64
    for col in ds_info.numeric_cols
        if !(eltype(df[!, col]) <: Number)
            parsed = tryparse.(Float64, strip.(string.(df[!, col])))
            df[!, col] = map(x -> (x === nothing || x === missing) ? 0.0 : x, parsed)
        end
    end

    return df
end

# ----------------------------------------------------------------------------
# 3) Helper Function for One-Hot Encoding
# ----------------------------------------------------------------------------
"""
build_one_hot_matrix(df, cat_cols) -> Matrix{Float64}

Creates a one-hot (dummy) matrix for the given categorical columns in `df`.
Returns a matrix of size (n_rows) x (sum_of_unique_values_in_cat_cols).
"""
function build_one_hot_matrix(df::DataFrame, cat_cols::Vector{Symbol})
    if isempty(cat_cols)
        return zeros(Float64, nrow(df), 0)  # No categorical columns => return empty matrix
    end

    out_mats = Matrix{Float64}[]
    for col in cat_cols
        # Gather unique values (sorted)
        vals = sort(unique(df[!, col]))
        n = nrow(df)
        k = length(vals) - 1  # Use (n-1) columns to avoid multicollinearity

        # One-hot encode without the last category (acts as a reference)
        M = zeros(Float64, n, k)
        for (j, v) in enumerate(vals[1:end-1])  # Exclude last category
            for i in 1:n
                if df[i, col] == v
                    M[i, j] = 1.0
                end
            end
        end
        push!(out_mats, M)
    end

    # Concatenate all one-hot encoded columns
    return hcat(out_mats...)
end

# ----------------------------------------------------------------------------
# 4) Main Function: get_dataset
# ----------------------------------------------------------------------------
"""
get_dataset(ds_name; force_download=false, frac=1.0, train_ratio=0.8)

Loads a dataset, optionally downloads it, samples rows, creates +1/-1 labels,
converts numeric columns to a matrix, one-hot encodes categorical columns,
splits into train/test, and standardizes numeric columns.

Returns (X_train, y_train, X_test, y_test).
"""
function get_dataset(ds_name::Symbol; force_download::Bool=false, frac::Real=1.0, train_ratio::Real=0.8)
    # 1) Load CSV
    df = load_dataset_csv(ds_name; force_download=force_download)

    # 2) Sample fraction of rows if needed
    Random.seed!(123)
    n_rows = nrow(df)
    keep_size = Int(floor(n_rows * frac))
    if keep_size < n_rows
        chosen = sample(1:n_rows, keep_size; replace=false)
        df = df[chosen, :]
    end

    # 3) Create binary labels
    ds_info = DATASET_CONFIG[ds_name]
    label_fn = ds_info.label_fn
    labels = [label_fn(row) for row in eachrow(df)]

    # 4) Numeric matrix
    numeric_cols = ds_info.numeric_cols
    X_num = Matrix(df[:, numeric_cols])  # might be empty if none numeric
    X_num = coalesce.(X_num, 0.0)

    # 5) One-hot encode categorical columns
    if length(ds_info.categorical_cols) == 0
        X_cats = zeros(Float64, nrow(df), 0)
    else
        X_cats = build_one_hot_matrix(df, ds_info.categorical_cols)
    end

    # 6) Combine numeric + categorical into single feature matrix
    if size(X_num, 2) == 0
        X = X_cats
    else
        X = hcat(X_num, X_cats)
    end

    # 7) Shuffle & split
    N = size(X, 1)
    indices = shuffle(1:N)
    train_size = Int(floor(N * train_ratio))
    train_idx = indices[1:train_size]
    test_idx  = indices[train_size+1:end]

    X_train = X[train_idx, :]
    y_train = labels[train_idx]
    X_test  = X[test_idx, :]
    y_test  = labels[test_idx]

    # 8) Standardize numeric columns only, using training set stats
    #    numeric columns are the first size(X_num, 2) in X
    num_dim = size(X_num, 2)
    if num_dim > 0
        train_mean = mean(X_train[:, 1:num_dim], dims=1)
        train_std  = std(X_train[:, 1:num_dim], dims=1)
        train_std  = replace(train_std, 0 => 1e-8)

        X_train[:, 1:num_dim] = (X_train[:, 1:num_dim] .- train_mean) ./ train_std
        X_test[:, 1:num_dim]  = (X_test[:, 1:num_dim]  .- train_mean) ./ train_std
    end

    # Ensure y in {-1, +1}, stored as Float64
    y_train = Float64.(y_train)
    y_test  = Float64.(y_test)

    return X_train, y_train, X_test, y_test
end

function get_dataset(ds_name::String; force_download::Bool=false, frac::Real=1.0, train_ratio::Real=0.8)
    # convert ds_name from String to Symbol
    # useful when usign the python call 
    return get_dataset(Symbol(ds_name); force_download=force_download, frac=frac, train_ratio=train_ratio)
end

end # module GetData