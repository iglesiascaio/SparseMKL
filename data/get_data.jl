module GetData

using CSV
using DataFrames
using Random
using Statistics
using StatsBase    # for sample()
using Downloads    # for download()
using Printf       # for formatted messages if needed

# Define the directory for local data storage.
const DATA_DIR = "../data"

# ----------------------------------------------------------------------------
# 1) Dataset Configuration Dictionary (15 datasets)
# ----------------------------------------------------------------------------
# Each dataset is defined by a NamedTuple containing:
#   - url: URL string from UCI.
#   - local_filename: File name to store locally.
#   - has_header: Whether the CSV file has a header row.
#   - rename_map: A Vector{Symbol} specifying new column names.
#   - label_fn: A function that takes a DataFrame row and returns 1 or -1.
#   - numeric_cols: A Vector{Symbol} with the names of columns to be used as features.
# ----------------------------------------------------------------------------

const DATASET_CONFIG = Dict{Symbol, NamedTuple}(
    # 1) IRIS
    :iris => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        local_filename = "iris.csv",
        has_header     = false,
        rename_map     = [:sepal_length, :sepal_width, :petal_length, :petal_width, :species],
        label_fn       = row -> (row[:species] == "Iris-setosa" ? 1 : -1),
        numeric_cols   = [:sepal_length, :sepal_width]
    ),

    # 2) ADULT (Note: using has_header=true to skip the header row already present locally)
    :adult => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        local_filename = "adult.csv",
        has_header     = true,   # Updated to true to avoid parsing header strings as data
        rename_map     = [
            :age,
            :workclass,
            :fnlwgt,
            :education,
            :education_num,
            :marital_status,
            :occupation,
            :relationship,
            :race,
            :sex,
            :capital_gain,
            :capital_loss,
            :hours_per_week,
            :native_country,
            :income
        ],
        label_fn       = row -> (row[:income] == " >50K" ? 1 : -1),
        numeric_cols   = [:age, :education_num, :capital_gain, :capital_loss, :hours_per_week]
    ),

    # 3) WINE (binarized: class 1 → +1; others → -1)
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
        ]
    ),

    # 4) BREASTCANCER (Wisconsin Diagnostic, wdbc.data; M = +1, B = -1)
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
        ]
    ),

    # 5) IONOSPHERE
    :ionosphere => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        local_filename = "ionosphere.csv",
        has_header     = false,
        rename_map     = [Symbol("feat$i") for i in 1:34] ∪ [:label],
        label_fn       = row -> (row[:label] == "g" ? 1 : -1),
        numeric_cols   = [Symbol("feat$i") for i in 1:34]
    ),

    # 6) SPAMBASE
    :spambase => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        local_filename = "spambase.csv",
        has_header     = false,
        rename_map     = [Symbol("feat$i") for i in 1:57] ∪ [:is_spam],
        label_fn       = row -> (row[:is_spam] == 1 ? 1 : -1),
        numeric_cols   = [Symbol("feat$i") for i in 1:57]
    ),

    # 7) BANKNOTE AUTHENTICATION
    :banknote => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        local_filename = "banknote.csv",
        has_header     = false,
        rename_map     = [:variance, :skewness, :curtosis, :entropy, :class],
        label_fn       = row -> (row[:class] == 1 ? 1 : -1),
        numeric_cols   = [:variance, :skewness, :curtosis, :entropy]
    ),

    # 8) PIMA INDIANS DIABETES
    :pimaindians => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
        local_filename = "pimaindians.csv",
        has_header     = false,
        rename_map     = [
            :n_pregnant, :plasma_glucose, :diastolic_bp, :triceps_thickness,
            :serum_insulin, :bmi, :pedigree, :age, :class
        ],
        label_fn       = row -> (row[:class] == 1 ? 1 : -1),
        numeric_cols   = [:n_pregnant, :plasma_glucose, :diastolic_bp, :triceps_thickness,
                          :serum_insulin, :bmi, :pedigree, :age]
    ),

    # 9) HEART DISEASE (Cleveland)
    :heart => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        local_filename = "heart.csv",
        has_header     = false,
        rename_map     = [
            :age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach,
            :exang, :oldpeak, :slope, :ca, :thal, :num
        ],
        label_fn       = row -> (row[:num] == 0 ? -1 : 1),
        numeric_cols   = [:age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach,
                          :exang, :oldpeak, :slope, :ca, :thal]
    ),

    # 10) GERMAN CREDIT (numeric version)
    :german => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric",
        local_filename = "german_numeric.csv",
        has_header     = false,
        rename_map     = [Symbol("feat$i") for i in 1:24] ∪ [:class],
        label_fn       = row -> (row[:class] == 1 ? 1 : -1),
        numeric_cols   = [Symbol("feat$i") for i in 1:24]
    ),

    # 11) HABERMAN
    :haberman => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data",
        local_filename = "haberman.csv",
        has_header     = false,
        rename_map     = [:age, :operation_year, :positive_axillary_nodes, :survival_status],
        label_fn       = row -> (row[:survival_status] == 1 ? 1 : -1),
        numeric_cols   = [:age, :operation_year, :positive_axillary_nodes]
    ),

    # 12) MAMMOGRAPHIC MASSES
    :mammographic => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data",
        local_filename = "mammographic.csv",
        has_header     = false,
        rename_map     = [:BI_RADS, :age, :shape, :margin, :density, :severity],
        label_fn       = row -> (row[:severity] == 1 ? 1 : -1),
        numeric_cols   = [:BI_RADS, :age, :shape, :margin, :density]
    ),

    # 13) BLOOD TRANSFUSION
    :blood => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data",
        local_filename = "blood.csv",
        has_header     = true,  # This dataset has a header row.
        rename_map     = [:recency, :frequency, :monetary, :time, :donated_march],
        label_fn       = row -> (row[:donated_march] == 1 ? 1 : -1),
        numeric_cols   = [:recency, :frequency, :monetary, :time]
    ),

    # 14) PARKINSON'S
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
        ]
    ),

    # 15) SONAR
    :sonar => (
        url            = "https://archive.ics.uci.edu/ml/machine-learning-databases/sonar/sonar.all-data",
        local_filename = "sonar.csv",
        has_header     = false,
        rename_map     = [Symbol("feat$i") for i in 1:60] ∪ [:label],
        label_fn       = row -> (row[:label] == "R" ? -1 : 1),
        numeric_cols   = [Symbol("feat$i") for i in 1:60]
    )
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

    # Read CSV; if has_header is true, use header row.
    df = CSV.read(local_path, DataFrame; header = ds_info.has_header ? 1 : 0)

    # Rename columns if rename_map length matches number of columns.
    rename_map = ds_info.rename_map
    if length(rename_map) == ncol(df)
        rename!(df, rename_map)
    else
        @warn "For dataset $(ds_name), rename_map size ($(length(rename_map))) does not match number of columns ($(ncol(df)))."
    end

    # Convert each numeric column from string to Float64 if necessary.
    for col in ds_info.numeric_cols
        if !(eltype(df[!, col]) <: Number)
            # Try to parse each value.
            parsed = tryparse.(Float64, strip.(string.(df[!, col])))
            # Replace any value that is either `nothing` or `missing` with 0.0.
            df[!, col] = map(x -> (x === nothing || x === missing) ? 0.0 : x, parsed)
        end
    end

    return df
end

# ----------------------------------------------------------------------------
# 3) Main Function: get_dataset
# ----------------------------------------------------------------------------
"""
    get_dataset(ds_name::Symbol; force_download=false, frac=1.0, train_ratio=0.8)

Downloads (if necessary) and loads the dataset specified by `ds_name`.
- Reads the CSV and renames columns.
- Optionally samples a fraction `frac` of rows.
- Applies the label function to generate binary labels (+1 or -1).
- Extracts numeric features (coalescing missing values to 0).
- Shuffles and splits into training and testing sets.
- Standardizes features using training set statistics.

Returns `(X_train, y_train, X_test, y_test)`.
"""
function get_dataset(ds_name::Symbol; force_download::Bool=false, frac::Real=1.0, train_ratio::Real=0.8)
    # 1) Load CSV
    df = load_dataset_csv(ds_name; force_download=force_download)

    # 2) Sample fraction of rows if needed.
    Random.seed!(123)
    n_rows = nrow(df)
    keep_size = Int(floor(n_rows * frac))
    if keep_size < n_rows
        chosen = sample(1:n_rows, keep_size; replace=false)
        df = df[chosen, :]
    end

    # 3) Create binary labels using the label function.
    ds_info = DATASET_CONFIG[ds_name]
    label_fn = ds_info.label_fn
    labels = [label_fn(row) for row in eachrow(df)]

    # 4) Extract numeric features as a matrix.
    numeric_cols = ds_info.numeric_cols
    X = Matrix(df[:, numeric_cols])
    X = coalesce.(X, 0)

    # 5) Shuffle and split into training and testing sets.
    N = size(X, 1)
    indices = shuffle(1:N)
    train_size = Int(floor(N * train_ratio))
    train_idx = indices[1:train_size]
    test_idx  = indices[train_size+1:end]

    X_train = X[train_idx, :]
    y_train = labels[train_idx]
    X_test  = X[test_idx, :]
    y_test  = labels[test_idx]

    # 6) Standardize features based on training data.
    train_mean = mean(X_train, dims=1)
    train_std  = std(X_train, dims=1)
    train_std  = replace(train_std, 0=>1e-8)

    X_train = (X_train .- train_mean) ./ train_std
    X_test  = (X_test  .- train_mean) ./ train_std

    return X_train, y_train, X_test, y_test
end

# ----------------------------------------------------------------------------
# 4) Convenience Wrappers for IRIS and ADULT (Matching old usage)
# ----------------------------------------------------------------------------
function get_iris_data(; force_download=false)
    return get_dataset(:iris; force_download=force_download, frac=1.0, train_ratio=0.8)
end

function get_adult_data(; force_download=false, frac=1.0)
    return get_dataset(:adult; force_download=force_download, frac=frac, train_ratio=0.8)
end

end  # module GetData
