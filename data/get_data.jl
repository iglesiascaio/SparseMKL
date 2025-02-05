module GetData

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

    const DATA_DIR = "../data"  # Define local data storage directory

    function get_iris_data(; force_download::Bool = false)  # Renamed the parameter to avoid conflict
        file_path = joinpath(DATA_DIR, "iris.csv")

        if force_download || !isfile(file_path)
            println("Downloading Iris dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
            local_file = download(url)  # No conflict with the renamed parameter
            iris = CSV.read(local_file, DataFrame, header = false)

            # Rename columns
            rename!(iris, [:sepal_length, :sepal_width, :petal_length, :petal_width, :species])

            # Save dataset locally
            if !isdir(DATA_DIR)
                mkdir(DATA_DIR)
            end
            CSV.write(file_path, iris)
        else
            println("Loading Iris dataset from local storage...")
            iris = CSV.read(file_path, DataFrame)
        end

        # Convert to binary classification (Iris-setosa vs. not Iris-setosa)
        iris.binary_class = iris.species .== "Iris-setosa"  
        iris = filter(row -> row.binary_class == 1 || row.binary_class == 0, iris)

        # Encode labels as +1 and -1
        labels = [b ? 1 : -1 for b in iris.binary_class]

        # Select only the first two features (Sepal Length and Sepal Width)
        features = Matrix(iris[:, [:sepal_length, :sepal_width]])

        # Set random seed for reproducibility
        Random.seed!(123)

        # Shuffle indices and split the dataset
        n = size(features, 1)
        indices = shuffle(1:n)

        train_ratio = 0.8
        train_size = Int(floor(train_ratio * n))

        train_indices = indices[1:train_size]
        test_indices = indices[train_size+1:end]

        X_train = features[train_indices, :]
        y_train = labels[train_indices]
        X_test = features[test_indices, :]
        y_test = labels[test_indices]

        # Compute normalization parameters from training set only
        train_mean = mean(X_train, dims = 1)
        train_std = std(X_train, dims = 1)

        # Normalize training and testing sets using training statistics
        X_train = (X_train .- train_mean) ./ train_std
        X_test = (X_test .- train_mean) ./ train_std

        return X_train, y_train, X_test, y_test
    end

    function get_adult_data(; force_download::Bool = false, frac=0.33)  # Renamed parameter to avoid conflict
        file_path = joinpath(DATA_DIR, "adult.csv")

        if force_download || !isfile(file_path)
            println("Downloading Adult dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            local_file = download(url)  # No conflict with renamed parameter
            adult = CSV.read(local_file, DataFrame, header = false)

            rename!(
                adult,
                [
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
                    :income,
                ],
            )

            # Save dataset locally
            if !isdir(DATA_DIR)
                mkdir(DATA_DIR)
            end
            CSV.write(file_path, adult)
        else
            println("Loading Adult dataset from local storage...")
            adult = CSV.read(file_path, DataFrame)
        end

        # Filter a fraction of the dataset
        Random.seed!(456)
        n_rows = size(adult, 1)
        sample_indices = sample(1:n_rows, Int(floor(n_rows * frac)), replace = false)
        adult_filtered = adult[sample_indices, :]

        # Encode labels as +1 and -1
        adult_filtered.income_binary = adult_filtered.income .== " >50K"
        labels = [b ? 1 : -1 for b in adult_filtered.income_binary]

        # Select numeric features
        features = Matrix(
            adult_filtered[:, [:age, :education_num, :capital_gain, :capital_loss, :hours_per_week]],
        )

        # Handle missing values
        features = coalesce.(features, 0)

        # Set random seed for reproducibility
        Random.seed!(123)

        # Shuffle and split dataset
        n = size(features, 1)
        indices = shuffle(1:n)

        train_ratio = 0.8
        train_size = Int(floor(train_ratio * n))

        train_indices = indices[1:train_size]
        test_indices = indices[train_size+1:end]

        X_train = features[train_indices, :]
        y_train = labels[train_indices]
        X_test = features[test_indices, :]
        y_test = labels[test_indices]

        # Normalize training and testing sets
        train_mean = mean(X_train, dims = 1)
        train_std = std(X_train, dims = 1)

        X_train = (X_train .- train_mean) ./ train_std
        X_test = (X_test .- train_mean) ./ train_std

        return X_train, y_train, X_test, y_test
    end

end
