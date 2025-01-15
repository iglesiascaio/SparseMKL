
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
        

    function get_iris_data()
        # Load dataset (Iris dataset)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        iris = CSV.read(download(url), DataFrame, header = false)

        # Rename columns
        rename!(iris, [:sepal_length, :sepal_width, :petal_length, :petal_width, :species])

        # Convert to binary classification (Iris-setosa vs. not Iris-setosa)
        iris.binary_class = iris.species .== "Iris-setosa"  # Setosa = 1, others = 0
        iris = filter(row -> row.binary_class == 1 || row.binary_class == 0, iris)

        # Encode labels as +1 and -1
        labels = [b ? 1 : -1 for b in iris.binary_class]

        # Select only the first two features (Sepal Length and Sepal Width)
        features = Matrix(iris[:, [:sepal_length, :sepal_width]])

        # Set random seed for reproducibility
        Random.seed!(123)

        # Shuffle indices and split the dataset
        n = size(features, 1)  # Number of samples
        indices = shuffle(1:n)

        # Define the split ratio
        train_ratio = 0.8
        train_size = Int(floor(train_ratio * n))

        # Split into training and testing sets
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

    function get_adult_data()

        # Load dataset (Adult dataset)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        adult = CSV.read(download(url), DataFrame, header = false)

        # Rename columns
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

        # Filter half of the dataset randomly
        Random.seed!(123)  # Set random seed for reproducibility
        n_rows = size(adult, 1)
        sample_indices = sample(1:n_rows, Int(floor(n_rows / 20)), replace = false)
        adult_filtered = adult[sample_indices, :]

        # Encode labels as +1 and -1 for binary classification (>50K vs. <=50K)
        adult_filtered.income_binary = adult_filtered.income .== " >50K"  # Income >50K is 1, <=50K is 0
        labels = [b ? 1 : -1 for b in adult_filtered.income_binary]

        # Select numeric features only for now (categorical features can be one-hot encoded later)
        features = Matrix(
            adult_filtered[
                :,
                [:age, :education_num, :capital_gain, :capital_loss, :hours_per_week],
            ],
        )

        # Handle missing values if any
        # (Adult dataset doesn't have missing numeric values, but adding safeguard)
        features = coalesce.(features, 0)

        # Set random seed for reproducibility (again for splitting)
        Random.seed!(123)

        # Shuffle indices and split the filtered dataset
        n = size(features, 1)  # Number of samples after filtering
        indices = shuffle(1:n)

        # Define the split ratio
        train_ratio = 0.8
        train_size = Int(floor(train_ratio * n))

        # Split into training and testing sets
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

        # Quick exploration of the filtered dataset
        println("Filtered dataset summary:")
        println("Number of samples: ", n)
        println("Number of features: ", size(features, 2))
        println("Training set size: ", size(X_train, 1))
        println("Testing set size: ", size(X_test, 1))
        println("Positive class ratio in training: ", mean(y_train .== 1))
        println("Positive class ratio in testing: ", mean(y_test .== 1))

        return X_train, y_train, X_test, y_test


    end

end
