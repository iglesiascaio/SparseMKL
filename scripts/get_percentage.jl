include("../data/get_data.jl")
using .GetData: get_dataset

# Function to compute percentage of +1 labels
function compute_positive_percentage(ds_name::Symbol)
    _, y_train, _, y_test = GetData.get_dataset(ds_name; frac=1.0, train_ratio=0.8)
    y_all = vcat(y_train, y_test)
    pos_frac = count(==(1), y_all) / length(y_all)
    return round(pos_frac * 100, digits=1)
end

# Datasets you are using
datasets = [:iris, :wine, :breastcancer, :ionosphere, :spambase, :banknote, :heart, :haberman, :mammographic, :parkinsons]

# Compute and print
for ds in datasets
    pos_pct = compute_positive_percentage(ds)
    println("Dataset $(ds): $(pos_pct)% positive (+1) labels")
end
