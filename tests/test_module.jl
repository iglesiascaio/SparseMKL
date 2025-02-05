using Infiltrator, Debugger, Revise

# Include the GetData module
include("../data/get_data.jl")
using .GetData: get_iris_data, get_adult_data  # Import only the get_iris_data function from GetData

# Call the function from the module
X_train, y_train, X_test, y_test = get_iris_data(force_download=true)

X_train, y_train, X_test, y_test  = get_adult_data(force_download=true)