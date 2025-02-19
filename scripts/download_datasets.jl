
include("../data/get_data.jl")
using .GetData: get_adult_data, get_dataset

function download_all_datasets()
    for ds_name in keys(GetData.DATASET_CONFIG)
        println("Downloading dataset: ", ds_name)
        try
            GetData.get_dataset(ds_name; force_download=true)
            println("Successfully downloaded: ", ds_name)
        catch e
            println("Failed to download ", ds_name, ": ", e)
        end
    end
end

# Run the function to download all datasets
download_all_datasets()
