#!/usr/bin/env julia

# Include the module with the formulations
include("../src/Lower_Bound_models/lower_bound_formulations.jl")
using .LowerBoundFormulations

# Run the main function to execute all methods and datasets
LowerBoundFormulations.main()