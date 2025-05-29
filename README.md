# Sparse Multiple Kernel Learning (SMKL)

Software supplement for the manuscript **“Sparse Multiple Kernel Learning: Alternating Best Response and Semidefinite Relaxations”** by Dimitris Bertsimas, Caio de Próspero Iglesias, and Nicholas A. G. Johnson.

## Introduction

Support Vector Machines (SVMs) aim to construct predictive classifiers by embedding inputs $x_i \in \mathbb{R}^m$ into a high- (possibly infinite-) dimensional feature space and learning a hyperplane that separates labels $y_i \in \{-1,1\}$. Rather than choosing a single kernel $K$ a priori, Multiple Kernel Learning (MKL) considers convex combinations of $q$ candidate kernels $\{K_1,\dots,K_q\}$, with weights $\beta \in \mathbb{R}^q_+$, to jointly learn a suitable representation and classifier.

In standard MKL, $\beta$ lies in the probability simplex, encouraging sparsity only implicitly. In this work, we consider a more structured variant—**Sparse MKL**—where an explicit cardinality constraint $\|\beta\|_0 \leq k_0$ is imposed. We study the following nonconvex-concave problem:

$\min_{\beta \in \mathbb{R}^q_+} \max_{\alpha \in \mathbb{R}^n_+} \sum_{i=1}^n \alpha_i - \frac{1}{2} (y \circ \alpha)^T \left( \sum_{j=1}^q \beta_j K_j \right) (y \circ \alpha) + \lambda |β|_2^2$

subject to:

$\sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C, \quad \|\beta\|_1 = 1, \quad \|\beta\|_0 \leq k_0.$

We provide a scalable alternating best-response algorithm to compute high-quality feasible solutions, as well as SDP and SOCP relaxations to compute lower bounds and evaluate optimality gaps.

## Installation and set-up

In order to run this software, you must install a recent version of Julia from http://julialang.org/downloads/, and a recent version of the Mosek solver (academic licenses are freely available at https://www.mosek.com/products/academic-licenses/). The most recent version of Julia at the time this code was last tested was Julia 1.10.1 using Mosek version 10.1.31.

Several packages must be installed in Julia before the code can be run.  These packages can be found in "SparseLowRankSoftware.jl". The code was last tested using the following package versions:

### Main packages used

#### Julia packages

- JuMP v0.23.2  
- LIBSVM v0.8.0  
- MosekTools v0.15.1  
- StatsBase v0.34.2  

#### Python packages

- pandas: 2.2.3  
- numpy: 1.26.4  
- scikit-learn: 1.6.0  
- torch: 2.5.1  
- julia: 0.6.2  
- MKLpy: 0.6  


## Use of the main routines

* **`train_sparse_mkl(X, y, C, K_list, λ; k0, …)`**
  Alternating best-response heuristic that returns $\alpha, \beta$ and the combined kernel. We also have the option here to choose if we want to use warm start and which method to use. 
* **`LowerBoundFormulations.main()`**
  Dispatches the full-SDP, 3×3-SDP and SOC relaxations and writes lower-bound CSVs to `results.csv`.

Example:

```julia
# include other files
include("../src/Sparse_MKL/sparse_multi_kernel.jl")
include("../data/get_data.jl")
include("../src/MKL/multi_kernel.jl")

# import relevant functions
using .SparseMKL: train_sparse_mkl
using .GetData: get_dataset
using .MKL: compute_kernels

# set kernels
kernels = [
    Dict(:type => "linear", :params => Dict()),
    Dict(:type => "polynomial", :params => Dict(:degree => 2, :c => 1.0)),
    Dict(:type => "rbf",       :params => Dict(:gamma => 0.5)),
    Dict(:type => "sigmoid",   :params => Dict(:gamma => 0.5, :c0 => 1.0)),
    Dict(:type => "laplacian", :params => Dict(:gamma => 0.3)),
]

# get dataset
Xtr, ytr, Xte, yte = get_dataset(:iris; force_download=true, frac=1.0, train_ratio=0.8)
Klist  = compute_kernels(Xtr, Xtr, kernels)
α, β, K, obj = train_sparse_mkl(Xtr, ytr, 50.0, Klist, 0.1; k0=3) # chooses k0=3, meaning at most 3 kernels will be used non-zero
```

Experiment reproduction:

```bash
cd scripts; julia
include("run_experiments.jl")     # heuristic method (our SMKL)          
include("run_lowerbound_formulations.jl")  # relaxations
```

Note that we ran this multiple times choosing different parameters in the code (e.g. warm-start-method, cross-validation, etc.)

Moreover, there is also a script `run_benchmark.py` for running other Multiple Kernel Algorithms from MKLpy. 

## Citing sparseMKL

If you use this code, please cite the accompanying manuscript:

```bibtex
@unpublished{bertsimas2025sparseMKL,
  author  = {Dimitris Bertsimas and Caio de Pr{\'o}spero Iglesias and Nicholas A. G. Johnson},
  title   = {Sparse Multiple Kernel Learning: Alternating Best Response and Semidefinite Relaxations},
  note    = {Manuscript under review},
  year    = {2025}
}
```

## Thank you

Thank you for your interest in SMKL. Feel free to open an issue or contact us directly:

* **Dimitris Bertsimas**  — [dbertsim@mit.edu](mailto:dbertsim@mit.edu)
* **Caio de Próspero Iglesias** — [caiopigl@mit.edu](mailto:caiopigl@mit.edu)
* **Nicholas A. G. Johnson** — [nagj@mit.edu](mailto:nagj@mit.edu)
