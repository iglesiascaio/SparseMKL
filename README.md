# Sparse Multiple Kernel Learning (SMKL)

A reference implementation accompanying the paper:

> **Bertsimas, D., Iglesias, C. d P., & Johnson, N. A. G. (2025). *Sparse Multiple Kernel Learning: Alternating Best Response and Semidefinite Relaxations.***

This repository provides:

* heuristic algorithm for cardinality–constrained multiple-kernel learning that provides a good feasible solution;
* different types of semidefinite programming and second–order–cone relaxations that furnish provable lower bounds;
* experimental pipeline used to generate results reported in the paper.

---

## Repository layout

```
├── data/                 # raw datasets + helper script (Julia)
├── src/                  # solver source code
│   ├── MKL/              # baseline dense MKL implementation
│   ├── Sparse_MKL/       # alternating best-response heuristic (our SMKL)
│   └── Lower_Bound_models/  # MISDP & SDP/SOCP relaxations
├── scripts/              # experiment launchers (Julia / Python / Bash)
├── notebooks/            # result analysis and plotting (Jupyter)
├── results/              # pre-computed CSV outputs for every experiment
└── README.md             # you are here
```

---

## Quick start

### 1 · Clone and set up the environment

```bash
# clone
git clone https://github.com/iglesiascaio/SparseMKL.git
cd SparseMKL

# create an isolated Julia environment
autoactivate=true julia --project -e 'using Pkg; Pkg.instantiate()'

# (optional) create a Python venv for `MKLpy` baselines
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # installs MKLpy, numpy, pandas, ...
```

All Julia dependencies are pinned in `Project.toml`/`Manifest.toml` (Julia ≥ 1.10.0); the baseline solvers in `MKLpy` require Python ≥ 3.10.

### 2 · Download the datasets (once)

```bash
julia --project data/get_data.jl     # caches CSVs inside data/
```

The ten UCI binary-classification tasks used in the paper are fetched automatically (`iris`, `wine`, `breastcancer`, `ionosphere`, `spambase`, `banknote`, `heart`, `haberman`, `mammographic`, `parkinsons`).

### 3 · Run Experiments

TODO: parameters needed

```bash
# warm-start from pre-computed lower-bound solutions
julia --project scripts/run_experiments.jl          

# or perform a full 10-fold CV from scratch
CROSS_VALIDATION=true julia --project scripts/run_experiments.jl
```

Outputs are saved in CSV files.

### 4 · Compute lower-bound relaxations 

TODO: parameters needed

```bash
# produces SOC, 3×3-SDP and full-SDP bounds
julia --project scripts/run_lowerbound_formulations.jl
```

Depending on the dataset size and the chosen relaxation, solving the relaxations may require several GB of RAM.

---


## Implementation notes

* **SMKL heuristic** (`src/Sparse_MKL/`): alternating best-response with

  * SMO updates for the SVM dual (via `LIBSVM.jl`),
  * a Greedy Selector & Simplex Projector (GSSP) for the exact l0–simplex projection;
* **Lower bounds** (`src/Lower_Bound_models/`):

  * full SDP relaxation,
  * SOC relaxation (all 2×2 minors, plus optional random projections),
  * 3x3-SDP (all 3×3 principal minors). 

All models are expressed with JuMP; MOSEK is the chosen optimizer for SDP/SOCP subproblems. 

---

## Citing this work

If you use this codebase, please cite our paper:

```bibtex
@misc{bertsimas2025sparseMKL,
  author       = {Dimitris Bertsimas and Caio de Pr{\'o}spero Iglesias and Nicholas A. G. Johnson},
  title        = {Sparse Multiple Kernel Learning: Alternating Best Response and Semidefinite Relaxations},
  year         = {2025},
  eprint       = {TODO-UPDATE HERE},  
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

---

## License

The source code is released under the MIT License (see `LICENSE`).

---

## Contact

For questions, please open an issue or contact **Caio de Próspero Iglesias** (`caiopigl@mit.edu`).
