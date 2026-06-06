# Unsymmetric policy-iteration figure code

This repository contains a compact standalone implementation for three figures based on an unsymmetric grid-based policy-iteration method for the controlled Van der Pol oscillator.

## Repository layout

```text
.
├── plot_initial_phase_portraits.py
├── plot_policy_iteration_errors.py
├── plot_nested_domains.py
├── functions/
│   ├── auxFunctions.py
│   ├── kernel.py
│   ├── model.py
│   ├── observer.py
│   └── plotStyle.py
├── data/
├── figures/
└── requirements.txt
```

`functions/kernel.py` contains the product-kernel implementation.  The class structure is intentionally small so that further kernels can be added.

`functions/model.py` contains the model interface and the controlled Van der Pol oscillator.  Further models can be added by following the same method layout.

`functions/auxFunctions.py` contains the unsymmetric policy-iteration routine, the shrinking-domain variant, reference-data generation, grid construction, and trajectory simulation helpers.

## Figure scripts

```bash
python plot_initial_phase_portraits.py
python plot_policy_iteration_errors.py
python plot_nested_domains.py
```

The scripts write their output to `figures/`:

```text
figures/initial_phase_portraits.pdf
figures/policy_iteration_errors.pdf
figures/nested_domain_portrait.pdf
```

The convergence plot computes finite-horizon reference values and stores them in `data/` after the first run.  Subsequent runs reuse the cached values.

## Requirements

Install the Python packages with

```bash
pip install -r requirements.txt
```

The PDF plots use Matplotlib with LaTeX rendering, so a working LaTeX installation is needed for the final typography.
