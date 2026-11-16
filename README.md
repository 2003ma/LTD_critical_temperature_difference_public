## Related Paper

This repository contains the code and data accompanying the following paper:

Momoko Amemiya and Yuki Izumida, “[Determining Critical Temperature Differences of Low-Temperature-Differential Stirling Engines: Nonlinear Dynamics Approach](https://arxiv.org/abs/2607.26539),” arXiv:2607.26539 (2026).

Figure, table, and equation numbers below refer to this paper.

## Project Structure

```text
.
├── README.md
├── bifurcation_plot (Figure 4)
│   └── param
│       ├── change_alpha
│       ├── change_beta
│       ├── change_gamma
│       ├── change_pair
│       └── change_sigma
├── bifurcation_point_nodim
│   └── param
│       ├── approximate_analytical_solution (Table I, Eq. 14)
│       │   ├── bif_pi.py
│       │   └── results_pi
│       └── numerical_solution (Table I, Eq. 13)
│           ├── bif_pi_brentq.py
│           └── results_pi
├── calculate_tau_tilde
│   ├── calculate.py
│   └── tau_tilde.txt
├── deltaT_omega_plot (Figure 2)
│   ├── experiment_plot.csv
│   ├── near_bif
│   │   ├── plot_minus.jl
│   │   ├── plot_plus.jl
│   │   ├── plots_given_params_2d_minus
│   │   │   └── alpha_0.157_beta_0.954_tau_0.0137_theta0_0
│   │   └── plots_given_params_2d_plus
│   │       └── alpha_0.157_beta_0.954_tau_0.0137_theta0_0
│   ├── plot_all_range.jl
│   └── plots_given_params_2d
│       └── alpha_0.157_beta_0.954_tau_0.0137_allrange
├── legend_plot (Figure 4)
│   ├── common_legend.pdf
│   ├── common_legend.png
│   └── legend.py
├── omega_deltaT
│   └── param
│       ├── dim (Numerical estimates for the caption of Figure 3)
│       │   ├── omega_deltaT_julia.jl
│       │   └── results
│       └── no_dim (Numerical estimates for Figure 4, Table I)
│           ├── tau
│           │   ├── omega_deltaT.jl
│           │   └── results
│           └── no_tau
│               ├── omega_deltaT.jl
│               └── results
└── phase_diagram (Figure 3)
    ├── dim
    │   ├── plot_dimensional_manifolds.py
    │   └── results_manifold_dimensional
    └── nodim
        └── plot_dimensional_manifolds.py
```

## Running the scripts

- Julia: `julia filename.jl`
- Python: `python3 filename.py`

## Rounding of numerical critical temperature estimates

For critical temperature estimates obtained from the ODE simulations in `omega_deltaT`, the sampled temperature difference on the nonrotating side immediately adjacent to the rotation–cessation boundary is selected from the simulation output. During post-processing, this value is rounded from N significant digits to N−1 significant digits for use in the plots and derived datasets. The number of retained significant digits depends on the intended use and may differ between figures, tables, and datasets. For Figure 4, the plotting scripts read the adopted values from the comparison CSV files; the scripts do not perform this rounding automatically.

## Experimental points in the Gamma plot

For the experimental points shown as diamonds in Figure 4(b), use the first measured temperature difference in each trial at which the absolute rotation frequency is at most 1e-5 Hz. For the no-added-load condition, average these temperature differences over the six positive-temperature-difference trials; for L1, L2, and L4, use the value from the corresponding loaded trial. Divide the resulting temperature differences by T_top to obtain the plotted dimensionless values.

## License

This project is licensed under the [MIT License](LICENSE).
