# Changes since the course version

This project started as my exam project for the *Machine Learning and Deep Learning* course (University of Calabria, A.Y. 2025/2026). The report and slides of that version, in Italian, are in [`docs/course/`](docs/course/); the numbers in them come from one run per configuration and are superseded by [`results/`](results/).

The method is the same. What changed is the engineering around it and the way it is evaluated.

## Evaluation

- **Five seeds per configuration instead of one.** A seed fixes both the attacks sampled into the training set and the initial weights, so variants are compared pair by pair on identical inputs, with a Wilcoxon signed-rank test. With a single run, differences of a few thousandths of AUROC cannot be told apart from initialisation noise.
- **A vanilla baseline.** The same VAE + planar flow trained on all data with no rejection, to measure what the GRAnD rejection itself contributes.
- **Rejection diagnostics.** The training labels, never used by the model, measure after the fact how many injected attacks the split puts in `S` or `U` and how many remain in `L`.
- **KL weight sensitivity.** The course code weighted the KL term by 0.05; the paper uses 1. Both are now run and reported.
- **An unsupervised alarm threshold.** Besides AUROC and AUPRC, precision, recall and F1 at a threshold chosen without labels (the boundary `u` of the nominal set on the training scores).

## Method details

- **Invertible planar flows.** `f(z) = z + u·tanh(wᵀz + b)` is a bijection only when `wᵀu ≥ −1`. The vector `u` is now reparametrised as in Rezende & Mohamed (2015, App. A) so that the condition holds by construction and `log|det J|` is a valid density correction.
- **Deterministic scores.** The free energy depends on the latent sample. Test scores are now the mean over 16 latent draws from a seeded generator, so the same model always gets the same AUROC, and Integrated Gradients explains a smooth, fixed function.
- **One energy everywhere.** Training, scoring and explanations all use the same `free_energy` function, with the KL weight in the configuration.
- **Preprocessing fitted on the training set actually used.** The min-max scaler no longer sees the attacks left out of a given training set. The one-hot vocabulary still comes from the whole KDDTrain+ file, which keeps 122 inputs at every contamination level.
- **AE-SAD scheduling.** Negative learning now ramps up over epochs 10–20, and structural early stopping cannot fire before every scheduled loss term is fully active. In the course version the ramp started at epoch 25 while early stopping could end training as early as epoch 35.

## Engineering

- A Python package (`src/grand`) with a single configuration object and named ablation variants, replacing notebook cells and module-level settings.
- A `grand` command line for download, training, the experiment grid, the report, the decomposition of the score into its two terms, and explanations.
- A resumable, parallel experiment grid that writes every run to `results/runs.csv`.
- The Generalized Pareto fit uses SciPy's maximum likelihood instead of R through `rpy2`: same estimator, no R installation.
- NSL-KDD is downloaded from a public mirror and checked against SHA-256 checksums instead of being shipped with the code.
- 36 tests, run on every push by GitHub Actions. They check the mathematics (flow log-determinant against the autograd Jacobian, recovery of a known GPD quantile, completeness of Integrated Gradients) and the protocol (exact contamination, labels never reaching the training).
- The five threshold policies of the course version (POT through R, POT through SciPy, the same with EMA smoothing, POT with cap, double quantile) are now three, POT, POT with empirical cap and double quantile, each with optional EMA smoothing. The curriculum on `t_q`, never evaluated, was dropped.
