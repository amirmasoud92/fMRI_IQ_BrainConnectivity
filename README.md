# Predicting intelligence from a single naturalistic fMRI run

Analysis code for a study of individual differences in general intelligence predicted from
movie-watching fMRI in AOMIC-ID1000 (*n* = 877), with generalisation to AOMIC-PIOP1 (*n* = 215) and
AOMIC-PIOP2 (*n* = 225). The repository contains the processing, modelling, statistical and plotting
code behind the reported results: feature extraction, model fitting and evaluation, confound control,
corrected inference for cross-validated comparisons, spatial null models, cross-collection
generalisation, and the figures.

The evaluation design is fixed in code: a hold-out set of 176 participants was sealed before any final
model was assessed, and model comparisons use a corrected resampled *t*-test rather than a naive paired
test on fold scores.

## Layout

```
config/       pipeline.yaml: data locations, atlas, preprocessing and model settings
src/          library code imported by the scripts
  data/         BIDS loading, confound regression, region time series, graph construction
  stats/        corrected resampled t-test, Bayesian correlated t-test, verdicts
  models/       graph autoencoder (withdrawn from the paper; evaluated in Fig. S1)
  training/     training loop and losses of the graph autoencoder
  evaluation/   metrics of the graph autoencoder
  visualization/  figure style, export, cortical surface rendering
  io_utils.py    result writing, hashing and the record saved beside each result table
scripts/
  extraction/       region time series, connectivity, morphometry and diffusion features
  prediction/       hold-out sealing and evaluation, confound benchmark, comparison table
  confounds/        confound-isolated test sets, spline and within-sex controls
  models/           representation, decoder and diagnostic analyses
  brain_maps/       Haufe forward weights and spatial null models
  generalisation/   transfer to PIOP1 and PIOP2, heterogeneity and equivalence
  nested_cv_evaluation/ training and nested cross-validation evaluation of the graph autoencoder
  figures/          one script per figure, plus make_figures.py
tests/        unit tests of the inference, confound control and figure-style code
```

## Requirements

Python 3.12 with numpy, scipy, pandas, scikit-learn, statsmodels, patsy, nilearn, nibabel, neuromaps,
brainsmash, h5py, torch, torch-geometric, matplotlib, joblib, threadpoolctl, tqdm, loguru and pyyaml.
Pinned versions are in `requirements.txt`.

PyTorch is used for linear algebra on the covariance matrices and runs on CPU or GPU; the graph
autoencoder in `scripts/nested_cv_evaluation/` needs a GPU to train in reasonable time.

## Data

The three AOMIC collections and their fMRIPrep, FreeSurfer and diffusion derivatives are public
(OpenNeuro ds003097, ds002785, ds002790). Set their locations in `config/pipeline.yaml`:

```yaml
paths:
  bids_root: .                       # AOMIC-ID1000
  piop1_root: PIOP1
  piop2_root: ../PIOP2
  fmriprep_dir: derivatives/fmriprep
  external_derivatives_dir: ../derivatives
```

Scripts are run from the repository root, and relative paths resolve from there.

Extraction writes to `outputs/`, and every later stage reads the result tables it finds there. The
figures are built from those tables alone, so they can be regenerated without repeating the extraction
if the tables are available. `outputs/` and `figures/` are not part of this repository; they are
deposited with the paper.

## Pipeline

Each stage depends on the ones above it.

| Stage | Directory | Notes |
| --- | --- | --- |
| Region time series and features | `scripts/extraction/` | hours per collection; writes `outputs/connectivity/` |
| Sealed hold-out and prediction | `scripts/prediction/` | `seal_holdout.py` first, then `evaluate_holdout.py` |
| Confound control | `scripts/confounds/` | confound-isolated test sets, spline and within-sex models |
| Representations and decoders | `scripts/models/` | one script per analysis |
| Forward maps and spatial nulls | `scripts/brain_maps/` | surface spin test and variogram surrogates |
| Generalisation to PIOP | `scripts/generalisation/` | frozen pipeline, no refitting |
| Graph autoencoder | `scripts/nested_cv_evaluation/` | the withdrawn model, kept for Fig. S1 |
| Figures | `scripts/figures/make_figures.py` | writes `figures/<name>/` |

Most scripts take command-line arguments and describe them under `--help`. Several analyses write their
result tables through `src/io_utils.py`, which saves a record of the arguments, the library versions and
the hashes of the inputs and the output beside each table.

To regenerate every figure from existing result tables:

```
python scripts/figures/make_figures.py            # all figures
python scripts/figures/make_figures.py Fig2 FigS3 # a subset
```

Each figure script writes the plotted values beside the image as CSV, so every panel can be checked
against the numbers it was drawn from.

## Figures

`scripts/figures/fig_N.py` draws figure N. The analyses behind each figure are:

| Figure | Analyses |
| --- | --- |
| Fig. 1 | `prediction/compile_comparison_table.py`, `prediction/seal_holdout.py` |
| Fig. 2 | `prediction/evaluate_holdout.py`, `prediction/holdout_metrics.py`, `prediction/confound_benchmark.py`, `prediction/multimodal_stack.py` |
| Fig. 3 | `confounds/confound_controls.py` |
| Fig. 4 | `models/ceiling_curves.py`, `prediction/compile_comparison_table.py` |
| Fig. 5 | `brain_maps/haufe_forward_maps.py`, `brain_maps/spatial_nulls_surface.py`, `generalisation/leave_one_collection_out.py`, `generalisation/cross_collection_transfer.py`, `generalisation/cross_collection_pooling.py` |
| Fig. S1 | `nested_cv_evaluation/nested_cv_comparison.py` |
| Fig. S2, S5 | `prediction/compile_comparison_table.py` |
| Fig. S3 | `prediction/evaluate_holdout.py`, `prediction/holdout_metrics.py` |
| Fig. S4 | `confounds/confound_controls.py` |
| Fig. S6 | `brain_maps/spatial_nulls_surface.py`, `brain_maps/spatial_nulls_centroid.py` |
| Fig. S7 | `generalisation/heterogeneity_and_equivalence.py`, `generalisation/piop_accuracy_prediction.py` |
| Fig. S8 | `brain_maps/haufe_forward_maps.py`, `generalisation/cross_collection_transfer.py` |

## Tests

Each test file is a script that runs its own cases and exits non-zero on failure:

```
python tests/test_cv_inference.py
```

The tests cover the corrected inference, the confound-control internals, the hold-out metrics, the
inter-subject correlation bookkeeping and the figure style module. Most run on synthetic data; two read
extracted time series and need `outputs/` to be populated.

## Notes

The graph autoencoder that the project began with is retained under `src/models/`, `src/training/` and
`scripts/nested_cv_evaluation/` because the paper reports its evaluation (Fig. S1); it is not part of the
reported prediction pipeline. The configuration file keeps its original project name for the same
reason.

The hold-out set was sealed once; `seal_holdout.py` reproduces the split from its recorded seed and
manifest, and `evaluate_holdout.py` refuses to run against a manifest whose hash does not match.
