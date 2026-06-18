# Reconstructing the Han Dynasty Western Regions from the *Book of Han* Using Physics-Based Spatial Optimization

By Lin Yun-Han

## Project Overview

This project reconstructs the relative spatial layout of states and sites in the Han dynasty Western Regions based on distance and direction descriptions extracted from the *Book of Han* (*Hanshu*, 漢書). The project treats historical geographic reconstruction as a constrained spatial optimization problem: textual distance records, directional statements, external site-position references, and anchor constraints are transformed into computable spatial constraints.

The core method, **PhysicsSim**, models historical places as particles in a damped physical system. Distance descriptions are represented as spring-like constraints, direction descriptions are represented as directional correction forces, and optional repulsion regularization is used to examine whether spatial separation improves reconstruction stability. The resulting layouts are evaluated using distance stress, directional consistency, and held-out site-position RMSE.

This repository contains the code, data processing scripts, experiment pipelines, and visualization tools used to reproduce the reconstruction experiments.

## Research Goals

This project aims to:

* Convert textual descriptions from the *Book of Han* into structured distance and direction constraints.
* Reconstruct the relative spatial configuration of Han dynasty Western Regions states.
* Compare physics-based reconstruction with MDS-based baseline models.
* Evaluate model performance using quantitative metrics rather than visual inspection alone.
* Analyze uncertainty arising from sparse textual data, random initialization, hyperparameter selection, and external site-position matching.

## Data

The main historical source used in the current version is:

* *Book of Han* (*Hanshu*, 漢書), especially the Western Regions material.

The structured input data include:

```text
data/distance_edges_verified.csv
data/direction_edges_verified.csv
data/site_rmse_points.csv
data/ini_data.csv
```

The `site_rmse_points.csv` file contains external site-position reference points. These are not treated as absolute ground truth. Instead, they are used as an external benchmark for evaluating how well the reconstructed layout aligns with plausible archaeological or historical site positions.

The current site-position design separates:

```text
3 anchor points
8 held-out test points
```

The anchors are used for alignment and hyperparameter validation. The held-out test points are reserved for final evaluation using `RMSE_test_km`.

## Methods

### PhysicsSim

PhysicsSim is a damped physics-based reconstruction framework. It integrates several types of constraints:

| Component                | Role                                                      |
| ------------------------ | --------------------------------------------------------- |
| Distance constraints     | Preserve historical distance relations                    |
| Direction constraints    | Enforce qualitative direction statements                  |
| Anchor constraints       | Align selected nodes to external site-position references |
| Repulsion regularization | Test whether spatial separation improves layout stability |
| Damping                  | Stabilize the physical simulation process                 |

The model is designed not only to produce a final layout, but also to support diagnostic analysis such as convergence behavior, directional violations, node displacement, and error visualization.

### Baseline Models

The repository also includes baseline comparison models:

| Model                        | Information used                 |
| ---------------------------- | -------------------------------- |
| SMACOF / Stress Majorization | Distance only                    |
| DirectedMDS                  | Distance + direction             |
| PhysicsSim-DistOnly          | Distance only                    |
| PhysicsSim-NoDir             | Distance + repulsion             |
| PhysicsSim-NoRep             | Distance + direction             |
| PhysicsSim-Full              | Distance + direction + repulsion |

The ablation variants allow the contribution of direction constraints and repulsion regularization to be evaluated separately.

## Evaluation Metrics

The main quantitative metrics are:

| Metric                        | Meaning                                              |
| ----------------------------- | ---------------------------------------------------- |
| `E_distance_stress`           | Distance reconstruction error                        |
| `E_direction_vr`              | Direction violation rate                             |
| `E_direction_mae`             | Mean angular error                                   |
| `RMSE_test_km`                | RMSE against held-out site-position benchmark points |
| `min_pairwise_distance_km`    | Minimum spacing between reconstructed nodes          |
| `median_pairwise_distance_km` | Median spacing between reconstructed nodes           |

The final evaluation focuses on held-out test sites rather than anchor points.

## Experiment Pipeline

### 1. Preflight Checks

Run these commands from the project root:

```bash
python -m scripts.check_site_points
python -m scripts.check_direction_data
python -m scripts.rebuild_ini_data
```

These checks verify the site-position file, direction-edge format, and initialization data.

### 2. Single PhysicsSim Reconstruction

```bash
python -m run_paper_script.paper_run ch4 --seed 0 --plot
```

This runs a single PhysicsSim reconstruction and produces reconstruction outputs for inspection.

### 3. Baseline Visualization

```bash
python -m run_paper_script.paper_run ch5-baseline --model StressMajorization --vis
python -m run_paper_script.paper_run ch5-baseline --model DirectedMDS --vis
```

These commands generate baseline layouts for comparison.

### 4. Hyperparameter Optimization

The current HPO design uses three-anchor leave-one-out validation. The held-out test sites are not used for selecting hyperparameters.

Example HPO command:

```bash
python -m run_paper_script.paper_run ch5-hparam-kfold \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --alpha-min -1 \
  --alpha-max 1.5 \
  --alpha-step 0.5 \
  --beta-min -2 \
  --beta-max 0.5 \
  --beta-step 0.5 \
  --outdir outputs/ch5_hparam_anchor_loo_grid
```

The hyperparameters are represented as:

```text
alpha = log10(w_dir / w_dis)
beta  = log10(w_reg / w_dis)
```

After the grid search, Pareto non-dominated candidates are inspected. A final candidate can be selected manually from the Pareto front:

```bash
python -m scripts.select_hpo_candidate \
  --source-hpo-outdir outputs/ch5_hparam_anchor_loo_grid \
  --alpha 1 \
  --beta 0.5 \
  --outdir outputs/ch5_hparam_anchor_loo_grid_manual_alpha_1_beta_0.5
```

The selected candidate is then used for benchmark and ablation experiments.

### 5. Ablation Study

The formal ablation study compares four PhysicsSim variants and two baseline models under the same anchors, held-out test sites, random seeds, and final HPO-selected parameters.

Smoke test:

```bash
python -m run_paper_script.paper_run ch5-ablation \
  --hpo-outdir outputs/ch5_hparam_anchor_loo_grid_manual_alpha_1_beta_0.5 \
  --seeds 0 \
  --outdir outputs/ch5_ablation_smoke
```

Formal 100-seed ablation:

```bash
python -m run_paper_script.paper_run ch5-ablation \
  --hpo-outdir outputs/ch5_hparam_anchor_loo_grid_manual_alpha_1_beta_0.5 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99 \
  --outdir outputs/ch5_ablation_final_100seed
```

The ablation output includes per-seed results, summary statistics, paired differences, and win-rate comparisons.

### 6. Visualization

Example visualization commands:

```bash
python -m run_paper_script.paper_run ch6-visualize --model PhysicsSim --seed 0 --no-wait
python -m run_paper_script.paper_run ch6-visualize --model SMACOF --no-wait
python -m run_paper_script.paper_run ch6-visualize --model DC-SMACOF --no-wait
python -m run_paper_script.paper_run ch6-map
```

## Output Files

Generated experiment outputs are written to the `outputs/` directory. Large output folders may be omitted from the GitHub repository and regenerated using the commands above.

Typical ablation outputs include:

```text
ablation_runs_by_seed.csv
ablation_summary.csv
ablation_paired_differences.csv
ablation_config.json
ablation_final_positions_y_up_sim.csv
```

Typical HPO outputs include:

```text
grid_runs_by_seed.csv
grid_summary_cv.csv
pareto_front_3d.csv
selected_final_summary.json
selected_candidate_summary.csv
```

## Repository Structure

```text
MDS_model/                 Baseline MDS and DirectedMDS models
data/                      Verified distance, direction, and site-position data
library/                   Core model, metrics, geometry, and visualization modules
run_paper_script/          Reproducible experiment entry points
scripts/                   Utility scripts and data checks
tests/                     Contract and pipeline tests
outputs/                   Generated experiment outputs
```

## Installation

This project uses Python 3.8+.

Recommended dependencies include:

```bash
pip install numpy pandas scipy scikit-learn matplotlib pymunk pygame pyproj
```

Some visualization functions may require additional plotting or GIS-related packages depending on the execution environment.

## Notes on Interpretation

This project does not claim to recover a definitive historical map. The reconstruction is a model-based approximation derived from sparse and uncertain historical descriptions. External site-position references are used for evaluation, but they are not treated as absolute ground truth.

The main contribution is a reproducible framework for transforming historical textual spatial descriptions into computable constraints, evaluating reconstruction quality with multiple metrics, and analyzing the uncertainty and sensitivity of the resulting map.

## Acknowledgements

This project builds on earlier work on the geography of the Han dynasty Western Regions and on prior discussions of Chen Shih-Liang’s mileage hypothesis. The current reconstruction data are based on structured distance and direction information extracted from the *Book of Han*.

Special thanks to the teachers, advisors, and researchers who provided feedback during the development of this project.
