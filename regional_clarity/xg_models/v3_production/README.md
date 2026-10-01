---
license: mit
library_name: xgboost
pipeline_tag: tabular-regression
tags:
  - xgboost
  - tabular-regression
  - ensemble
  - remote-sensing
  - landsat
  - water-quality
  - water-clarity
  - secchi-disc-depth
  - limnology
metrics:
  - rmse
  - mae
  - r_squared
model-index:
  - name: intermountain-west-regional-clarity-sdd-landsat
    results:
      - task:
          type: tabular-regression
          name: Secchi disc depth estimation
        dataset:
          name: AquaMatch SDD–Landsat matchups, fixed HUC8 holdout (CO, ID, MT, NM, UT, WY)
          type: aquamatch-sdd-landsat-matchups
          split: holdout
        metrics:
          - type: rmse
            value: 1.259
            name: Holdout RMSE (m)
          - type: mae
            value: 0.940
            name: Holdout MAE (m)
          - type: r_squared
            value: 0.626
            name: Holdout R²
---

# Intermountain West Regional Secchi Disc Depth Model

An ensemble of 40 XGBoost regressors that estimates Secchi disc depth (SDD, in meters) for lakes and reservoirs from Landsat Collection 2 surface reflectance. It also uses antecedent weather and static catchment characteristics. The model covers six states in the intermountain West (Colorado, Idaho, Montana, New Mexico, Utah, and Wyoming) across Landsat 4, 5, 7, 8, and 9, 1984–2023.

On a fixed holdout of whole HUC8 basins that no ensemble member trained on, the model reaches an RMSE of 1.26 m, an MAE of 0.94 m, and an R² of 0.63. It is most accurate below about 6 m SDD and underestimates very clear water (see [Limitations](#limitations)).

The full workflow, with code and outputs for every step, is published at <https://rossyndicate.github.io/regional-clarity-RS-model/>.

## Model details

| | |
|---|---|
| **Developed by** | B Steele, ROSSyndicate, Colorado State University |
| **Model type** | Ensemble of gradient-boosted regression trees (XGBoost 3.2, `reg:squarederror`) |
| **Ensemble** | 10 independent seeds (601–610) × 4 cross-validation folds = 40 boosters; the prediction is their unweighted mean |
| **Input** | One row per Landsat observation at a waterbody location: harmonized surface reflectance and spectral indices, antecedent gridMET weather, LakeCat catchment metrics, and a shoreline flag (26–33 features per seed; see [Features](#features)) |
| **Output** | Secchi disc depth, in meters |
| **License** | MIT |
| **Source code** | <https://github.com/rossyndicate/regional-clarity-RS-model> |
| **Contact** | B Steele (repository maintainer) |

## Intended use

**Primary use.** The model estimates SDD for lakes and reservoirs in the six-state region on dates with a usable Landsat overpass. This extends clarity records in space and time beyond in situ sampling. For example, at locations with in situ SDD, the model adds about 660,000 quality-controlled location-days of estimates across 1,796 locations ([step 07](https://rossyndicate.github.io/regional-clarity-RS-model/regional_clarity/07_regional_application.html)).

**Intended users.** Limnologists, water managers, and researchers who are studying regional and long-term patterns in water clarity.

**Appropriate inputs.** Inputs must be prepared the same way as the training data:

- Landsat Collection 2 surface reflectance from AquaMatch siteSR, using DSWE1 (high-confidence water) pixels;
- bands harmonized across sensors to the Landsat 7 reference using the AquaMatch handoff coefficients;
- the same scene QA as the training data;
- open-water-season scenes only;
- every feature inside the model's area of applicability.

The [Applicability checks](#applicability-checks) section gives the exact criteria.

**Out of scope.**

- Waterbodies outside the six-state region. The model has not been evaluated there, and the area-of-applicability check does not account for geographic distance.
- Ice-season scenes (outside day of year 70–320).
- Reflectance products other than AquaMatch siteSR with LS7-referenced harmonization, such as un-harmonized surface reflectance, other atmospheric corrections, or other sensors such as Sentinel-2.
- Regulatory or compliance decisions for a single waterbody on a single date. Individual estimates carry about ±1.3 m of typical error. The model is better suited to patterns aggregated across dates or sites.
- Rivers and streams. Training locations are lakes and reservoirs.

## How to use

The model is published on the Hugging Face Hub as [`bgsteele/intermountain-west-regional-clarity-sdd-landsat`](https://huggingface.co/bgsteele/intermountain-west-regional-clarity-sdd-landsat). It contains the full ensemble:

```
seed601/ … seed610/
  backward_elim_summary.json   # that seed's selected features ("final_features")
  final_tune_xgboost.json      # that seed's tuned hyperparameters
  final_eval_summary.json      # that seed's CV and holdout metrics
  xgboost_fold{1-4}.json       # the four fitted boosters (XGBoost JSON format)
ensemble_mean_abs_shap.csv     # ensemble SHAP ranking (also used to weight the AOA check)
```

The seed folders are also tracked in the source repository under [`regional_clarity/xg_models/v3_production/`](https://github.com/rossyndicate/regional-clarity-RS-model/tree/main/regional_clarity/xg_models/v3_production). `ensemble_mean_abs_shap.csv` is not, but [step 05](https://rossyndicate.github.io/regional-clarity-RS-model/regional_clarity/05_evaluate_ensemble.html) regenerates it.

Each seed has its own feature set, so every booster has to be given its seed's features, in order. The code below loads all 40 members and returns the ensemble mean. `X` is a pandas DataFrame with one row per observation and columns named as in [Features](#features). Spectral indices and coarsened site features must be computed first, using `add_spectral_indices()` and `coarsen_site_features()` from [`regional_clarity/python/features.py`](https://github.com/rossyndicate/regional-clarity-RS-model/blob/main/regional_clarity/python/features.py) in the source repository.

```python
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from huggingface_hub import snapshot_download

from features import add_spectral_indices, coarsen_site_features  # from the source repository


def load_ensemble(model_dir):
    members = []
    for seed_dir in sorted(Path(model_dir).glob("seed*")):
        feats = json.loads((seed_dir / "backward_elim_summary.json").read_text())["final_features"]
        for fold in range(1, 5):
            booster = xgb.Booster()
            booster.load_model(str(seed_dir / f"xgboost_fold{fold}.json"))
            members.append((feats, booster))
    return members


def predict_sdd(members, X):
    return np.mean([b.predict(xgb.DMatrix(X[feats])) for feats, b in members], axis=0)


model_dir = snapshot_download("bgsteele/intermountain-west-regional-clarity-sdd-landsat")   # or a local copy of the folder
members = load_ensemble(model_dir)   # 40 members
X = coarsen_site_features(add_spectral_indices(X))
sdd_m = predict_sdd(members, X)
```

This code reproduces the published holdout RMSE (1.2589 m) when it is run on the training repository's holdout table.

On macOS, set `OMP_NUM_THREADS` and `KMP_DUPLICATE_LIB_OK=TRUE` before importing XGBoost alongside other OpenMP libraries. The [repository README](https://github.com/rossyndicate/regional-clarity-RS-model#python-environment) explains why. Exact package versions are listed in `requirements-lock.txt` in the source repository.

## Training data

The training data are matchups between in situ SDD and same-location Landsat observations. Both come from AquaMatch data packages on the Environmental Data Initiative (EDI) repository, all at revision 1:

| EDI package | Product | Used for |
|---|---|---|
| [`edi.1856.1`](https://doi.org/10.6073/pasta/542a305e8484ae5abc33881bd7761308) | AquaMatch Secchi disc depth | Target variable (in situ SDD) |
| [`edi.2254.1`](https://doi.org/10.6073/pasta/f85622d6d32ef7fe6cff8d63c3b947c9) | siteSR | Landsat surface reflectance and temperature at sampling sites, scene-level metadata, and the site list |
| [`edi.2114.1`](https://doi.org/10.6073/pasta/941f3cb046f5e7fbe04d5811989ed810) | lakeSR | Cross-sensor handoff coefficients used to harmonize all Landsat missions to Landsat 7 |

The matchups were built from these packages in [steps 00–02a](https://rossyndicate.github.io/regional-clarity-RS-model/):

- **SDD quality control.** SDD records were filtered for quality, including per-site RANSAC outlier screening.
- **Scene QA.** Landsat observations passed scene QA (DSWE share and count, no clouds in the buffer, surface temperature 0–40 °C, scene cloud cover < 20%) and per-site band RANSAC.
- **Matching.** Each SDD sample was matched to the closest image within ±5 days. A pair was excluded if more than 25.4 mm of rain fell on a single day between the image and the sample.
- **Harmonization.** Bands from all sensors were harmonized to the Landsat 7 reference.

| | All matchups | CV pool (training) | Holdout |
|---|---|---|---|
| Matchups | 11,143 | 8,937 | 2,206 |
| Sites | 1,462 | 1,081 | 381 |
| HUC8 basins | 195 | 153 | 42 |

- **Period:** 1984-05-12 to 2023-11-10. About 85% of matchups fall in May–October.
- **Missions:** Landsat 5 (4,883 matchups), Landsat 7 (4,551), Landsat 8 (1,511), Landsat 9 (150), and Landsat 4 (48).
- **Target:** SDD ranges from 0.1 to 17.2 m (median 2.9 m, mean 3.3 m). Only 11.6% of matchups exceed 6 m.
- **Splits:** All splits are by HUC8, so every basin falls entirely on one side of every split ([step 03](https://rossyndicate.github.io/regional-clarity-RS-model/regional_clarity/03_split_data.html)). The holdout is about 20% of HUC8s, drawn once (holdout seed 12) and never used for training, tuning, or feature selection. For each ensemble seed, the remaining HUC8s are reshuffled into four size-balanced CV folds.

## Training procedure

Each of the 10 seeds runs the full pipeline independently ([step 04](https://rossyndicate.github.io/regional-clarity-RS-model/regional_clarity/04_make_models.html)). Every selection decision uses out-of-fold CV only.

1. **Candidate pool (45 features).** The pool contains the harmonized bands, curated spectral indices, LakeCat catchment metrics, elevation, `shore_flag`, and 1/3/7/30-day antecedent weather.
2. **Correlation pruning.** Within each cluster of features with |r| > 0.95, only the feature most correlated with SDD is kept. This step reduces the pool to 34 features.
3. **Gap-aware tuning.** A 25-trial random search is run. Among trials whose mean validation RMSE is within 2% of the best, the trial with the smallest train–validation gap is chosen.
4. **Backward elimination with shadow decoys.** At each step, the weakest real feature is dropped if its permutation importance does not beat the best row-shuffled shadow copy. Elimination stops when CV RMSE degrades more than 0.5% from the best.
5. **Final gap-aware tuning.** A 40-trial search is run on the selected feature set.
6. **Final training.** Four fold models are trained, each with early stopping (5,000 rounds maximum, patience 250) against its own validation fold.

The models are trained unweighted. Every seed converged on shallow, heavily regularized trees:

| Hyperparameter | Search space | Selected across seeds |
|---|---|---|
| `max_depth` | 2, 3, 4 | 3 (all seeds) |
| `eta` | 0.01, 0.03, 0.05 | 0.01 (9 seeds), 0.03 (1) |
| `subsample` | 0.6, 0.7, 0.8 | 0.6–0.8 |
| `colsample_bytree` | 0.4, 0.5, 0.7 | 0.4–0.7 |
| `min_child_weight` | 3, 5, 7, 10 | 3–10 |
| `reg_alpha` | 0, 0.1, 1, 5 | 0–5 |
| `reg_lambda` | 1, 5, 10, 20 | 1–20 |

The exact values for each seed are in `seed*/final_tune_xgboost.json`.

## Features

Each seed keeps between 26 and 33 features. Twenty-one features were kept by every seed (the *unanimous core*). The table shows how many of the 10 seeds kept each feature. Features dropped by correlation pruning (red, blue, and SWIR2 bands; GR ratio; most 1- and 3-day weather windows) are not shown.

| Group | Feature | Seeds | Definition |
|---|---|---|---|
| Optical | `green_corr7`, `nir_corr7`, `temp_corr7` | 10 | Median surface reflectance (green, NIR) and surface temperature, harmonized to Landsat 7 |
| | `swir1_corr7` | 9 | SWIR1 surface reflectance, harmonized to Landsat 7 |
| | `BR`, `BG`, `NR` | 10 | Blue/red, blue/green, and NIR/red band ratios |
| | `fai`, `NDVI`, `NDSSI`, `MNDWI` | 10 | Floating Algae Index, NDVI, NDSSI ((blue − NIR)/(blue + NIR)), and MNDWI (Xu 2006) |
| | `NDWI` | 9 | NDWI (McFeeters 1996) |
| | `atm_corr_LaSRC` | 10 | 1 for Landsat 8/9 (LaSRC atmospheric correction), 0 for Landsat 4/5/7 (LEDAPS) |
| Site | `catchment_area_sqkm` | 10 | LakeCat catchment area, winsorized at 617.1 km² and binned to 0.1 log10 units |
| | `pct_impervious_2006`, `pct_forest_2006`, `pct_cropland_2006`, `pct_wetland_2006` | 10 | LakeCat catchment land cover (NLCD 2006), rounded to whole percent |
| | `pct_urban_2006` | 8 | LakeCat catchment urban land cover, rounded to whole percent |
| | `elevation_m` | 9 | Site elevation |
| | `shore_flag` | 10 | 1 if the site is within 230 m of the matched waterbody's shoreline or falls outside its polygon |
| Weather | `srad_Wm2_prev1`, `tmean_degC_prev30`, `tmin_degC_prev30`, `tmax_degC_prev30`, `srad_Wm2_prev30` | 10 | gridMET summaries over the *N* days strictly before the image date: mean for `srad` and `tmean`, minimum for `tmin`, maximum for `tmax`, and total for `precip` |
| | `tmean_degC_prev7`, `srad_Wm2_prev7` | 7 | |
| | `tmin_degC_prev7` | 6 | |
| | `precip_mm_prev30` | 5 | |
| | `precip_mm_prev1` | 4 | |
| | `tmax_degC_prev7` | 3 | |
| | `precip_mm_prev3` | 1 | |

The catchment and land-cover features are coarsened because they are constant for every observation on a waterbody. Without coarsening, a model could use them to identify a waterbody instead of describing it. The 617.1 km² cap is fixed at the training-data 99th percentile and must be applied unchanged at inference. XGBoost handles missing values natively, and band ratios that divide by zero are set to missing.

**Attribution.** Ensemble mean |SHAP| on the holdout, computed over the unanimous core ([step 05](https://rossyndicate.github.io/regional-clarity-RS-model/regional_clarity/05_evaluate_ensemble.html)), splits as follows: 76% optical, 14% site, and 10% weather. The top features are `BG` (0.48 m), `green_corr7` (0.48 m), `BR` (0.25 m), `MNDWI` (0.23 m), `fai` (0.18 m), and `catchment_area_sqkm` (0.15 m).

## Evaluation

### Holdout

Predictions are averaged to one per site-date before scoring (1,833 site-dates from 2,206 matchups).

| Metric | Value |
|---|---|
| RMSE | 1.259 m |
| MAE | 0.940 m |
| Bias (predicted − observed) | +0.148 m |
| R² | 0.626 |
| MAPE | 41.5% |
| sMAPE | 31.8% |

Error grows with water clarity. Below 4 m, the model runs high by about 0.4–0.6 m. From 4 to 6 m it is close to unbiased. Above 6 m, predictions are compressed toward the regional mean:

| Observed SDD | n | RMSE (m) | Bias (m) |
|---|---|---|---|
| 0–2 m | 616 | 1.00 | +0.57 |
| 2–4 m | 671 | 1.07 | +0.41 |
| 4–6 m | 385 | 1.20 | −0.09 |
| 6–10 m | 148 | 2.30 | −1.86 |
| > 10 m | 13 | 3.64 | −3.42 |

### Robustness

- **Choice of holdout.** Holdout RMSE depends on which basins are held out. The same 10 seed configurations were retrained against 16 different holdout draws ([step 06](https://rossyndicate.github.io/regional-clarity-RS-model/regional_clarity/06_test_set_sensitivity.html)). Holdout RMSE across those draws ranged from 1.01 to 1.98 m (mean 1.46 m, SD 0.26 m). The production holdout sits at the 19th percentile of that range, so 1.26 m is on the favorable side of what to expect for a new set of basins. Most of the spread comes from how much of Lake Powell falls in the holdout.
- **Seed-to-seed spread.** For the typical holdout prediction, the per-point standard deviation across the 10 seeds is 0.12 m, and pairwise correlation between seeds is 0.99, so the members agree closely.
- **CV vs. holdout.** Mean out-of-fold CV RMSE across seeds is 1.47 m (range 1.43–1.53 m), within the range of the holdout draws above.

### Superseded matchups

An SDD sample often has more than one image within ±5 days. Training kept only the closest image, so the other images were never used in training (at CV-pool sites they share SDD samples with training matchups). Of 2,811 such matchups, 2,714 pass every applicability check. On those, RMSE is 1.20 m (bias +0.03 m, R² 0.72). At holdout sites only, RMSE is 1.39 m (n = 582, R² 0.61) ([step 07](https://rossyndicate.github.io/regional-clarity-RS-model/regional_clarity/07_regional_application.html)).

## Applicability checks

An estimate should be used only when it passes all of the checks that [step 07](https://rossyndicate.github.io/regional-clarity-RS-model/regional_clarity/07_regional_application.html) applies:

1. The scene passes the same siteSR QA as the training data.
2. The Landsat mission is one represented in training (Landsat 4, 5, 7, 8, or 9).
3. The day of year is between 70 and 320, the central 98% of training matchups.
4. Every model input is available.
5. Every input falls within the training range.
6. The observation is inside the area of applicability (AOA; Meyer & Pebesma 2021). The dissimilarity index (DI) is computed in standardized feature space over the unanimous core, with each feature weighted by its ensemble mean |SHAP|. The threshold is DI ≤ 0.385, the upper whisker of the training pool's cross-fold nearest-neighbor DI.
7. The estimate is no shallower than the training minimum (0.1 m).

Across the region, 88% of QA-passing location-days pass every check. Step 07 writes the AOA threshold, mean training distance, AOA features, and seasonal window to `aoa_parameters.json`.

## Limitations

- **Clear-water compression.** Above about 6 m SDD, water-leaving reflectance is weak relative to the noise floor of atmospheric correction. Like other Landsat clarity models, this model regresses toward the mean in that range, with a bias of about −1.9 m at 6–10 m and about −3.4 m above 10 m. Trends in very clear lakes will be damped.
- **Sparsely sampled, distinctive waterbodies.** Lake Powell is a very large, deep reservoir whose clarity varies widely between years. The deep, forested-catchment lakes of the Kootenai–Pend Oreille–Spokane basin (HUC4 1701), such as Flathead Lake and Lake Pend Oreille, sit in sparse tails of the training distribution. These two groups account for a large share of the clearest observations and of the error. The errors reflect real physical regimes, not bad data.
- **Shallow, turbid water.** Below 2 m SDD, predictions are biased high by about 0.5 m.
- **Point estimates only.** The ensemble spread (about 0.12 m) reflects model variance, not total predictive uncertainty. Typical error is about 1.3 m, and up to about 2 m for unfavorable sets of basins.
- **Static catchment features.** Land cover is fixed at NLCD 2006 for all dates, so land-use change is not represented.
- **Sensor balance.** Landsat 8/9 make up 15% of training matchups. Landsat 4 contributes only 48 matchups.

## References

- Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution*, 12(9), 1620–1633.
- McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425–1432.
- Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. *International Journal of Remote Sensing*, 27(14), 3025–3033.

## Citation

Please cite the model by its Zenodo DOI, and cite the data products it was trained on.

> Steele, B.G. (YYYY). *Intermountain West Regional Clarity Remote Sensing Model* (vX.Y.Z) [Software]. Zenodo. <https://doi.org/10.5281/zenodo.XXXXXXX>

```bibtex
@software{steele_regional_clarity_rs_model,
  author    = {Steele, B.G.},
  title     = {Intermountain West Regional Clarity Remote Sensing Model},
  version   = {vX.Y.Z},
  year      = {YYYY},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/rossyndicate/regional-clarity-RS-model},
  note      = {Production ensemble v3 (seeds 601--610)}
}
```

The DOI above is the Zenodo concept DOI, which always resolves to the latest release. Zenodo also assigns a DOI to each release; cite that one to pin the exact version.

**Data products** (recommended citations from EDI):

- De La Torre, J., B.G. Steele, M.R. Brousil, M.F. Meyer, K. Willi, and M.R. Ross. 2025. AquaMatch Secchi Disk Depth Data from Water Quality Portal: ~1970-2024 ver 1. Environmental Data Initiative. <https://doi.org/10.6073/pasta/542a305e8484ae5abc33881bd7761308>
- Steele, B.G., M.R. Brousil, K.R. Willi, M.F. Meyer, and M.R. Ross. 2026. SiteSR: AquaMatch Landsat Collection 2 surface reflectance and surface temperature datasets for remote-sensing visible Water Quality Portal and National Water Information System sites in the United States and Territories, 1983-2024 ver 1. Environmental Data Initiative. <https://doi.org/10.6073/pasta/f85622d6d32ef7fe6cff8d63c3b947c9>
- Steele, B.G., M.R. Brousil, K.R. Willi, M.F. Meyer, and M.R. Ross. 2026. lakeSR: AquaMatch Landsat Collection 2 surface reflectance and surface temperature datasets for centrally-located points of water bodies greater than 1 hectare in the United States and territories, 1983-2024 ver 1. Environmental Data Initiative. <https://doi.org/10.6073/pasta/941f3cb046f5e7fbe04d5811989ed810>
