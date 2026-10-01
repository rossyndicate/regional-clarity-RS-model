# Regional Remote Sensing Clarity Model

<!-- Zenodo DOI badge: after the first GitHub release is archived, replace XXXXXXX with the
     record number of the *concept* DOI (it resolves to the latest release) and uncomment.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

Primary repository contact: B Steele

This repository builds a regional model of Secchi disc depth (SDD), a standard measure of water clarity, for lakes and reservoirs in Colorado, Idaho, Montana, New Mexico, Utah, and Wyoming. The model is trained on in situ SDD matched to Landsat Collection 2 surface reflectance. Both come from the AquaMatch data products published on the Environmental Data Initiative (EDI) repository (see [Data sources](#data-sources)).

This repository is covered by the MIT use license. We request that all downstream uses of this work be available to the public when possible.

**Rendered workflow:** <https://rossyndicate.github.io/regional-clarity-RS-model/>. Every step below is also a page on that site, with its code and outputs.

## How to cite

If you use this model or code, please cite this repository by its Zenodo DOI, and cite the data products listed under [Data sources](#data-sources).

> Steele, B.G. (YYYY). *Intermountain West Regional Clarity Remote Sensing Model* (vX.Y.Z) [Software]. Zenodo. <https://doi.org/10.5281/zenodo.XXXXXXX>

Zenodo assigns a DOI to each GitHub release, plus a concept DOI that always resolves to the latest release. The citation above uses the concept DOI. To cite the exact release you used, take that release's version DOI from Zenodo instead.

## Data sources

All in situ and remote sensing inputs come from three AquaMatch data packages on EDI. [Step 00](regional_clarity/00_download_AquaMatch_data.Rmd) downloads them with the [`AquaMatchr`](https://github.com/AquaSat/AquaMatchr) R package (Brousil & Ross 2026), which requires an EDI account and API key (see [Secrets/credentials](#secretscredentials)). The production model was built from revision 1 of each package.

| EDI package | Product | Used for |
|---|---|---|
| [`edi.1856.1`](https://doi.org/10.6073/pasta/542a305e8484ae5abc33881bd7761308) | AquaMatch Secchi disc depth | Target variable (in situ SDD) |
| [`edi.2254.1`](https://doi.org/10.6073/pasta/f85622d6d32ef7fe6cff8d63c3b947c9) | siteSR | Landsat surface reflectance and temperature at sampling sites, scene-level metadata, and the site list |
| [`edi.2114.1`](https://doi.org/10.6073/pasta/941f3cb046f5e7fbe04d5811989ed810) | lakeSR | Cross-sensor handoff coefficients used to harmonize all Landsat missions to Landsat 7 |

Recommended citations, as provided by EDI:

- De La Torre, J., B.G. Steele, M.R. Brousil, M.F. Meyer, K. Willi, and M.R. Ross. 2025. AquaMatch Secchi Disk Depth Data from Water Quality Portal: ~1970-2024 ver 1. Environmental Data Initiative. <https://doi.org/10.6073/pasta/542a305e8484ae5abc33881bd7761308>
- Steele, B.G., M.R. Brousil, K.R. Willi, M.F. Meyer, and M.R. Ross. 2026. SiteSR: AquaMatch Landsat Collection 2 surface reflectance and surface temperature datasets for remote-sensing visible Water Quality Portal and National Water Information System sites in the United States and Territories, 1983-2024 ver 1. Environmental Data Initiative. <https://doi.org/10.6073/pasta/f85622d6d32ef7fe6cff8d63c3b947c9>
- Steele, B.G., M.R. Brousil, K.R. Willi, M.F. Meyer, and M.R. Ross. 2026. lakeSR: AquaMatch Landsat Collection 2 surface reflectance and surface temperature datasets for centrally-located points of water bodies greater than 1 hectare in the United States and territories, 1983-2024 ver 1. Environmental Data Initiative. <https://doi.org/10.6073/pasta/941f3cb046f5e7fbe04d5811989ed810>

Access software:

- Brousil, M., and M. Ross. 2026. AquaMatchr: Download, Join, and Analyze AquaMatch Data Products. R package version 0.0.0.9000. <https://github.com/AquaSat/AquaMatchr>

## Model card

The model card for the production ensemble is [`regional_clarity/xg_models/v3_production/README.md`](regional_clarity/xg_models/v3_production/README.md). It is formatted for the Hugging Face Hub (YAML metadata, including evaluation results), so that folder can be pushed to the Hub as is. The Quarto site shows the same file as its [Model card](https://rossyndicate.github.io/regional-clarity-RS-model/model_card.html) page (`model_card.qmd`), so edit the card in the model folder only.

## Workflow

Data preparation and application run in R (`.Rmd`). Modeling runs in Python notebooks (`.ipynb`) that share the modules in `regional_clarity/python/`. Arrows are labeled with the main file each step hands to the next. Data files live under `regional_clarity/aquamatch_files/` and models under `regional_clarity/xg_models/`; neither is tracked in git.

<!-- keep in sync with pipeline_diagram.mmd, which the Quarto site renders -->
```mermaid
flowchart TD
  AM[("AquaMatch<br/>SDD + siteSR")] --> S00

  subgraph PREP ["Data preparation (R)"]
    S00["00 · Download AquaMatch data"]
    S01["01 · Filter AquaMatch data"]
    S02["02a · Make matchups"]
    SC["02b · Site characteristics"]
    WX["02c · Weather summaries"]
  end

  subgraph MODEL ["Modeling (Python)"]
    S03["03 · Split data"]
    S04["04 · Make models"]
    S05["05 · Evaluate ensemble"]
    S06["06 · Test-set sensitivity"]
  end

  subgraph APPLY ["Application (R)"]
    S07["07 · Regional application"]
  end

  S00 -->|"sdd_cache.parquet<br/>siteSR_DSWE1_LS7corr.parquet"| S01
  S01 -->|"sdd_quality_filter.parquet<br/>siteSR_DSWE1_regional_band_filtered.parquet"| S02
  S02 -->|"filtered_regional_sdd.parquet"| S03
  S02 --> SC
  S02 --> WX
  SC -->|"site_characteristics.feather"| S03
  WX -->|"weather_summaries.feather"| S03
  S03 -->|"base_with_splits.parquet"| S04
  S04 -->|"v3_production/seed601–610/"| S05
  S04 -->|"per-seed features + parameters"| S06
  S05 -->|"ensemble_mean_abs_shap.csv"| S07
  S04 -->|"40 fold models"| S07

  classDef src fill:#eeeeee,stroke:#555555,color:#111111
  classDef rstep fill:#ffffff,stroke:#1f5fa8,stroke-width:2px,color:#111111
  classDef pystep fill:#ffffff,stroke:#a85a00,stroke-width:2px,color:#111111
  class AM src
  class S00,S01,S02,SC,WX,S07 rstep
  class S03,S04,S05,S06 pystep
  style PREP fill:#e3eefb,stroke:#1f5fa8,color:#0b2e55
  style MODEL fill:#fdf0e0,stroke:#a85a00,color:#4a2800
  style APPLY fill:#e3eefb,stroke:#1f5fa8,color:#0b2e55
```

| Step | File | What it does |
|---|---|---|
| 00 | [`00_download_AquaMatch_data.Rmd`](regional_clarity/00_download_AquaMatch_data.Rmd) | Download AquaMatch SDD and siteSR; apply Landsat 7-referenced handoffs. Run by hand only when AquaMatch changes. |
| 01 | [`01_filter_AquaMatch_Data.Rmd`](regional_clarity/01_filter_AquaMatch_Data.Rmd) | Scope to the 6-state HUC4 region; SDD QC and RANSAC; siteSR scene QA and band RANSAC. |
| 02a | [`02a_make_matches.Rmd`](regional_clarity/02a_make_matches.Rmd) | 5-day matchups, rain-event filter, closest image per sample. |
| 02b | [`02b_pull_site_characteristics.Rmd`](regional_clarity/02b_pull_site_characteristics.Rmd) | Elevation and LakeCat catchment features per site. Cached; re-run only when the site set changes. |
| 02c | [`02c_pull_weather_summaries.Rmd`](regional_clarity/02c_pull_weather_summaries.Rmd) | gridMET antecedent-weather features (1/3/7/30-day windows). Cached; re-run only when the site set or date range changes. |
| 03 | [`03_split_data.ipynb`](regional_clarity/03_split_data.ipynb) | Join features; fixed HUC8 holdout and CV folds. |
| 04 | [`04_make_models.ipynb`](regional_clarity/04_make_models.ipynb) | Per-seed feature selection, tuning, and 4-fold XGBoost (10 seeds × 4 folds). |
| 05 | [`05_evaluate_ensemble.ipynb`](regional_clarity/05_evaluate_ensemble.ipynb) | Ensemble holdout performance and SHAP. |
| 06 | [`06_test_set_sensitivity.ipynb`](regional_clarity/06_test_set_sensitivity.ipynb) | Holdout RMSE across 16 holdout draws. |
| 07 | [`07_regional_application.Rmd`](regional_clarity/07_regional_application.Rmd) | Apply the ensemble at in situ SDD locations, with training-domain and AOA checks. |

Manuscript tables and figures are built by [`manuscript_tables_figures.Rmd`](regional_clarity/asv2_manuscript/manuscript_tables_figures.Rmd), kept for reproducibility and not part of the rendered site.

Each file opens with a header listing its purpose, inputs, outputs, the working directory to run it from, and links to the previous and next steps.

## Python environment

The modeling notebooks (03–06) and `regional_clarity/python/` run in a virtual environment built with Python 3.11. The production run used 3.11.16. From the repository root:

```
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m ipykernel install --user --name regional_clarity --display-name "regional_clarity" \
  --env KMP_DUPLICATE_LIB_OK TRUE --env OMP_NUM_THREADS 4 --env MKL_NUM_THREADS 4 --env VECLIB_MAXIMUM_THREADS 4
```

- `requirements-lock.txt` pins every package, including transitive dependencies, to the versions used for the production results. `requirements.txt` lists only the direct dependencies, with the same pins, for reading or for a looser install.
- The notebooks' kernel is named `regional_clarity`, so the last command makes them open in this environment in Jupyter, VS Code, and Positron.
- The `--env` flags are required on macOS. Without them, XGBoost and LightGBM load competing OpenMP runtimes and oversubscribe the CPU, and backward elimination in 04 can stall for hours. Registering the kernel this way sets them for every notebook run. If you run the modules outside Jupyter, export the same four variables first.
- Run notebooks from `regional_clarity/`, which Jupyter does by default. To execute one headlessly: `cd regional_clarity && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 06_test_set_sensitivity.ipynb`.

## Building the site

The site is a [Quarto](https://quarto.org) website (`_quarto.yml`, `index.qmd`). It builds into `_site/` (not tracked) and is published to the `gh-pages` branch, which GitHub Pages serves. Quarto ships with RStudio and Positron. From the repository root:

```
quarto render                                          # whole site, into _site/
quarto render regional_clarity/02a_make_matches.Rmd     # one page
quarto preview                                         # local preview with live reload
quarto publish gh-pages                                # render and push to the gh-pages branch
```

- **Notebooks are never executed by Quarto.** Their pages use the outputs saved in the `.ipynb`, so run a notebook and save it before rendering.
- **Rmds are executed, then frozen.** Results are stored in `_freeze/` (tracked in git) and reused until that Rmd's source changes. Commit `_freeze/` along with the Rmd so the site can be rebuilt without the data. The slow steps inside each Rmd are cached to disk, so a re-render mostly reloads those caches.
- The two `pull_*` Rmds are shown as code only (`execute: eval: false` in their front matter) so a site build never re-runs the web fetches.
- Keep the diagram above in sync with `pipeline_diagram.mmd`.

## Secrets/credentials

API keys and other credentials are stored locally in a `.Renviron` file (untracked by git). This workflow requires an API key for accessing the
EDI data repository. Copy `.Renviron.example` to `.Renviron` and fill in real values. R loads `.Renviron` automatically at session start; access 
values in code with `Sys.getenv("KEY_NAME")`.
