# Code for TUNE Benchmark Experiments

This directory contains Python scripts for running the main machine-learning experiments described in the TUNE benchmark study.  
All scripts assume that you run them from the **repository root** and that input CSV files are located in the `data/` directory (see `data/README.md` for details).

Before running anything, create an environment and install dependencies from the root of the repo:

```bash
python -m venv .venv
source .venv/bin/activate        # on macOS / Linux
# .venv\Scripts\activate         # on Windows

pip install -r requirements.txt


All scripts share the same basic conventions:

Input data is passed via --csv (a path to a CSV file).

Columns for question id, version and gold label default to: QuestionNum, SurveyVersion, ActualAnswer.

Models are specified via --models (default: lr,knn,rf,xgb).

Outputs (metrics, predictions, ablation tables) are written under the directory provided by --outdir (default: outputs).


Common Arguments

Most scripts accept the following core arguments:

--csv (str, required): Path to the input CSV.

--sep (str, optional): CSV delimiter; if omitted, pandas will try to auto-detect.

--qid (str, default: QuestionNum): Column name for the question ID.

--version (str, default: SurveyVersion): Column name for the survey version.

--gold (str, default: ActualAnswer): Column name for the ground-truth label.

--models (str, default: lr,knn,rf,xgb): Comma-separated list of models to run.

--outdir (str, default: outputs): Output directory for all generated CSV files.

--random_state (int, default: 42): Random seed for reproducibility.

Model codes:

lr – Logistic Regression

knn – k-Nearest Neighbors

rf – Random Forest

xgb – XGBoost 

1. s1_experiment.py — Scenario 1: Per-User ML (Human Features Only)
Per-user ML with leakage-safe engineered behavioral features (clicks, times, confidence, explanations, user metadata).
No TUS features are used. Splitting is leave-one-version-out (LOVO) by SurveyVersion.

Key arguments

--user (str, required): User identifier column (e.g., ByWho).

--ablation (str, default: none): One of

none – use all feature groups

drop_group – drop each feature group once

only_group – use only one group at a time

Example:
python code/s1_experiment.py \
  --csv data/Feature_Engineered.csv \
  --user ByWho \
  --models lr,knn,rf,xgb \
  --ablation none \
  --outdir results/s3_peruser/

2. s2_experiment.py — Scenario 2: Per-User ML with Human + TUS Features
Per-user ML combining human behavioral features with TUS scores (e.g., Santos, Starnie, D3L).
Supports several ablation modes over both human groups and TUS features.

Additional arguments

--user (str, required): User identifier column (e.g., ByWho).

--tus_features (str, default: Santos,Starnie,D3L): Comma-separated list of TUS columns.

--ablation (str, default: none): One of

none

drop_group – drop each feature group (including TUS) once

only_group – only one feature group at a time

lofo_tus – leave-one-TUS-feature-out (keep other groups)

only_tus – TUS features only (and each TUS alone)

Example:
python code/s2_experiment.py \
  --csv data/Feature_Engineered.csv \
  --user ByWho \
  --tus_features Santos,Starnie,D3L \
  --models lr,knn,rf,xgb \
  --ablation none \
  --outdir results/s4_peruser_tus/

3. s3_experiment.py — Scenario 3: Crowd Aggregation (No TUS)

Crowd-level models that aggregate individual votes and behavioral signals into per-question predictions, without TUS features.
Includes per-feature and group ablations.

Additional arguments

--vote_col (str, default: SurveyAnswer): Column with individual yes/no decisions.

--conf_col (str, default: ConfidenceLevel): Confidence column.

--time_col (str, default: DecisionTime): Decision time column.

--clicks_col (str, default: ClickCount): Click count column.

--user (str, default: ByWho): User ID; used for observational Top-10 user metrics.

--ablation (str, default: none): One of

none

lofo

singletons

groups

both

all

Example:
python code/s3_experiment.py \
  --csv data/Compiled_Version.csv \
  --vote_col SurveyAnswer \
  --conf_col ConfidenceLevel \
  --time_col DecisionTime \
  --clicks_col ClickCount \
  --user ByWho \
  --models lr,knn,rf,xgb \
  --ablation none \
  --outdir results/s5_crowd_no_tus/

4. s4_experiment.py — Scenario 4: Crowd Aggregation + TUS Features
Crowd-aggregation models that combine:

Aggregated human signals (votes, confidence, time, clicks), and

TUS scores at the question level (e.g., Santos, Starnie, D3L),

with per-version metrics, best-model summaries.

Example:
python code/s4_experiment.py \
  --csv data/Compiled_Version_TUS.csv \
  --vote_col SurveyAnswer \
  --conf_col ConfidenceLevel \
  --time_col DecisionTime \
  --clicks_col ClickCount \
  --user ByWho \
  --tus_features Santos,Starnie,D3L \
  --models lr,knn,rf,xgb \
  --ablation none \
  --outdir results/s4_crowd_tus/

5. TUS_experiment.py — Baseline ML on TUS 

Example:
python code/TUS_experiment.py \
  --csv data/Compiled_Version_TUS.csv \
  --num_features Santos,Starnie,D3L \
  --cat_features "" \
  --models lr,knn,rf,xgb \
  --ablation none \
  --outdir results/tus_baseline/




