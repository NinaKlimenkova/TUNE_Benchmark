## Data Directory

This folder contains all datasets used in the TUNE Benchmark, including raw survey exports, intermediate compiled tables, engineered features, and merged data with TUS method outputs.
These files support all ML scenarios described in the benchmark.

# 1. Compiled_Version.xlsx

This is the primary compiled dataset used for Scenarios 3 and 5 (crowd-based models without TUS features).
It merges per-user responses with table-pair metadata and core behavioral signals.

Typical Columns Include:

QuestionNum – ID of the table pair

SurveyVersion – Survey version (1–4)

ByWho – Anonymous respondent ID

SurveyAnswer – Human unionability judgment (0/1)

ActualAnswer – Ground-truth label

ConfidenceLevel – Self-reported confidence (0–100)

DecisionTime – Time (ms/sec) spent before submitting

ClickCount – Number of UI interactions

Explanation – Free-text justification

# 2. Compiled_Version_TUS.xlsx

This version extends Compiled_Version.xlsx by adding outputs from TUS methods:

SANTOS scores

Starmie scores

D3L scores

# 3. Feature_Engineered.csv

A leakage-safe, feature-engineered file derived from Compiled_Version_TUS.xlsx.
It includes:

Normalized and aggregated behavioral features

Text-derived features from explanations (if used)

Bucketed or transformed time/click signals

# 4. Qualtrics export.csv

The raw survey export from Qualtrics before cleaning or merging.
This file includes:

Every response as recorded by Qualtrics

Metadata (timestamps, user agent, etc.)

Free-text explanations

Version assignment for each respondent

Column naming conventions defined by Qualtrics

Used For:

Audit and reproducibility

Regenerating compiled files

Verifying behavioral signal extraction

Research on survey behavior patterns

# 5. RS_Compiled Version.xlsx

A cleaned and simplified version of the compiled dataset.
Maintains the same row-level granularity but may include:

Additional quality-control annotations

Intermediate processing steps

Alternative feature derivations

Version-control of earlier stages
