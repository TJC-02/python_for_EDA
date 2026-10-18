# Python for exploratory data analysis

A CSV analysis workflow tested with Python 3.12. It generates data quality
reports, descriptive statistics, distributions, correlations, and optional
outcome counts without modifying the input or imputing missing values.

## Setup

In the prepared cloud environment:

```bash
source /workspace/eda-environment/activate.sh
cd /workspace/python_for_EDA
```

For a new local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the main tools; `requirements-lock.txt` pins the full
tested environment, including Jupyter dependencies. To reproduce that environment,
install with `python -m pip install -r requirements-lock.txt` instead.

## Run

```bash
python eda.py /path/to/manufacturing_dataset.csv --target 'Pass/Fail' --output outputs/manufacturing
python -m unittest -v
```

Omit `--target` for other datasets. Input datasets belong in ignored `data/`
or an external location. Reports under `outputs/` are ignored by Git.
Use a fresh output directory for each dataset so older optional plots cannot
be mistaken for current results.

Reports include `overview.json`, `column_profile.csv`, and
`descriptive_statistics.csv`. Plots are generated when applicable: up to 20
columns with missing values, distributions and pairwise Pearson correlations
for the first 12 varying numeric predictors, and target counts. Change the
numeric feature limit with `--max-features`. The outcome is excluded from
predictor plots; target charts are omitted when there are over 30 unique values.

Constant and entirely missing columns are reported separately. Infinite values
are counted in the overview and treated as missing for numeric plots.
Correlations use available pairs and do not establish causation. Numbered
manufacturing sensor columns have no inferred physical interpretation.
The workflow performs no imputation, row removal, or predictive modeling.
