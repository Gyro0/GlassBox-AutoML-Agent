# GlassBox AutoML

GlassBox AutoML is a transparent machine learning library built from scratch with NumPy. The aim is to provide an end-to-end AutoML pipeline that remains readable, explainable, and easy to debug.

## Project Goal

The project covers the full machine learning workflow inside the `glassbox/` package:

- exploratory data analysis
- preprocessing
- models
- evaluation
- hyperparameter optimization
- agent-level AutoFit integration

Core library modules are built from scratch with NumPy only. No Scikit-Learn code belongs inside `glassbox/`.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If you need local development tools:

```bash
python3 -m pip install pytest jupyter
```

## Quick Start

Run the full AutoFit pipeline on the included sample CSV:

```python
from glassbox.agent import auto_fit

report = auto_fit(
    "data/sample.csv",
    target_column="purchased",
    task="auto",
    search="random",
    time_budget=20,
)

print(report["best_model"])
print(report["cv_score"])
print(report["eda_summary"]["overview"])
```

The returned report is JSON-safe and includes:

- EDA overview, numerical profile table, correlations, and outlier rows
- selected task type
- candidate model leaderboard
- best model, best parameters, and cross-validation score
- feature importances or coefficient-style importances when available

## Manual Workflow Example

```python
import numpy as np

from glassbox.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from glassbox.models import RandomForestClassifier
from glassbox.evaluation.classification import classification_report

X_num = np.array([
    [22.0, 32000.0],
    [24.0, np.nan],
    [42.0, 76000.0],
])
X_cat = np.array([["basic"], ["basic"], ["plus"]], dtype=object)
y = np.array([0, 0, 1])

X_num = SimpleImputer(strategy="mean").fit_transform(X_num)
X_num = StandardScaler().fit_transform(X_num)
X_cat = OneHotEncoder().fit_transform(X_cat)
X = np.hstack([X_num, X_cat])

model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
model.fit(X, y)
predictions = model.predict(X)

print(classification_report(y, predictions))
```

## Model Zoo

Classification:

- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GaussianNaiveBayes`
- `KNearestNeighbors(task="classification")`

Regression:

- `LinearRegression`
- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `KNearestNeighbors(task="regression")`

## Demo And Benchmarks

Launch the notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

Run the Scikit-Learn comparison benchmark:

```bash
python3 benchmarks/sklearn_comparison.py
```

Scikit-Learn is used only in the benchmark script. The `glassbox/` package itself remains NumPy-only.

## Repository Structure

```text
GlassBox-AutoML-Agent/
|-- .github/
|   `-- pull_request_template.md
|-- glassbox/
|   |-- agent/
|   |-- eda/
|   |-- evaluation/
|   |-- models/
|   |-- optimization/
|   |-- preprocessing/
|   `-- utils/
|-- tests/
|-- notebooks/
|-- benchmarks/
|-- data/
|-- README.md
|-- pyproject.toml
`-- requirements.txt
```

## Testing

Run the test suite from the repository root:

```bash
python3 -m pytest -q
```
