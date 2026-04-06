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
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you need local development tools:

```bash
pip install pytest jupyter
```

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

