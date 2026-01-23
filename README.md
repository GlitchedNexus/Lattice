# Lattice

A small, in-progress machine learning library written in Python. The goal is to
keep implementations readable and easy to extend as new models are added over
time.

## What’s inside

Implemented models and utilities:

- Linear regression: `LeastSquares`, `LeastSquaresBias`
- Decision trees: `DecisionTree`, `DecisionStumpErrorRate`, `DecisionStumpInfoGain`
- Randomized trees/forests: `RandomTree`, `RandomForest`
- Clustering: `KMeans`, `KMedians`
- Naive Bayes (binary features): `NaiveBayes`, `NaiveBayesLaplace` (in progress)
- Plotting/helpers: distance utilities, dataset loading, plotting helpers

## Install (local dev)

This project targets Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Basic examples:

```python
import numpy as np
from lattice.kmeans import KMeans

X = np.random.randn(100, 2)
model = KMeans(X, k=3, plot=False, log=False)
labels = model.get_assignments(X)
```

```python
import numpy as np
from lattice.random_forest import RandomForest

X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

forest = RandomForest(num_trees=10, max_depth=3)
forest.fit(X, y)
preds = forest.predict(X)
```

## Project layout

- `src/lattice/`: core implementations
- `tests/`: unit tests
- `main.py`: placeholder entry point

## Roadmap

- Add additional models and training utilities over time
- Improve API consistency and documentation
- Expand tests and examples

## Development

Run tests:

```bash
pytest
```
