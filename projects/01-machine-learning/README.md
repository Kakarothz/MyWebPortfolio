# 01 — Machine Learning Pipeline

**Status: Implemented**

Reproducible classical ML demo using scikit-learn's built-in Iris dataset. The pipeline performs a stratified train/test split, standardization, logistic-regression training, evaluation and model serialization.

## Run
```bash
pip install -r requirements.txt
python train.py
python predict.py 5.1 3.5 1.4 0.2
```

Training creates `artifacts/model.joblib` and `artifacts/metrics.json`. Generated artifacts are ignored by Git. No accuracy value is claimed in this README because results should come from an actual run in the target environment.
