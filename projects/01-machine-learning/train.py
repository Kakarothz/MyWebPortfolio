"""Reproducible classical ML portfolio demo using scikit-learn's built-in Iris dataset."""
from pathlib import Path
import json, joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

OUT = Path(__file__).parent / "artifacts"
OUT.mkdir(exist_ok=True)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "classification_report": classification_report(y_test, pred, output_dict=True),
}
joblib.dump(model, OUT / "model.joblib")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({"accuracy": metrics["accuracy"]}, indent=2))
