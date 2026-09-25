from pathlib import Path
import argparse, joblib

p = argparse.ArgumentParser()
p.add_argument("features", nargs=4, type=float, help="Four Iris features")
args = p.parse_args()
model = joblib.load(Path(__file__).parent / "artifacts" / "model.joblib")
print(int(model.predict([args.features])[0]))
