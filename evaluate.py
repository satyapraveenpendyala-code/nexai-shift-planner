import joblib, pandas as pd, numpy as np
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from train_model import add_features
b = joblib.load("artifacts/model.joblib"); t = pd.read_csv("artifacts/test_split.csv")
X, y = add_features(t), t.machine_failure.values
p = b["model"].predict_proba(X)[:, 1]
for thr in [b["threshold"], 0.5]:
    pr = (p >= thr).astype(int)
    print(f"\n=== threshold {thr} ===  accuracy {accuracy_score(y,pr):.3f}  "
          f"balanced {balanced_accuracy_score(y,pr):.3f}  MCC {matthews_corrcoef(y,pr):.3f}")
    print(classification_report(y, pr, target_names=["no failure", "failure"], digits=3))