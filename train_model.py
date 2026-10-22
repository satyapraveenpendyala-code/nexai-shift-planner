"""
train_model.py - AI4I 2020 failure predictor + cost-based alert threshold (PS 2.1)

Run:   python train_model.py
Needs: pip install pandas numpy scikit-learn xgboost ucimlrepo joblib matplotlib
Data:  downloads AI4I 2020 via ucimlrepo (id=601), or uses ./ai4i2020.csv if present.

Outputs (artifacts/):
  model.joblib     -> failure model + 4 failure-mode models + threshold + feature list
  metrics.json     -> held-out test metrics (report these honestly)
  cost_curve.csv   -> cost vs threshold (validation split)
  cost_curve.png   -> same, as a chart for the README / pitch
  test_split.csv   -> unseen rows the app replays as "shifts"
"""
import json
import re
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
COST_FALSE_ALARM = 1.0   # one unnecessary inspection
COST_MISSED = 20.0       # stated assumption: a missed failure costs 20x a false alarm
MODES = ["twf", "hdf", "pwf", "osf"]  # RNF left out: random by design, not predictable
OUT = Path("artifacts")

RAW = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
FEATURES = ["type_l", "type_m", "type_h", *RAW, "temp_diff", "power_w", "strain"]


def load_data() -> pd.DataFrame:
    csv = Path("ai4i2020.csv")
    if csv.exists():
        df = pd.read_csv(csv)
    else:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=601)
        df = pd.concat([d.data.features, d.data.targets], axis=1)
    # "Air temperature [K]" -> "air_temperature"
    df.columns = [re.sub(r"\s*\[.*?\]", "", c).strip().lower().replace(" ", "_")
                  for c in df.columns]
    return df.drop(columns=["udi", "product_id"], errors="ignore")  # IDs leak, teach nothing


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics features from the dataset's own failure definitions. Reused by the app."""
    X = pd.DataFrame(index=df.index)
    for t in "LMH":
        X[f"type_{t.lower()}"] = (df["type"] == t).astype(int)
    for c in RAW:
        X[c] = df[c].astype(float)
    X["temp_diff"] = X.process_temperature - X.air_temperature            # heat dissipation
    X["power_w"] = X.torque * X.rotational_speed * 2 * np.pi / 60          # power failure
    X["strain"] = X.tool_wear * X.torque                                   # overstrain
    return X[FEATURES]


def make_model(y: pd.Series) -> XGBClassifier:
    pos = y.mean()
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=(1 - pos) / pos,       # handles the ~3.4% failure rate
        eval_metric="aucpr", random_state=SEED, n_jobs=-1,
    )


def cost_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        pred = p >= t
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        rows.append((t, fp, fn, fp * COST_FALSE_ALARM + fn * COST_MISSED))
    return pd.DataFrame(rows, columns=["threshold", "false_alarms", "missed", "cost"])


def main():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    X, y = add_features(df), df["machine_failure"]

    # 60 / 20 / 20 stratified: train / validation (threshold) / test (report only)
    idx_trval, idx_test = train_test_split(df.index, test_size=0.2, stratify=y, random_state=SEED)
    idx_tr, idx_val = train_test_split(idx_trval, test_size=0.25, stratify=y[idx_trval], random_state=SEED)

    # 1) Failure model
    clf = make_model(y[idx_tr]).fit(X.loc[idx_tr], y[idx_tr])

    # 2) Failure-mode models (one per mode; modes can co-occur)
    mode_models = {m: make_model(df.loc[idx_tr, m]).fit(X.loc[idx_tr], df.loc[idx_tr, m]) for m in MODES}

    # 3) Cost-optimal threshold, chosen on VALIDATION only
    curve = cost_curve(y[idx_val].values, clf.predict_proba(X.loc[idx_val])[:, 1])
    best = curve.loc[curve.cost.idxmin()]
    thr = float(best.threshold)

    # 4) Honest evaluation on the untouched TEST split
    p_test = clf.predict_proba(X.loc[idx_test])[:, 1]
    y_test = y[idx_test].values
    pred = (p_test >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    base = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    base.fit(X.loc[idx_tr], y[idx_tr])

    # mode accuracy on true failures that have a known mode
    ft = df.loc[idx_test]
    ft = ft[(ft.machine_failure == 1) & (ft[MODES].sum(axis=1) > 0)]
    mode_p = np.column_stack([mode_models[m].predict_proba(X.loc[ft.index])[:, 1] for m in MODES])
    pred_mode = np.array(MODES)[mode_p.argmax(axis=1)]
    mode_acc = float(np.mean([ft.loc[i, pm] == 1 for i, pm in zip(ft.index, pred_mode)]))

    metrics = {
        "test_rows": int(len(idx_test)), "test_failures": int(y_test.sum()),
        "pr_auc": round(average_precision_score(y_test, p_test), 4),
        "roc_auc": round(roc_auc_score(y_test, p_test), 4),
        "baseline_logreg_pr_auc": round(average_precision_score(y_test, base.predict_proba(X.loc[idx_test])[:, 1]), 4),
        "threshold": thr,
        "cost_ratio_missed_to_false_alarm": COST_MISSED / COST_FALSE_ALARM,
        "precision_at_threshold": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall_at_threshold": round(recall_score(y_test, pred), 4),
        "f1_at_threshold": round(f1_score(y_test, pred), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "test_cost_model": float(fp * COST_FALSE_ALARM + fn * COST_MISSED),
        "test_cost_default_0.5": float(((p_test >= 0.5) & (y_test == 0)).sum() * COST_FALSE_ALARM
                                       + ((p_test < 0.5) & (y_test == 1)).sum() * COST_MISSED),
        "test_cost_no_model": float(y_test.sum() * COST_MISSED),  # never inspect
        "mode_pr_auc": {m: round(average_precision_score(df.loc[idx_test, m],
                        mode_models[m].predict_proba(X.loc[idx_test])[:, 1]), 4) for m in MODES},
        "mode_accuracy_on_true_failures": round(mode_acc, 4),
        "notes": "AI4I 2020 is synthetic. RNF excluded (random failures). Threshold tuned on validation, metrics on test.",
    }

    joblib.dump({"model": clf, "mode_models": mode_models, "features": FEATURES,
                 "threshold": thr, "cost_false_alarm": COST_FALSE_ALARM,
                 "cost_missed": COST_MISSED}, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    curve.to_csv(OUT / "cost_curve.csv", index=False)
    df.loc[idx_test].to_csv(OUT / "test_split.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(curve.threshold, curve.cost, label="total cost")
    plt.axvline(thr, color="red", ls="--", label=f"optimal = {thr:.2f}")
    plt.xlabel("alert threshold"); plt.ylabel(f"cost (false alarm = 1, missed = {COST_MISSED:g})")
    plt.title("Cost vs alert threshold (validation split)"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "cost_curve.png", dpi=120)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
