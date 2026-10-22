"""
app.py - Maintenance Shift Planner (NexAI Hack 2026, PS 2.1)
Run: streamlit run app.py   (needs artifacts/ from train_model.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

from train_model import FEATURES, MODES, add_features

ART = Path("artifacts")
MODE_NAME = {"twf": "Tool Wear", "hdf": "Heat Dissipation", "pwf": "Power", "osf": "Overstrain"}
SKILL = {"twf": "Mechanical", "osf": "Mechanical", "hdf": "Electrical", "pwf": "Electrical"}
ROSTER = ["Electrical", "Mechanical", "General", "Electrical", "Mechanical"]  # filled in this order
ACTION = {
    "twf": "Replace tool insert; check wear log.",
    "osf": "Reduce load; inspect tool and torque limiter.",
    "hdf": "Check cooling and air flow; clean heat exchanger.",
    "pwf": "Check drive power and electrical supply limits.",
}
NICE = {"type_l": "Type L", "type_m": "Type M", "type_h": "Type H", "air_temperature": "Air temp (K)",
        "process_temperature": "Process temp (K)", "rotational_speed": "Speed (rpm)", "torque": "Torque (Nm)",
        "tool_wear": "Tool wear (min)", "temp_diff": "Temp difference (K)", "power_w": "Power (W)",
        "strain": "Strain (min·Nm)"}

st.set_page_config(page_title="Maintenance Shift Planner", layout="wide")


@st.cache_resource
def load():
    b = joblib.load(ART / "model.joblib")
    test = pd.read_csv(ART / "test_split.csv").reset_index(drop=True)
    test.insert(0, "machine_id", [f"M-{i:04d}" for i in range(len(test))])
    curve = pd.read_csv(ART / "cost_curve.csv")
    metrics = json.loads((ART / "metrics.json").read_text())
    return b, test, curve, metrics


def predict(b, df):
    X = add_features(df)
    p = b["model"].predict_proba(X)[:, 1]
    mp = np.column_stack([b["mode_models"][m].predict_proba(X)[:, 1] for m in MODES])
    # TreeSHAP contributions computed natively by XGBoost (last column = bias)
    contrib = b["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    return X, p, np.array(MODES)[mp.argmax(1)], contrib


def best_threshold(curve, ratio):
    c = curve.assign(cost=curve.false_alarms + ratio * curve.missed)
    return float(c.loc[c.cost.idxmin(), "threshold"]), c


def dispatch(ranked, n_tech):
    """Greedy, skill-aware: highest expected loss first; matching specialist, else generalist."""
    free = [f"Tech {i+1} ({ROSTER[i]})" for i in range(n_tech)]
    out = []
    for _, r in ranked.iterrows():
        need = SKILL[r["mode"]]
        pick = next((t for t in free if need in t), None) or next((t for t in free if "General" in t), None)
        if pick:
            free.remove(pick)
        out.append(pick or "Deferred: no matching technician free")
    return out


b, test, curve, metrics = load()

# ---------------- sidebar ----------------
st.sidebar.header("Shift settings")
n_tech = st.sidebar.slider("Available technicians", 1, 5, 3)
shift_size = st.sidebar.slider("Machine readings per shift", 20, 200, 100, step=10)
n_shifts = len(test) // shift_size
shift_no = st.sidebar.number_input(f"Replay shift (1–{n_shifts})", 1, n_shifts, 1)
ratio = st.sidebar.slider("Cost of missed failure ÷ cost of false alarm", 5, 50, int(b["cost_missed"]))
with st.sidebar.expander("Failure-mode cost weights (stated assumptions)"):
    penalty = {m: st.number_input(MODE_NAME[m], 1, 50, d, key=m)
               for m, d in {"pwf": 25, "osf": 20, "hdf": 12, "twf": 10}.items()}
st.sidebar.caption("AI4I 2020 is synthetic data. Replay mode: readings come from the held-out test split.")

thr, c_curve = best_threshold(curve, ratio)

st.title("Maintenance Shift Planner")
st.caption("XGBoost predicts failure risk → cost model sets the alert threshold → "
           "limited technicians are assigned by expected loss and skill.")

tab_plan, tab_model, tab_one = st.tabs(["Shift plan", "Model & cost curve", "Check one machine"])

# ---------------- shift plan ----------------
with tab_plan:
    shift = test.iloc[(shift_no - 1) * shift_size: shift_no * shift_size].reset_index(drop=True)
    X, p, mode, contrib = predict(b, shift)
    shift = shift.assign(p=p, mode=mode)
    shift["expected_loss"] = shift.p * shift["mode"].map(penalty)
    alerts = shift[shift.p >= thr].sort_values("expected_loss", ascending=False).copy()
    alerts["assigned"] = dispatch(alerts, n_tech)
    served = alerts[~alerts.assigned.str.startswith("Deferred")]

    k = st.columns(5)
    k[0].metric("Readings in shift", len(shift))
    k[1].metric("Alerts (p ≥ threshold)", len(alerts))
    k[2].metric("Assigned", len(served))
    k[3].metric("Deferred", len(alerts) - len(served))
    cover = served.expected_loss.sum() / alerts.expected_loss.sum() if len(alerts) else 1.0
    k[4].metric("Expected loss covered", f"{cover:.0%}")
    st.caption(f"Alert threshold **{thr:.2f}**, chosen on the validation split for a {ratio}:1 cost ratio. "
               f"Ground truth in this replay: {int(shift.machine_failure.sum())} actual failures, "
               f"{int(served.machine_failure.sum())} of them assigned a technician.")

    if alerts.empty:
        st.success("No machine crosses the alert threshold this shift.")
    else:
        view = alerts.assign(
            **{"P(failure)": alerts.p.round(3), "Predicted mode": alerts["mode"].map(MODE_NAME),
               "Expected loss": alerts.expected_loss.round(2), "Technician": alerts.assigned,
               "Why": [f"p={r.p:.2f} × {MODE_NAME[r['mode']]} weight {penalty[r['mode']]}" for _, r in alerts.iterrows()]})
        st.dataframe(view[["machine_id", "P(failure)", "Predicted mode", "Expected loss", "Technician", "Why"]],
                     hide_index=True, width="stretch")

        st.subheader("Work orders")
        for n, (i, r) in enumerate(served.iterrows(), 1):
            top = np.argsort(-contrib[i])[:3]
            causes = ", ".join(f"{NICE[FEATURES[j]]} = {X.iloc[i, j]:.1f}" for j in top)
            with st.expander(f"WO-{shift_no:02d}{n:02d} · {r.machine_id} · {r.assigned} · "
                             f"{MODE_NAME[r['mode']]} · p={r.p:.2f}"):
                st.markdown(f"**Root cause (SHAP, top 3):** {causes}  \n**Suggested action:** {ACTION[r['mode']]}")
                st.bar_chart(pd.Series(contrib[i][top], index=[NICE[FEATURES[j]] for j in top]),
                             horizontal=True)

# ---------------- model & cost ----------------
with tab_model:
    m = st.columns(4)
    m[0].metric("PR-AUC (test)", metrics["pr_auc"], f"baseline {metrics['baseline_logreg_pr_auc']}")
    m[1].metric("ROC-AUC (test)", metrics["roc_auc"])
    m[2].metric("Recall @ trained threshold", f"{metrics['recall_at_threshold']:.0%}")
    m[3].metric("Precision @ trained threshold", f"{metrics['precision_at_threshold']:.0%}")
    st.caption(f"Test split: {metrics['test_rows']} rows, {metrics['test_failures']} failures. "
               f"Test cost (20:1): model {metrics['test_cost_model']:.0f} vs 0.5 cutoff "
               f"{metrics['test_cost_default_0.5']:.0f} vs no model {metrics['test_cost_no_model']:.0f}.")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(c_curve.threshold, c_curve.cost)
    ax.axvline(thr, color="red", ls="--", label=f"optimal = {thr:.2f}")
    ax.set_xlabel("alert threshold"); ax.set_ylabel(f"cost (missed = {ratio}, false alarm = 1)")
    ax.set_title("Cost vs threshold (validation split)"); ax.legend(); ax.grid(alpha=.3)
    st.pyplot(fig)
    st.caption("Limitations: synthetic data; tool-wear (TWF) and random (RNF) failures are not predictable; "
               "cost ratio and mode weights are assumptions you can change in the sidebar.")

# ---------------- single prediction ----------------
with tab_one:
    c = st.columns(3)
    row = pd.DataFrame([{
        "type": c[0].selectbox("Machine type", ["L", "M", "H"]),
        "air_temperature": c[1].number_input("Air temp (K)", 290.0, 310.0, 300.0),
        "process_temperature": c[2].number_input("Process temp (K)", 300.0, 320.0, 310.0),
        "rotational_speed": c[0].number_input("Speed (rpm)", 1100, 2900, 1500),
        "torque": c[1].number_input("Torque (Nm)", 3.0, 80.0, 40.0),
        "tool_wear": c[2].number_input("Tool wear (min)", 0, 260, 100),
    }])
    _, p1, m1, _ = predict(b, row)
    st.metric("Failure probability", f"{p1[0]:.1%}")
    st.write(f"Predicted mode if it fails: **{MODE_NAME[m1[0]]}** · "
             f"{'⚠️ Above alert threshold' if p1[0] >= thr else '✅ Below alert threshold'} ({thr:.2f})")
