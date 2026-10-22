"""
app.py - Maintenance Shift Planner (NexAI Hack 2026, PS 2.1)
Flow: choose profile (Manager / Technician) -> log in or sign up (SQLite, hashed passwords) -> dashboard.
Run: streamlit run app.py   (needs artifacts/ from train_model.py)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

import db
import rag
from train_model import FEATURES, MODES, add_features

ART = Path("artifacts")
MODE_NAME = {"twf": "Tool wear", "hdf": "Heat dissipation", "pwf": "Power", "osf": "Overstrain"}
SKILL = {"twf": "Mechanical", "osf": "Mechanical", "hdf": "Electrical", "pwf": "Electrical"}
ACTION = {"twf": "Replace tool insert; check wear log.",
          "osf": "Reduce load; inspect tool and torque limiter.",
          "hdf": "Check cooling and air flow; clean heat exchanger.",
          "pwf": "Check drive power and electrical supply limits."}
NICE = {"type_l": "Type L", "type_m": "Type M", "type_h": "Type H", "air_temperature": "Air temp (K)",
        "process_temperature": "Process temp (K)", "rotational_speed": "Speed (rpm)", "torque": "Torque (Nm)",
        "tool_wear": "Tool wear (min)", "temp_diff": "Temp difference (K)", "power_w": "Power (W)",
        "strain": "Strain (min·Nm)"}

st.set_page_config(page_title="Maintenance Shift Planner", layout="centered")
ss = st.session_state
ss.setdefault("page", "home")
ss.setdefault("role", None)
ss.setdefault("user", None)


@st.cache_resource
def setup():
    db.init()
    b = joblib.load(ART / "model.joblib")
    test = pd.read_csv(ART / "test_split.csv").reset_index(drop=True)
    test.insert(0, "machine_id", [f"M-{i:04d}" for i in range(len(test))])
    return b, test, pd.read_csv(ART / "cost_curve.csv"), json.loads((ART / "metrics.json").read_text())


def predict(b, df):
    X = add_features(df)
    p = b["model"].predict_proba(X)[:, 1]
    mp = np.column_stack([b["mode_models"][m].predict_proba(X)[:, 1] for m in MODES])
    contrib = b["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]  # TreeSHAP
    return X, p, np.array(MODES)[mp.argmax(1)], contrib


@st.cache_data(show_spinner="Retrieving SOPs and writing work order...")
def sop_work_order(machine_id, tech, mode, p, causes):
    """RAG copilot: retrieve SOP sections and write a cited work order (cached per machine)."""
    return rag.work_order({"machine_id": machine_id, "tech": tech, "mode": mode, "p": p, "causes": causes})


def threshold_for(curve, ratio):
    c = curve.assign(cost=curve.false_alarms + ratio * curve.missed)
    return float(c.loc[c.cost.idxmin(), "threshold"]), c


def plan_shift(cfg):
    """Predict -> cost threshold -> rank by expected loss -> skill-aware greedy dispatch."""
    thr, c_curve = threshold_for(curve, cfg["ratio"])
    s = cfg["shift_size"]
    shift = test.iloc[(cfg["shift_no"] - 1) * s: cfg["shift_no"] * s].reset_index(drop=True)
    X, p, mode, contrib = predict(b, shift)
    shift = shift.assign(p=p, mode=mode)
    shift["loss"] = shift.p * shift["mode"].map({m: cfg[m] for m in MODES})
    alerts = shift[shift.p >= thr].sort_values("loss", ascending=False).copy()
    free = db.TECH_SLOTS[:cfg["n_tech"]]
    assigned = []
    for m in alerts["mode"]:
        pick = next((t for t in free if SKILL[m] in t), None) or next((t for t in free if "General" in t), None)
        if pick:
            free = [t for t in free if t != pick]
        assigned.append(pick or "Deferred")
    alerts["assigned"] = assigned
    return thr, c_curve, shift, X, contrib, alerts


def go(page, role=None):
    ss.page = page
    if role:
        ss.role = role
    st.rerun()


b, test, curve, metrics = setup()

# ================= PAGE 1: choose profile =================
if ss.page == "home":
    st.title("Maintenance Shift Planner")
    st.write("Predicts machine failures and assigns your limited technicians where a breakdown would cost the most.")
    st.write("")
    st.subheader("Choose your profile")
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("### 🧑‍💼 Manager")
        st.caption("Plan the shift, set technicians and costs, view model performance.")
        if st.button("Continue as Manager", width="stretch", type="primary"):
            go("auth", "Manager")
    with c2.container(border=True):
        st.markdown("### 🔧 Technician")
        st.caption("See your assigned machines and why each one was flagged.")
        if st.button("Continue as Technician", width="stretch", type="primary"):
            go("auth", "Technician")
    st.caption("AI4I 2020 dataset (synthetic) · NexAI Hack 2026")

# ================= PAGE 2: log in / sign up =================
elif ss.page == "auth":
    st.title(f"{ss.role} access")
    if st.button("← Back to profiles"):
        go("home")
    login, signup = st.tabs(["Log in", "Sign up"])
    with login:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary", width="stretch"):
                user = db.verify(u, p, ss.role)
                if user:
                    ss.user = user
                    go("app")
                else:
                    st.error(f"Wrong username or password, or this is not a {ss.role} account.")
        demo = "manager / manager123" if ss.role == "Manager" else "tech1, tech2 or tech3 / tech123"
        st.caption(f"Demo account: {demo}")
    with signup:
        with st.form("signup"):
            u = st.text_input("Choose a username")
            p = st.text_input("Choose a password", type="password")
            slot = st.selectbox("Your technician slot", db.TECH_SLOTS) if ss.role == "Technician" else None
            if st.form_submit_button("Create account", width="stretch"):
                ok, msg = db.create_user(u, p, ss.role, slot)
                (st.success if ok else st.error)(msg)

# ================= PAGE 3: dashboards =================
else:
    user = ss.user
    st.sidebar.markdown(f"**{user['username']}** · {user['role']}")
    if user["tech_slot"]:
        st.sidebar.caption(user["tech_slot"])
    if st.sidebar.button("Log out", width="stretch"):
        ss.user = None
        go("home")

    cfg = db.get_settings()

    # ---------------- MANAGER ----------------
    if user["role"] == "Manager":
        st.sidebar.divider()
        st.sidebar.subheader("Shift settings")
        new = dict(cfg)
        new["n_tech"] = st.sidebar.number_input("Technicians on shift", 1, db.MAX_TECH, cfg["n_tech"])
        new["shift_size"] = st.sidebar.number_input("Machines in shift", 20, 500, cfg["shift_size"], step=10)
        n_shifts = len(test) // new["shift_size"]
        new["shift_no"] = st.sidebar.number_input(f"Shift (1–{n_shifts})", 1, n_shifts, min(cfg["shift_no"], n_shifts))
        new["ratio"] = st.sidebar.number_input("Missed failure cost (× false alarm)", 1, 100, cfg["ratio"])
        with st.sidebar.expander("Failure cost weights (assumptions)"):
            for m in ["pwf", "osf", "hdf", "twf"]:
                new[m] = st.number_input(MODE_NAME[m], 1, 50, cfg[m], key=f"w_{m}")
        if new != cfg:
            db.save_settings(new)  # technicians see this plan when they log in
            cfg = new

        thr, c_curve, shift, X, contrib, alerts = plan_shift(cfg)
        served = alerts[alerts.assigned != "Deferred"]

        st.title(f"Shift {cfg['shift_no']} plan")
        st.caption(f"{len(shift)} machines · {cfg['n_tech']} technicians · alert threshold {thr:.2f}")
        k = st.columns(4)
        k[0].metric("Alerts", len(alerts))
        k[1].metric("Assigned", len(served))
        k[2].metric("Deferred", len(alerts) - len(served))
        k[3].metric("Risk covered", f"{served.loss.sum() / alerts.loss.sum():.0%}" if len(alerts) else "100%")

        tab_plan, tab_model, tab_one = st.tabs(["Shift plan", "Cost curve & model", "Check a machine"])
        with tab_plan:
            if alerts.empty:
                st.success("No machine needs attention this shift.")
            else:
                st.dataframe(pd.DataFrame({
                    "Priority": range(1, len(alerts) + 1),
                    "Machine": alerts.machine_id,
                    "Failure risk": alerts.p.map("{:.0%}".format),
                    "Likely failure": alerts["mode"].map(MODE_NAME),
                    "Expected loss": alerts.loss.round(1),
                    "Assigned to": alerts.assigned,
                }), hide_index=True, width="stretch")
                st.caption("Ranked by expected loss = failure risk × failure cost weight; "
                           "matching specialist first, generalist as backup.")

                st.subheader("Why was this machine flagged? (SHAP)")
                pick = st.selectbox("Machine", alerts.machine_id,
                                    format_func=lambda m: f"{m} · {MODE_NAME[alerts.set_index('machine_id').loc[m, 'mode']]}")
                i = shift.index[shift.machine_id == pick][0]
                top = np.argsort(-np.abs(contrib[i]))[:5]
                st.bar_chart(pd.Series(contrib[i][top], index=[NICE[FEATURES[j]] for j in top]),
                             horizontal=True, height=220)
                st.caption("Bars to the right push failure risk up; to the left push it down. "
                           + ", ".join(f"{NICE[FEATURES[j]]} = {X.iloc[i, j]:.1f}" for j in top))
            st.info(f"Replay check: {int(shift.machine_failure.sum())} machines in this shift actually failed; "
                    f"{int(served.machine_failure.sum())} of them were assigned a technician.")

        with tab_model:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.plot(c_curve.threshold, c_curve.cost, color="#1f6feb", lw=2, label="total cost")
            ax.axvline(thr, color="#d1242f", ls="--", label=f"chosen threshold {thr:.2f}")
            ax.set_xlabel("alert threshold")
            ax.set_ylabel(f"cost (missed = {cfg['ratio']}, false alarm = 1)")
            ax.set_title("Cost vs alert threshold (validation split)")
            ax.grid(alpha=.3); ax.legend()
            st.pyplot(fig)
            st.caption("The threshold with the lowest total cost is chosen. Change the missed-failure cost in the sidebar to see it shift.")
            m = st.columns(4)
            m[0].metric("PR-AUC", metrics["pr_auc"], f"baseline {metrics['baseline_logreg_pr_auc']}", delta_color="off")
            m[1].metric("ROC-AUC", metrics["roc_auc"])
            m[2].metric("Recall", f"{metrics['recall_at_threshold']:.0%}")
            m[3].metric("Precision", f"{metrics['precision_at_threshold']:.0%}")
            st.caption(f"Held-out test split: {metrics['test_rows']} machines, {metrics['test_failures']} failures. "
                       "Limitations: synthetic data; tool-wear failures are hard to predict; cost weights are assumptions.")

        with tab_one:
            c = st.columns(3)
            row = pd.DataFrame([{
                "type": c[0].selectbox("Type", ["L", "M", "H"]),
                "air_temperature": c[1].number_input("Air temp (K)", 290.0, 310.0, 298.0),
                "process_temperature": c[2].number_input("Process temp (K)", 300.0, 320.0, 308.5),
                "rotational_speed": c[0].number_input("Speed (rpm)", 1100, 2900, 1500),
                "torque": c[1].number_input("Torque (Nm)", 3.0, 80.0, 40.0),
                "tool_wear": c[2].number_input("Tool wear (min)", 0, 260, 50),
            }])
            _, p1, m1, _ = predict(b, row)
            st.metric("Failure risk", f"{p1[0]:.1%}")
            st.write(f"Likely failure: **{MODE_NAME[m1[0]]}** · "
                     f"{'needs a technician' if p1[0] >= thr else 'no action needed'} (threshold {thr:.2f})")

    # ---------------- TECHNICIAN ----------------
    else:
        thr, c_curve, shift, X, contrib, alerts = plan_shift(cfg)
        me = user["tech_slot"]
        mine = alerts[alerts.assigned == me]
        st.title("My work orders")
        st.caption(f"{me} · Shift {cfg['shift_no']} (set by the manager)")

        if me not in db.TECH_SLOTS[:cfg["n_tech"]]:
            st.info("You are not on this shift.")
        elif mine.empty:
            st.success("No machines assigned to you this shift.")
        for i, r in mine.iterrows():
            top = np.argsort(-contrib[i])[:3]
            with st.container(border=True):
                st.subheader(f"{r.machine_id} · {MODE_NAME[r['mode']]}")
                a, c2 = st.columns(2)
                a.metric("Failure risk", f"{r.p:.0%}")
                a.markdown(f"**What to do:** {ACTION[r['mode']]}")
                c2.markdown("**Why it was flagged (SHAP)**")
                c2.bar_chart(pd.Series(contrib[i][top], index=[NICE[FEATURES[j]] for j in top]),
                             horizontal=True, height=160)
                causes = ", ".join(f"{NICE[FEATURES[j]]} = {X.iloc[i, j]:.1f}" for j in top)
                c2.caption(causes)

                if st.button("Show SOP steps", key=f"rag_{r.machine_id}"):
                    ss.setdefault("sop_open", set()).add(r.machine_id)  # keep it open across reruns
                if r.machine_id in ss.get("sop_open", set()):
                    wo = sop_work_order(r.machine_id, me, r["mode"], float(r.p), causes)
                    (st.warning if wo["status"] == "escalate" else st.markdown)(wo["text"])
                    st.caption({"llm": "Source: LLM from retrieved SOPs",
                                "template": "Template from retrieved SOPs (no LLM key)"
                                if wo.get("reason") == "no key" else "Template from retrieved SOPs (LLM unavailable)",
                                "none": "No matching SOP"}[wo["source"]])
                    if wo["hits"]:
                        ids = [h["id"] for h in wo["hits"]]
                        cited = [x for x in ids if x in wo["text"]] or ids
                        st.caption("Cited: " + ", ".join(cited))
                    st.caption("Illustrative team-written SOPs, not approved maintenance procedures.")