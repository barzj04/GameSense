"""Streamlit dashboard.  Run:  streamlit run app.py"""
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import FEATURES, GOAL_X, GOAL_Y

st.set_page_config(page_title="xG Scout", layout="wide")
GOAL_C, MISS_C = "#e4572e", "#6c8ead"


@st.cache_data
def load():
    shots = pd.read_pickle("data/shots_xg.pkl")
    players = pd.read_csv("data/players.csv")
    teams = pd.read_csv("data/teams.csv")
    calib = pd.read_csv("data/calibration.csv")
    metrics = json.load(open("data/metrics.json"))
    fin = json.load(open("data/finishing.json"))
    return shots, players, teams, calib, metrics, fin


@st.cache_resource
def model():
    return joblib.load("data/xg_model.joblib")


def half_pitch(fig):
    line = dict(color="#888", width=1.5)
    shapes = [
        dict(type="rect", x0=60, y0=0, x1=120, y1=80, line=line),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=line),      # penalty box
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=line),      # 6-yard box
        dict(type="rect", x0=120, y0=36, x1=122, y1=44, line=line),      # goal
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=line),      # centre circle (half)
    ]
    fig.update_layout(shapes=shapes, plot_bgcolor="rgba(0,0,0,0)", height=520,
                      xaxis=dict(range=[58, 123], visible=False),
                      yaxis=dict(range=[-2, 82], visible=False, scaleanchor="x", autorange="reversed"),
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


shots, players, teams, calib, metrics, fin = load()
sel = metrics["selected_model"]

st.title("xG Scout")
st.caption("Who creates good chances, and who is just finishing hot? Built on StatsBomb open data.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Shots analysed", f"{metrics['n_shots']:,}")
c2.metric(f"Model AUC ({sel})", f"{metrics[sel]['auc']:.3f}")
c3.metric("Shot-quality repeatability (r)", f"{fin['split_half']['shot_quality_r']:.2f}")
c4.metric("Finishing repeatability (r)", f"{fin['split_half']['finishing_r']:.2f}")

tabs = st.tabs(["Shot map", "xG calculator", "Finishing scout", "Team shot selection", "Model"])

# ---------------- 1. Shot map ----------------
with tabs[0]:
    a, b = st.columns(2)
    team = a.selectbox("Team", ["All"] + sorted(shots["team"].unique()))
    pool = shots if team == "All" else shots[shots["team"] == team]
    player = b.selectbox("Player", ["All"] + sorted(pool["player"].unique()))
    s = pool if player == "All" else pool[pool["player"] == player]
    s = s[~s["is_penalty"]]
    fig = go.Figure()
    for is_goal, name, col in [(0, "No goal", MISS_C), (1, "Goal", GOAL_C)]:
        t = s[s["goal"] == is_goal]
        fig.add_trace(go.Scatter(
            x=t["x"], y=t["y"], mode="markers", name=name,
            marker=dict(size=4 + 16 * t["xg"], color=col, opacity=0.65 if is_goal else 0.35,
                        line=dict(width=0.5, color="white")),
            text=t["player"] + "<br>xG " + t["xg"].round(2).astype(str),
            hoverinfo="text"))
    st.plotly_chart(half_pitch(fig), width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("Shots (non-penalty)", len(s))
    m2.metric("Goals vs xG", f"{int(s['goal'].sum())} vs {s['xg'].sum():.1f}")
    m3.metric("xG per shot", f"{s['xg'].mean():.3f}" if len(s) else "–")
    st.caption("Bubble size = xG (chance quality).")

# ---------------- 2. xG calculator ----------------
with tabs[1]:
    st.subheader("How good is this chance?")
    l, r = st.columns([1, 1.3])
    x = l.slider("Distance from own goal line (x, 0–120)", 60.0, 119.0, 108.0, 0.5)
    y = l.slider("Width position (y, 0–80; 40 = centre)", 0.0, 80.0, 40.0, 0.5)
    head = l.checkbox("Header")
    press = l.checkbox("Under pressure")
    ft = l.checkbox("First-time shot")
    one = l.checkbox("One-on-one with keeper")
    n_cone = l.slider("Defenders between ball and goal", 0, 6, 1)
    near = l.slider("Nearest defender (yards)", 0.5, 15.0, 3.0, 0.5)

    dx, dy = GOAL_X - x, y - GOAL_Y
    ang = np.arctan2(8 * dx, dx ** 2 + dy ** 2 - 16)
    ang = ang + np.pi if ang < 0 else ang
    dist = float(np.hypot(dx, dy))
    med = shots[~shots["is_penalty"]][FEATURES].median()
    row = med.copy()
    row.update({"distance": dist, "angle": ang, "dist_x_angle": dist * ang, "n_def_in_cone": n_cone,
                "nearest_def": near, "n_def_within_3": int(near < 3), "is_head": int(head),
                "under_pressure": int(press), "first_time": int(ft), "one_on_one": int(one),
                "free_kick": 0, "from_corner": 0, "from_counter": 0, "from_set_piece": 0,
                "gk_in_cone": 1})
    xg = float(model().predict_proba(pd.DataFrame([row[FEATURES]]))[:, 1][0])
    r.metric("Expected goals (xG)", f"{xg:.2f}", help="Probability this shot is scored")
    r.markdown(f"An average player scores this about **{xg * 100:.0f} times in 100**.")
    fig = go.Figure(go.Scatter(x=[x], y=[y], mode="markers",
                               marker=dict(size=14 + 60 * xg, color=GOAL_C)))
    r.plotly_chart(half_pitch(fig).update_layout(height=380, showlegend=False),
                   width="stretch")

# ---------------- 3. Finishing scout ----------------
with tabs[2]:
    sh = fin["split_half"]
    st.subheader("Is finishing a skill or a hot streak?")
    st.markdown(
        f"We split each player's matches into two random halves, 50 times over, and checked whether "
        f"the same players stand out in both halves (across {sh['n_players']} players):\n"
        f"- **Shot quality** (xG per shot): r = **{sh['shot_quality_r']:.2f}**. Players who find good "
        f"positions keep finding them.\n"
        f"- **Finishing** (goals minus xG per shot): r = **{sh['finishing_r']:.2f}**. Beating your xG in one "
        f"half tells you almost nothing about the other half.\n\n"
        "**For a recruiter:** pay for players who *get into* high-xG positions, not for one hot finishing season.")
    bar = pd.DataFrame({"Metric": ["Shot quality (xG/shot)", "Finishing (goals − xG)"],
                        "r": [sh["shot_quality_r"], sh["finishing_r"]]})
    st.plotly_chart(px.bar(bar, x="r", y="Metric", orientation="h", range_x=[-0.3, 1], height=220,
                           color="Metric", color_discrete_sequence=[MISS_C, GOAL_C])
                    .update_layout(showlegend=False, xaxis_title="Split-half correlation"),
                    width="stretch")

    st.markdown("#### Chance getters: shot volume × shot quality")
    p = players.copy()
    fig = px.scatter(p, x="shots", y="xg_per_shot", size="xg", color="goals_minus_xg",
                     color_continuous_scale="RdBu", color_continuous_midpoint=0,
                     hover_name="player", hover_data=["team", "goals", "xg"], height=520,
                     labels={"shots": "Non-penalty shots", "xg_per_shot": "xG per shot",
                             "goals_minus_xg": "Goals − xG"})
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Player table")
    show = p[["player", "team", "shots", "goals", "xg", "goals_minus_xg", "xg_per_shot", "p_luck",
              "skill", "skill_lo", "skill_hi"]].rename(columns={
        "p_luck": "P(average finisher does this)", "skill": "Shrunk finishing ×",
        "skill_lo": "90% low", "skill_hi": "90% high"})
    st.dataframe(show.round(3).sort_values("goals_minus_xg", ascending=False),
                 width="stretch", hide_index=True)
    st.caption("Shrunk finishing × uses an empirical-Bayes Gamma-Poisson model; 1.00 = average. "
               + ("On this data the best-fitting prior has almost no between-player spread, so every "
                  "player is pulled to ~1.00, consistent with the split-half result."
                  if fin.get("alpha_at_bound") else ""))

# ---------------- 4. Team shot selection ----------------
with tabs[3]:
    st.subheader("Who takes good shots?")
    t = teams.sort_values("xg_per_shot")
    st.plotly_chart(px.bar(t, x="xg_per_shot", y="team", orientation="h", color="low_value_share",
                           color_continuous_scale="Reds", height=40 + 22 * len(t),
                           labels={"xg_per_shot": "xG per shot", "low_value_share": "Share of shots < 0.05 xG"}),
                    width="stretch")
    worst = teams.sort_values("xg_if_half_recycled", ascending=False).iloc[0]
    st.markdown(f"- **{worst.team}** took **{worst.low_value_share:.0%}** of their shots from very low-value "
                f"positions (< 0.05 xG). If half of those possessions were worked into an average good chance, "
                f"that is worth about **{worst.xg_if_half_recycled:.1f} extra expected goals**.")
    st.dataframe(teams.round(3), width="stretch", hide_index=True)

# ---------------- 5. Model ----------------
with tabs[4]:
    st.subheader("Model performance (held-out matches)")
    rows = {k: v for k, v in metrics.items()
            if isinstance(v, dict) and "log_loss" in v}
    st.dataframe(pd.DataFrame(rows).T.rename_axis("model").reset_index(), hide_index=True,
                 width="stretch")
    st.caption("Lower log loss / Brier = better. StatsBomb's own commercial xG is shown as a benchmark.")
    fig = px.line(calib, x="predicted", y="observed", color="model", markers=True, height=420,
                  labels={"predicted": "Predicted xG", "observed": "Actual goal rate"})
    fig.add_trace(go.Scatter(x=[0, 0.8], y=[0, 0.8], mode="lines", name="Perfect",
                             line=dict(dash="dash", color="grey")))
    st.plotly_chart(fig, width="stretch")
    coefs = pd.Series(metrics["logistic_coefficients_std"]).sort_values()
    st.plotly_chart(px.bar(coefs, orientation="h", height=420,
                           labels={"value": "Standardised effect on log-odds of scoring", "index": ""})
                    .update_layout(showlegend=False), width="stretch")
