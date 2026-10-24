"""GameSense dashboard.  Run:  streamlit run app.py"""
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import FEATURES, GOAL_X, GOAL_Y

st.set_page_config(page_title="GameSense", page_icon="⚽", layout="wide")

# ---------- look & feel ----------
PITCH, STRIPE, CHALK = "#1F6B3A", "#1A5E33", "#FFFFFF"
INK, MUTED, PAPER, LINE = "#17231D", "#5D6B63", "#F4F7F5", "#D9E2DC"
YELLOW, RED, MISS = "#F5B800", "#C8433A", "rgba(255,255,255,0.38)"
FONT = "Barlow, sans-serif"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input {{ font-family: 'Barlow', sans-serif; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ padding-top: 1.2rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ font-family: 'Barlow Condensed', sans-serif !important; color: {INK}; letter-spacing: .2px; }}
h3 {{ font-size: 1.9rem !important; }}
.gs-band {{ background: repeating-linear-gradient(90deg, {PITCH} 0 90px, {STRIPE} 90px 180px);
           border-radius: 6px; padding: 34px 40px 30px; color: {CHALK}; position: relative; overflow: hidden; }}
.gs-band::after {{ content: ""; position: absolute; right: -70px; top: 50%; width: 220px; height: 220px;
                  border: 3px solid rgba(255,255,255,.35); border-radius: 50%; transform: translateY(-50%); }}
.gs-brand {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700; font-size: 1.15rem; opacity: .85; }}
.gs-head {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700; font-size: 3.3rem; line-height: 1.02;
           margin: 6px 0 10px; max-width: 16ch; }}
.gs-sub {{ font-size: 1.12rem; max-width: 62ch; opacity: .92; margin-bottom: 26px; }}
.gs-finds {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 28px; max-width: 980px; }}
.gs-find {{ border-top: 3px solid {YELLOW}; padding-top: 10px; font-size: 1.02rem; line-height: 1.45; }}
.gs-find b {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-size: 1.35rem; display: block; margin-bottom: 4px; }}
.gs-note {{ color: {MUTED}; font-size: .95rem; }}
.gs-big {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700; font-size: 3.6rem; line-height: 1; color: {PITCH}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 2px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-size: 1.2rem; font-weight: 600; padding: 10px 14px; }}
.stTabs [aria-selected="true"] {{ color: {PITCH} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {PITCH}; height: 3px; }}
footer, #MainMenu {{ visibility: hidden; }}
@media (max-width: 800px) {{
  .gs-finds {{ grid-template-columns: 1fr; gap: 18px; }}
  .gs-head {{ font-size: 2.4rem; }}
  .gs-band {{ padding: 24px 20px; }}
  .gs-band::after {{ display: none; }}
}}
</style>
""", unsafe_allow_html=True)


def style(fig, height=460):
    fig.update_layout(font=dict(family=FONT, color=INK, size=14), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.06, x=0, title=None))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig


def pitch(fig, height=540):
    """Half pitch with mowing stripes; goal on the right. StatsBomb units are yards."""
    line = dict(color=CHALK, width=2)
    shapes = [dict(type="rect", x0=x0, x1=x0 + 6, y0=-20, y1=100, fillcolor=PITCH if i % 2 else STRIPE,
                   line=dict(width=0), layer="below") for i, x0 in enumerate(range(30, 150, 6))]
    shapes += [
        dict(type="rect", x0=60, y0=0, x1=120, y1=80, line=line),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=line),
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=line),
        dict(type="rect", x0=120, y0=36, x1=122.5, y1=44, line=line),
        dict(type="path", path="M 60,30 Q 50,40 60,50", line=line),
        dict(type="path", path="M 102,32.7 Q 97,40 102,47.3", line=line),
    ]
    fig.update_layout(shapes=shapes, height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=STRIPE,
                      font=dict(family=FONT, color=INK, size=14),
                      xaxis=dict(range=[59, 123.5], visible=False, fixedrange=True),
                      yaxis=dict(range=[-1, 81], visible=False, scaleanchor="x", autorange="reversed",
                                 fixedrange=True),
                      margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=0.99, x=0.01, yanchor="top", title=None,
                                  font=dict(color=CHALK), bgcolor="rgba(0,0,0,0.25)"))
    return fig


@st.cache_data
def load():
    shots = pd.read_pickle("data/shots_xg.pkl")
    players = pd.read_csv("data/players.csv")
    teams = pd.read_csv("data/teams.csv")
    calib = pd.read_csv("data/calibration.csv")
    metrics = json.load(open("data/metrics.json"))
    fin = json.load(open("data/finishing.json"))
    buy = pd.read_csv("data/buy_low.csv")
    sell = pd.read_csv("data/sell_high.csv")
    try:  # show the names fans know ("Lionel Messi", not his full legal name)
        names = json.load(open("data/names.json", encoding="utf-8"))
        for df in (shots, players, buy, sell):
            df["player"] = df["player"].map(lambda n: names.get(n, n))
    except FileNotFoundError:
        pass
    return shots, players, teams, calib, metrics, fin, buy, sell


@st.cache_resource
def model():
    return joblib.load("data/xg_model.joblib")


shots, players, teams, calib, metrics, fin, buy, sell = load()
sel = metrics["selected_model"]
pt, sh = fin["predictive_test"], fin["split_half"]
open_play = shots[~shots["is_penalty"]]

PLAIN = {"player": "Player", "team": "Team", "shots": "Shots", "goals": "Goals",
         "xg": "Expected goals", "goals_minus_xg": "Goals above expected",
         "xg_per_shot": "Avg chance quality", "luck_check": "Luck check"}

# ---------- header ----------
st.markdown(f"""
<div class="gs-band">
  <div class="gs-brand">GameSense</div>
  <div class="gs-head">Is he a great finisher, or just on a lucky run?</div>
  <div class="gs-sub">We looked at {len(open_play):,} shots from {metrics['n_matches']:,} matches to find out
  what really predicts goals, so clubs can sign the right players.</div>
  <div class="gs-finds">
    <div class="gs-find"><b>Good positions are a real skill</b>Players who get good chances keep getting them.
      Players who score more than they should usually don't keep it up.</div>
    <div class="gs-find"><b>Judge chances, not goals</b>To predict a player's future goals, look at the quality of
      his chances, not his goal tally. It's {pt['error_reduction_pct']:.0f}% more accurate.</div>
    <div class="gs-find"><b>Accurate and tested</b>Our model is almost as accurate as a paid industry model, and it
      worked on World Cup and Euro matches it had never seen.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown('<p class="gs-note"><b>Expected goals</b> = how likely a shot is to go in, from 0 to 1, based on '
            'thousands of similar shots. A penalty is about 0.75; a shot from the halfway line is close to 0.</p>',
            unsafe_allow_html=True)

tabs = st.tabs(["Shot map", "Rate a chance", "Skill or luck?", "Players to watch",
                "Team shooting habits", "How accurate is it?"])

# ---------- Shot map ----------
with tabs[0]:
    a, b = st.columns(2)
    team = a.selectbox("Team", ["All teams"] + sorted(shots["team"].unique()))
    pool = open_play if team == "All teams" else open_play[open_play["team"] == team]
    player = b.selectbox("Player", ["All players"] + sorted(pool["player"].unique()))
    s = pool if player == "All players" else pool[pool["player"] == player]
    plot = s.sample(6000, random_state=0) if len(s) > 6000 else s
    fig = go.Figure()
    for is_goal, name, col in [(0, "Missed / saved", MISS), (1, "Goal", YELLOW)]:
        t = plot[plot["goal"] == is_goal]
        fig.add_trace(go.Scatter(
            x=t["x"], y=t["y"], mode="markers", name=name,
            marker=dict(size=5 + 18 * t["xg"], color=col, line=dict(width=0)),
            text=t["player"] + "<br>Chance of scoring: " + (t["xg"] * 100).round(0).astype(int).astype(str) + "%",
            hoverinfo="text"))
    st.plotly_chart(pitch(fig), width="stretch")
    st.markdown(
        f"**{len(s):,} shots**, **{int(s['goal'].sum())} goals**. An average player would have scored "
        f"**{s['xg'].sum():.0f}** from the same chances. Bigger dots are better chances."
        + (" Showing 6,000 random shots to keep the map readable." if len(s) > 6000 else ""))

# ---------- Rate a chance ----------
with tabs[1]:
    l, r = st.columns([1, 1.25], gap="large")
    with l:
        st.markdown("### Set up the shot")
        out = st.slider("Distance from the goal line (yards)", 1, 40, 12)
        side = st.slider("Distance from the middle of the goal (yards)", -30, 30, 0,
                         help="Negative = left side, positive = right side")
        near = st.slider("Closest defender (yards away)", 1, 15, 3)
        n_cone = st.slider("Defenders blocking the path to goal", 0, 6, 1)
        head = st.toggle("Header")
        press = st.toggle("Shooter is under pressure")
        ft = st.toggle("Hit first time (no touch to control)")
        one = st.toggle("One-on-one with the keeper")
    x, y = GOAL_X - out, GOAL_Y + side
    dx, dy = GOAL_X - x, y - GOAL_Y
    ang = np.arctan2(8 * dx, dx ** 2 + dy ** 2 - 16)
    ang = ang + np.pi if ang < 0 else ang
    dist = float(np.hypot(dx, dy))
    row = open_play[FEATURES].median()
    row.update({"distance": dist, "angle": ang, "dist_x_angle": dist * ang, "n_def_in_cone": n_cone,
                "nearest_def": near, "n_def_within_3": int(near < 3), "is_head": int(head),
                "under_pressure": int(press), "first_time": int(ft), "one_on_one": int(one),
                "free_kick": 0, "from_corner": 0, "from_counter": 0, "from_set_piece": 0, "gk_in_cone": 1})
    xg = float(model().predict_proba(pd.DataFrame([row[FEATURES]]))[:, 1][0])
    with r:
        st.markdown(f'<div class="gs-big">{xg * 100:.0f}%</div>', unsafe_allow_html=True)
        st.markdown(f"chance of scoring. An average player puts this away about **{xg * 100:.0f} times in 100**.")
        fig = go.Figure(go.Scatter(x=[x], y=[y], mode="markers", showlegend=False,
                                   marker=dict(size=18 + 50 * xg, color=YELLOW, line=dict(color=CHALK, width=2))))
        st.plotly_chart(pitch(fig, 400), width="stretch")

# ---------- Skill or luck? ----------
with tabs[2]:
    st.markdown("### Can players keep it up?")
    st.markdown(
        f"We split each player's games into two random groups and checked if the same players stood out in both. "
        f"A score of 1 means they always do; 0 means it's pure chance. ({sh['n_players']} players, repeated 50 times.)")
    bar = pd.DataFrame({"What we measured": ["Getting good chances", "Scoring more than expected"],
                        "Score": [sh["shot_quality_r"], sh["finishing_r"]]})
    fig = px.bar(bar, x="Score", y="What we measured", orientation="h", text="Score", range_x=[-0.2, 1],
                 color="What we measured", color_discrete_sequence=[PITCH, YELLOW])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", width=0.55)
    st.plotly_chart(style(fig, 230).update_layout(showlegend=False, yaxis_title=None,
                                                    xaxis_title="How repeatable it is (0 = luck, 1 = skill)"),
                    width="stretch")
    st.markdown("**What it means:** a striker who scored loads more than expected last season will probably "
                "drop back to normal. A striker who keeps getting into good positions will keep doing it.")

    st.markdown("### Every player at a glance")
    p = players.copy()
    fig = px.scatter(p, x="shots", y="xg_per_shot", size="xg", color="goals_minus_xg",
                     color_continuous_scale=[RED, "#E9ECE8", PITCH], color_continuous_midpoint=0,
                     hover_name="player", hover_data={"team": True, "goals": True, "xg": ":.1f"},
                     labels={"shots": "Shots taken", "xg_per_shot": "Avg chance quality",
                             "goals_minus_xg": "Goals above<br>expected", "xg": "Expected goals"})
    st.plotly_chart(style(fig, 500), width="stretch")
    st.caption("Higher up = gets better chances. Further right = shoots more. Green scored more than expected, red less.")

    st.markdown("### Look up a player")
    search = st.text_input("Search by name", placeholder="e.g. Kane")
    show = p[["player", "team", "shots", "goals", "xg", "goals_minus_xg", "xg_per_shot", "luck_check"]]
    if search:
        show = show[show["player"].str.contains(search, case=False)]
    st.dataframe(show.sort_values("goals_minus_xg", ascending=False).rename(columns=PLAIN).round(2),
                 width="stretch", hide_index=True)
    st.caption("Luck check: could an average finisher have scored this many (or this few) from the same chances "
               "just by luck?")

# ---------- Players to watch ----------
with tabs[3]:
    st.markdown("### Who to buy, who to sell")
    st.markdown("Goals tend to catch up with chance quality. These players (30+ shots) look set to change direction.")
    cols = ["player", "team", "shots", "goals", "xg", "goals_minus_xg", "luck_check"]
    st.markdown(f"#### Buy: due more goals ({len(buy)})")
    st.markdown('<p class="gs-note">They get good chances but scored at least 2 fewer goals than expected.</p>',
                unsafe_allow_html=True)
    st.dataframe(buy[cols].rename(columns=PLAIN).round(1), hide_index=True, width="stretch", height=320)
    st.markdown(f"#### Sell: likely to cool off ({len(sell)})")
    st.markdown('<p class="gs-note">They get ordinary chances but scored at least 3 more goals than expected.</p>',
                unsafe_allow_html=True)
    st.dataframe(sell[cols].rename(columns=PLAIN).round(1), hide_index=True, width="stretch", height=320)
    both = pd.concat([buy.assign(group="Buy"), sell.assign(group="Sell")])
    fig = px.scatter(both, x="xg", y="goals", color="group", hover_name="player", hover_data=["team", "shots"],
                     color_discrete_map={"Buy": PITCH, "Sell": RED},
                     labels={"xg": "Goals an average player would score", "goals": "Goals actually scored"})
    mx = float(max(both["xg"].max(), both["goals"].max())) + 2
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Exactly as expected",
                             line=dict(dash="dash", color=MUTED)))
    fig.update_traces(marker=dict(size=11), selector=dict(mode="markers"))
    st.plotly_chart(style(fig, 460), width="stretch")
    st.caption("Below the dashed line = scored fewer than their chances deserved. Penalties not included.")

# ---------- Team shooting habits ----------
with tabs[4]:
    st.markdown("### Which teams take good shots?")
    comp_of = shots.groupby("team")["competition"].agg(lambda c: c.mode().iat[0])
    tm = teams.assign(competition=teams["team"].map(comp_of))
    comps = sorted(tm["competition"].dropna().unique())
    comp = st.selectbox("Competition", comps,
                        index=comps.index("Premier League") if "Premier League" in comps else 0)
    tm = tm[tm["competition"] == comp].sort_values("xg_per_shot")
    fig = px.bar(tm, x="xg_per_shot", y="team", orientation="h", color="low_value_share",
                 color_continuous_scale=["#E9ECE8", YELLOW, RED],
                 labels={"xg_per_shot": "Avg chance quality per shot", "team": "",
                         "low_value_share": "Share of<br>hopeful shots"})
    st.plotly_chart(style(fig, 60 + 26 * len(tm)), width="stretch")
    w = tm.sort_values("xg_if_half_recycled", ascending=False).iloc[0]
    st.markdown(f"**{w.team}** took **{w.low_value_share:.0%}** of their shots from spots that almost never "
                f"score (under 5% chance). Passing instead on half of those could have been worth about "
                f"**{w.xg_if_half_recycled:.0f} extra goals** over the period.")
    st.caption("Hopeful shot = less than a 5% chance of scoring, e.g. long-range efforts through a crowd.")

# ---------- How accurate is it? ----------
with tabs[5]:
    st.markdown("### How good is the model?")
    names = {"baseline_constant": "Guessing the average every time",
             "logistic": "Our simple model", "lightgbm": "Our main model",
             "statsbomb_xg_benchmark": "StatsBomb's paid model (industry standard)",
             "generalisation_international": "Our model on World Cup + Euro (never seen before)"}
    rows = [{"Model": names[k], "Accuracy score (higher is better)": round(v["auc"], 3),
             "Goals predicted": round(v["pred_goals"]), "Goals actually scored": v["actual_goals"]}
            for k, v in metrics.items() if k in names]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Tested on matches the model never saw while learning. Accuracy score (AUC): 0.5 is a coin flip, "
               "1.0 is perfect; 0.8 is very good for football, where luck plays a big part.")

    l, r = st.columns(2, gap="large")
    with l:
        st.markdown("#### Do the percentages match reality?")
        c = calib[calib["model"] == ("LightGBM" if sel == "lightgbm" else "Logistic")]
        fig = go.Figure([
            go.Scatter(x=[0, .8], y=[0, .8], mode="lines", name="Perfect match", line=dict(dash="dash", color=MUTED)),
            go.Scatter(x=c["predicted"], y=c["observed"], mode="lines+markers", name="Our model",
                       line=dict(color=PITCH, width=3), marker=dict(size=8))])
        fig.update_xaxes(title="What the model said", tickformat=".0%")
        fig.update_yaxes(title="How often it went in", tickformat=".0%")
        st.plotly_chart(style(fig, 380), width="stretch")
        st.caption("When the model says 30%, about 30% of those shots really go in.")
    with r:
        st.markdown("#### Goals or chances: which predicts the future?")
        comp_df = pd.DataFrame({"Predict using": ["Past goals", "Past chance quality"],
                                "Average miss (goals)": [pt["mae_goals_based"], pt["mae_xg_based"]]})
        fig = px.bar(comp_df, x="Predict using", y="Average miss (goals)", text="Average miss (goals)",
                     color="Predict using", color_discrete_sequence=[MUTED, PITCH])
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", width=0.5)
        st.plotly_chart(style(fig, 380).update_layout(showlegend=False, xaxis_title=None), width="stretch")
        st.caption(f"We used the first half of each season to predict the second half for "
                   f"{pt['n_player_seasons']} players. Chance quality missed by "
                   f"{pt['error_reduction_pct']:.0f}% less.")

st.divider()
st.markdown('<p class="gs-note">Built with StatsBomb Open Data: the 2015/16 top five European leagues, '
            'World Cup 2022 and Euro 2024. Penalties are left out of player comparisons.</p>',
            unsafe_allow_html=True)
