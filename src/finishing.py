"""Finishing skill with empirical-Bayes shrinkage, and team shot-selection analysis.

Model: goals_i ~ Poisson(skill_i * xG_i),  skill_i ~ Gamma(alpha, alpha)  (prior mean 1 = average finisher)
Marginal is negative binomial -> estimate alpha by max likelihood, then
posterior skill_i ~ Gamma(alpha + goals_i, alpha + xG_i).
A player with few shots is pulled toward 1.0; only sustained over-performance survives.

Usage:  python -m src.finishing
"""
import json
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

MIN_SHOTS = 20


def _neg_loglik(log_alpha, g, x):
    a = np.exp(log_alpha)
    return -np.sum(gammaln(a + g) - gammaln(a) - gammaln(g + 1)
                   + a * np.log(a / (a + x)) + g * np.log(x / (a + x)))


def fit_alpha(goals, xg) -> float:
    r = optimize.minimize_scalar(_neg_loglik, bounds=(-3, 8), args=(goals, xg), method="bounded")
    return float(np.exp(r.x))


def player_table(shots: pd.DataFrame, min_shots: int = MIN_SHOTS):
    s = shots[~shots["is_penalty"]]
    p = (s.groupby(["player"])
           .agg(team=("team", lambda t: t.mode().iat[0]), shots=("goal", "size"),
                goals=("goal", "sum"), xg=("xg", "sum"))
           .reset_index())
    p = p[p["shots"] >= min_shots].copy()
    alpha = fit_alpha(p["goals"].values, p["xg"].values)
    a_post, b_post = alpha + p["goals"], alpha + p["xg"]
    p["raw_ratio"] = p["goals"] / p["xg"]
    p["skill"] = a_post / b_post                                   # posterior mean multiplier
    p["skill_lo"] = stats.gamma.ppf(0.05, a_post, scale=1 / b_post)
    p["skill_hi"] = stats.gamma.ppf(0.95, a_post, scale=1 / b_post)
    p["p_above_avg"] = 1 - stats.gamma.cdf(1.0, a_post, scale=1 / b_post)
    p["goals_minus_xg"] = p["goals"] - p["xg"]
    # chance an exactly-average finisher scores at least this many goals from this xG
    p["p_luck"] = stats.poisson.sf(p["goals"] - 1, p["xg"])
    p["xg_per_shot"] = p["xg"] / p["shots"]
    return p.sort_values("skill", ascending=False), alpha


def split_half(shots: pd.DataFrame, min_shots: int = MIN_SHOTS, seed: int = 0) -> dict:
    """Is finishing repeatable? Correlate (goals - xG)/shot between two random halves of each
    player's matches. r near 0 => over-performance is mostly noise."""
    s = shots[~shots["is_penalty"]]
    rng = np.random.default_rng(seed)
    rs = []
    for k in range(50):
        m = s[["player", "match_id"]].drop_duplicates()
        m["half"] = rng.integers(0, 2, len(m))
        t = s.merge(m, on=["player", "match_id"])
        h = (t.groupby(["player", "half"]).agg(n=("goal", "size"), g=("goal", "sum"), x=("xg", "sum"))
              .unstack("half"))
        h = h[(h["n"] >= min_shots // 2).all(axis=1)]
        a = (h[("g", 0)] - h[("x", 0)]) / h[("n", 0)]
        b = (h[("g", 1)] - h[("x", 1)]) / h[("n", 1)]
        rs.append(np.corrcoef(a, b)[0, 1])
    # same test for shot quality (xG per shot) - a skill we expect to be repeatable
    q = []
    for k in range(50):
        m = s[["player", "match_id"]].drop_duplicates()
        m["half"] = rng.integers(0, 2, len(m))
        t = s.merge(m, on=["player", "match_id"])
        h = t.groupby(["player", "half"]).agg(n=("goal", "size"), x=("xg", "mean")).unstack("half")
        h = h[(h["n"] >= min_shots // 2).all(axis=1)]
        q.append(np.corrcoef(h[("x", 0)], h[("x", 1)])[0, 1])
    return {"finishing_r": round(float(np.mean(rs)), 3),
            "shot_quality_r": round(float(np.mean(q)), 3), "n_players": int(len(h))}


def team_shot_selection(shots: pd.DataFrame) -> pd.DataFrame:
    s = shots[~shots["is_penalty"]].copy()
    s["low_value"] = s["xg"] < 0.05
    t = (s.groupby("team")
           .agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"),
                xg_per_shot=("xg", "mean"), low_value_share=("low_value", "mean"),
                avg_distance=("distance", "mean"))
           .reset_index())
    # counterfactual: replace low-value shots with the team's own median "good" shot? keep it simple:
    # xG thrown away = low-value shots x (league mean xG/shot of non-low-value shots - their xG) x 0.5 conversion of possessions
    good_mean = s.loc[~s["low_value"], "xg"].mean()
    lv = s[s["low_value"]].groupby("team")["xg"].agg(["size", "sum"])
    t = t.merge(lv.rename(columns={"size": "low_value_shots", "sum": "low_value_xg"}),
                left_on="team", right_index=True, how="left").fillna(0)
    t["xg_if_half_recycled"] = 0.5 * (t["low_value_shots"] * good_mean - t["low_value_xg"])
    return t.sort_values("xg_per_shot", ascending=False)


def main():
    shots = pd.read_pickle("data/shots_xg.pkl")
    players, alpha = player_table(shots)
    teams = team_shot_selection(shots)
    players.to_csv("data/players.csv", index=False)
    teams.to_csv("data/teams.csv", index=False)
    summary = {"alpha": round(alpha, 2), "n_players": int(len(players)),
               "players_95pct_above_avg": int((players["p_above_avg"] > 0.95).sum()),
               "players_95pct_below_avg": int((players["p_above_avg"] < 0.05).sum()),
               "alpha_at_bound": bool(alpha > 2000),
               "split_half": split_half(shots)}
    with open("data/finishing.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(players.head(10)[["player", "team", "shots", "goals", "xg", "raw_ratio", "skill",
                            "skill_lo", "skill_hi"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
