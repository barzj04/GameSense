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
    # chance an average finisher scores this few goals or fewer (for under-performers)
    p["p_unlucky"] = stats.poisson.cdf(p["goals"], p["xg"])
    p["luck_check"] = [luck_label(g, x, a, b) for g, x, a, b in
                       zip(p["goals"], p["xg"], p["p_luck"], p["p_unlucky"])]
    p["xg_per_shot"] = p["xg"] / p["shots"]
    return p.sort_values("skill", ascending=False), alpha


def luck_label(goals, xg, p_over, p_under) -> str:
    """Plain-language verdict: could an exactly-average finisher produce this record by chance?"""
    if goals >= xg:
        if p_over < 0.01:
            return "Hard to explain by luck"
        if p_over < 0.10:
            return "Possibly skill, possibly luck"
        return "Within normal luck"
    if p_under < 0.01:
        return "Unusually unlucky / poor"
    if p_under < 0.10:
        return "Possibly unlucky"
    return "Within normal luck"


def predictive_test(shots: pd.DataFrame, min_each: int = 10) -> dict:
    """Does past xG predict FUTURE goals better than past goals do?
    For each player-season, order matches by date and split into first / second half.
    Forecast second-half goals two ways, scaled by second-half shot volume:
      goals-based: (first-half goals / shots) x second-half shots
      xG-based:    (first-half xG    / shots) x second-half shots
    """
    s = shots[~shots["is_penalty"]]
    key = ["player", "competition", "season"]
    m = s[key + ["match_id", "date"]].drop_duplicates(key + ["match_id"]).sort_values("date")
    m["rank"] = m.groupby(key).cumcount()
    m["n"] = m.groupby(key)["match_id"].transform("size")
    m["half"] = (m["rank"] >= m["n"] / 2).astype(int)
    t = s.merge(m[key + ["match_id", "half"]], on=key + ["match_id"])
    h = (t.groupby(key + ["half"]).agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"))
          .unstack("half"))
    h = h[(h["shots"] >= min_each).all(axis=1)]
    s1, s2 = h[("shots", 0)], h[("shots", 1)]
    g1, g2, x1 = h[("goals", 0)], h[("goals", 1)], h[("xg", 0)]
    pred_goals = g1 / s1 * s2
    pred_xg = x1 / s1 * s2
    mae_g = float(np.mean(np.abs(pred_goals - g2)))
    mae_x = float(np.mean(np.abs(pred_xg - g2)))
    return {
        "n_player_seasons": int(len(h)),
        "r_past_goals_vs_future_goals": round(float(np.corrcoef(g1 / s1, g2 / s2)[0, 1]), 3),
        "r_past_xg_vs_future_goals": round(float(np.corrcoef(x1 / s1, g2 / s2)[0, 1]), 3),
        "mae_goals_based": round(mae_g, 3),
        "mae_xg_based": round(mae_x, 3),
        "error_reduction_pct": round(100 * (1 - mae_x / mae_g), 1),
    }


def shortlists(players: pd.DataFrame, min_shots: int = 30):
    """Buy-low: get good chances, haven't converted (expect goals to rise).
    Sell-high: average chances, finishing well above xG (expect goals to fall)."""
    q = players[players["shots"] >= min_shots]
    med = q["xg_per_shot"].median()
    cols = ["player", "team", "shots", "goals", "xg", "goals_minus_xg", "xg_per_shot", "luck_check"]
    buy = q[(q["xg_per_shot"] >= med) & (q["goals_minus_xg"] <= -2)].sort_values("xg", ascending=False)
    sell = q[(q["xg_per_shot"] < med) & (q["goals_minus_xg"] >= 3)].sort_values("goals_minus_xg",
                                                                                   ascending=False)
    return buy[cols], sell[cols]


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
    buy, sell = shortlists(players)
    buy.to_csv("data/buy_low.csv", index=False)
    sell.to_csv("data/sell_high.csv", index=False)
    teams.to_csv("data/teams.csv", index=False)
    summary = {"alpha": round(alpha, 2), "n_players": int(len(players)),
               "players_95pct_above_avg": int((players["p_above_avg"] > 0.95).sum()),
               "players_95pct_below_avg": int((players["p_above_avg"] < 0.05).sum()),
               "alpha_at_bound": bool(alpha > 2000),
               "split_half": split_half(shots),
               "predictive_test": predictive_test(shots),
               "n_buy_low": int(len(buy)), "n_sell_high": int(len(sell))}
    with open("data/finishing.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(players.head(10)[["player", "team", "shots", "goals", "xg", "raw_ratio", "skill",
                            "skill_lo", "skill_hi"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
