"""Shot features: geometry + freeze-frame (defender / keeper positions at the moment of the shot).

StatsBomb pitch is 120 x 80; attacking goal at x=120, posts at y=36 and y=44.
"""
import json
import numpy as np
import pandas as pd

GOAL_X, POST_L, POST_R, GOAL_Y = 120.0, 36.0, 44.0, 40.0

NUM = ["distance", "angle", "dist_x_angle", "n_def_in_cone", "gk_dist_to_goal",
       "gk_off_line", "nearest_def", "n_def_within_3"]
BOOL = ["is_head", "under_pressure", "first_time", "follows_dribble", "one_on_one",
        "free_kick", "from_corner", "from_counter", "from_set_piece", "gk_in_cone"]
FEATURES = NUM + BOOL


def _in_triangle(px, py, ax, ay, bx, by, cx, cy):
    """Vectorised point-in-triangle test (barycentric signs)."""
    def s(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)
    d1 = s(px, py, ax, ay, bx, by)
    d2 = s(px, py, bx, by, cx, cy)
    d3 = s(px, py, cx, cy, ax, ay)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def freeze_feats(x, y, ff_json: str) -> dict:
    ff = json.loads(ff_json) if isinstance(ff_json, str) else (ff_json or [])
    opp = [p for p in ff if not p["teammate"]]
    out = dict(n_def_in_cone=0, gk_dist_to_goal=np.nan, gk_off_line=np.nan,
               gk_in_cone=False, nearest_def=np.nan, n_def_within_3=0)
    if not opp:
        return out
    px = np.array([p["x"] for p in opp])
    py = np.array([p["y"] for p in opp])
    gk = np.array([p["gk"] for p in opp])
    cone = _in_triangle(px, py, x, y, GOAL_X, POST_L, GOAL_X, POST_R)
    d = np.hypot(px - x, py - y)
    out["n_def_in_cone"] = int((cone & ~gk).sum())
    out["n_def_within_3"] = int(((d < 3) & ~gk).sum())
    if (~gk).any():
        out["nearest_def"] = float(d[~gk].min())
    if gk.any():
        gx, gy = px[gk][0], py[gk][0]
        out["gk_dist_to_goal"] = float(np.hypot(GOAL_X - gx, GOAL_Y - gy))
        out["gk_off_line"] = float(GOAL_X - gx)
        out["gk_in_cone"] = bool(cone[gk][0])
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dx = GOAL_X - d["x"]
    dy = d["y"] - GOAL_Y
    d["distance"] = np.hypot(dx, dy)
    ang = np.arctan2(8 * dx, dx ** 2 + dy ** 2 - 16)          # angle subtended by the posts
    d["angle"] = np.where(ang < 0, ang + np.pi, ang)
    d["dist_x_angle"] = d["distance"] * d["angle"]

    d["is_head"] = d["body_part"].eq("Head")
    d["free_kick"] = d["shot_type"].eq("Free Kick")
    d["from_corner"] = d["play_pattern"].eq("From Corner")
    d["from_counter"] = d["play_pattern"].eq("From Counter")
    d["from_set_piece"] = d["play_pattern"].isin(["From Free Kick", "From Throw In", "From Corner"])

    ff = pd.DataFrame([freeze_feats(a, b, c) for a, b, c in zip(d["x"], d["y"], d["freeze_frame"])],
                      index=d.index)
    d = pd.concat([d, ff], axis=1)
    # sensible fills when no freeze frame (e.g. some older matches)
    d["gk_dist_to_goal"] = d["gk_dist_to_goal"].fillna(d["gk_dist_to_goal"].median())
    d["gk_off_line"] = d["gk_off_line"].fillna(d["gk_off_line"].median())
    d["nearest_def"] = d["nearest_def"].fillna(d["nearest_def"].median())

    for c in BOOL:
        d[c] = d[c].astype(int)
    d["goal"] = d["outcome"].eq("Goal").astype(int)
    d["is_penalty"] = d["shot_type"].eq("Penalty")
    return d


def load(path="data/shots.pkl") -> pd.DataFrame:
    return build_features(pd.read_pickle(path))
