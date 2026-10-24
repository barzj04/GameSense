"""Download StatsBomb open-data shots (no login) -> data/shots.pkl.

Usage:
  python -m src.load            # default: 2015/16 big-5 leagues + Euro 2024 + World Cup 2022
  python -m src.load --quick    # Premier League 2015/16 only (fast test)

Only shot events are kept, cached per match in data/cache/, so re-runs are instant.
"""
import json
import sys
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE = pathlib.Path("data/cache")
CACHE.mkdir(parents=True, exist_ok=True)

# (competition_id, season_id)
DEFAULT = [(2, 27), (11, 27), (9, 27), (12, 27), (7, 27),   # 2015/16 PL, La Liga, BL, Serie A, Ligue 1
           (55, 282), (43, 106)]                            # Euro 2024, World Cup 2022
QUICK = [(2, 27)]

S = requests.Session()


def fetch(url: str):
    r = S.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def matches(comp: int, season: int) -> pd.DataFrame:
    m = fetch(f"{BASE}/matches/{comp}/{season}.json")
    return pd.DataFrame([{
        "match_id": x["match_id"],
        "competition": x["competition"]["competition_name"],
        "season": x["season"]["season_name"],
        "home": x["home_team"]["home_team_name"],
        "away": x["away_team"]["away_team_name"],
        "date": x["match_date"],
    } for x in m])


def shots_for_match(match_id: int) -> list:
    f = CACHE / f"{match_id}.json"
    if f.exists():
        return json.loads(f.read_text())
    ev = fetch(f"{BASE}/events/{match_id}.json")
    shots = []
    for e in ev:
        if e["type"]["name"] != "Shot":
            continue
        s = e["shot"]
        shots.append({
            "match_id": match_id,
            "id": e["id"],
            "minute": e["minute"],
            "period": e["period"],
            "team": e["team"]["name"],
            "player": e["player"]["name"],
            "x": e["location"][0], "y": e["location"][1],
            "under_pressure": bool(e.get("under_pressure", False)),
            "play_pattern": e["play_pattern"]["name"],
            "body_part": s["body_part"]["name"],
            "shot_type": s["type"]["name"],
            "technique": s["technique"]["name"],
            "first_time": bool(s.get("first_time", False)),
            "follows_dribble": bool(s.get("follows_dribble", False)),
            "one_on_one": bool(s.get("one_on_one", False)),
            "outcome": s["outcome"]["name"],
            "sb_xg": s.get("statsbomb_xg"),
            "freeze_frame": [
                {"x": p["location"][0], "y": p["location"][1],
                 "teammate": p["teammate"], "gk": p["position"]["name"] == "Goalkeeper"}
                for p in s.get("freeze_frame", [])
            ],
        })
    f.write_text(json.dumps(shots))
    return shots


def build(comps, out="data/shots.pkl") -> pd.DataFrame:
    meta = pd.concat([matches(c, s) for c, s in comps], ignore_index=True)
    meta.to_pickle("data/matches.pkl")
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(shots_for_match, m): m for m in meta["match_id"]}
        for f in as_completed(futs):
            try:
                rows.extend(f.result())
            except Exception as e:
                print(f"skip match {futs[f]}: {e}")
            done += 1
            if done % 50 == 0 or done == len(futs):
                print(f"{done}/{len(futs)} matches")
    df = pd.DataFrame(rows).merge(meta, on="match_id", how="left")
    df["freeze_frame"] = df["freeze_frame"].apply(json.dumps)   # keep as JSON string
    df.to_pickle(out)
    print(f"saved {len(df)} shots from {meta.match_id.nunique()} matches -> {out}")
    return df


if __name__ == "__main__":
    build(QUICK if "--quick" in sys.argv else DEFAULT)
