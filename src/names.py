"""Map StatsBomb full legal names to the names fans know (e.g. 'Lionel Andrés Messi Cuccittini' -> 'Lionel Messi').

Uses the `player_nickname` field in StatsBomb lineup files. Cached in data/names.json.
Usage:  python -m src.names
"""
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .load import BASE, fetch

OUT = pathlib.Path("data/names.json")


def lineup_names(match_id: int) -> dict:
    out = {}
    for team in fetch(f"{BASE}/lineups/{match_id}.json"):
        for p in team["lineup"]:
            if p.get("player_nickname"):
                out[p["player_name"]] = p["player_nickname"]
    return out


def build() -> dict:
    ids = pd.read_pickle("data/matches.pkl")["match_id"].tolist()
    names = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for d in pool.map(lineup_names, ids):
            names.update(d)
    OUT.write_text(json.dumps(names, ensure_ascii=False, indent=0))
    print(f"saved {len(names)} short names -> {OUT}")
    return names


def short_name(full: str, names: dict) -> str:
    return names.get(full, full)


if __name__ == "__main__":
    build()
