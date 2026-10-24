# xG Scout

> TODO: one-sentence pitch in your own words.

**Live demo:** TODO (Streamlit Cloud link)

## Problem
TODO: Clubs pay huge fees for strikers after one prolific season. But is "being a good finisher" a stable skill, or mostly variance? And which players and teams actually create good chances?

## Data
StatsBomb Open Data (free, no login). Every shot includes location, body part, play pattern, and a **freeze frame**: the position of every player at the moment of the shot.
- Default: 2015/16 Premier League, La Liga, Bundesliga, Serie A, Ligue 1 + Euro 2024 + World Cup 2022
- Quick mode: 2015/16 Premier League (~9.8k shots)

## Method
1. **Features:** distance, angle to goal, defenders inside the shot cone, keeper position, nearest defender, header, pressure, first-time shot, one-on-one, set-piece/counter context
2. **xG models:** logistic regression vs LightGBM, evaluated on **held-out matches** (grouped split, so there is no leakage within a match); StatsBomb's commercial xG used as a benchmark. Penalties are handled separately (empirical conversion rate).
3. **Out-of-fold xG:** every shot is scored by a model that never saw its match, so player and team analysis is not inflated.
4. **Finishing skill:**
   - Split-half repeatability (random match halves, 50 repeats)
   - Empirical-Bayes Gamma-Poisson shrinkage: goals ~ Poisson(skill × xG), skill ~ Gamma(α, α)
5. **Team shot selection:** share of low-value (< 0.05 xG) shots and the xG lost to them.

## Results (TODO: update after your full run)
| | Log loss | Brier | AUC |
|---|---|---|---|
| Constant baseline | 0.320 | 0.088 | 0.50 |
| Logistic (ours) | 0.272 | 0.078 | 0.777 |
| LightGBM (ours) | 0.274 | 0.078 | 0.772 |
| StatsBomb xG (commercial) | 0.265 | 0.075 | 0.788 |

*(Quick-mode PL 2015/16 numbers)*

**Headline finding:** shot quality is repeatable (split-half r ≈ 0.66); finishing over-performance is not (r ≈ −0.11).
TODO: say this in your own words, plus one named player example.

## What this means for a recruiter / coach
TODO:
- Recruit for chance *getting* (xG per shot × volume), not one hot finishing season
- Coaching target: shot selection (see the Team tab)

## Dashboard
Shot map · xG calculator · Finishing scout · Team shot selection · Model diagnostics

## Limitations
TODO: no tracking data beyond freeze frames, mostly a single season per player, xG ignores shot placement (post-shot xG), 2015/16 data is older.

## Run it
```bash
pip install -r requirements.txt
python -m src.load            # or: python -m src.load --quick
python -m src.model
python -m src.finishing
streamlit run app.py
```

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Please credit StatsBomb as their licence requires.
