# GameSense

> TODO: one-sentence pitch in your own words.

**Live app:** https://gamesense.streamlit.app (TODO: update once you rename the app URL)

## The problem
TODO (your words): Clubs pay huge fees for strikers after one big season. But is a player scoring lots because he's a great finisher, or because he got lucky? Getting that wrong costs millions.

## What we found
1. **Getting into good positions is a real skill. Scoring more than expected is mostly luck.** We split each player's games into two random groups. Players with good chances in one group also had good chances in the other (score 0.67 out of 1). Players who out-scored their chances in one group didn't do it in the other (score −0.04, basically zero).
2. **Chance quality predicts future goals better than past goals do.** Using the first half of a season to predict the second half (590 players), chance quality was 22% more accurate than goal tallies.
3. **The model is accurate and works on new competitions.** It is almost as accurate as StatsBomb's paid model (0.814 vs 0.816). It learned only from club football and still predicted 254 goals at World Cup 2022 + Euro 2024, where 250 were actually scored.

TODO: add one player story. Our 2015/16 "buy" list includes Edin Džeko (Roma, 6 goals from chances worth 12) and Romelu Lukaku (Everton, 17 from 21.5). Check how they did the next season before using this.

## How it works (simple version)
- **Expected goals (xG):** for every shot, the chance it goes in (0 to 1), learned from 41,000+ real shots. It looks at distance, angle, defenders in the way, the keeper's position, headers, pressure and more.
- **Goals above expected:** actual goals minus expected goals. Positive means a player scored more than an average player would from the same chances.
- **Luck check:** how likely an average finisher would match a player's record just by luck.

## Data
StatsBomb Open Data (free): 41,562 shots from 1,666 matches. That's the 2015/16 Premier League, La Liga, Serie A, Ligue 1 and part of the Bundesliga, plus World Cup 2022 and Euro 2024. Every shot comes with the positions of all players at that moment.

## Method (technical)
- Models: logistic regression and LightGBM, tested on matches held out from training (no match appears in both)
- Every shot is scored by a model that never saw that match (5-fold, grouped by match)
- Skill test: split-half correlation over 50 random splits, plus Gamma-Poisson empirical-Bayes shrinkage
- Future test: first half of season → second half, forecast error compared
- Generalisation: trained on club leagues, tested on international tournaments
- Penalties use the real conversion rate and are left out of player comparisons

| Model | Accuracy (AUC) | Goals predicted | Actual goals |
|---|---|---|---|
| Guessing the average | 0.500 | 784 | 806 |
| Our simple model | 0.810 | 788 | 806 |
| Our main model | 0.814 | 786 | 806 |
| StatsBomb's paid model | 0.816 | 764 | 806 |
| Our model on World Cup + Euro | 0.797 | 254 | 250 |

## Who it's for
TODO (your words):
- **Recruiters:** sign players who get good chances, not players on a hot streak (see "Players to watch")
- **Coaches:** cut down on hopeful long shots (see "Team shooting habits")
- **Fans and media:** check if a striker's form is likely to last

## Limitations
TODO (your words): only shot data (no full tracking), mostly one season per player, doesn't measure where the shot was placed, club data is from 2015/16, Bundesliga coverage is partial.

## Run it yourself
```bash
pip install -r requirements.txt
python -m src.load        # downloads the data (~5 min); add --quick for Premier League only
python -m src.names       # short player names
python -m src.model
python -m src.finishing
streamlit run app.py
```

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Please credit StatsBomb as their licence requires.
