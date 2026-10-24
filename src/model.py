"""Train & evaluate xG models; write out-of-fold predictions for every shot.

Usage:  python -m src.model
Outputs: data/shots_xg.pkl, data/metrics.json, data/calibration.csv
"""
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve

from .features import load, FEATURES

SEED = 42


def make_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def make_lgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=4,
                              min_child_samples=200, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, verbose=-1)


def scores(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score_loss(y, p), 4),
            "auc": round(roc_auc_score(y, p), 4), "pred_goals": round(float(p.sum()), 1),
            "actual_goals": int(y.sum())}


def main():
    d = load()
    pens = d[d["is_penalty"]]
    np_ = d[~d["is_penalty"]].reset_index(drop=True)          # model non-penalty shots
    X, y, g = np_[FEATURES], np_["goal"], np_["match_id"]

    # ---- 1. held-out test (20% of matches, never seen in training) ----
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, g))
    base_rate = y.iloc[tr].mean()
    logit = make_logit().fit(X.iloc[tr], y.iloc[tr])
    gbm = make_lgbm().fit(X.iloc[tr], y.iloc[tr])
    res = {
        "n_shots": int(len(np_)), "n_matches": int(g.nunique()), "n_test_shots": int(len(te)),
        "baseline_constant": scores(y.iloc[te], np.full(len(te), base_rate)),
        "logistic": scores(y.iloc[te], logit.predict_proba(X.iloc[te])[:, 1]),
        "lightgbm": scores(y.iloc[te], gbm.predict_proba(X.iloc[te])[:, 1]),
    }
    sb = np_.iloc[te]["sb_xg"]
    if sb.notna().all():
        res["statsbomb_xg_benchmark"] = scores(y.iloc[te], sb.values)

    # logistic coefficients = interpretable story for the README
    coefs = pd.Series(logit[-1].coef_[0], index=FEATURES).sort_values()
    res["logistic_coefficients_std"] = coefs.round(3).to_dict()
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=FEATURES)
    res["lgbm_gain_importance"] = (imp / imp.sum()).sort_values(ascending=False).round(3).to_dict()

    # calibration on the test set
    rows = []
    for name, p in [("Logistic", logit.predict_proba(X.iloc[te])[:, 1]),
                    ("LightGBM", gbm.predict_proba(X.iloc[te])[:, 1])]:
        frac, mean = calibration_curve(y.iloc[te], p, n_bins=10, strategy="quantile")
        rows += [dict(model=name, predicted=m, observed=f) for m, f in zip(mean, frac)]
    pd.DataFrame(rows).to_csv("data/calibration.csv", index=False)

    # pick the better model on held-out log loss
    best_name = min(["logistic", "lightgbm"], key=lambda k: res[k]["log_loss"])
    make = make_logit if best_name == "logistic" else make_lgbm
    res["selected_model"] = best_name

    # ---- 2. out-of-fold xG for every shot (used for player/team analysis, no leakage) ----
    oof = np.zeros(len(np_))
    for a, b in GroupKFold(n_splits=5).split(X, y, g):
        oof[b] = make().fit(X.iloc[a], y.iloc[a]).predict_proba(X.iloc[b])[:, 1]
    np_["xg"] = oof
    pen_rate = pens["goal"].mean() if len(pens) else 0.76
    pens = pens.assign(xg=pen_rate)
    res["penalty_conversion"] = round(float(pen_rate), 3)

    out = pd.concat([np_, pens], ignore_index=True)
    out.drop(columns=["freeze_frame"]).to_pickle("data/shots_xg.pkl")

    # final model on all data (for the xG calculator in the app)
    joblib.dump(make().fit(X, y), "data/xg_model.joblib")

    with open("data/metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if "coef" not in k and "importance" not in k}, indent=2))


if __name__ == "__main__":
    main()
