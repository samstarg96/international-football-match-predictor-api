"""Train the match outcome model and save the files the API needs.

Downloads the latest international results, rebuilds each team's Elo rating and recent
form by replaying every match in date order, evaluates the model, then refits it on all
the data. app.py imports match_features and predict_proba from here, so the API builds
features and makes predictions in exactly the same way as training does.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
ANALYSIS_CUTOFF = "2026-06-10"  # end of the data used in my original analysis
CLASSES = ["A", "D", "H"]  # away win, draw, home win (the order sklearn uses)
SIDE_SWAP = {"home_": "away_", "away_": "home_"}

# Elo K-factor (World Football Elo ratings): bigger matches move ratings more. Anything not listed is 30.
FINALS = ["AFC Asian Cup", "African Cup of Nations", "Copa América", "CONCACAF Championship", "Gold Cup",
          "Oceania Nations Cup", "UEFA Euro", "Confederations Cup", "CONMEBOL–UEFA Cup of Champions"]
QUALIFIERS = [f"{name} qualification" for name in FINALS + ["FIFA World Cup"]]
K_FACTOR = {"FIFA World Cup": 60, "Friendly": 20, "UEFA Nations League": 40, "CONCACAF Nations League": 40}
K_FACTOR.update({name: 50 for name in FINALS})
K_FACTOR.update({name: 40 for name in QUALIFIERS})


def outcome(home_score, away_score):
    return "H" if home_score > away_score else "A" if home_score < away_score else "D"


def new_team():
    # form holds the last 5 games as [goals for, goals against, points, Elo change]
    return {"elo": 1500.0, "played": 0, "form": []}


def match_features(home, away, neutral, tournament):
    """Features for one fixture, built only from the two teams' ratings and form before kick-off."""
    row = {"home_elo": home["elo"], "away_elo": away["elo"], "elo_diff": home["elo"] - away["elo"],
           "abs_elo_diff": abs(home["elo"] - away["elo"]), "mean_elo": (home["elo"] + away["elo"]) / 2,
           "k_factor": K_FACTOR.get(tournament, 30), "neutral": int(neutral),
           "is_friendly": int(tournament == "Friendly"), "is_world_cup": int(tournament == "FIFA World Cup")}
    for side, team in (("home", home), ("away", away)):
        form = np.array(team["form"], dtype=float).reshape(-1, 4)
        if len(form) >= 3:
            goals_for, goals_against, points = form[:, :3].mean(axis=0)
            elo_change = form[:, 3].sum()
        else:  # too few recent games: leave missing and let the imputer fill it
            goals_for = goals_against = points = elo_change = np.nan
        row.update({f"{side}_goals_for": goals_for, f"{side}_goals_against": goals_against,
                    f"{side}_goal_diff": goals_for - goals_against, f"{side}_points": points,
                    f"{side}_elo_change": elo_change})
    return row


def update_teams(home, away, home_score, away_score, neutral, tournament):
    """Elo update with a 100-point home advantage; bigger winning margins move ratings more."""
    home_advantage = 0 if neutral else 100
    expected = 1 / (1 + 10 ** (-(home["elo"] - away["elo"] + home_advantage) / 400))
    actual = {"H": 1.0, "D": 0.5, "A": 0.0}[outcome(home_score, away_score)]
    margin = abs(home_score - away_score)
    margin_weight = 1 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8
    change = K_FACTOR.get(tournament, 30) * margin_weight * (actual - expected)
    for team, scored, conceded, delta in [(home, home_score, away_score, change),
                                          (away, away_score, home_score, -change)]:
        points = 3 if scored > conceded else 1 if scored == conceded else 0
        team["elo"] += delta
        team["played"] += 1
        team["form"] = (team["form"] + [[scored, conceded, points, delta]])[-5:]


def replay(matches):
    """Go through every match in date order, recording its features before applying its result."""
    teams, rows = {}, []
    for m in matches.itertuples(index=False):
        home = teams.setdefault(m.home_team, new_team())
        away = teams.setdefault(m.away_team, new_team())
        row = match_features(home, away, m.neutral, m.tournament)
        row["date"] = m.date
        row["result"] = outcome(m.home_score, m.away_score)
        row["provisional"] = min(home["played"], away["played"]) < 30  # ratings need ~30 games to settle
        rows.append(row)
        update_teams(home, away, m.home_score, m.away_score, m.neutral, m.tournament)
    return pd.DataFrame(rows), teams


def make_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))


def swap_sides(df):
    """The same fixtures with the two teams listed the other way round."""
    renamed = {col: SIDE_SWAP[col[:5]] + col[5:] for col in df.columns if col[:5] in SIDE_SWAP}
    return df.rename(columns=renamed)[df.columns].assign(elo_diff=-df["elo_diff"])


def with_mirrored_neutral(data):
    """Add each neutral-venue match a second time with the teams swapped, since order means nothing there."""
    neutral = data[data["neutral"] == 1]
    mirrored = swap_sides(neutral).assign(result=neutral["result"].map({"H": "A", "D": "D", "A": "H"}))
    return pd.concat([data, mirrored], ignore_index=True)


def predict_proba(model, X):
    """Probabilities in CLASSES order. At neutral venues, average both team orders so order can't matter."""
    probs = model.predict_proba(X)
    swapped = model.predict_proba(swap_sides(X))[:, ::-1]  # the swapped fixture's home win is our away win
    return np.where(X[["neutral"]].to_numpy(dtype=bool), (probs + swapped) / 2, probs)


def score(train, test, features):
    """Log loss of the model and of an Elo-only model, both trained on `train` and scored on `test`."""
    train = with_mirrored_neutral(train)
    model = make_model().fit(train[features], train["result"])
    elo_only = make_model().fit(train[["elo_diff"]], train["result"])
    return (log_loss(test["result"], predict_proba(model, test[features]), labels=CLASSES),
            log_loss(test["result"], elo_only.predict_proba(test[["elo_diff"]]), labels=CLASSES))


def evaluate(data, features):
    """Rolling yearly test from 2014, plus a test on matches played after my original analysis."""
    years = list(range(2014, data["date"].max().year))  # complete years only
    yearly = [score(data[data["date"].dt.year < year], data[data["date"].dt.year == year], features)
              for year in years]
    after = data["date"] > ANALYSIS_CUTOFF
    model_after, elo_after = score(data[~after], data[after], features)
    return {"rolling_yearly": {"years": f"{years[0]}-{years[-1]}",
                               "model_log_loss": round(float(np.mean([m for m, _ in yearly])), 4),
                               "elo_only_log_loss": round(float(np.mean([e for _, e in yearly])), 4),
                               "years_model_better": sum(m < e for m, e in yearly)},
            "since_analysis": {"after": ANALYSIS_CUTOFF, "matches": int(after.sum()),
                               "model_log_loss": round(model_after, 4), "elo_only_log_loss": round(elo_after, 4)}}


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else DATA_URL  # or pass a local CSV path
    matches = pd.read_csv(source, parse_dates=["date"]).dropna(subset=["home_score", "away_score"])
    matches = matches[matches["tournament"] != "Olympic Games"].drop_duplicates(["date", "home_team", "away_team"])
    matches = matches.astype({"home_score": int, "away_score": int}).sort_values("date", kind="stable")

    data, teams = replay(matches)
    data = data[(data["date"] >= "2000-01-01") & ~data["provisional"]]
    features = [col for col in data.columns if col not in ("date", "result", "provisional")]

    evaluation = evaluate(data, features)
    print(json.dumps(evaluation, indent=2))
    rolling = evaluation["rolling_yearly"]
    if rolling["model_log_loss"] >= rolling["elo_only_log_loss"]:  # catches a broken pipeline or bad data
        sys.exit("The model is no better than Elo alone, so it has not been saved.")

    everything = with_mirrored_neutral(data)
    model = make_model().fit(everything[features], everything["result"])
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")
    Path("artifacts/teams.json").write_text(json.dumps(teams))
    metadata = {"model_version": pd.Timestamp.now("UTC").strftime("%Y%m%d-%H%M%S"),
                "data_as_of": str(matches["date"].max().date()), "training_matches": len(data),
                "evaluation": evaluation}
    Path("artifacts/metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved a model trained on {len(data):,} matches (data up to {metadata['data_as_of']}).")
