"""Prediction API. Loads the model and team ratings saved by train.py and serves forecasts."""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train import match_features, predict_proba

model = joblib.load("artifacts/model.joblib")
teams = json.loads(Path("artifacts/teams.json").read_text())
metadata = json.loads(Path("artifacts/metadata.json").read_text())

app = FastAPI(title="International football match predictor")


class Fixture(BaseModel):
    home_team: str = Field(examples=["England"])
    away_team: str = Field(examples=["France"])
    neutral: bool = False
    tournament: str = Field("Friendly", examples=["Friendly", "FIFA World Cup", "UEFA Euro qualification"])


@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata["model_version"], "data_as_of": metadata["data_as_of"]}


@app.get("/teams")
def rankings():
    ranked = sorted(teams, key=lambda name: teams[name]["elo"], reverse=True)
    return [{"rank": i, "team": name, "elo": round(teams[name]["elo"], 1)} for i, name in enumerate(ranked, start=1)]


@app.post("/predict")
def predict(fixture: Fixture):
    for name in (fixture.home_team, fixture.away_team):
        if name not in teams:
            raise HTTPException(404, f"Unknown team '{name}'. GET /teams lists the valid names.")
    if fixture.home_team == fixture.away_team:
        raise HTTPException(422, "home_team and away_team must be different.")
    row = match_features(teams[fixture.home_team], teams[fixture.away_team], fixture.neutral, fixture.tournament)
    away_win, draw, home_win = predict_proba(model, pd.DataFrame([row]))[0]
    return {"home_team": fixture.home_team, "away_team": fixture.away_team,
            "probabilities": {"home_win": round(home_win, 4), "draw": round(draw, 4), "away_win": round(away_win, 4)}}


@app.get("/performance")
def performance():
    return metadata["evaluation"]
