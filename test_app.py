"""Tests for the prediction API. Run `python train.py` first so the model files exist."""
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def predict(home, away, **options):
    return client.post("/predict", json={"home_team": home, "away_team": away, **options})


def test_probabilities_are_valid():
    assert client.get("/health").json()["status"] == "ok"
    probs = predict("England", "France").json()["probabilities"]
    assert abs(sum(probs.values()) - 1) < 1e-3
    assert all(0 < p < 1 for p in probs.values())


def test_bad_requests_are_rejected():
    assert predict("Englund", "France").status_code == 404
    assert predict("England", "England").status_code == 422
    assert predict("England", "France", neutral="sometimes").status_code == 422


def test_stronger_team_and_home_side_are_favoured():
    ranked = client.get("/teams").json()
    probs = predict(ranked[0]["team"], ranked[-1]["team"], neutral=True).json()["probabilities"]
    assert probs["home_win"] > probs["away_win"]
    at_home = predict("England", "France").json()["probabilities"]["home_win"]
    at_neutral_venue = predict("England", "France", neutral=True).json()["probabilities"]["home_win"]
    assert at_home > at_neutral_venue


def test_team_order_does_not_matter_at_a_neutral_venue():
    first = predict("England", "France", neutral=True).json()["probabilities"]
    second = predict("France", "England", neutral=True).json()["probabilities"]
    assert first["home_win"] == second["away_win"] and first["draw"] == second["draw"]


def test_performance_reports_both_evaluations():
    performance = client.get("/performance").json()
    assert performance["rolling_yearly"]["model_log_loss"] < performance["rolling_yearly"]["elo_only_log_loss"]
    assert performance["since_analysis"]["matches"] > 0
