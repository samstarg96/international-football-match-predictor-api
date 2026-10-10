# Football match predictor API

A small API that predicts international football results. Give it a fixture, such as England v France, and it returns the probability of a home win, a draw and an away win.

It follows on from my earlier analysis, [International football match prediction and World Cup simulation](https://github.com/samstarg96/International-football-match-prediction-and-World-Cup-simulation), where I compared logistic regression, XGBoost, random forests and simple baselines and chose logistic regression. This project is a simplified version of that pipeline: it retrains the chosen model on the latest results and serves its predictions with FastAPI.

## How it works

1. `train.py` downloads the latest results from [martj42/international_results](https://github.com/martj42/international_results) and goes through every match in date order, rebuilding each team's Elo rating and its form over the last five games. Each match's features are recorded before its result is applied, so the model only learns from information that was available before kick-off.
2. It evaluates the model (see below) and stops without saving anything if the model does no better than a simple Elo-only model. This is a basic check before a retrained model replaces the previous one.
3. It refits the model on all the data and saves the model, the current team ratings and the evaluation results to `artifacts/`.
4. `app.py` loads those files and serves predictions, using the same feature code as training.

To pick up new results, rerun `train.py` and restart the API.

## Running it

With Python 3.13 (on Windows, activate with `.venv\Scripts\activate` instead):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py        # downloads the data, evaluates and trains (about 10 seconds)
uvicorn app:app        # then open http://localhost:8000/docs
pytest                 # runs the tests
```

Or with Docker, which trains the model while building the image:

```bash
docker build -t football-predictor-api .
docker run -p 8000:8000 football-predictor-api
```

The page at `/docs` lets you try each endpoint in the browser.

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | Confirms the API is running, with the model version and the date of the latest result used |
| `GET /teams` | Current Elo rankings, which also shows the valid team names |
| `POST /predict` | Home win, draw and away win probabilities for a fixture |
| `GET /performance` | The evaluation results from the last training run |

Example request to `/predict` (`neutral` defaults to `false` and `tournament` to `"Friendly"`):

```json
{"home_team": "England", "away_team": "France"}
```

Response:

```json
{"home_team": "England", "away_team": "France",
 "probabilities": {"home_win": 0.5277, "draw": 0.2777, "away_win": 0.1946}}
```

An unknown team name returns a 404, and the same team on both sides returns a 422.

## The model

The model is a multinomial logistic regression. Its features are both teams' Elo ratings (plus their difference, absolute difference and average), the importance of the match, whether it is at a neutral venue, a friendly or a World Cup match, and each team's average goals scored, goals conceded and points, and total Elo change, over its last five games.

Compared with my original analysis, this version uses the recorded final score rather than reconstructing the 90-minute result, and it updates Elo ratings after each match rather than in daily batches. At neutral venues, each match is used in training in both team orders and predictions are averaged over both orders, so the order the teams are listed in doesn't change the result.

## Evaluation

Results are measured with log loss, where lower is better (guessing a third for each outcome scores 1.099). Each training run reports two tests. These figures are from a run on 10 October 2026. The yearly figures should stay the same until 2026 is complete and added as a new year, while the second test grows as new results come in:

| Test | Matches | Model | Elo only |
|---|---:|---:|---:|
| Each year from 2014 to 2025, trained only on earlier years | about 11,000 | 0.882 | 0.892 |
| Matches after my original analysis (11 June to 6 October 2026) | 470 | 0.874 | 0.880 |

The model beat the Elo-only version in 11 of the 12 years. The improvement is small, about 1%: most of the gain over guessing comes from the Elo ratings themselves.

The yearly test is the more stable of the two, but I used the same years to choose the model in my original analysis, so it is slightly optimistic. The second test uses matches that played no part in choosing the model, so it is a cleaner check, but it is still small and its difference is within noise. This set grows each time the model is retrained on new results, so it also works as a simple way to monitor performance over time.

## Limitations

- The model is retrained whenever `train.py` is run or the Docker image is rebuilt, which downloads the latest results. In a real deployment this would run on a schedule, for example after each international window.
- The data comes from a volunteer-maintained dataset. It is usually updated a few days after each international window, but there is no guarantee.
- Team names must match the dataset exactly (for example "United States", not "USA"). `/teams` lists them, including some defunct teams such as Yugoslavia.

## Files

- `train.py`: downloads the data, builds the features, evaluates and trains the model
- `app.py`: the API
- `test_app.py`: tests for the API (run `train.py` first)
- `Dockerfile` and `.github/workflows/ci.yml`: the container and a GitHub Actions workflow that trains, tests and builds the image on every push

## Data and credits

Match results are from [martj42/international_results](https://github.com/martj42/international_results), released under CC0. The ratings follow the World Football Elo method (match importance weighting, goal margin multiplier and a 100-point home advantage).
