Bettor Project README
Overview
The Bettor project is a Python-based tool designed for predictive analysis and Elo-based ranking in football (soccer). It provides functionality to manage and update Elo ratings, train machine learning models for match outcome prediction, and calculate probabilities for match outcomes using various algorithms.

Features
Elo Rating Management:

Update team Elo ratings based on match outcomes.
Initialize or reset Elo ratings for teams.
Store and manage Elo history.
Data Management:

Automatically download football match data from a specified source.
Process historical match data and prepare it for training models.
Prediction Models:

Train and compare machine learning models (DecisionTreeClassifier, XGBoost).
Predict match outcomes using trained models or Bradley-Terry odds.
Utility Functions:

Load, process, and store data in a structured format.
Serialize and deserialize models and data using JSON and Pickle.

Usage
1. Initialize the Bettor Class
python
Copy code
from bettor import Bettor

bettor = Bettor(path="/path/to/project_root")
2. Update Elo Ratings
python
Copy code
elo_dict = bettor.update_all_elos(elo_dict={})
3. Make Predictions
python
Copy code
bettor.make_prediction(HomeTeam="TeamA", AwayTeam="TeamB")
4. Train a Model
python
Copy code
bettor.train_model(max_depth=5)
5. Calculate Odds
python
Copy code
odds_ml = bettor.get_odds_ml("TeamA", "TeamB", elo_dict, model)
odds_bt = bettor.calculate_odds_bradley("TeamA", "TeamB", elo_dict)
Folder Structure
current/: Contains the latest downloaded match data.
prev/: Contains historical match data for training.
league.json: Configuration for league codes and identifiers.
last_update.json: Tracks the last data update date and current season.
elo_dict.json: Stores the Elo ratings of teams.
bettor.pkl: Serialized machine learning model.
Key Functions
update_all_elos: Updates Elo ratings for all teams based on match results.
make_prediction: Predicts the outcome of a match.
train_model: Trains and serializes the best machine learning model.
download_new_data: Fetches the latest match data from a specified source.
calculate_odds_bradley: Calculates match odds using the Bradley-Terry model.
