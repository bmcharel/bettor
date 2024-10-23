import pandas as pd
import numpy as np
import os
import xgboost as xgb
import requests
from datetime import *
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import re
import requests

class Bettor:

    def __init__(self, path):
        
        if not os.path.exists(path):
            raise ValueError(f"The folder {path} does not exist.")
        self.path = path  # Initialize folder path
        
    def update_all_elos(self, elo_dict):

        self.download_new_data()

        folder_path = f'{self.path}/current/'
        
        Files = self.get_all_csv_files_in_folder(folder_path)

        df = self.load_data(folder_path, files=Files, edit_global = True)
        Teams = df['HomeTeam'].unique()
        for team in Teams:
            if team not in elo_dict:
                elo_dict[team] = {'elo': 1000, 'prev_date': "2016-01-01"}
        
        Teams = df['AwayTeam'].unique()
        for team in Teams:
            if team not in elo_dict:
                elo_dict[team] = {'elo': 1000, 'prev_date': "2016-01-01"}

        # Load the JSON data from the file
        last_update = self.load_json_as_dict(file_path=f'{self.path}/last_update.json')
        
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        # Process each match
        for index, row in df.iterrows():
            #prev = elo_dict[row['HomeTeam']]['prev_date']
            prev = pd.to_datetime(elo_dict[row['HomeTeam']]['prev_date']).date()
            if pd.notna(row['Date']):
                if row['Date'] > prev:   
                    self.update_elo(row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG'], elo_dict)
        
        # Update the date field with today's date
        last_update['last_update'] = str(date.today())
        
        # Write the updated JSON data back to the file
        with open(self.path+'/last_update.json', 'w') as json_file:
            json.dump(last_update, json_file, indent=4)

        # Open a file in write mode and save the dictionary as JSON
        with open(self.path+'/elo_dict.json', 'w') as json_file:
            json.dump(elo_dict, json_file, indent=4)  # 'indent=4' makes the JSON readable

        return elo_dict

    def make_prediction(self, HomeTeam, AwayTeam):
        # Load the JSON data from the file
        elo_dict = self.load_json_as_dict(file_path=f'{self.path}/elo_dict.json')
        
        last_update = self.read_date_from_json(file_path=f'{self.path}/last_update.json')
        last_update = datetime.strptime(last_update, '%Y-%m-%d').date()

        if last_update < date.today():
            elo_dict = self.update_all_elos(elo_dict=elo_dict)

        model = pickle.load(open(self.path+'/bettor.pkl', 'rb'))

        print(f"ML odds{self.get_odds_ml(HomeTeam, AwayTeam, elo_dict, model=model)}")
        
        print(f"Bradley Terry odds{self.calculate_odds_bradley(HomeTeam, AwayTeam, elo_dict)}")


    def train_model(self, max_depth):
        folder_path = f'{self.path}/prev/'
        files = ['ENG1_2324.csv', 'ENG1_2223.csv', 'ENG1_2122.csv', 'ENG1_2021.csv', 'ENG1_1920.csv', 'ENG1_1819.csv']
        
        df = self.load_data(folder_path, files=files, edit_global=True)

        elo_dict = self.initialize_elo(df=df, training=True)

        df['Result'] = df.apply(lambda row: self.determine_victor(row['FTHG'], row['FTAG']), axis=1)

        # Initialize ELO ratings before the first game
        df['HomeElo'] = df['HomeTeam'].apply(lambda x: elo_dict[x])
        df['AwayElo'] = df['AwayTeam'].apply(lambda x: elo_dict[x])
        df = df.reset_index(drop=True)
        # Process each match

        current_season = None
        
        # Process each match
        for index, row in df.iterrows():
            season = row['season']
            match_date = str(row['Date'].date())

            if current_season is None:
                current_season = season
            
            elif season != current_season:
                for Team in elo_dict:
                    elo_dict[Team] = (elo_dict[Team] + 1000) / 2
                
                current_season = season

            self.update_elo_training(row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG'], elo_dict)

            #Update the Elo ratings in the DataFrame for the next matches
            if index + 1 < len(df):
                df.loc[index + 1, 'HomeElo'] = elo_dict[df.loc[index + 1, 'HomeTeam']]
                df.loc[index + 1, 'AwayElo'] = elo_dict[df.loc[index + 1, 'AwayTeam']]

        # Filter to create training data: Games before 2024
        training_data = df[(df['Date'] < pd.Timestamp('2023-08-15'))]

        # Filter to create test data: Games from 2024 onwards
        test_data = df[df['Date'] >= pd.Timestamp('2023-08-15')]

        X_train = training_data.drop(columns=['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result', 'season'])
        y_train_scored = training_data['Result']
        X_test = test_data.drop(columns=['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result', 'season'])
        y_test= test_data['Result']

        #Train the tree
        dt_classifier = DecisionTreeClassifier(max_depth=max_depth)
        dt_classifier.fit(X_train, y_train_scored)

        y_pred = dt_classifier.predict(X_test)

        accuracy_dt = accuracy_score(y_test, y_pred)

        #Train XGB
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train, y_train_scored)

        y_pred = xgb_model.predict(X_test)

        accuracy_xgb = accuracy_score(y_test, y_pred)
        print(accuracy_dt)
        print(accuracy_xgb)

        if accuracy_dt > accuracy_xgb:
            pickle.dump(dt_classifier, open(self.path+'/bettor.pkl', 'wb'))
        else:
            pickle.dump(dt_classifier, open(self.path+'/bettor.pkl', 'wb'))
        

    def initialize_elo(self, df, training=False):

        Teams = df['HomeTeam'].unique()
        # Initialize Elo ratings
        if training:
            elo_dict = {team: 1000 for team in Teams}
        else:
            elo_dict = {team: {'elo': 1000, 'prev_date': "2016-01-01"} for team in Teams}

        return elo_dict

    def create_global_elo_history(self):
        folder_path = f'{self.path}/prev/'
        Files = self.get_all_csv_files_in_folder(folder_path)
        df = self.load_data(folder_path, files=Files, edit_global=True)

        elo_dict = self.initialize_elo(df=df)

        current_season = None
        
        # Process each match
        for index, row in df.iterrows():
            season = row['season']
            match_date = str(row['Date'].date())

            if current_season is None:
                current_season = season
            
            elif season != current_season:
                for key in elo_dict:
                    elo_dict[key]['elo'] = (elo_dict[key]['elo'] + 1000) / 2
                
                current_season = season

            self.update_elo(row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG'], elo_dict)

            # Update the 'last_game_date' for both teams
            elo_dict[row['HomeTeam']]['prev_date'] = match_date
            elo_dict[row['AwayTeam']]['prev_date'] = match_date
        
        for key in elo_dict:
            elo_dict[key]['elo'] = (elo_dict[key]['elo'] + 1000) / 2

        # Open a file in write mode and save the dictionary as JSON
        file_path = f'{self.path}/elo_dict.json'
        with open(file_path, 'w') as json_file:
            json.dump(elo_dict, json_file, indent=4)  # 'indent=4' makes the JSON readable

    def load_data(self, folder_path, files, edit_global=False):

        dataframes = []
        
        for file in files:
            os.chdir(folder_path)
            df_file = pd.read_csv(file, encoding='ISO-8859-1', sep=',')
            if edit_global:
                df_file['season'] = os.path.basename(file)[-8:-4]
            dataframes.append(df_file)
        
        df_combined = pd.concat(dataframes, axis=0, ignore_index=True)

        df_combined['Date'] = pd.to_datetime(df_combined['Date'], format='%d/%m/%Y')
        df_combined = df_combined.sort_values(by='Date', ascending=True)

        if edit_global:
            df_compact = df_combined[['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'season']]
        else:
            df_compact = df_combined[['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]

        return df_compact

    def update_elo(self, home_team, away_team, home_goals, away_goals, elo_dict):
        home_elo = elo_dict[home_team]['elo']
        away_elo = elo_dict[away_team]['elo']
        elo_diff = (home_elo * 1.025) - away_elo

        # Calculate the adjustment factor based on elo difference
        adjustment_factor = elo_diff // 20
        if adjustment_factor > 50:
            adjustment_factor == 50

        goal_factor = home_goals - away_goals

        if home_goals > away_goals:  # Home win
            elo_dict[home_team]['elo'] += 50 - (adjustment_factor + goal_factor)
            elo_dict[away_team]['elo'] -= 50 + (adjustment_factor - goal_factor)
        elif home_goals < away_goals:  # Away win
            elo_dict[home_team]['elo'] -= 50 + (adjustment_factor + goal_factor)
            elo_dict[away_team]['elo'] += 50 - (adjustment_factor - goal_factor)
        else:  # Draw
            if adjustment_factor > 0:  # Home was stronger
                elo_dict[home_team]['elo'] -= adjustment_factor
                elo_dict[away_team]['elo'] += adjustment_factor
            elif adjustment_factor < 0:  # Away was stronger
                elo_dict[home_team]['elo'] += -adjustment_factor
                elo_dict[away_team]['elo'] -= -adjustment_factor

        if elo_dict[home_team]['elo'] < 0:
            elo_dict[home_team]['elo'] == 0
        if elo_dict[away_team]['elo'] < 0:
            elo_dict[away_team]['elo'] == 0

    def read_date_from_json(self, file_path, name='last_update'):
        # Load the JSON data from the file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Extract and return the date field
        return data.get(name)

    def update_elo_training(self, home_team, away_team, home_goals, away_goals, elo_dict):
        home_elo = elo_dict[home_team]
        away_elo = elo_dict[away_team]
        elo_diff = (home_elo * 1.025) - away_elo

        # Calculate the adjustment factor based on elo difference
        adjustment_factor = elo_diff // 20
        if adjustment_factor > 50:
            adjustment_factor == 50

        goal_factor = home_goals - away_goals

        if home_goals > away_goals:  # Home win
            elo_dict[home_team] += 50 - (adjustment_factor + goal_factor)
            elo_dict[away_team] -= 50 + (adjustment_factor - goal_factor)
        elif home_goals < away_goals:  # Away win
            elo_dict[home_team] -= 50 + (adjustment_factor + goal_factor)
            elo_dict[away_team] += 50 - (adjustment_factor - goal_factor)
        else:  # Draw
            if adjustment_factor > 0:  # Home was stronger
                elo_dict[home_team] -= adjustment_factor
                elo_dict[away_team] += adjustment_factor
            elif adjustment_factor < 0:  # Away was stronger
                elo_dict[home_team] += -adjustment_factor
                elo_dict[away_team] -= -adjustment_factor

        if elo_dict[home_team] < 0:
            elo_dict[home_team] == 0
        if elo_dict[away_team] < 0:
            elo_dict[away_team] == 0

    def read_date_from_json(self, file_path, name='last_update'):
        # Load the JSON data from the file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Extract and return the date field
        return data.get(name)

    def load_json_as_dict(self, file_path):
        # Open the JSON file and load its contents into a Python dictionary
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data

    def determine_victor(self, x, y):
        if x == y:
            return 1
        elif x > y:
            return 0
        else:
            return 2
        
    def download_new_data(self):
        #should download the files from the website and store them in the current folder. Files should replace existing files with the same names#
        base_url = "https://www.football-data.co.uk/mmz4281"

        league = self.load_json_as_dict(file_path=f'{self.path}/league.json')

        for league_letter, league_code in league.items():
            save_folder = f'{self.path}/current/'
            dict = self.load_json_as_dict(file_path=f'{self.path}/last_update.json')
            current_season = dict['season']
            start_year = int(current_season[:2])
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'
            }

            # Increment the start year
            next_start_year = start_year + 1

            # Construct the next season identifier
            next_season = f"{next_start_year:02d}{(next_start_year + 1) % 100:02d}"

            url = f"{base_url}/{next_season}/{league_letter}.csv"
            file_name = f"{league_code}_{next_season}.csv"
            file_path = os.path.join(save_folder, file_name)
            
            #try:
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                file_path = os.path.join(save_folder, file_name)
                url = f"{base_url}/{current_season}/{league_letter}.csv"
                file_name = f"{league_code}_{current_season}.csv"
                file_path = os.path.join(save_folder, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                save_folder = f'{self.path}/prev/'
                file_path = os.path.join(save_folder, file_name)

                response = requests.get(url, headers=headers)
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                
                dict['season'] = next_season
                with open(self.path+'/last_update.json', 'w') as json_file:
                    json.dump(dict, json_file, indent=4)

            else:
                url = f"{base_url}/{current_season}/{league_letter}.csv"
                file_name = f"{league_code}_{current_season}.csv"
                file_path = os.path.join(save_folder, file_name)
                response = requests.get(url, headers=headers)
                with open(file_path, 'wb') as file:
                    file.write(response.content)

        print("Download Done")

                
                    
    def get_odds_ml(self, HomeTeam, AwayTeam, elo_dict, model):
        data = {    'HomeElo':[elo_dict[HomeTeam]['elo']],
                    'AwayElo':[elo_dict[AwayTeam]['elo']],
        }

        df = pd.DataFrame(data)

        prediction = model.predict_proba(df)

        return prediction

    def calculate_odds_bradley(self, HomeTeam, AwayTeam, elo_dict):
        Home_elo = elo_dict[HomeTeam]['elo'] * 1.025
        Away_elo = elo_dict[AwayTeam]['elo']

        Home_odds = Home_elo/(Home_elo + Away_elo)
        Away_odds = Away_elo/(Home_elo + Away_elo)
        Tie_odds = (1 - (1.75*abs(Home_odds - Away_odds))) * 0.33

        Final_home = (1 - Tie_odds) * Home_odds
        Final_away = (1 - Tie_odds) * Away_odds

        return Final_home, Tie_odds, Final_away

    def get_all_csv_files_in_folder(self, folder_path):
        # Get all files in the specified folder
        all_files = os.listdir(folder_path)
        
        # Filter only CSV files
        csv_files = [os.path.join(folder_path, f) for f in all_files if f.endswith('.csv')]
        
        return csv_files