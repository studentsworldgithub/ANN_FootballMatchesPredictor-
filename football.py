################################################################################
############################## This Code and ANN developed By Majdi Awad #######
############################## MIT License #####################################
################################################################################

import numpy as np
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def connect_to_database(host, user, password, database):
    try:
        conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
        cursor = conn.cursor()
        return conn, cursor
    except mysql.connector.Error as e:
        print(f"Error connecting to the database: {e}")
        raise

def fetch_training_data(cursor):
    try:
        query = """
        SELECT matches.home_score, matches.away_score, matches.home_xg, matches.away_xg, 
               matches.home_penalty, matches.away_penalty, teams_ranking.team_rank, teams_ranking.previous_rank,
               teams_ranking.points, teams_ranking.previous_points, world_cup.attendance, world_cup.attendance_avg,
               world_cup.matches
        FROM matches
        JOIN teams_ranking ON matches.home_team_code = teams_ranking.team_code
        JOIN world_cup ON matches.year = world_cup.year
        """

        cursor.execute(query)
        data = cursor.fetchall()
        return np.array(data)
    except mysql.connector.Error as e:
        print(f"Error fetching training data: {e}")
        raise

def preprocess_data(data):
    X = data[:, 2:]  # Features (excluding home_score and away_score)
    y = data[:, :2]  # Labels (home_score and away_score)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_neural_network(X, y):
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
    model.fit(X, y)
    return model

def predict_scores(model, scaler, home_team_code, away_team_code, cursor):
    # Retrieve relevant features for prediction from the database
    query = f"""
    SELECT home_xg, away_xg, home_penalty, away_penalty,
           team_rank, previous_rank, points, previous_points,
           attendance, attendance_avg, matches
    FROM matches
    JOIN teams_ranking ON matches.home_team_code = teams_ranking.team_code
    JOIN world_cup ON matches.year = world_cup.year
    WHERE home_team_code = '{home_team_code}' AND away_team_code = '{away_team_code}'
    """

    cursor.execute(query)
    input_features = np.array(cursor.fetchall())

    if input_features.size == 0:
        print("Data not found for the entered teams.")
        return

    # Check if the number of features matches the model input size
    if input_features.shape[1] != len(model.coefs_[0]):
        print(f"Number of features in the input ({input_features.shape[1]}) does not match the model input size.")
        return

    # Preprocess the input features using the scaler fitted during training
    input_features_scaled = scaler.transform(input_features)

    # Predict home_score and away_score
    prediction = model.predict(input_features_scaled)
    home_score_prediction, away_score_prediction = prediction[0][0], prediction[0][1]

    print(f"Predicted home_score: {home_score_prediction:.2f}, Predicted away_score: {away_score_prediction:.2f}")

def main():
    # Database connection parameters
    host = "localhost"
    user = "root"
    password = "Majdi@00800"
    database = "football"

    # Connect to the database
    conn, cursor = connect_to_database(host, user, password, database)

    try:
        # Fetch training data from the database
        training_data = fetch_training_data(cursor)

        # Preprocess the training data
        X, y, scaler = preprocess_data(training_data)

        # Train the Neural Network model
        model = train_neural_network(X, y)

        # Accept user input for home_team_code and away_team_code
        home_team_code = input("Enter the home_team_code: ")
        away_team_code = input("Enter the away_team_code: ")

        # Predict scores for the entered teams
        predict_scores(model, scaler, home_team_code, away_team_code, cursor)

    finally:
        # Close the database connection
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()

################################################################################
############################## This Code and ANN developed By Majdi Awad #######
############################## MIT License #####################################
################################################################################
