import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Load the dataset
file_path = 'Basketball_dataset.xlsx'  # Replace with your file path
data = pd.ExcelFile(file_path)

# Initialize variables
team_sheets = data.sheet_names[1:]  # Skip the first sheet
team_metrics = {}
march_madness_games = []

# remove seed and training whitespace from team names (because inconsistent seeds caused key errors)
def remove_seed_spaces(team_name):
    if "Saint Peter" in team_name:
        a = 0
    output = ""
    i = 0
    # remove seed and weird backslash characters
    while i < len(team_name):
        if not team_name[i].isalnum() and not team_name[i]==" ":
            break 
        else:
            output = output + team_name[i]
            i+=1
            
    # remove trailing whitespaces
    output = list(output)
    i = len(output)-1
    while i >= 0:
        if output[i] != " ":
            break 
        else:
            del output[i]
            i-=1
    output = "".join(output)
    # edge case for St vs State. Make all State
    if output[-1] == "t" and output[-2] == "S":
        output = output + "ate"
    # UCONN vs Connecticut
    if output == "Connecticut":
        output = "UConn"
    # Western Kentucky vs Kentucky
    if output == "Kentucky":
        output = "Western Kentucky"
    # UNC vs North Carolina
    if output == "North Carolina":
        output = "UNC"
    # McNeese State vs McNeese
    if "McNeese" in output:
        output = "McNeese"
    # Fla Atlantic vs Florida Atlantic
    if "Atlantic" in team_name:
        output = "Fla Atlantic"
    if "Brigham" in output:
        output = "BYU"
    return output

# Process each team's sheet
for sheet in team_sheets:
    # Load team data
    team_data = data.parse(sheet)
    
    # Extract team name from sheet name
    team_name = sheet.split('(')[0].strip()
    team_name = remove_seed_spaces(team_name)

    # Separate regular season and tournament games
    regular_season = team_data[team_data['Type'] != 'NCAA']
    march_madness = team_data[team_data['Type'] == 'NCAA']
    conference_games = team_data[team_data['Type'] == 'CTOURN']
    
    # Compute team metrics (win/loss ratio and point differential) for the regular season
    total_wins = (regular_season['W/L'] == 'W').sum()
    total_games = len(regular_season)
    win_loss_ratio = total_wins / total_games if total_games > 0 else 0
    if win_loss_ratio < 0:
        a = 0
    point_differential = (regular_season['Tm'] - regular_season['Opp']).sum()
    total_points_for = regular_season['Tm']
    w_l_game_num = []
    point_diff = []
    for i in range(0,24):
        point_diff.append(regular_season['Tm'][i] - regular_season['Opp'][i])

        if (regular_season['W/L'] == 'W')[i]:
            w_l_game_num.append(1)
        else:
            w_l_game_num.append(0)
    
        
    
        
    # Save the metrics
    team_metrics[team_name] = {
        'win_loss_ratio': win_loss_ratio,
        'point_differential': point_differential,
        'total_wins': total_wins,
        'W/L_list' : w_l_game_num,
        'point_diff': point_diff,

    }
    
    # Append March Madness games to the list
    for _, game in march_madness.iterrows():
        march_madness_games.append({
            'team': team_name,
            'opponent': remove_seed_spaces(game['Opponent']),
            'W/L': game['W/L'],
            'Tm': game['Tm'],
            'Opp': game['Opp']
        })

# Convert March Madness games to a NumPy array
march_madness_array = np.array([
    [remove_seed_spaces(game['team']), remove_seed_spaces(game['opponent']), game['W/L'], game['Tm'], game['Opp']]
    for game in march_madness_games
], dtype=object)

# Prepare X and Y matrices
features = []
targets = []

for game in march_madness_games:
    team = remove_seed_spaces(game['team'])
    opponent = remove_seed_spaces(game['opponent'])
    
    # Get metrics for both teams
    team_metrics_team = team_metrics[team]
    for key in team_metrics.keys():
        if "Saint P" in key:
            a = 0
    team_metrics_opponent = team_metrics[opponent]
    
    # Calculate feature differences
    win_loss_diff = team_metrics_team['win_loss_ratio'] - team_metrics_opponent['win_loss_ratio']
    point_diff_diff = team_metrics_team['point_differential'] - team_metrics_opponent['point_differential']
    
    # Append features and target
    features.append([win_loss_diff, point_diff_diff,*team_metrics_team['W/L_list'],
                     *team_metrics_team['point_diff'],*team_metrics_opponent['W/L_list'],*team_metrics_opponent['point_diff']])

    targets.append(1 if game['W/L'] == 'W' else 0)
    
# Convert features and targets to NumPy arrays
X = np.array(features)
y = np.array(targets)
print("X.shape " +str(X.shape))
print("y.shape " + str(y.shape))
# Save the processed data (optional)
np.savetxt('X_matrix.csv', X, delimiter=',', header='Win/Loss Ratio Diff,Point Differential Diff', comments='')
np.savetxt('y_vector.csv', y, delimiter=',', header='Outcome', comments='')

print("X and y matrices are ready!")

def create_model(isNN):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if isNN:
        # Build the neural network model
        model = Sequential([
            Dense(64, input_dim=X_train.shape[1], activation='relu'),  # First hidden layer
            Dropout(0.3),  # Dropout layer to prevent overfitting
            Dense(32, activation='relu'),  # Second hidden layer
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

        # Evaluate the model on the test set
        y_pred = (model.predict(X_test) > 0.5).astype(int)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
    else:
        # Create a pipeline with polynomial features and logistic regression
        degree = 4  # You can experiment with different degrees
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('logistic', LogisticRegression(penalty = 'l2',C=5.0))
        ])

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

    print("Model Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", report)

create_model(False)