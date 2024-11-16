import numpy as np
import pandas as pd

# Load the dataset
file_path = 'Basketball_dataset.xlsx'  # Replace with your file path
data = pd.ExcelFile(file_path)

# Initialize variables
team_sheets = data.sheet_names[1:]  # Skip the first sheet
team_metrics = {}
march_madness_games = []

# remove seed and training whitespace from team names (because inconsistent seeds caused key errors)
def remove_seed_spaces(team_name):
    output = ""
    i = 0
    # remove seed and weird backslash characters
    while i < len(team_name):
        if team_name[i] == "(" or "\\":
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
    return "".join(output)


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
    
    # Compute team metrics (win/loss ratio and point differential) for the regular season
    total_wins = (regular_season['W/L'] == 'W').sum()
    total_games = len(regular_season)
    win_loss_ratio = total_wins / total_games if total_games > 0 else 0
    point_differential = (regular_season['Tm'] - regular_season['Opp']).sum()

    # Save the metrics
    team_metrics[team_name] = {
        'win_loss_ratio': win_loss_ratio,
        'point_differential': point_differential
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
    team_metrics_opponent = team_metrics[opponent]
    
    # Calculate feature differences
    win_loss_diff = team_metrics_team['win_loss_ratio'] - team_metrics_opponent['win_loss_ratio']
    point_diff_diff = team_metrics_team['point_differential'] - team_metrics_opponent['point_differential']
    
    # Append features and target
    features.append([win_loss_diff, point_diff_diff])
    targets.append(1 if game['W/L'] == 'W' else 0)

# Convert features and targets to NumPy arrays
X = np.array(features)
y = np.array(targets)

# Save the processed data (optional)
np.savetxt('X_matrix.csv', X, delimiter=',', header='Win/Loss Ratio Diff,Point Differential Diff', comments='')
np.savetxt('y_vector.csv', y, delimiter=',', header='Outcome', comments='')

print("X and y matrices are ready!")
