import requests
import pandas as pd
import pulp

# Importing the data from free FPL API
FPL_BASE_URL = 'https://fantasy.premierleague.com/api'


# Fetching the data from specified endpoints
def fetch_fpl_data(endpoint: str) -> dict:
    url = f"{FPL_BASE_URL}/{endpoint}/"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


# Loading the player, team and fixture details into a dictionary of data frames
def load_raw_data() -> dict:
    # Fetch JSON
    bootstrap = fetch_fpl_data('bootstrap-static')
    fixtures = fetch_fpl_data('fixtures')

    # Normalize into DataFrames
    players_df = pd.DataFrame(bootstrap['elements'])
    teams_df = pd.DataFrame(bootstrap['teams'])
    fixtures_df = pd.DataFrame(fixtures)

    return {
        'players': players_df,
        'teams': teams_df,
        'fixtures': fixtures_df
    }


# Features

def engineer_features(players: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()
    # Rename cost (prices are in tenths of million)
    df['cost_m'] = df['now_cost'] / 10
    # Total points last season
    df['total_points'] = df['total_points']
    # Basic form: average points over last 5 games
    df['form'] = df['form'].astype(float)

    # Placeholder for fixture difficulty
    df['next_5_fixt_diff'] = 0

    return df[['id', 'first_name', 'second_name', 'element_type', 'team', 'cost_m', 'total_points', 'form',
               'next_5_fixt_diff']]


# Optimisation model

def optimize_team(df: pd.DataFrame, budget: float = 100.0) -> pd.DataFrame:
    """
    Solves ILP to pick 15 players within budget maximizing predicted points.
    """
    players = df['id'].tolist()
    cost = dict(zip(df['id'], df['cost_m']))
    # Here use 'total_points' as a placeholder for predicted points
    predicted_points = dict(zip(df['id'], df['total_points']))
    positions = dict(zip(df['id'], df['element_type']))  # 1=GK,2=DEF,3=MID,4=FWD
    clubs = dict(zip(df['id'], df['team']))

    # Initialize model
    model = pulp.LpProblem('FPL_Team_Optimization', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('pick', players, cat='Binary')

    # Objective: maximize total predicted points
    model += pulp.lpSum(predicted_points[i] * x[i] for i in players)

    # Budget constraint
    model += pulp.lpSum(cost[i] * x[i] for i in players) <= budget

    # Squad size
    model += pulp.lpSum(x[i] for i in players) == 15

    # Position constraints
    model += pulp.lpSum(x[i] for i in players if positions[i] == 1) == 2  # GK
    model += pulp.lpSum(x[i] for i in players if positions[i] == 2) == 5  # DEF
    model += pulp.lpSum(x[i] for i in players if positions[i] == 3) == 5  # MID
    model += pulp.lpSum(x[i] for i in players if positions[i] == 4) == 3  # FWD

    # Max 3 per club
    for club_id in set(clubs.values()):
        model += pulp.lpSum(x[i] for i in players if clubs[i] == club_id) <= 3

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_ids = [i for i in players if x[i].value() == 1]
    return df[df['id'].isin(selected_ids)]


# Main execution

if __name__ == '__main__':
    data = load_raw_data()
    feats = engineer_features(data['players'], data['fixtures'])
    team = optimize_team(feats)
    print(team[['first_name', 'second_name', 'element_type', 'team', 'cost_m', 'total_points']])
