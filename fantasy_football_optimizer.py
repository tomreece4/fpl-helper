import requests
import pandas as pd
import pulp

FPL_BASE_URL = 'https://fantasy.premierleague.com/api'

def fetch_fpl_data(endpoint: str) -> dict:
    url = f"{FPL_BASE_URL}/{endpoint}/"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def load_raw_data() -> dict:
    bootstrap = fetch_fpl_data('bootstrap-static')
    fixtures = fetch_fpl_data('fixtures')

    players_df = pd.DataFrame(bootstrap['elements'])
    teams_df = pd.DataFrame(bootstrap['teams'])
    fixtures_df = pd.DataFrame(fixtures)

    return {
        'players': players_df,
        'teams': teams_df,
        'fixtures': fixtures_df
    }

def engineer_features(players: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()
    df['cost_m'] = df['now_cost'] / 10
    df['form'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)

    df['next_5_fixt_diff'] = 0  # Placeholder for future fixture difficulty modeling

    return df[['id', 'first_name', 'second_name', 'element_type', 'team', 'cost_m', 'total_points', 'form',
               'next_5_fixt_diff']]

def optimize_team(df: pd.DataFrame, budget: float = 100.0) -> pd.DataFrame:
    players = df['id'].tolist()
    cost = dict(zip(df['id'], df['cost_m']))
    predicted_points = dict(zip(df['id'], df['total_points']))
    positions = dict(zip(df['id'], df['element_type']))
    clubs = dict(zip(df['id'], df['team']))

    model = pulp.LpProblem('FPL_Team_Optimization', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('pick', players, cat='Binary')

    model += pulp.lpSum(predicted_points[i] * x[i] for i in players)
    model += pulp.lpSum(cost[i] * x[i] for i in players) <= budget
    model += pulp.lpSum(x[i] for i in players) == 15

    model += pulp.lpSum(x[i] for i in players if positions[i] == 1) == 2
    model += pulp.lpSum(x[i] for i in players if positions[i] == 2) == 5
    model += pulp.lpSum(x[i] for i in players if positions[i] == 3) == 5
    model += pulp.lpSum(x[i] for i in players if positions[i] == 4) == 3

    for club_id in set(clubs.values()):
        model += pulp.lpSum(x[i] for i in players if clubs[i] == club_id) <= 3

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_ids = [i for i in players if x[i].value() == 1]
    return df[df['id'].isin(selected_ids)]

if __name__ == '__main__':
    data = load_raw_data()
    feats = engineer_features(data['players'], data['fixtures'])
    team = optimize_team(feats)
    print(team[['first_name', 'second_name', 'element_type', 'team', 'cost_m', 'total_points']])
