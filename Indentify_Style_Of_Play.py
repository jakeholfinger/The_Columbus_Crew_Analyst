import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

REQUIRED_MATCH_FILES = {
    'Match_Attributes.csv', 'Average_Positions.csv', 'Full_Team_Statistics.csv',
    'Match_Momentum.csv', 'Player_Event_Data.csv', 'Player_Heatmaps.csv',
    'Player_Statistics.csv', 'Shot_Map.csv',
}

def get_files(currentPath):
    '''Returns a list of all visible files inside the folder at currentPath'''
    return [f for f in os.listdir(currentPath) if not f.startswith('.')]

def get_data(year, league, team, date):

    # Gather team's match data for the season
    team_path = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data/{league.replace(" ", "_")}_Data/{team.replace(" ", "_")}_Data'
    matches_data = {}
    for match_name in get_files(team_path):
        match_path = os.path.join(team_path, match_name)
        match_data = {}
        for file_name in get_files(match_path):
            match_data[file_name] = pd.read_csv(os.path.join(match_path, file_name))

        matches_data[match_name] = match_data

    # Folders are named "M-D-YYYY_vs_Opponent" (e.g. "2-21-2026_vs_Portland_Timbers").
    # Parse the date prefix with strptime so we sort chronologically within a year.
    def folder_date(folder_name):
        return datetime.strptime(folder_name.split('_')[0], '%m-%d-%Y')

    sorted_matches = dict(sorted(matches_data.items(), key=lambda item: folder_date(item[0])))

    # Keep only matches that kicked off strictly before the target date.
    month, day, yr = date.split('-')
    target_date = datetime(int(yr), int(month), int(day)).date()
    filtered_matches = {
        folder: data for folder, data in sorted_matches.items()
        if folder_date(folder).date() < target_date
    }

    return filtered_matches

def get_league_features(year, league, date):
    '''Extracts match features for every team in the league, as a scaling reference'''
    league_path = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data/{league.replace(" ", "_")}_Data'
    league_features = []
    for team_folder in get_files(league_path):
        team = team_folder.removesuffix('_Data')
        team_matches = get_data(year, league, team, date)
        for match_folder, match_data in team_matches.items():
            if not REQUIRED_MATCH_FILES.issubset(match_data.keys()):
                print(f'Skipping {team_folder}/{match_folder} for league scaling (missing CSVs)')
                continue
            league_features.append(extract_features(match_data, team))

    return pd.DataFrame(league_features)

def safe_div(numerator, denominator):
    '''Returns NaN instead of raising ZeroDivisionError when denominator is 0'''
    return numerator / denominator if denominator else np.nan

def compute_momentum_change_rate(momentum, threshold=5):

    filtered_momentum = momentum[momentum['value'].abs() >= threshold].copy()

    filtered_momentum['sign'] = np.where(filtered_momentum['value'] > 0, 1, -1)

    filtered_momentum['previous_sign'] = filtered_momentum['sign'].shift(1)

    filtered_momentum['momentum_change'] = filtered_momentum['sign'] + filtered_momentum['previous_sign']

    num_changes = (filtered_momentum['momentum_change'] == 0).sum()

    change_rate = num_changes / len(momentum)

    return change_rate

def extract_features(match_data, team):

    team = team.replace('_', ' ')

    match_attributes = match_data['Match_Attributes.csv']
    avg_positions = match_data['Average_Positions.csv']
    team_stats = match_data['Full_Team_Statistics.csv']
    momentum = match_data['Match_Momentum.csv']
    events = match_data['Player_Event_Data.csv']
    player_heatmaps = match_data['Player_Heatmaps.csv']
    player_stats = match_data['Player_Statistics.csv']
    shots = match_data['Shot_Map.csv']

    features = {}

    team_row = team_stats['Team Name'] == team

    def team_stat(column):
        return team_stats.loc[team_row, column].iloc[0]

    # POSSESSION & PASS STYLE

    features['possession_pct'] = team_stat('Ball possession')

    team_touches = player_stats['touches'].sum()
    features['touches_in_opp_penalty_area_per_touch'] = team_stat('Touches in penalty area') / team_touches
    features['final_third_entries_per_touch'] = team_stat('Final third entries') / team_touches

    passes = events[events['Event Type'] == 'pass']
    num_prg_passes = len(passes[passes['Pass End X Coord'] > passes['Player X Coord']])
    num_passes = len(passes)
    features['prg_pass_pct'] = safe_div(num_prg_passes, num_passes)

    num_opp_half_passes = len(passes[passes['Player X Coord'] > 50.0])
    features['opp_half_pass_pct'] = safe_div(num_opp_half_passes, num_passes)

    prg_carries = events[events['Event Type'] == 'ball-carry']
    features['avg_prg_carry_gain'] = (prg_carries['Pass End X Coord'] - prg_carries['Player X Coord']).mean()


    # DEFENSE

    features['recoveries'] = team_stat('Recoveries')
    features['tackles'] = team_stat('Total tackles')
    features['interceptions'] = team_stat('Interceptions')

    def_actions = events[events['Event Type'] == 'Defensive Action']
    features['avg_x_def_action'] = def_actions['Player X Coord'].mean()

    features['avg_def_position'] = avg_positions[avg_positions['player.position'] == 'D']['averageX'].mean()

    gk_stats = player_stats[player_stats['Position'] == 'G']
    if 'totalKeeperSweeper' in player_stats.columns:
        features['num_sweeper_keeper'] = gk_stats['totalKeeperSweeper'].sum()
    else:
        features['num_sweeper_keeper'] = 0


    # DIRECTNESS

    features['aerial_duels'] = team_stat('Aerial duels')

    pass_dist = np.sqrt((passes['Pass End X Coord'] - passes['Player X Coord'])**2 + (passes['Pass End Y Coord'] - passes['Player Y Coord'])**2)
    features['avg_pass_dist'] = pass_dist.mean()

    gk_passes = passes[passes['Player'].isin(gk_stats['Player Name'])]
    gk_pass_dist = np.sqrt((gk_passes['Pass End X Coord'] - gk_passes['Player X Coord'])**2 + (gk_passes['Pass End Y Coord'] - gk_passes['Player Y Coord'])**2)
    features['avg_gk_pass_length'] = gk_pass_dist.mean()

    features['long_ball_pct'] = safe_div(team_stat('Long balls'), team_stat('Passes'))

    # TRANSITION SPEED
    is_home = match_attributes['homeTeam.name'].iloc[0] == team
    fast_break_shots = shots[(shots['situation'] == 'fast-break') & (shots['isHome'] == is_home)]
    features['fast_break_shot_pct'] = safe_div(len(fast_break_shots), len(shots))

    features['momentum_volatility'] = momentum['value'].std()
    features['momentum_changes'] = compute_momentum_change_rate(momentum)

    # WIDTH
    features['num_crosses'] = team_stat('Crosses')
    features['team_width'] = player_heatmaps['Y'].std()

    # SET PIECES?
    
    return features

def fit_scaler(reference_df):
    scaler = MinMaxScaler()
    scaler.fit(reference_df)

    return scaler

def normalize_features(df, scaler):

    scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return scaled_df

def score_axes(df):

    # POSSESSION
    df['possession_score'] = (
        (df['possession_pct'] * 0.24)
        + (df['prg_pass_pct'] * 0.22)
        + (df['opp_half_pass_pct'] * 0.18)
        + (df['final_third_entries_per_touch'] * 0.14)
        + (df['touches_in_opp_penalty_area_per_touch'] * 0.14)
        + (df['avg_prg_carry_gain'] * 0.08)
    )

    # DEFENSE
    df['defense_score'] = (
        (df['avg_x_def_action'] * 0.40)
        + (df['avg_def_position'] * 0.25)
        + (df['recoveries'] * 0.20)
        + (df['tackles'] * 0.075)
        + (df['interceptions'] * 0.075)
    )

    # DIRECTNESS
    df['direct_score'] = (
        (df['long_ball_pct'] * 0.30)
        + (df['avg_gk_pass_length'] * 0.30)
        + (df['avg_pass_dist'] * 0.25)
        + (df['aerial_duels'] * 0.15)
    )

    # TRANSITION
    df['transition_score'] = (
        (df['fast_break_shot_pct'] * 0.45)
        + (df['momentum_volatility'] * 0.30)
        + (df['momentum_changes'] * 0.25)
    )

    # WIDTH
    df['width_score'] = (
        (df['team_width'] * 0.60)
        + (df['num_crosses'] * 0.40)
    )

    # SET PIECES

    return df

CENTROIDS = {
    'Tiki-Taka':          [0.88, 0.40, 0.08, 0.15, 0.25],
    'Vertical Possession':[0.72, 0.60, 0.12, 0.25, 0.35],
    'High Press':         [0.55, 0.75, 0.25, 0.50, 0.32],
    'Wide/Crossing':      [0.52, 0.42, 0.35, 0.38, 0.90],
    'Counter-Attack':     [0.35, 0.18, 0.45, 0.88, 0.38],
    'Direct':             [0.42, 0.30, 0.88, 0.42, 0.48],
    'Low Block':          [0.28, 0.08, 0.42, 0.22, 0.28],
}

def classify_matches(df):

    axis_cols = ['possession_score', 'defense_score', 'direct_score', 'transition_score', 'width_score']
    archetypes = []

    for __, match in df[axis_cols].iterrows():

        team_vector = match.values.astype(float)

        centroid_distances = {
            archetype: np.linalg.norm(team_vector - np.array(centroid_vector))
            for archetype, centroid_vector in CENTROIDS.items()
        }

        archetypes.append(min(centroid_distances, key=centroid_distances.get))

    df = df.copy()
    df['archetype'] = archetypes

    return df

def classify_team_style(df, half_life=5):

    team_style_dict = {}

    # Rows are in ascending chronological order (oldest first, most recent last —
    # matching the convention in Pre_Match_Report.loadMatchData), so games_ago must
    # count down to 0 at the last row, not up from it.
    games_ago = np.arange(len(df))[::-1]
    weights = 0.5 ** (games_ago / half_life)
    df['weight'] = weights

    # Weighted axis scores for radar chart
    axis_cols = ['possession_score', 'defense_score', 'direct_score', 'transition_score', 'width_score']
    weighted_axis_scores = (df[axis_cols].multiply(df['weight'], axis=0)).sum() / df['weight'].sum()
    team_style_dict['axis_scores'] = weighted_axis_scores.to_dict()

    # Weighted archetype
    weighted_archetypes = df.groupby('archetype')['weight'].sum().sort_values(ascending=False)
    team_style_dict['primary_archetype'] = weighted_archetypes.index[0]
    team_style_dict['secondary_archetype'] = weighted_archetypes.index[1]

    # Weighted archetype consistency
    team_style_dict['primary_archetype_consistency'] = (df['archetype'] == team_style_dict['primary_archetype']).mean()
    team_style_dict['secondary_archetype_consistency'] = (df['archetype'] == team_style_dict['secondary_archetype']).mean()

    return team_style_dict

def main(matches_data=None, year=2026, league='MLS', team='Columbus_Crew', date='7-22-2026'):

    if matches_data is None:
        matches_data = get_data(year, league, team, date)

    matches_features = []

    for match_folder, match_data in matches_data.items():

        if not REQUIRED_MATCH_FILES.issubset(match_data.keys()):
            print(f'Skipping {match_folder} (missing CSVs)')
            continue

        match_features = extract_features(match_data, team)

        matches_features.append(match_features)

    feature_df = pd.DataFrame(matches_features)

    league_feature_df = get_league_features(year, league, date)
    scaler = fit_scaler(league_feature_df)
    scaled_df = normalize_features(feature_df, scaler)

    scored_df = score_axes(scaled_df)

    match_classifications = classify_matches(scored_df)

    team_style = classify_team_style(match_classifications)

    return team_style

if __name__ == "__main__":
    main()