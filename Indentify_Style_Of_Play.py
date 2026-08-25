import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# The files every feature below is read from. Split in two because SofaScore's coverage is not
# uniform across the competitions this report is run over.
#
# CORE is what a match needs before anything at all can be computed from it. Player_Event_Data.csv
# sits outside that set on purpose: it powers six of the twenty-one features, and there are whole
# competitions it is never published for — measured, 0 of 288 NCAA match folders across 190 teams
# have it, against essentially all of MLS. Requiring it there skipped every match, left an empty
# feature frame, and took the whole report down (see main).
CORE_MATCH_FILES = {
    'Match_Attributes.csv', 'Average_Positions.csv', 'Full_Team_Statistics.csv',
    'Match_Momentum.csv', 'Player_Heatmaps.csv', 'Player_Statistics.csv', 'Shot_Map.csv',
}
FULL_MATCH_FILES = CORE_MATCH_FILES | {'Player_Event_Data.csv'}

# Kept as an alias so anything importing the old name still works.
REQUIRED_MATCH_FILES = FULL_MATCH_FILES

# Features that can only be computed from Player_Event_Data.csv, listed so the reduced-mode log can
# name what it lost rather than just saying it lost something.
EVENT_DERIVED_FEATURES = (
    'prg_pass_pct', 'opp_half_pass_pct', 'avg_prg_carry_gain',
    'avg_x_def_action', 'avg_pass_dist', 'avg_gk_pass_length',
)

# Below this many matches the recency weighting has nothing to work with and the consistency
# figures stop measuring anything — at one match the team's archetype and that match's archetype
# are the same thing, so consistency is 100% whatever the team did.
STYLE_THIN_SAMPLE_MATCHES = 3

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
            try:
                match_data[file_name] = pd.read_csv(os.path.join(match_path, file_name))
            except Exception:
                # A file that won't parse is treated as absent, which the required-files checks
                # downstream already know how to handle — the same tolerance
                # Pre_Match_Report.loadMatchData has always had, and this function was missing.
                #
                # Not hypothetical: 68 CSVs in the archive are zero-length, 48 of them
                # Player_Heatmaps.csv and 20 Goalkeeper_Spatial_Points.csv, concentrated in NCAA.
                # pd.read_csv raises EmptyDataError on those, and one such file anywhere in a
                # league killed the whole league-scaling scan.
                continue

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

def get_league_features(year, league, date, required_files=None):
    '''Extracts match features for every team in the league, as a scaling reference.

    `required_files` must be the same set the scouted team was extracted under — see main. Fitting
    the scaler on a richer feature set than the team can supply leaves it holding columns it will
    never be asked to transform, and a narrower one leaves the team's columns unscaled.'''
    if required_files is None:
        required_files = FULL_MATCH_FILES
    league_path = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data/{league.replace(" ", "_")}_Data'
    league_features = []
    skipped = 0
    for team_folder in get_files(league_path):
        team = team_folder.removesuffix('_Data')
        team_matches = get_data(year, league, team, date)
        for match_folder, match_data in team_matches.items():
            if not required_files.issubset(match_data.keys()):
                skipped += 1
                continue
            league_features.append(extract_features(match_data, team))

    if skipped:
        # One line rather than one per match: a competition without event data used to print a
        # "Skipping ... for league scaling" line for every fixture in it, which on NCAA's 288
        # folders buried everything else the run had to say.
        print(f'League scaling: skipped {skipped} match(es) missing required CSVs.')

    return pd.DataFrame(league_features)

def _extract_matches(matches_data, team, required_files, quiet=False):
    '''Features for every match that has `required_files`, oldest first.'''
    matches_features = []
    for match_folder, match_data in matches_data.items():
        if not required_files.issubset(match_data.keys()):
            if not quiet:
                print(f'Skipping {match_folder} (missing CSVs)')
            continue
        matches_features.append(extract_features(match_data, team))
    return matches_features

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
    # Optional — see CORE_MATCH_FILES. Every feature derived from it yields NaN when it is absent,
    # and main() drops the resulting all-NaN columns and rescales the axis weights over what is
    # left, rather than the match being thrown away whole.
    events = match_data.get('Player_Event_Data.csv')
    player_heatmaps = match_data['Player_Heatmaps.csv']
    player_stats = match_data['Player_Statistics.csv']
    shots = match_data['Shot_Map.csv']

    features = {}

    team_row = team_stats['Team Name'] == team

    def team_stat(column):
        '''One column of this team's row, or NaN when the competition doesn't carry it.

        Coverage varies COLUMN BY COLUMN, not just file by file: NCAA's Full_Team_Statistics.csv
        has no 'Final third entries' and no 'Recoveries' at all, where MLS has both. Indexing a
        missing column raised and killed the run, so absence is now just another empty feature.'''
        if column not in team_stats.columns:
            return np.nan
        values = team_stats.loc[team_row, column]
        return values.iloc[0] if len(values) else np.nan

    # POSSESSION & PASS STYLE

    features['possession_pct'] = team_stat('Ball possession')

    team_touches = player_stats['touches'].sum()
    features['touches_in_opp_penalty_area_per_touch'] = team_stat('Touches in penalty area') / team_touches
    features['final_third_entries_per_touch'] = team_stat('Final third entries') / team_touches

    if events is not None:
        passes = events[events['Event Type'] == 'pass']
        num_prg_passes = len(passes[passes['Pass End X Coord'] > passes['Player X Coord']])
        num_passes = len(passes)
        features['prg_pass_pct'] = safe_div(num_prg_passes, num_passes)

        num_opp_half_passes = len(passes[passes['Player X Coord'] > 50.0])
        features['opp_half_pass_pct'] = safe_div(num_opp_half_passes, num_passes)

        prg_carries = events[events['Event Type'] == 'ball-carry']
        features['avg_prg_carry_gain'] = (prg_carries['Pass End X Coord'] - prg_carries['Player X Coord']).mean()
    else:
        passes = None
        features['prg_pass_pct'] = np.nan
        features['opp_half_pass_pct'] = np.nan
        features['avg_prg_carry_gain'] = np.nan


    # DEFENSE

    features['recoveries'] = team_stat('Recoveries')
    features['tackles'] = team_stat('Total tackles')
    features['interceptions'] = team_stat('Interceptions')

    if events is not None:
        def_actions = events[events['Event Type'] == 'Defensive Action']
        features['avg_x_def_action'] = def_actions['Player X Coord'].mean()
    else:
        features['avg_x_def_action'] = np.nan

    features['avg_def_position'] = avg_positions[avg_positions['player.position'] == 'D']['averageX'].mean()

    gk_stats = player_stats[player_stats['Position'] == 'G']
    if 'totalKeeperSweeper' in player_stats.columns:
        features['num_sweeper_keeper'] = gk_stats['totalKeeperSweeper'].sum()
    else:
        features['num_sweeper_keeper'] = 0


    # DIRECTNESS

    features['aerial_duels'] = team_stat('Aerial duels')

    if passes is not None:
        pass_dist = np.sqrt((passes['Pass End X Coord'] - passes['Player X Coord'])**2 + (passes['Pass End Y Coord'] - passes['Player Y Coord'])**2)
        features['avg_pass_dist'] = pass_dist.mean()

        gk_passes = passes[passes['Player'].isin(gk_stats['Player Name'])]
        gk_pass_dist = np.sqrt((gk_passes['Pass End X Coord'] - gk_passes['Player X Coord'])**2 + (gk_passes['Pass End Y Coord'] - gk_passes['Player Y Coord'])**2)
        features['avg_gk_pass_length'] = gk_pass_dist.mean()
    else:
        features['avg_pass_dist'] = np.nan
        features['avg_gk_pass_length'] = np.nan

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

# Each axis is a weighted mean of min-max scaled features. Held as data rather than inline
# arithmetic so score_axes can rescale over whichever features a competition actually supplies —
# see there. Weights within an axis sum to 1.
AXIS_WEIGHTS = {
    'possession_score': {
        'possession_pct': 0.24,
        'prg_pass_pct': 0.22,
        'opp_half_pass_pct': 0.18,
        'final_third_entries_per_touch': 0.14,
        'touches_in_opp_penalty_area_per_touch': 0.14,
        'avg_prg_carry_gain': 0.08,
    },
    'defense_score': {
        'avg_x_def_action': 0.40,
        'avg_def_position': 0.25,
        'recoveries': 0.20,
        'tackles': 0.075,
        'interceptions': 0.075,
    },
    'direct_score': {
        'long_ball_pct': 0.30,
        'avg_gk_pass_length': 0.30,
        'avg_pass_dist': 0.25,
        'aerial_duels': 0.15,
    },
    'transition_score': {
        'fast_break_shot_pct': 0.45,
        'momentum_volatility': 0.30,
        'momentum_changes': 0.25,
    },
    'width_score': {
        'team_width': 0.60,
        'num_crosses': 0.40,
    },
}

AXIS_COLUMNS = list(AXIS_WEIGHTS)

def score_axes(df):
    '''Adds the five axis scores, rescaling each over the features actually present.

    An axis is a weighted MEAN of features already min-max scaled to [0, 1], so when a competition
    doesn't supply one of them the honest move is to redistribute its weight across the rest rather
    than treat the feature as zero — scoring a missing feature as 0 would drag the axis toward the
    low end and misplace the team against CENTROIDS. Rescaling keeps the axis in [0, 1] and
    directly comparable to the centroids, at the cost of assuming the absent features would have
    scored like the present ones.

    How much of an axis that assumption is carrying varies. On NCAA data, transition and width lose
    nothing, while possession is rebuilt from 38% of its intended weight and defense from 40% —
    which is what main()'s `limited` flag exists to tell the reader.'''
    for axis, weights in AXIS_WEIGHTS.items():
        present = {feature: weight for feature, weight in weights.items() if feature in df.columns}
        total = sum(present.values())
        if not total:
            # No feature of this axis survived. Nothing sensible to put in the column, and a zero
            # would read as "this team does none of it" rather than "we cannot tell".
            df[axis] = np.nan
            continue
        df[axis] = sum(df[feature] * (weight / total) for feature, weight in present.items())

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

def rank_centroids(vector):
    '''Returns archetype names ordered nearest-first by Euclidean distance from
    `vector` to each CENTROIDS entry.'''
    centroid_distances = {
        archetype: np.linalg.norm(vector - np.array(centroid_vector))
        for archetype, centroid_vector in CENTROIDS.items()
    }
    return sorted(centroid_distances, key=centroid_distances.get)

def classify_matches(df):

    axis_cols = ['possession_score', 'defense_score', 'direct_score', 'transition_score', 'width_score']
    archetypes = []

    for __, match in df[axis_cols].iterrows():
        team_vector = match.values.astype(float)
        archetypes.append(rank_centroids(team_vector)[0])

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

    # Primary/secondary archetype = nearest/second-nearest centroid to the team's own
    # recency-weighted average profile (weighted_axis_scores above) — not a plurality
    # vote across individual match labels. A vote lets a team's archetype get decided
    # by whichever bucket its noisier/less-representative matches happen to cluster
    # into (e.g. several matches split across possession-flavored archetypes while a
    # smaller set consistently lands in one defense-flavored one), even when that
    # bucket isn't close to the team's actual average — voting was also fully
    # disconnected from weighted_axis_scores despite both being computed here.
    ranked_archetypes = rank_centroids(weighted_axis_scores.values.astype(float))
    team_style_dict['primary_archetype'] = ranked_archetypes[0]
    team_style_dict['secondary_archetype'] = ranked_archetypes[1]

    # Weighted archetype consistency
    team_style_dict['primary_archetype_consistency'] = (df['archetype'] == team_style_dict['primary_archetype']).mean()
    team_style_dict['secondary_archetype_consistency'] = (df['archetype'] == team_style_dict['secondary_archetype']).mean()

    return team_style_dict

def main(matches_data=None, year=2026, league='MLS', team='Columbus_Crew', date='7-22-2026'):

    if matches_data is None:
        matches_data = get_data(year, league, team, date)

    # TWO PASSES, and the order is what keeps well-covered competitions unaffected.
    #
    # Pass one demands every file, exactly as before. A team playing across competitions — an MLS
    # side with cup fixtures — keeps having its thin matches skipped in favour of its complete
    # ones, and is scored on the full feature set. Only when that leaves NOTHING does pass two
    # relax to CORE_MATCH_FILES, which is the difference between a reduced report and no report at
    # all for a competition like NCAA where no match anywhere carries event data.
    required_files = FULL_MATCH_FILES
    matches_features = _extract_matches(matches_data, team, required_files)

    if not matches_features:
        required_files = CORE_MATCH_FILES
        matches_features = _extract_matches(matches_data, team, required_files, quiet=True)
        if matches_features:
            print(f'No {team} match has Player_Event_Data.csv — falling back to the features that '
                  f'do not need it.')

    feature_df = pd.DataFrame(matches_features)
    if feature_df.empty or not len(feature_df.columns):
        print(f'No {team} match has even the core files '
              f'({", ".join(sorted(CORE_MATCH_FILES))}) — skipping style of play.')
        return None

    # Features nothing could be computed for come back all-NaN. Dropped rather than imputed: a
    # made-up value would be indistinguishable from a measured one in the output, and MinMaxScaler
    # cannot take a NaN anyway. score_axes redistributes their weight.
    unavailable = [column for column in feature_df.columns if feature_df[column].isna().all()]
    if unavailable:
        feature_df = feature_df.drop(columns=unavailable)
        print(f'Style of play is running without {len(unavailable)} feature(s) this competition '
              f'does not provide: {", ".join(sorted(unavailable))}')
    if feature_df.empty or not len(feature_df.columns):
        print(f'No usable style-of-play feature for {team} — skipping style of play.')
        return None

    # The scaler has to be fitted on the SAME columns it will be asked to transform, so the league
    # reference is built under the same rules and then narrowed to whatever survived above.
    league_feature_df = get_league_features(year, league, date, required_files)
    league_feature_df = league_feature_df.reindex(columns=feature_df.columns)
    league_feature_df = league_feature_df.dropna(axis=0, how='any')
    if league_feature_df.empty:
        print(f'No team in {league} yields a usable scaling reference — skipping style of play. '
              f'The axis scores are league-relative, so there is nothing to scale against.')
        return None

    scaler = fit_scaler(league_feature_df)
    scaled_df = normalize_features(feature_df, scaler)

    scored_df = score_axes(scaled_df)

    match_classifications = classify_matches(scored_df)

    team_style = classify_team_style(match_classifications)

    # What the caller needs to know before leaning on this. Two things make a classification worth
    # less than it looks: features the competition never supplied, and too few matches to average
    # over. Both are reported through one flag, because to a reader they mean the same thing.
    limitations = []
    if unavailable:
        limitations.append(f'{len(unavailable)} feature(s) unavailable: {", ".join(sorted(unavailable))}')
    if len(feature_df) < STYLE_THIN_SAMPLE_MATCHES:
        # At one match the archetype IS the match, so consistency is forced to 100% by
        # construction and would actively mislead if read as agreement across games.
        limitations.append(f'only {len(feature_df)} match(es) of data')
    team_style['limited'] = bool(limitations)
    team_style['limitations'] = limitations

    return team_style

if __name__ == "__main__":
    main()