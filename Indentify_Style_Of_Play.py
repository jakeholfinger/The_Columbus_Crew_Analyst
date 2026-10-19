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

# Column get_league_features tags each row with. Named rather than inlined because main() has to
# strip it before the frame is handed to the scaler — a team name is not a feature — and put it
# back alongside the scored rows for fit_centroids.
LEAGUE_TEAM_COLUMN = '__team'

def get_league_features(year, league, date, required_files=None):
    '''Extracts match features for every team in the league, as a scaling reference.

    Rows carry LEAGUE_TEAM_COLUMN so the same pass can also supply the per-team vectors
    fit_centroids places the archetypes against.

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
            # Tagged with its team, and left in the chronological order get_data returns, so the
            # same rows can serve twice: as the scaler's reference (which ignores the tag) and as
            # the per-team vectors fit_centroids needs. See LEAGUE_TEAM_COLUMN.
            features = extract_features(match_data, team)
            features[LEAGUE_TEAM_COLUMN] = team
            league_features.append(features)

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

# How quickly a match stops counting, in matches. One constant rather than a default repeated at
# each call site, because the scouted team's vector and the league reference vectors fit_centroids
# is built from HAVE to be weighted identically — a team compared against a reference computed a
# different way is being measured with two rulers.
STYLE_HALF_LIFE = 5

# Below this many teams the league is not a distribution, and fit_centroids leaves the archetypes
# where they are. Eight is the point at which a per-axis standard deviation over five axes stops
# being dominated by whichever two teams happen to be at the ends.
STYLE_MIN_REFERENCE_TEAMS = 8

def recency_weighted_axes(df, half_life=STYLE_HALF_LIFE):
    '''One team's recency-weighted axis vector, from rows in ascending chronological order.

    Oldest first, most recent last — the convention get_data and Pre_Match_Report.loadMatchData
    both return — so games_ago counts DOWN to 0 at the last row rather than up from it.'''
    games_ago = np.arange(len(df))[::-1]
    weights = 0.5 ** (games_ago / half_life)
    return (df[AXIS_COLUMNS].multiply(weights, axis=0)).sum() / weights.sum()

def league_team_vectors(league_scored, league_teams, half_life=STYLE_HALF_LIFE):
    '''One recency-weighted axis vector per team in the league reference, built by exactly the
    path the scouted team's own vector is built by.'''
    frame = league_scored.copy()
    frame[LEAGUE_TEAM_COLUMN] = list(league_teams)
    # groupby keeps each group's rows in their original order, and get_league_features appends a
    # team's matches oldest-first, so the recency weighting lands the right way round.
    return pd.DataFrame([recency_weighted_axes(group, half_life)
                         for _, group in frame.groupby(LEAGUE_TEAM_COLUMN, sort=True)])

def fit_centroids(team_vectors):
    '''CENTROIDS moved onto the part of the space this league's teams actually occupy.

    Returns (centroids, reason_declined).

    WHY THIS EXISTS. The centroids are written on a textbook 0-1 scale and span 0.08 to 0.90, but
    an axis score cannot reach those ends: it is a weighted mean of three to six min-max scaled
    features, averaged again over a season of matches, and two rounds of averaging pull everything
    to the middle. Measured across MLS 2026 the five axes span only 0.118 (transition) to 0.346
    (defense) of the range the centroids are laid out over.

    The consequence is not a small bias, it is the classifier not working. Every team sits in a
    small cloud near the centre of a space whose landmarks are all far outside it, so the nearest
    landmark is decided by which one happens to sit closest to the middle rather than by how the
    team plays. The three centroids nearest the league average — High Press at 0.377 from it,
    Vertical Possession at 0.417, Low Block at 0.428 — took every single team, 16/12/2 of 30,
    while the four further out (Tiki-Taka, Wide/Crossing, Counter-Attack, Direct, all 0.52 to
    0.61 away) were unreachable. New England were labelled High Press on a 3% margin while sitting
    26th of 30 for how high up the pitch they defend.

    THE FIX is a per-axis affine map: each axis of the constellation is recentred on the league's
    mean for that axis and rescaled to its standard deviation. Affine per axis, so every
    archetype keeps its position RELATIVE to the others on every axis — the tactical meaning the
    coordinates encode is untouched — while the constellation as a whole ends up the same size as
    the cloud it is being compared against. Once the two are on the same scale, direction decides
    the label instead of proximity to the average. Checked afterwards: all seven archetypes are
    used, Vancouver (1st in the league for defensive-action height and recoveries) is the High
    Press team, Houston (30th and 30th) is the Low Block one.

    WHAT IT DOES NOT FIX. A team with no tilt at all still gets a label. New England move from
    High Press to Counter-Attack, and their transition score is exactly the league mean, so the
    new answer is no better founded than the old one — only differently wrong. What is needed
    there is for the classifier to be able to say "no distinctive style", which is a separate
    change to how the margin between the top two archetypes is reported.

    The map is refitted per competition rather than baked in, because the axis scores are already
    league-relative — the scaler is fitted on the league's own feature distribution — so a fixed
    map would be right for whichever competition it was measured on and wrong for the rest.'''
    if len(team_vectors) < STYLE_MIN_REFERENCE_TEAMS:
        return CENTROIDS, (f'archetypes not fitted to the league: {len(team_vectors)} team(s) of '
                           f'reference, fewer than {STYLE_MIN_REFERENCE_TEAMS}')

    middle = team_vectors[AXIS_COLUMNS].mean()
    spread = team_vectors[AXIS_COLUMNS].std(ddof=0)
    # An axis nothing could be computed for, or one every team scores identically on, has no scale
    # to map onto. Declining WHOLESALE rather than passing that axis through untouched: a
    # constellation half in league units and half in textbook ones is a worse object than either.
    if not np.isfinite(spread).all() or (spread <= 0).any():
        return CENTROIDS, 'archetypes not fitted to the league: an axis has no spread to fit to'

    coordinates = np.array([CENTROIDS[archetype] for archetype in CENTROIDS], dtype=float)
    fitted = middle.values + ((coordinates - coordinates.mean(axis=0))
                              * (spread.values / coordinates.std(axis=0, ddof=0)))
    return ({archetype: list(vector) for archetype, vector in zip(CENTROIDS, fitted)}, None)

# How much further the runner-up archetype has to be before the nearest one is reported on its
# own, as a fraction of the nearest one's distance. Read off the real distribution rather than
# picked: across MLS 2026 the margins fall in a tight group of five teams between 0.025 and 0.050
# and then jump to 0.098, so 0.10 sits in the gap. It flags six teams of thirty — about one in
# five sides has no single style the five axes can separate, which is a believable number and
# exactly the case the classifier used to paper over.
STYLE_AMBIGUOUS_MARGIN = 0.10

def centroid_distances(vector, centroids=CENTROIDS):
    '''Euclidean distance from `vector` to each entry of `centroids`, by archetype.'''
    return {archetype: float(np.linalg.norm(vector - np.array(centroid_vector)))
            for archetype, centroid_vector in centroids.items()}

def rank_centroids(vector, centroids=CENTROIDS):
    '''Returns archetype names ordered nearest-first by Euclidean distance from
    `vector` to each entry of `centroids`.'''
    distances = centroid_distances(vector, centroids)
    return sorted(distances, key=distances.get)

def classify_matches(df, centroids=CENTROIDS):

    archetypes = []

    for __, match in df[AXIS_COLUMNS].iterrows():
        team_vector = match.values.astype(float)
        archetypes.append(rank_centroids(team_vector, centroids)[0])

    df = df.copy()
    df['archetype'] = archetypes

    return df

def classify_team_style(df, centroids=CENTROIDS, half_life=STYLE_HALF_LIFE):

    team_style_dict = {}

    # Weighted axis scores for radar chart
    weighted_axis_scores = recency_weighted_axes(df, half_life)
    team_style_dict['axis_scores'] = weighted_axis_scores.to_dict()

    # Primary/secondary archetype = nearest/second-nearest centroid to the team's own
    # recency-weighted average profile (weighted_axis_scores above) — not a plurality
    # vote across individual match labels. A vote lets a team's archetype get decided
    # by whichever bucket its noisier/less-representative matches happen to cluster
    # into (e.g. several matches split across possession-flavored archetypes while a
    # smaller set consistently lands in one defense-flavored one), even when that
    # bucket isn't close to the team's actual average — voting was also fully
    # disconnected from weighted_axis_scores despite both being computed here.
    distances = centroid_distances(weighted_axis_scores.values.astype(float), centroids)
    ranked_archetypes = sorted(distances, key=distances.get)
    team_style_dict['primary_archetype'] = ranked_archetypes[0]
    team_style_dict['secondary_archetype'] = ranked_archetypes[1]

    # HOW MUCH the nearest archetype won by, and whether that is enough to name it alone.
    #
    # Nearest-centroid always returns something, so without this a team with no distinguishable
    # style reads exactly like one with a pronounced one. New England are the case that prompted
    # it: middling on all five axes, and whichever centroid they came out nearest was decided by
    # margins in the third decimal place. Reporting the pair is the honest answer — they play a
    # bit of both — and it costs nothing when a team really does have one style, because then the
    # margin is wide and only the one name is printed.
    #
    # Relative to the nearest distance rather than absolute, because the distances themselves
    # scale with how far the team sits from the middle of the league, and a fixed gap would mean
    # different things for a team on the edge of the distribution and one near its centre.
    nearest, runner_up = distances[ranked_archetypes[0]], distances[ranked_archetypes[1]]
    margin = (runner_up - nearest) / nearest if nearest else 0.0
    team_style_dict['archetype_margin'] = margin
    team_style_dict['archetype_ambiguous'] = margin < STYLE_AMBIGUOUS_MARGIN

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
    # Pulled off BEFORE the reindex, which would drop it, and kept aligned to the rows that survive
    # so fit_centroids can group by it. A boolean mask rather than dropna for exactly that reason.
    league_teams = league_feature_df.pop(LEAGUE_TEAM_COLUMN)
    league_feature_df = league_feature_df.reindex(columns=feature_df.columns)
    usable_rows = league_feature_df.notna().all(axis=1)
    league_feature_df = league_feature_df[usable_rows]
    league_teams = league_teams[usable_rows]
    if league_feature_df.empty:
        print(f'No team in {league} yields a usable scaling reference — skipping style of play. '
              f'The axis scores are league-relative, so there is nothing to scale against.')
        return None

    scaler = fit_scaler(league_feature_df)

    # The archetypes are placed against this league before anything is compared to them — see
    # fit_centroids for why they cannot be used where they are written. The reference goes through
    # the identical scale/score/weight path the scouted team does, so the two are commensurable.
    league_scored = score_axes(normalize_features(league_feature_df, scaler))
    centroids, centroid_note = fit_centroids(league_team_vectors(league_scored, league_teams))

    scaled_df = normalize_features(feature_df, scaler)

    scored_df = score_axes(scaled_df)

    match_classifications = classify_matches(scored_df, centroids)

    team_style = classify_team_style(match_classifications, centroids)

    # What the caller needs to know before leaning on this. Two things make a classification worth
    # less than it looks: features the competition never supplied, and too few matches to average
    # over. Both are reported through one flag, because to a reader they mean the same thing.
    limitations = []
    if unavailable:
        limitations.append(f'{len(unavailable)} feature(s) unavailable: {", ".join(sorted(unavailable))}')
    if centroid_note:
        # Unfitted archetypes are the pre-fit behaviour, which is known to hand more than half a
        # league the same label — see fit_centroids. A reader has to be told when that is what
        # they are looking at.
        limitations.append(centroid_note)
    if len(feature_df) < STYLE_THIN_SAMPLE_MATCHES:
        # At one match the archetype IS the match, so consistency is forced to 100% by
        # construction and would actively mislead if read as agreement across games.
        limitations.append(f'only {len(feature_df)} match(es) of data')
    team_style['limited'] = bool(limitations)
    team_style['limitations'] = limitations
    team_style['centroids_fitted'] = centroid_note is None

    return team_style

if __name__ == "__main__":
    main()