import pandas as pd
import numpy as np
import os
from datetime import datetime
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# Constants
PITCH_LENGTH_YARDS = 115
PITCH_WIDTH_YARDS = 74

# Fixed figure-fraction gap between a chart's requested bbox bottom edge (bbox[1]) and
# its notes box below it. Shared by generate_relative_gk_pass_map and
# generate_defensive_action_map, whose charts both sit on bbox[1]=0.50, so their notes
# boxes land on the same horizontal line rather than each chart computing its own offset.
NOTES_OFFSET = 0.02

# Most entries the key-pass legend will name individually before rolling the remainder
# into a single "Other" row. Bounds the legend's height the same way the Potential
# Absences table caps itself in Pre_Match_Report.generatePredictedLineup; the list is
# already sorted by key passes per 90, so the cut drops the least creative players.
# Tokens that belong to the surname rather than being a middle name, used when shortening
# names for the key-pass legend. Lowercase-compared, so it covers both "van Dijk" and the
# capitalised Arabic/Iberian patronymics.
SURNAME_PARTICLES = {
    'abou', 'abu', 'al', 'ben', 'bin', 'da', 'das', 'de', 'del', 'della', 'der', 'di',
    'do', 'dos', 'du', 'el', 'ibn', 'la', 'le', 'los', 'mac', 'mc', 'san', 'santa',
    'ten', 'ter', 'van', 'von',
}

KEY_PASS_LEGEND_MAX = 15

# A key-pass legend entry is greyed out below this share of the minutes the team actually
# played, because per-90 rates built on a cameo are noise dressed as form: measured on
# real data, a Pumas substitute with ONE key pass in 79 minutes (26% of available) scored
# 1.14/90 and outranked a near-ever-present with two in 255. Expressed as a share rather
# than a fixed minute count so it adapts to sample size on its own — 30% is ~91 minutes
# for a 3-match opponent and ~490 for an 18-match one. Sub-threshold players are still
# listed and still plotted, just visibly de-emphasised, since a rotation player can start.
KEY_PASS_MINUTES_QUALIFIED = 0.30

# How much wider each notes box is than the chart bbox it sits under, split evenly on
# both sides of the shared center (see notes_x0 at each call site).
NOTES_WIDTH_PAD = 0.04

# Fixed colour-scale half-ranges for page two's four team-vs-league diff heatmaps: each
# is used as vmin=-SCALE, vmax=+SCALE. FIXED, not derived per team — every one of these
# charts previously pinned its scale to its own team's np.nanmax(np.abs(diff)), which
# made the same colour mean a different magnitude in every opponent's report (measured on
# the buildup chart, that per-team max swung 5.4x, from 0.0025 for Club Necaxa to 0.0137
# for Querétaro, with both teams' worst cell rendering identically saturated). A fixed
# scale is what lets a reader tell "close to league average" from "genuine outlier".
#
# Values are the p95 of |team − league| measured across every team, taken from MLS rather
# than Liga MX. The two leagues' spreads differ ~2x, but that is sampling noise, not
# tactical variance: MLS has 18 matches/team vs Liga MX's 3, noise scales as 1/sqrt(n),
# and sqrt(18/3) = 2.45 — which matches the observed p95 ratios (buildup 2.36x, press
# 2.67x) almost exactly. MLS is therefore the honest calibration; a thin-sample league
# will clip until its season accumulates, which is the correct reading of data too noisy
# to act on. Re-measure these if the underlying metric definitions change.
GK_PASS_DIFF_SCALE = 5.0        # % of a keeper's own passes (MLS p95 5.20)
DEF_ACTION_DIFF_SCALE = 1.5     # % of a team's own defensive actions (MLS p95 1.33)
BUILDUP_DIFF_SCALE = 0.002      # press value per opponent action (MLS p95 0.0022)
PRESSING_DIFF_SCALE = 0.002     # press threat per build-out action (MLS p95 0.0018)

#%%
def diff_label_color(value, scale, cmap):
    '''Black or white cell-label text, whichever contrasts with the fill underneath.

    The diff heatmaps are drawn on a fixed symmetric scale (vmin=-scale, vmax=+scale),
    and both ends of the diverging colormap are built from team colors — so for a club
    with a dark primary or secondary, a saturated cell renders near-black and hardcoded
    black label text disappears into it. That got much more common once the colour scale
    was fixed rather than per-team normalised (a fifth of cells clip on a thin-sample
    league), and a clipped cell is exactly the one worth reading.

    Rather than assume which end is dark, this asks the actual colormap what colour the
    cell got and switches on its Rec. 709 relative luminance, so it adapts to whatever
    team colors the report is being generated in. NaN cells are never painted (the
    "No Actions" case), so they keep the default black.'''

    if not np.isfinite(value):
        return 'black'
    normalized = np.clip((value + scale) / (2 * scale), 0, 1)
    red, green, blue = cmap(normalized)[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return 'white' if luminance < 0.5 else 'black'

#%%
_current_squad_cache = {}

def load_current_squad(team_matches):
    """Set of player names on the scouted team's CURRENT roster, or None if it can't be
    determined.

    Uses SofaScore's squad endpoint rather than inferring from recent appearances, because
    the local inference gets it backwards in both directions — checked against real
    Columbus Crew data: Wessam Abou Ali had not featured in the last five matchday squads
    but IS still on the roster (injured, not gone), while Diego Rossi appeared in those
    same five and is NOT (departed). "Hasn't played lately" and "has left" are different
    facts and only the endpoint knows the second.

    The team ID is recovered from the cached Match_Attributes rather than taken as a
    parameter: the scouted team is the only side appearing in every one of its own
    fixtures, so intersecting the home/away IDs across matches leaves exactly it.

    Returns None (rather than an empty set) when the team can't be resolved or the request
    fails, so callers can tell "nobody has left" apart from "we don't know" and skip the
    greying instead of marking the whole squad as departed.
    """
    team_ids = None
    for match_files in team_matches.values():
        attrs = match_files.get('Match_Attributes.csv')
        if attrs is None or 'homeTeam.id' not in attrs.columns:
            continue
        ids = {int(attrs['homeTeam.id'].iloc[0]), int(attrs['awayTeam.id'].iloc[0])}
        team_ids = ids if team_ids is None else (team_ids & ids)

    if not team_ids or len(team_ids) != 1:
        return None
    team_id = team_ids.pop()

    if team_id in _current_squad_cache:
        return _current_squad_cache[team_id]

    import Pre_Match_Report  # deferred: Pre_Match_Report imports this module in turn
    squad_json = Pre_Match_Report.scrapeURLData(f'https://www.sofascore.com/api/v1/team/{team_id}/players')
    if not squad_json:
        return None

    squad = {entry['player']['name'] for entry in squad_json.get('players', []) if entry.get('player')}
    squad = squad or None
    _current_squad_cache[team_id] = squad
    return squad

#%%
def abbreviate_player_name(name):
    '''"Sekou Tidiany Bangoura" -> "S. Bangoura".

    The key-pass legend's width is set by its single longest entry, so one long name
    pushes the whole legend into the chart beside it — measured: Columbus Crew's list ran
    to x=0.613 and overlapped the conceded map starting at 0.600, while Pumas' shorter
    names stopped at 0.587 and cleared it. Shortening bounds the width by the longest
    SURNAME rather than the longest full name.

    Mononyms ("Juninho") are returned untouched instead of becoming "J. Juninho". The
    pitch cards in generatePredictedLineup already display surnames only, so abbreviating
    here is consistent with the report's existing style.'''
    parts = str(name).split()
    if len(parts) < 2:
        return str(name)

    # Walk back from the end while the preceding token is a surname particle, so compound
    # surnames survive. Taking only the last token turned "Wessam Abou Ali" into "W. Ali",
    # which is simply the wrong name — his surname is "Abou Ali". Middle names must still
    # be dropped though ("Sekou Tidiany Bangoura" -> "S. Bangoura"), and nothing about the
    # capitalisation or position distinguishes "Abou" from "Tidiany", so a particle list is
    # the only thing that separates the two cases.
    surname_start = len(parts) - 1
    while surname_start > 1 and parts[surname_start - 1].lower() in SURNAME_PARTICLES:
        surname_start -= 1

    return f"{parts[0][0]}. {' '.join(parts[surname_start:])}"

#%%
def draw_notes_box(page, bbox):
    '''Empty bordered rectangle reserving space for a real fillable PDF field, added in a
    post-processing pass over the finished report (matplotlib alone can't create
    interactive form fields — see PDF_Form_Fields.py). Returns bbox unchanged so callers
    can register it for that pass.'''
    ax = page.add_axes(bbox)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#888888')
    return bbox

#%%
def compute_total_minutes(matches):

    def get_col(df, col, default=0):
        '''Safe column lookup: 0 if the column doesn't exist (e.g. injuryTime3/4,
        which are only present when extra time was played) or its value is NaN.'''
        series = df.get(col)
        if series is None:
            return default
        val = series.iloc[0]
        return val if pd.notna(val) else default

    total_minutes = 0

    for match, match_files in matches.items():
        match_attributes = match_files.get('Match_Attributes.csv')

        # Match_Attributes.csv is sometimes absent (confirmed gaps in this
        # dataset) — fall back to a flat 90 rather than crashing on the lookups
        # below, all of which assume a real DataFrame.
        if match_attributes is None:
            total_minutes += 90
            continue

        # Every finished match has both regulation halves regardless of whether
        # either team scored in them — homeScore.period1/2 are goal counts, not
        # "did this period happen" flags, so they aren't used to gate this.
        try:
            minutes = match_attributes['defaultPeriodLength'].iloc[0] * 2
        except:
            minutes = 90

        minutes += get_col(match_attributes, 'time.injuryTime1')
        minutes += get_col(match_attributes, 'time.injuryTime2')

        # homeScore.extra1/extra2 are NaN (not 0, not missing) when extra time
        # wasn't played, and bool(nan) is True in Python — pd.notna() avoids
        # treating every match as if it went to extra time.
        if pd.notna(match_attributes.get('homeScore.extra1', pd.Series([None])).iloc[0]):
            try:
                minutes += match_attributes['defaultOvertimeLength'].iloc[0]
            except:
                minutes += 15
            minutes += get_col(match_attributes, 'time.injuryTime3')

        if pd.notna(match_attributes.get('homeScore.extra2', pd.Series([None])).iloc[0]):
            try:
                minutes += match_attributes['defaultOvertimeLength'].iloc[0]
            except:
                minutes += 15
            minutes += get_col(match_attributes, 'time.injuryTime4')

        total_minutes += minutes

    return total_minutes

#%%
def compute_player_minutes(matches):
    '''Total minutes played per player, summed across `matches`. Player_Statistics.csv's
    'Player Name' column joins cleanly against Player_Event_Data.csv's 'Player' column
    (confirmed on real data — every event-data player has a matching stats row; the only
    one-sided names are unused substitutes with NaN minutes, who never appear in event
    data either).'''
    minutes = {}
    for match_files in matches.values():
        stats = match_files.get('Player_Statistics.csv')
        if stats is None:
            continue
        for _, row in stats.dropna(subset=['minutesPlayed']).iterrows():
            minutes[row['Player Name']] = minutes.get(row['Player Name'], 0) + row['minutesPlayed']
    return minutes

#%%
def get_files(current_path):
    '''Returns a list of all visible files inside the folder at currentPath'''
    return [f for f in os.listdir(current_path) if not f.startswith('.')]

#%%
# load_league_data re-reads every team's Player_Event_Data.csv off disk, and page two of
# the report now asks for a league baseline four times (GK passes, defensive actions,
# buildup, pressing). It's a pure function of its arguments, so the results are memoised
# here rather than paying that cost once per chart. Keyed on the full argument tuple
# since `data_required` and `exclude_team` both change what gets loaded.
_league_data_cache = {}

def load_league_data(leagues, date, data_required, exclude_team=None):
    '''`leagues` is a list (same convention as loadMatchData) — every team's
    matches across all listed leagues are pooled. `exclude_team` skips a single
    team folder (used to keep a team's own matches out of its own league-average
    baseline).

    Results are cached in-process (see _league_data_cache). Callers treat the return
    value as read-only — none of the chart functions mutate the returned frames in
    place, they filter/copy first.'''

    cache_key = (tuple(leagues), date, tuple(data_required), exclude_team)
    if cache_key in _league_data_cache:
        return _league_data_cache[cache_key]

    matches = {}
    seen_match_ids = set()
    year = date.split('-')[2]
    for league in leagues:
        league_path = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data/{league.replace(" ", "_")}_Data'
        if not os.path.exists(league_path):
            print(f'Data for {league} not found. Please scrape relevant data.')
            continue
        for team_folder in get_files(league_path):
            team_name = team_folder.removesuffix('_Data')
            if exclude_team is not None and team_name == exclude_team:
                continue
            team_path = os.path.join(league_path, team_folder)
            if not os.path.isdir(team_path):
                continue
            for match_folder in get_files(team_path):
                match_path = os.path.join(team_path, match_folder)
                if not os.path.isdir(match_path):
                    continue
                scraped_files = {}
                for file in get_files(match_path):
                    if file in data_required:
                        file_path = os.path.join(match_path, file)
                        scraped_files[file] = pd.read_csv(file_path)

                # Match_Attributes.csv isn't always present (confirmed gaps in this
                # dataset) — read it if it exists and keep it in scraped_files (needed
                # downstream for per-90 minutes), skip the dedup/minutes info if not,
                # rather than crashing on a bare read.
                match_attributes_path = os.path.join(match_path, 'Match_Attributes.csv')
                match_attributes = None
                if os.path.exists(match_attributes_path):
                    match_attributes = pd.read_csv(match_attributes_path)
                    scraped_files['Match_Attributes.csv'] = match_attributes

                if match_attributes is not None and 'id' in match_attributes.columns:
                    match_id = match_attributes['id'].iloc[0]
                    if match_id in seen_match_ids:
                        print(f'Skipping duplicate match folder {match_folder} (match ID {match_id})')
                        continue
                    seen_match_ids.add(match_id)

                matches[match_folder] = scraped_files

    def folder_date(folder_name):
        return datetime.strptime(folder_name.split('_')[0], '%m-%d-%Y')

    sorted_matches = dict(sorted(matches.items(), key=lambda item: folder_date(item[0])))

    # Keep only matches that kicked off strictly before the target date.
    month, day, yr = date.split('-')
    target_date = datetime(int(yr), int(month), int(day)).date()
    filtered_matches = {
        folder: data for folder, data in sorted_matches.items()
        if folder_date(folder).date() < target_date
    }

    _league_data_cache[cache_key] = filtered_matches
    return filtered_matches

#%%
def load_actual_opponent_events(matches, leagues, date, team):
    '''`matches` (e.g. oppositionMatches) holds `team`'s own event data — each match's own
    Player_Event_Data.csv contains only `team`'s own events, never the actual team they
    played against that day (confirmed: a single match folder's file lists ~16-18 players,
    one full roster, not two). This finds, for each of `team`'s own fixtures, the real
    opponent's own event data for that SAME match — matched by SofaScore match ID rather
    than folder date, since the same match can appear under differently-dated folder names
    depending on scrape-time timezone (see loadMatchData/load_league_data).

    Returns a dict shaped like `matches` itself (folder name -> {filename: DataFrame}),
    just sourced from each real opponent's own team folder instead of `team`'s.'''
    target_ids = set()
    for match_files in matches.values():
        attrs = match_files.get('Match_Attributes.csv')
        if attrs is not None and 'id' in attrs.columns:
            target_ids.add(attrs['id'].iloc[0])

    if not target_ids:
        return {}

    # Pools every other team's matches across the given leagues, then keeps only the
    # folders whose match ID is one of team's own fixtures — i.e. exactly the real
    # opponents' own data for these specific matches, discarding everyone else pooled.
    league_matches = load_league_data(leagues, date, ['Player_Event_Data.csv'], exclude_team=team)

    opponent_matches = {}
    for match_folder, match_files in league_matches.items():
        attrs = match_files.get('Match_Attributes.csv')
        if attrs is not None and 'id' in attrs.columns and attrs['id'].iloc[0] in target_ids:
            opponent_matches[match_folder] = match_files

    return opponent_matches

#%%
def extract_gk_passes(matches):

    matches_gk_passes = []
    for match, match_files in matches.items():
        if 'Player_Event_Data.csv' not in match_files or 'Player_Statistics.csv' not in match_files:
            continue

        player_data = match_files['Player_Statistics.csv']
        gks = player_data[player_data['Position'] == 'G']['Player Name']

        event_data = match_files['Player_Event_Data.csv']
        match_gk_passes = event_data[(event_data['Event Type'] == 'pass') & (event_data['Player'].isin(gks))]
        matches_gk_passes.append(match_gk_passes)

    if not matches_gk_passes:
        return pd.DataFrame(columns=['Player', 'Event Type', 'Player X Coord', 'Player Y Coord'])

    gk_passes = pd.concat(matches_gk_passes)

    return gk_passes

# bbox[1]=0.4683 (this and its lower-row/def-action/pressing siblings) was solved, not
# guessed: measured the actual rendered pixel position of each chart's title (which sits
# well below the nominal bbox top due to aspect-lock shrink+centering) against a real
# page-two render, then solved for the y0 that makes three gaps equal — separator line to
# upper-row title, upper-row notes-box bottom to lower-row title, and lower-row notes-box
# bottom to the page's inner border. Changing bbox height/width invalidates this value.
def generate_relative_gk_pass_map(leagues, team, team_matches, date, team_colors, page, bbox=[0.13, 0.4683, 0.28, 0.42]):

    # extract_gk_passes needs Player_Statistics.csv to identify which player names
    # are goalkeepers — without it every league match fails that guard and the
    # league side of the comparison is silently always empty.
    league_matches = load_league_data(leagues, date, ['Player_Event_Data.csv', 'Player_Statistics.csv'], exclude_team=team)

    team_gk_passes = extract_gk_passes(team_matches)
    league_gk_passes = extract_gk_passes(league_matches)

    if team_gk_passes.empty or league_gk_passes.empty:
        print('No GK pass data available — skipping gk relative pass map.')
        return page, None

    def relative_vectors_yards(gk_passes):
        '''Pass End minus player touch location, in yards, per axis — origin (0,0)
        is where the keeper touched the ball, not a spot on the pitch. Converted to
        yards on each raw endpoint before differencing (linearly equivalent to
        scaling the difference, just matches this project's established
        opta-to-yards convention).'''
        start_x = gk_passes['Player X Coord'] / 100 * PITCH_LENGTH_YARDS
        start_y = gk_passes['Player Y Coord'] / 100 * PITCH_WIDTH_YARDS
        end_x = gk_passes['Pass End X Coord'] / 100 * PITCH_LENGTH_YARDS
        end_y = gk_passes['Pass End Y Coord'] / 100 * PITCH_WIDTH_YARDS
        dx, dy = end_x - start_x, end_y - start_y
        finite = np.isfinite(dx) & np.isfinite(dy)
        return dx[finite], dy[finite]

    team_dx, team_dy = relative_vectors_yards(team_gk_passes)
    league_dx, league_dy = relative_vectors_yards(league_gk_passes)

    # Relative pass vectors legitimately span negative values (backward/left of the
    # keeper's touch) — pitch.bin_statistic only bins points inside a real pitch's
    # [0,100] opta range and silently drops anything outside it, so this uses a
    # plain numpy histogram over a custom, off-pitch coordinate range instead.
    #
    # Extent is asymmetric (forward-biased), matching real keeper behavior — backward
    # passes rarely go more than ~20yd, forward punts can travel most of the pitch —
    # and its total span (115 x 74) matches PITCH_LENGTH_YARDS/PITCH_WIDTH_YARDS so
    # this panel renders at the same landscape aspect ratio as the real pitch in
    # generate_defensive_action_map, and its (6,5) bins are the same physical yards²
    # size as that map's, keeping per-90 counts comparable across the two panels.
    # Forward/backward is the x (horizontal) axis here to match that map's
    # "→ Attack →" orientation, rather than the old vertical layout.
    bins = (5, 4)
    x_extent = (-25, 100)
    y_extent = (-40, 40)
    x_edges = np.linspace(*x_extent, bins[0] + 1)
    y_edges = np.linspace(*y_extent, bins[1] + 1)

    team_gk_pass_grid, _, _ = np.histogram2d(team_dx, team_dy, bins=bins, range=[x_extent, y_extent])
    league_gk_pass_grid, _, _ = np.histogram2d(league_dx, league_dy, bins=bins, range=[x_extent, y_extent])

    # Convert both to each side's own share of their total GK passes (not a per-90 count)
    # before differencing — this isolates WHERE a keeper concentrates their passing from
    # HOW MUCH they pass overall, which per-90 counts conflate (a keeper who plays short
    # far less often than average would look uniformly "cold" everywhere in a per-90
    # diff, even if their actual distribution shape is completely normal).
    team_total = team_gk_pass_grid.sum()
    league_total = league_gk_pass_grid.sum()
    team_gk_pass_pct = team_gk_pass_grid / team_total * 100 if team_total else team_gk_pass_grid
    league_gk_pass_pct = league_gk_pass_grid / league_total * 100 if league_total else league_gk_pass_grid

    diff_gk_pass_grid = team_gk_pass_pct - league_gk_pass_pct

    # This is diverging data (more than league average vs. less, around a
    # meaningful zero) — team's primary color for "more passes than average" in a
    # zone, secondary color for "less," white at zero.
    cmap = LinearSegmentedColormap.from_list('diff_cmap', [team_colors[1], '#ffffff', team_colors[0]])

    ax = page.add_axes(bbox)
    ax.set_xlim(x_extent)
    ax.set_ylim(y_extent)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_color('#888888')

    # Dashed crosshair through (0,0) — the keeper's own touch location — standing in
    # for the real pitch outline that generate_defensive_action_map draws (there's no
    # actual pitch to draw here since these are relative displacement vectors, not
    # absolute position).
    ax.axvline(0, color='#888888', linewidth=1, linestyle='--')
    ax.axhline(0, color='#888888', linewidth=1, linestyle='--')

    # histogram2d's grid is (nx, ny); pcolormesh wants (ny, nx).
    mesh = ax.pcolormesh(x_edges, y_edges, diff_gk_pass_grid.T, cmap=cmap,
                         vmin=-GK_PASS_DIFF_SCALE, vmax=GK_PASS_DIFF_SCALE)

    # Redraw the crosshair on top of the heatmap fill so it's visible — pcolormesh
    # defaults to zorder=1, same convention as generate_defensive_action_map's
    # pitch-line redraw (there the artists default to zorder=0.9 and need bumping;
    # here the crosshair is redrawn fresh on top for the same reason: to guarantee
    # it renders above the mesh regardless of default zorder ordering).
    existing_artists = set(ax.get_children())
    ax.axvline(0, color='#888888', linewidth=1, linestyle='--')
    ax.axhline(0, color='#888888', linewidth=1, linestyle='--')
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # A dedicated colorbar axes, sized off ax's *actual* resolved position rather
    # than colorbar(ax=ax, ...)'s default "steal space from ax" behavior — the
    # latter shrinks ax a second time, which double-triggers the aspect enforcement
    # above and throws the colorbar/title out of alignment with the plot.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    # extend='both': the scale is now fixed (see the *_DIFF_SCALE constants), so a cell
    # can genuinely exceed it — arrow caps flag that instead of silently flattening an
    # extreme value into the same colour as an ordinary one.
    cbar = page.colorbar(mesh, cax=cbar_ax, extend='both')
    cbar.set_label('% of GK Passes (Team − League)', fontsize=8)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    for i, xc in enumerate(x_centers):
        for j, yc in enumerate(y_centers):
            ax.text(xc, yc, f'{diff_gk_pass_grid[i, j]:+.1f}%',
                    color=diff_label_color(diff_gk_pass_grid[i, j], GK_PASS_DIFF_SCALE, cmap), fontsize=7,
                     ha='center', va='center', zorder=mesh.get_zorder() + 2)

    ax.set_title('Relative GK Passes vs. League Average', fontsize=10, pad=12)
    ax.set_xlabel('Backwards ← Yards From Touch → Forward', labelpad=10, fontsize=8)
    ax.set_ylabel('Left ← Yards From Touch → Right', labelpad=10, fontsize=8)

    # bbox[1] is the requested box's bottom edge, not where the plot actually ends —
    # set_aspect('equal') shrinks and vertically centers the plot within that box. 0.06
    # is a fixed offset down from bbox[1] tuned to clear the tick labels and x-axis
    # label text below the plot (checked against a real render). generate_defensive_
    # action_map uses this same NOTES_OFFSET/bbox[1]=0.50 pairing so the two notes boxes
    # land on the same horizontal line. Centered on the left half of the page (x=0.25)
    # rather than under the chart's own bbox[0], which varies slightly chart to chart.
    # Anchored on the same TOP edge as before (old_y0 + old_height=0.06) and grown
    # downward from there — growing from y0 upward instead ate into the plot above it.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.25 - notes_width / 2
    notes_y0 = bbox[1] + NOTES_OFFSET - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox
#%%
def extract_def_actions(matches):

    matches_def_actions = []
    for match, match_files in matches.items():
        if 'Player_Event_Data.csv' not in match_files:
            continue
        event_data = match_files['Player_Event_Data.csv']
        match_def_actions = event_data[event_data['Event Type'] == 'Defensive Action']
        matches_def_actions.append(match_def_actions)

    if not matches_def_actions:
        # Includes Event Sub-Type even though generate_defensive_action_map itself never
        # reads it — generate_pressing_turnover_map filters on it via this same
        # extractor, and would otherwise KeyError on this empty-data fallback path.
        return pd.DataFrame(columns=['Player', 'Event Type', 'Event Sub-Type', 'Player X Coord', 'Player Y Coord'])

    def_actions = pd.concat(matches_def_actions)

    return def_actions

#%%
def generate_defensive_action_map(leagues, team, team_matches, date, team_colors, page, bbox=[0.59, 0.4683, 0.28, 0.42]):

    league_matches = load_league_data(leagues, date, ['Player_Event_Data.csv'], exclude_team=team)

    # Get defensive action data
    team_def_actions = extract_def_actions(team_matches)
    league_def_actions = extract_def_actions(league_matches)

    if team_def_actions.empty or league_def_actions.empty:
        print('No defensive action data available — skipping defensive action map.')
        return page, None

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background fill doesn't paint over the
    # heatmap when the lines get redrawn on top of it below.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)

    team_def_action_bins = pitch.bin_statistic(team_def_actions['Player X Coord'], team_def_actions['Player Y Coord'], statistic='count', bins=(6, 5))
    league_def_action_bins = pitch.bin_statistic(league_def_actions['Player X Coord'], league_def_actions['Player Y Coord'], statistic='count', bins=(6, 5))

    # Convert both to each side's own share of their total defensive actions (not a
    # per-90 count) before differencing — this isolates WHERE a team concentrates its
    # defending from HOW MUCH defending it does overall, which per-90 counts conflate
    # (a low-possession team that defends more everywhere would look uniformly "hot"
    # in a per-90 diff, even if its actual shape is unremarkable).
    team_total = team_def_action_bins['statistic'].sum()
    league_total = league_def_action_bins['statistic'].sum()
    team_def_action_bins['statistic'] = team_def_action_bins['statistic'] / team_total * 100 if team_total else team_def_action_bins['statistic']
    league_def_action_bins['statistic'] = league_def_action_bins['statistic'] / league_total * 100 if league_total else league_def_action_bins['statistic']

    # bin_statistic returns a dict, not an array — diff the 'statistic' grids
    # specifically and carry the rest (grid edges, etc.) from the team's own bins.
    diff_def_action_bins = dict(team_def_action_bins)
    diff_def_action_bins['statistic'] = team_def_action_bins['statistic'] - league_def_action_bins['statistic']

    # This is diverging data (more than league average vs. less, around a
    # meaningful zero) — team's primary color for "more defensive actions than
    # average" in a zone, secondary color for "less," white at zero.
    cmap = LinearSegmentedColormap.from_list('diff_cmap', [team_colors[1], '#ffffff', team_colors[0]])

    mesh = pitch.heatmap(diff_def_action_bins, ax=ax, cmap=cmap,
                         vmin=-DEF_ACTION_DIFF_SCALE, vmax=DEF_ACTION_DIFF_SCALE)

    # Redraw the pitch lines on top of the heatmap fill so they're visible —
    # mplsoccer's line artists default to zorder=0.9, below pcolormesh's
    # zorder=1, so drawing them again isn't enough on its own; each new artist's
    # zorder has to be bumped above the mesh's or they stay hidden underneath it.
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # A dedicated colorbar axes, sized off ax's *actual* resolved position rather
    # than colorbar(ax=ax, ...)'s default "steal space from ax" behavior — the
    # latter shrinks ax a second time, which double-triggers mplsoccer's aspect
    # enforcement and throws the colorbar/title out of alignment with the pitch.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    # extend='both': the scale is now fixed (see the *_DIFF_SCALE constants), so a cell
    # can genuinely exceed it — arrow caps flag that instead of silently flattening an
    # extreme value into the same colour as an ordinary one.
    cbar = page.colorbar(mesh, cax=cbar_ax, extend='both')
    cbar.set_label('% of Def Actions (Team − League)', labelpad=8, fontsize=8)

    # label_heatmap only accepts a single colour for every label, so the per-cell
    # contrast switch is applied afterwards to the artists it returns. They come back in
    # np.ravel order over the statistic grid (no cells are excluded — exclude_zeros /
    # exclude_nan both default to False), so they zip directly against the values.
    def_action_labels = pitch.label_heatmap(diff_def_action_bins, ax=ax, str_format='{:+.1f}%',
                                            color='black', fontsize=7, ha='center', va='center')
    for annotation, value in zip(def_action_labels, np.ravel(diff_def_action_bins['statistic'])):
        annotation.set_color(diff_label_color(value, DEF_ACTION_DIFF_SCALE, cmap))

    ax.set_title('Defensive Actions vs. League Average', fontsize=10, pad=12)
    #←
    ax.set_xlabel('→   Attack   →', labelpad=8, fontsize=8)

    # Shares NOTES_OFFSET and bbox[1]=0.50 with generate_relative_gk_pass_map so the two
    # notes boxes land on the same horizontal line (see that function for why bbox[1]
    # rather than the pitch's own resolved position is used as the anchor). Centered on
    # the right half of the page (x=0.75), mirroring the left-side charts. Anchored on
    # the same TOP edge as before (old_y0 + old_height=0.06) and grown downward from
    # there — growing from y0 upward instead ate into the plot above it.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.75 - notes_width / 2
    notes_y0 = bbox[1] + NOTES_OFFSET - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox

#%%
# Karun Singh's publicly published Expected Threat (xT) grid — an 8 (width) x 12 (length)
# matrix of possession-value by zone, fit on 2017/18 Premier League event data. Column 0 =
# deep in a team's own defensive third, column 11 = edge of the opponent's box; values rise
# toward the attacking goal. Source: https://karun.in/blog/data/open_xt_12x8_v1.json — the
# exact public data file reused across the analytics community (e.g. cited by the
# socceraction package) rather than re-derived per-project, since this project's own event
# volume is too thin to fit a stable 96-zone model from scratch.
XT_GRID = np.array([
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506,  0.02122129, 0.02756312, 0.03485072, 0.0379259 ],
    [0.00750072, 0.00878589, 0.00942382, 0.0105949,  0.01214719, 0.0138454,  0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.0088799,  0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.0241224,  0.02855202, 0.05491138, 0.06442595],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.0199707,  0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.0199707,  0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.0088799,  0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.0241224,  0.02855202, 0.05491138, 0.06442595],
    [0.00750072, 0.00878589, 0.00942382, 0.0105949,  0.01214719, 0.0138454,  0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506,  0.02122129, 0.02756312, 0.03485072, 0.0379259 ],
])

def lookup_xt(x, y, attacking_toward_100=True):
    '''xT for a team attacking toward x=100 (this file's opta-coordinate convention) at
    pitch position (x, y), each in 0-100. Pass attacking_toward_100=False for the
    opposite direction (mirrors x) — needed here because when the scouted team loses the
    ball, the opponent attacks toward the scouted team's OWN goal (x=0 in this file's own
    attacking-direction convention), not toward x=100.'''
    eval_x = x if attacking_toward_100 else (100 - x)
    col = np.clip((np.asarray(eval_x) / 100 * 12).astype(int), 0, 11)
    row = np.clip((np.asarray(y) / 100 * 8).astype(int), 0, 7)
    return XT_GRID[row, col]

#%%
def extract_offensive_turnover_actions(matches):

    matches_offensive_turnover_actions = []
    for match, match_files in matches.items():
        if 'Player_Event_Data.csv' not in match_files:
            continue
        event_data = match_files['Player_Event_Data.csv']
        # Crosses excluded: a blocked/cleared cross is a normal defensive contest in the
        # box, not a buildup mistake — including them concentrated a lot of "failures"
        # (crosses fail often but cheaply) in the attacking third, muddying the signal
        # this metric is meant to isolate (confirmed by inspecting the actual per-cell
        # numbers — that column's risk was coming from cross-failure rate, not genuine
        # buildup turnover cost).
        match_offensive_turnover_actions = event_data[event_data['Event Type'].isin(['pass', 'dribble'])]
        matches_offensive_turnover_actions.append(match_offensive_turnover_actions)

    if not matches_offensive_turnover_actions:
        return pd.DataFrame(columns=['Player', 'Event Type', 'Outcome', 'Player X Coord', 'Player Y Coord', 'Pass End X Coord', 'Pass End Y Coord'])

    turnover_actions = pd.concat(matches_offensive_turnover_actions)

    return turnover_actions

#%%
def buildup_grids(matches):
    '''Per-zone buildup risk/reward grids over the defensive half, for one pool of
    matches. Factored out of generate_buildup_turnover_map so the scouted team and the
    league-average baseline are computed by identical code — the chart differences the
    two. Returns the press_value / turnover_pct / total_actions grids plus bin edges.'''

    offensive_turnover_actions = extract_offensive_turnover_actions(matches)

    # Restrict to the team's own defensive half — the half they actually build out from.
    # (x=0 is a team's own goal in this file's coordinate convention, confirmed via
    # goalkeeper pass locations clustering around x=5-12.) The attacking half mixes in a
    # different phenomenon — congestion-forced turnovers near the opponent's box, driven
    # by a much higher failure rate on cheap-to-lose actions — that isn't about buildup
    # pressing at all and was muddying the signal this chart is meant to isolate.
    offensive_turnover_actions = offensive_turnover_actions[offensive_turnover_actions['Player X Coord'] <= 50]

    offensive_turnovers = offensive_turnover_actions[offensive_turnover_actions['Outcome'].isna()].copy()

    # A failed dribble is lost right where the dribbler was standing (Pass End X/Y Coord
    # is always NaN for dribbles — there's no separate end location). A failed pass/cross
    # is actually recovered by the opponent wherever it was headed, which can be a very
    # different zone than where it was struck from (e.g. a long ball intercepted at
    # midfield rather than lost at the passer's own feet).
    offensive_turnovers['End X Coord'] = np.where(
        offensive_turnovers['Event Type'] == 'dribble',
        offensive_turnovers['Player X Coord'], offensive_turnovers['Pass End X Coord'])
    offensive_turnovers['End Y Coord'] = np.where(
        offensive_turnovers['Event Type'] == 'dribble',
        offensive_turnovers['Player Y Coord'], offensive_turnovers['Pass End Y Coord'])

    valid_end = np.isfinite(offensive_turnovers['End X Coord'].astype(float)) & np.isfinite(offensive_turnovers['End Y Coord'].astype(float))
    offensive_turnovers = offensive_turnovers[valid_end]

    # xT of the opponent recovering the ball at (End X/Y Coord) — they attack toward THIS
    # team's own goal from there (x=0 in this file's own-attacking-direction convention),
    # the opposite direction from the standard xT grid orientation, hence
    # attacking_toward_100=False.
    offensive_turnovers['xT Conceded'] = lookup_xt(
        offensive_turnovers['End X Coord'].to_numpy(),
        offensive_turnovers['End Y Coord'].to_numpy(),
        attacking_toward_100=False,
    )

    # Reward side: most actions in this zone don't turn the ball over — they're
    # completed, and the team gains (a little or a lot of) xT by progressing. Without
    # this, a zone can look purely risky when it's actually net-positive because the
    # reward on the (much more frequent) successful attempts outweighs the turnover
    # cost. Passes only — successful dribbles have no end-location (Pass End X/Y Coord
    # is always NaN for dribbles), the same limitation the turnover side above already
    # works around, but there's no start-of-turn fallback for a *gain* the way there is
    # for a turnover's own position, so they're excluded here rather than approximated.
    successful_passes = offensive_turnover_actions[
        (offensive_turnover_actions['Outcome'] == True) & (offensive_turnover_actions['Event Type'] == 'pass')
    ].copy()
    valid_pass_end = np.isfinite(successful_passes['Pass End X Coord']) & np.isfinite(successful_passes['Pass End Y Coord'])
    successful_passes = successful_passes[valid_pass_end]

    # attacking_toward_100=True (not False like the turnover side) — the team still has
    # the ball, attacking their own x=100, same convention generate_pressing_turnover_
    # map's 'xT Won' uses.
    successful_passes['xT Start'] = lookup_xt(
        successful_passes['Player X Coord'].to_numpy(),
        successful_passes['Player Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )
    successful_passes['xT End'] = lookup_xt(
        successful_passes['Pass End X Coord'].to_numpy(),
        successful_passes['Pass End Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )
    successful_passes['xT Gained'] = successful_passes['xT End'] - successful_passes['xT Start']

    # Bin over just the defensive half (length 0-50) — pitch.bin_statistic always spans
    # the full pitch with no way to restrict its range, so this uses the same
    # custom-range approach as generate_relative_gk_pass_map for the same reason. Bins
    # keep the same physical cell size as the other charts in this file (16.67 x 14.8
    # units each), just half as many columns as the full-pitch (6,5) grid.
    #
    # Rotated so the goal sits at the top and the halfway line at the bottom: length
    # (0-50, 0=own goal) drives the screen-VERTICAL axis, width (0-100) drives the
    # screen-horizontal axis — the opposite of the other charts in this file, which are
    # all length-horizontal.
    length_bins, width_bins = 3, 5
    length_extent = (0, 50)
    width_extent = (0, 100)
    length_edges = np.linspace(*length_extent, length_bins + 1)
    width_edges = np.linspace(*width_extent, width_bins + 1)

    total_action_grid, _, _ = np.histogram2d(
        offensive_turnover_actions['Player Y Coord'], offensive_turnover_actions['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent])
    turnover_xt_grid, _, _ = np.histogram2d(
        offensive_turnovers['Player Y Coord'], offensive_turnovers['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent], weights=offensive_turnovers['xT Conceded'])
    turnover_count_grid, _, _ = np.histogram2d(
        offensive_turnovers['Player Y Coord'], offensive_turnovers['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent])
    gained_xt_grid, _, _ = np.histogram2d(
        successful_passes['Player Y Coord'], successful_passes['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent], weights=successful_passes['xT Gained'])

    # Some zones may see zero actions at all; guard against 0/0 rather than letting it warn.
    with np.errstate(invalid='ignore', divide='ignore'):
        risk_rate = np.where(total_action_grid > 0, turnover_xt_grid / total_action_grid, np.nan)
        reward_rate = np.where(total_action_grid > 0, gained_xt_grid / total_action_grid, np.nan)
        turnover_pct = np.where(total_action_grid > 0, turnover_count_grid / total_action_grid * 100, np.nan)
    # Signed from the PRESSING team's perspective (risk − reward, not reward − risk):
    # how much this zone costs the ball-playing team per action they attempt in it, i.e.
    # what a presser stands to gain there. Higher = better press target, which is the
    # direction a coach reads the chart in.
    press_value = risk_rate - reward_rate

    return {
        'press_value': press_value,
        'turnover_pct': turnover_pct,
        'total_actions': total_action_grid,
        'length_edges': length_edges,
        'width_edges': width_edges,
    }

#%%
def generate_buildup_turnover_map(leagues, team, team_matches, date, team_colors, page, bbox=[0.08, 0.0684, 0.29, 0.42]):

    team_grids = buildup_grids(team_matches)
    league_grids = buildup_grids(load_league_data(leagues, date, ['Player_Event_Data.csv'], exclude_team=team))

    # Difference the per-action RATES directly — deliberately NOT converting each side to
    # its own share of total first, the way generate_defensive_action_map does. That
    # chart wants to isolate WHERE a team defends from HOW MUCH; here "how much" is the
    # entire point. A team that simply loses the ball less often than the league should
    # read as less pressable everywhere, which is exactly the "sit off, they rarely give
    # it away" call this chart exists to support. Both sides are already per-action
    # rates, so they're directly comparable without re-normalising.
    press_value = team_grids['press_value'] - league_grids['press_value']
    turnover_pct = team_grids['turnover_pct'] - league_grids['turnover_pct']
    length_edges, width_edges = team_grids['length_edges'], team_grids['width_edges']

    # Diverging around a zero that is now genuinely meaningful: the league average. The
    # structural xT offset that made an ABSOLUTE zero meaningless here (a turnover
    # forfeits a whole accumulated stock while a completion adds only a small increment,
    # so the raw quantity comes out negative in nearly every zone for every team)
    # applies equally to every team, and so cancels in the difference.
    #
    # Symmetric vmin/vmax rather than the observed min/max: scaling to each team's own
    # range made the colours within-team normalised, so a side that rarely loses the
    # ball rendered just as hot as one that constantly does. A fixed symmetric scale
    # keeps the same colour meaning the same thing from one opponent's report to the next.
    cmap = LinearSegmentedColormap.from_list('cmap', [team_colors[1], '#ffffff', team_colors[0]])

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background fill doesn't paint over the
    # heatmap when the lines get redrawn on top of it below. Draw the FULL pitch (so the
    # goal/box/halfway-line geometry is correct), then crop to just the defensive half —
    # VerticalPitch maps pitch length to the y-axis with HIGH length at the top (its own
    # half=True shows the opposite/attacking half, the wrong end for this convention), so
    # ylim is set reversed (50, 0) rather than (0, 50) to put OUR goal (length=0) at the
    # top instead.
    pitch = VerticalPitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)
    ax.set_ylim(50, 0)

    # histogram2d's grid here is (width_bins, length_bins); pcolormesh wants
    # (length_bins, width_bins) since Y=length_edges, X=width_edges.
    mesh = ax.pcolormesh(width_edges, length_edges, press_value.T, cmap=cmap,
                         vmin=-BUILDUP_DIFF_SCALE, vmax=BUILDUP_DIFF_SCALE)

    # Redraw the pitch lines on top of the heatmap fill so they're visible — same
    # zorder-bump fix already used in generate_defensive_action_map (mplsoccer's line
    # artists default to zorder=0.9, below pcolormesh's zorder=1).
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    ax.set_ylim(50, 0)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # A dedicated colorbar axes, sized off ax's *actual* resolved position rather
    # than colorbar(ax=ax, ...)'s default "steal space from ax" behavior — the
    # latter shrinks ax a second time, which double-triggers mplsoccer's aspect
    # enforcement and throws the colorbar/title out of alignment with the pitch.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    # extend='both': the scale is now fixed (see the *_DIFF_SCALE constants), so a cell
    # can genuinely exceed it — arrow caps flag that instead of silently flattening an
    # extreme value into the same colour as an ordinary one.
    cbar = page.colorbar(mesh, cax=cbar_ax, extend='both')
    cbar.set_label('Press Value / Action (Team − League)', labelpad=8, fontsize=8)

    # Two-line label, both values now signed diffs against league average: press value
    # on top, turnover-rate gap in percentage points below it. The signs are meaningful
    # again (unlike the previous absolute version) — positive means more pressable than
    # a typical side, negative means they hold the ball better than most there, which is
    # what supports a "sit off" call. A zone with zero recorded actions is NaN (guarded
    # in buildup_grids to avoid a 0/0 warning) rather than a real rate — labeled
    # explicitly instead of printing the literal string "nan".
    width_centers = (width_edges[:-1] + width_edges[1:]) / 2
    length_centers = (length_edges[:-1] + length_edges[1:]) / 2
    for i, wc in enumerate(width_centers):
        for j, lc in enumerate(length_centers):
            label = 'No\nActions' if np.isnan(press_value[i, j]) else f'{press_value[i, j]:+.3f}\n{turnover_pct[i, j]:+.0f}% TO'
            ax.text(wc, lc, label, color=diff_label_color(press_value[i, j], BUILDUP_DIFF_SCALE, cmap),
                     fontsize=6.5, ha='center', va='center', zorder=mesh.get_zorder() + 2)

    ax.set_title('Buildup Vulnerability vs. League Average', fontsize=10, pad=12)
    ax.set_ylabel('←   Attack    ←', labelpad=8, fontsize=8)

    # bbox[1] is the requested box's bottom edge, not where the pitch actually ends —
    # VerticalPitch's aspect lock shrinks and vertically centers the pitch within that
    # box. 0.05 is a fixed offset down from bbox[1] tuned to clear the pitch and its
    # ylabel (checked against a real render). Centered on the left half of the page
    # (x=0.25) rather than under the chart's own bbox[0]. Anchored on the same TOP edge
    # as before (old_y0 + old_height=0.06) and grown downward from there — growing from
    # y0 upward instead ate into the plot above it.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.25 - notes_width / 2
    notes_y0 = bbox[1] + 0.05 - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox

#%%
def mirrored_buildout_actions(matches):
    '''Build-out actions (own defensive half) from `matches`, mirrored into the frame of
    the team pressing them.

    Each team's own Player_Event_Data.csv uses ITS OWN coordinate convention (0 = that
    team's own goal), so a team's build-out half (their own X <= 50) is the same
    real-world zone as their opponent's offensive half, just in the opposite numeric
    frame. Mirroring X (100 - X) lands it on the pressing team's shared grid. Only X is
    mirrored, not Y — this dataset's width axis isn't flipped between the two teams' own
    frames, only the attacking-direction (length) axis is (the same convention lookup_xt
    already relies on elsewhere in this file).'''

    actions = extract_offensive_turnover_actions(matches)
    actions = actions[actions['Player X Coord'] <= 50].copy()
    actions['Player X Coord'] = 100 - actions['Player X Coord']
    return actions

#%%
def pressing_grids(pressing_matches, buildout_actions):
    '''Per-zone press-threat grids over the offensive half, for one pool of matches.
    Factored out of generate_pressing_turnover_map so the scouted team and the league
    baseline are computed by identical code — the chart differences the two.

    `buildout_actions` is the denominator: build-out actions already mirrored into the
    pressing team's frame (see mirrored_buildout_actions). The caller supplies it rather
    than this function deriving it, because the team and league sides source it
    differently — see generate_pressing_turnover_map.'''

    # Numerator: the scouted team's own defensive actions in their offensive half that
    # actually win the ball back cleanly. Interceptions and ball-recoveries clearly
    # count; tackles do too, since this plot is about THIS team winning the ball (unlike
    # the buildup map's turnover count, which excludes tackles for an unrelated reason —
    # an imprecise xT-loss location on a won tackle). Clearances and blocks are left out
    # — defensive contact without necessarily gaining controlled possession.
    pressing_actions = extract_def_actions(pressing_matches)
    pressing_actions = pressing_actions[pressing_actions['Player X Coord'] >= 50]
    pressing_turnovers = pressing_actions[pressing_actions['Event Sub-Type'].isin(['interception', 'tackle', 'ball-recovery'])].copy()

    # Unlike a failed pass/dribble (buildup map), a defensive action has no separate "end
    # location" — the ball is won right where the defender is, so Player X/Y Coord is
    # both the bin location and the point to value.
    #
    # xT of THIS team now attacking from (Player X/Y Coord) — after winning the ball back
    # they attack toward their own x=100 goal (this file's convention), the same
    # direction the xT grid is already oriented for, hence attacking_toward_100=True (no
    # mirroring — the opposite of the buildup map, where the OPPONENT does the attacking
    # after the scouted team loses the ball).
    pressing_turnovers['xT Won'] = lookup_xt(
        pressing_turnovers['Player X Coord'].to_numpy(),
        pressing_turnovers['Player Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )

    # Bin over just the offensive half (length 50-100) — pitch.bin_statistic always spans
    # the full pitch with no way to restrict its range, so this uses the same
    # custom-range approach as generate_buildup_turnover_map for the same reason. Bins
    # keep the same physical cell size as the other charts in this file (16.67 x 14.8
    # units each), just half as many columns as the full-pitch (6,5) grid.
    #
    # Rotated so the opponent's goal sits at the top and the halfway line at the bottom:
    # length (50-100) drives the screen-VERTICAL axis, width (0-100) drives the screen-
    # horizontal axis — the opposite of the other charts in this file, which are all
    # length-horizontal.
    length_bins, width_bins = 3, 5
    length_extent = (50, 100)
    width_extent = (0, 100)
    length_edges = np.linspace(*length_extent, length_bins + 1)
    width_edges = np.linspace(*width_extent, width_bins + 1)

    total_action_grid, _, _ = np.histogram2d(
        buildout_actions['Player Y Coord'], buildout_actions['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent])
    turnover_xt_grid, _, _ = np.histogram2d(
        pressing_turnovers['Player Y Coord'], pressing_turnovers['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent], weights=pressing_turnovers['xT Won'])
    turnover_count_grid, _, _ = np.histogram2d(
        pressing_turnovers['Player Y Coord'], pressing_turnovers['Player X Coord'],
        bins=(width_bins, length_bins), range=[width_extent, length_extent])

    # Some zones may see zero actions at all; guard against 0/0 rather than letting it warn.
    with np.errstate(invalid='ignore', divide='ignore'):
        press_threat = np.where(total_action_grid > 0, turnover_xt_grid / total_action_grid, np.nan)
        win_pct = np.where(total_action_grid > 0, turnover_count_grid / total_action_grid * 100, np.nan)

    return {
        'press_threat': press_threat,
        'win_pct': win_pct,
        'total_actions': total_action_grid,
        'length_edges': length_edges,
        'width_edges': width_edges,
    }

#%%
def generate_pressing_turnover_map(leagues, team, team_matches, date, team_colors, page, bbox=[0.55, 0.0684, 0.29, 0.42]):

    # Team side denominator: the REAL opponents' own offensive activity (pass + dribble)
    # in this zone, not how much the scouted team merely defended there — normalizes
    # press value by how much the true opponent actually tried to play through it.
    # `team_matches` only holds the scouted team's own events (a match folder's
    # Player_Event_Data.csv is a single roster, never both sides), so the real opponents'
    # data for these specific fixtures is fetched separately.
    opponent_matches = load_actual_opponent_events(team_matches, leagues, date, team)
    team_grids = pressing_grids(team_matches, mirrored_buildout_actions(opponent_matches))

    # League side denominator does NOT need that per-fixture opponent matching. Pooled
    # over every team in the league, "build-out actions faced" and "build-out actions
    # performed" are the same multiset — each match contributes both teams' build-out
    # actions either way — so the league pool's own mirrored build-out actions are
    # already the correct denominator. This relies on the league being closed (every
    # opponent also having a scraped folder of their own), which was verified on this
    # dataset. It's also dramatically cheaper than calling load_actual_opponent_events
    # once per team.
    league_matches = load_league_data(leagues, date, ['Player_Event_Data.csv'], exclude_team=team)
    league_grids = pressing_grids(league_matches, mirrored_buildout_actions(league_matches))

    # Difference the per-action rates directly (see generate_buildup_turnover_map for why
    # these are NOT re-normalised to shares first).
    press_threat = team_grids['press_threat'] - league_grids['press_threat']
    press_win_pct = team_grids['win_pct'] - league_grids['win_pct']
    length_edges, width_edges = team_grids['length_edges'], team_grids['width_edges']

    # No sign flip here, unlike generate_buildup_turnover_map — that chart's raw quantity
    # was the ball-playing team's own value and had to be negated to read as the
    # presser's gain. This chart scouts the OPPOSITION's press (generateSecondPage passes
    # `opposition` as `team`), so a high value already means they win the ball back
    # dangerously there, i.e. a zone to avoid building through. Positive now means they
    # press that zone more dangerously than a typical side does; negative means it's a
    # relatively safe area to play through against them.
    cmap = LinearSegmentedColormap.from_list('cmap', [team_colors[1], '#ffffff', team_colors[0]])

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background fill doesn't paint over the
    # heatmap when the lines get redrawn on top of it below. Draw the FULL pitch (so the
    # goal/box/halfway-line geometry is correct), then crop to just the offensive half —
    # VerticalPitch's default already puts HIGH length (100, the opponent's goal) at the
    # top, which is exactly what's wanted here, so unlike generate_buildup_turnover_map
    # (which needs a reversed ylim to put length=0 at the top instead) this crop uses the
    # default direction as-is.
    pitch = VerticalPitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)
    ax.set_ylim(50, 100)

    # histogram2d's grid here is (width_bins, length_bins); pcolormesh wants
    # (length_bins, width_bins) since Y=length_edges, X=width_edges.
    mesh = ax.pcolormesh(width_edges, length_edges, press_threat.T, cmap=cmap,
                         vmin=-PRESSING_DIFF_SCALE, vmax=PRESSING_DIFF_SCALE)

    # Redraw the pitch lines on top of the heatmap fill so they're visible — same
    # zorder-bump fix already used in generate_defensive_action_map (mplsoccer's line
    # artists default to zorder=0.9, below pcolormesh's zorder=1).
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    ax.set_ylim(50, 100)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # A dedicated colorbar axes, sized off ax's *actual* resolved position rather
    # than colorbar(ax=ax, ...)'s default "steal space from ax" behavior — the
    # latter shrinks ax a second time, which double-triggers mplsoccer's aspect
    # enforcement and throws the colorbar/title out of alignment with the pitch.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    # extend='both': the scale is now fixed (see the *_DIFF_SCALE constants), so a cell
    # can genuinely exceed it — arrow caps flag that instead of silently flattening an
    # extreme value into the same colour as an ordinary one.
    cbar = page.colorbar(mesh, cax=cbar_ax, extend='both')
    cbar.set_label('Press Threat / Action (Team − League)', labelpad=8, fontsize=8)

    # Two-line label mirroring generate_buildup_turnover_map's: press threat on top, and
    # below it the share of build-out actions in that zone the opposition actually wins
    # — which separates a zone they win rarely but devastatingly from one they win often
    # and cheaply. Unsigned '{:.3f}' (values are small non-negative decimals and the
    # scale is relative). A zone with zero recorded actions is NaN (guarded above to
    # avoid a 0/0 warning) rather than a real rate — labeled explicitly instead of
    # printing the literal string "nan".
    width_centers = (width_edges[:-1] + width_edges[1:]) / 2
    length_centers = (length_edges[:-1] + length_edges[1:]) / 2
    for i, wc in enumerate(width_centers):
        for j, lc in enumerate(length_centers):
            label = 'No\nActions' if np.isnan(press_threat[i, j]) else f'{press_threat[i, j]:+.3f}\n{press_win_pct[i, j]:+.0f}% Won'
            ax.text(wc, lc, label, color=diff_label_color(press_threat[i, j], PRESSING_DIFF_SCALE, cmap),
                        fontsize=6.5, ha='center', va='center', zorder=mesh.get_zorder() + 2)

    ax.set_title('Press Threat vs. League Average', fontsize=10, pad=12)
    ax.set_ylabel('→   Attack    →', labelpad=8, fontsize=8)

    # bbox[1] is the requested box's bottom edge, not where the pitch actually ends —
    # VerticalPitch's aspect lock shrinks and vertically centers the pitch within that
    # box. 0.05 is a fixed offset down from bbox[1] tuned to clear the pitch and its
    # ylabel (checked against a real render). Centered on the right half of the page
    # (x=0.75), mirroring the left-side charts. Anchored on the same TOP edge as before
    # (old_y0 + old_height=0.06) and grown downward from there — growing from y0 upward
    # instead ate into the plot above it.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.75 - notes_width / 2
    notes_y0 = bbox[1] + 0.05 - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox

#%%
def extract_key_passes(matches):

    matches_key_passes = []
    for match, match_files in matches.items():
        if 'Player_Event_Data.csv' not in match_files:
            continue
        event_data = match_files['Player_Event_Data.csv']
        match_key_passes = event_data[event_data['Key Pass'] == True]
        matches_key_passes.append(match_key_passes)

    if not matches_key_passes:
        return pd.DataFrame(columns=['Player', 'Key Pass', 'Player X Coord', 'Player Y Coord', 'Pass End X Coord', 'Pass End Y Coord'])

    key_passes = pd.concat(matches_key_passes)

    # Corner kicks come through with the same Event Type as open-play crosses/passes (no
    # situation/set-piece flag exists in this data), but are struck from a standardized
    # placeholder coordinate right at the corner arc rather than a real tracked position
    # — confirmed empirically: of all crosses near a corner arc, 79% sit at just two
    # exactly-repeated coordinates (99.5/0.5 and 99.5/99.5), versus organically varied
    # positions for genuine open-play byline deliveries. Excluding them here (not just in
    # the chart) keeps every downstream user — the heatmap, the scatter dots, and the
    # per-90 counts — limited to open play.
    from_corner_arc = (key_passes['Player X Coord'] > 99) & ((key_passes['Player Y Coord'] < 1) | (key_passes['Player Y Coord'] > 99))
    key_passes = key_passes[~from_corner_arc]

    return key_passes

#%%
# bbox[0]=0.03 sits the pitch's left edge just inside the page's inner border, which
# generatePageTemplate puts at (6+5)pt / (72*8.5in) = 0.018 in figure fractions. The
# pitch is width-constrained inside this bbox (its aspect lock shrinks the axes
# vertically, not horizontally), so the drawn pitch starts at bbox[0] itself.
def generate_key_pass_map(team_matches, team_colors, page, bbox=[0.03, 0.50, 0.28, 0.42],
                          expected_lineup=None, expected_subs=None):

    key_passes = extract_key_passes(team_matches)

    if key_passes.empty:
        print('No key pass data available — skipping key pass map.')
        return page, None

    valid_end = np.isfinite(key_passes['Pass End X Coord']) & np.isfinite(key_passes['Pass End Y Coord'])
    key_passes = key_passes[valid_end].copy()

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background matches the page instead of
    # painting a separate fill behind it — same as every other chart in this file.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)

    # xT of where THIS team's key pass was delivered to — after playing it, they're
    # still attacking toward their own x=100 goal (this file's convention), the same
    # direction the xT grid is already oriented for, hence attacking_toward_100=True (no
    # mirroring — same reasoning as generate_pressing_turnover_map's 'xT Won').
    key_passes['xT Delivered'] = lookup_xt(
        key_passes['Pass End X Coord'].to_numpy(),
        key_passes['Pass End Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )

    # Background: binned by ORIGIN (Player X/Y Coord — where the passer had to be to
    # create it), valued by DELIVERY (Pass End Coord — how dangerous the resulting
    # position was). Answers "how much delivered danger has originated from this zone" —
    # an absolute magnitude, not diffed against league average (same reasoning
    # generate_buildup_turnover_map already uses for its own xT rate: this is a
    # self-normalized metric with meaningful units, and a prevention decision needs
    # absolute risk, not relative-to-league risk). Full pitch, same (6,5) grid
    # generate_defensive_action_map uses — key passes aren't restricted to one half the
    # way the turnover charts are.
    xt_bins = pitch.bin_statistic(key_passes['Player X Coord'], key_passes['Player Y Coord'],
                                   values=key_passes['xT Delivered'], statistic='sum', bins=(6, 5))

    cmap = LinearSegmentedColormap.from_list('cmap', ['#ffffff', team_colors[0]])
    max_val = np.nanmax(xt_bins['statistic'])
    if not max_val:
        max_val = 1

    # No per-cell numeric labels on this one — the scatter dots on top already carry
    # per-pass detail, and stacking text under them would clutter the chart rather than
    # clarify it.
    mesh = pitch.heatmap(xt_bins, ax=ax, cmap=cmap, vmin=0, vmax=max_val, zorder=1)

    # Redraw the pitch lines on top of the heatmap fill so they're visible — same
    # zorder-bump fix used by every other heatmap chart in this file.
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # One player at a time (rather than a single scatter with a c= color array) so
    # ax.legend() can auto-build a correct player -> color key with no extra work.
    # tab20 gives 20 visually distinct colors — comfortably covers a realistic number of
    # distinct key-pass takers across a date range; colors cycle rather than crash if a
    # squad ever produces more than 20. Sorted by per-90 descending (not alphabetically)
    # so the most creative players surface at the top of the key.
    player_minutes = compute_player_minutes(team_matches)

    def per_90(player):
        minutes = player_minutes.get(player, 0)
        count = (key_passes['Player'] == player).sum()
        return count / minutes * 90 if minutes else 0

    players = sorted(key_passes['Player'].dropna().unique(), key=per_90, reverse=True)
    named_players, other_players = players[:KEY_PASS_LEGEND_MAX], players[KEY_PASS_LEGEND_MAX:]

    # Minutes the team itself played, as the denominator for the qualification share.
    team_minutes = compute_total_minutes(team_matches)
    # None means "couldn't check", which must not grey anybody — see load_current_squad.
    current_squad = load_current_squad(team_matches)

    # Players the lineup model expects to feature in the match being scouted. The two
    # frames name the column differently — expected_lineup uses 'Player Name', while
    # expected_subs uses 'Player' — confirmed against live output. Every predicted sub
    # sat at 0.47-0.57 Sub In Probability, so no probability floor is applied: being in
    # the predicted squad at all is the signal.
    predicted_to_feature = set()
    for frame, column in ((expected_lineup, 'Player Name'), (expected_subs, 'Player')):
        if frame is not None and len(frame) and column in frame.columns:
            predicted_to_feature |= set(frame[column].dropna())

    unqualified_labels = []

    # tab20 is built as TEN HUE PAIRS — a saturated and a pale version of each colour at
    # adjacent indices — so walking it in order hands consecutive players two shades of
    # the same hue (#1/#2 dark and light blue, #3/#4 dark and light orange...) which at
    # this dot size are genuinely hard to tell apart. Taking the even indices first gives
    # all ten saturated hues before any pale variant appears, so the highest-ranked
    # players are maximally distinct and a pale shade only ever sits far down the list
    # from its dark partner.
    palette = plt.get_cmap('tab20')
    distinct_colors = [palette(i) for i in range(0, 20, 2)] + [palette(i) for i in range(1, 20, 2)]

    for i, player in enumerate(named_players):
        player_passes = key_passes[key_passes['Player'] == player]
        minutes = player_minutes.get(player, 0)
        # Greying answers one question: is this player expected on the pitch? The lineup
        # prediction answers it directly, so it wins outright — a predicted starter stays
        # black however few minutes they have banked. That matters: checked against the
        # real 8-16-2026 fixture, Jamal Thiaré (21% of minutes) and Mohamed Farsi (26%)
        # are both predicted to START, and a minutes-only rule greyed out two of the
        # eleven players the coach most needs to plan for. Their minutes are still printed
        # in the label, so the thin sample is visible without being buried.
        #
        # Minutes and roster remain the fallback for everyone the model didn't name. The
        # `or` also handles an unavailable prediction for free — when the predictor
        # returns nothing (it does exactly this for a date with no scheduled fixture,
        # giving XI=0/subs=0) the first term is simply always False and the rule collapses
        # back to the previous minutes-and-roster behaviour rather than greying the squad.
        enough_minutes = team_minutes and (minutes / team_minutes) >= KEY_PASS_MINUTES_QUALIFIED
        on_roster = current_squad is None or player in current_squad
        qualified = (player in predicted_to_feature) or (enough_minutes and on_roster)
        # Minutes shown alongside the rate so a reader can see for themselves what the
        # rate rests on — costs only ~0.008 of page width, and answers "why is this
        # player ranked here?" without needing a confidence model explained to them.
        label = f'{abbreviate_player_name(player)} ({per_90(player):.2f}, {minutes:.0f}\')'
        # Circles at the pass ORIGIN (Player X/Y Coord) — not the end location — per the
        # requested design: this chart is about who creates chances and from where, not
        # where they end up (already shown implicitly by "key pass" leading to a shot).
        ax.scatter(player_passes['Player X Coord'], player_passes['Player Y Coord'],
                   color=distinct_colors[i % len(distinct_colors)], s=45, edgecolors='black',
                   linewidths=0.5, alpha=0.65, label=label, zorder=3)
        unqualified_labels.append(not qualified)

    # Everyone past the cap still gets their passes plotted — dropping them would remove
    # real events from the chart — but shares one grey swatch and a single legend row, so
    # a deep squad can't grow the key without bound. They're the lowest per-90 creators by
    # construction, so the detail lost is the least interesting.
    if other_players:
        other_passes = key_passes[key_passes['Player'].isin(other_players)]
        ax.scatter(other_passes['Player X Coord'], other_passes['Player Y Coord'],
                   color='#999999', s=45, edgecolors='black', linewidths=0.5,
                   alpha=0.65, label=f'Other ({len(other_players)} players)', zorder=3)

    # A dedicated colorbar axes close against the pitch (same pattern as every other
    # chart in this file), with the player legend positioned further out so the two
    # side-by-side "keys" don't overlap.
    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    cbar = page.colorbar(mesh, cax=cbar_ax)
    cbar.set_label('ΣxT Delivered / Zone', labelpad=6, fontsize=8)

    # Set before the legend below, which measures the title to align itself against it.
    title_fontsize, title_pad = 10, 12
    ax.set_title('Key Passes', fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel('→    Attack    →', labelpad=8, fontsize=8)

    # Two things pin the legend here:
    #
    # 'upper left' rather than 'center left' — a vertically centered legend drifts with
    # the number of players in the key-pass list, so its top edge only lines up with the
    # title by coincidence and moves from one opponent to the next. Anchoring the top
    # fixes it regardless of how many entries there are.
    #
    # ...and the anchor sits ABOVE y=1.0, because y=1.0 is the axes' top edge, which is
    # where the pitch ends, not where the title sits — the title is `pad` points further
    # up. Converting that pad (plus ~0.8 * fontsize to reach the top of the capitals)
    # out of points and into axes fractions puts the two tops level, and keeps them level
    # if the figure size or font sizes change. get_position() is the aspect-resolved box
    # by this point, same assumption the colorbar placement above already relies on.
    axes_height_points = ax.get_position().height * page.get_figheight() * 72
    legend_top = 1.0 + (title_pad + 0.8 * title_fontsize) / axes_height_points

    # x=1.40 is as close as the legend can sit without running into the colorbar's
    # rotated label, which occupies the gap between pitch and legend (1.28 overlapped it).
    # labelspacing below matplotlib's 0.5 default: at the 15-player cap the legend runs
    # to 16 rows, which at default spacing bottomed out at y=0.608 and clipped the notes
    # box starting at 0.610. Tightening the gaps buys back the room without shrinking the
    # text, which at fontsize 6 has none to spare.
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.40, legend_top), fontsize=6,
                       # 'KP/90' rather than spelling it out: at fontsize 7 the legend TITLE is the
                       # widest element in the block (0.170 of page width spelled out, vs 0.117 for
                       # the longest player row), so it — not the rows — is what sets the legend's
                       # width and what pushed it over the conceded chart's left edge.
                       title='Player (KP/90, Mins)', title_fontsize=7, frameon=False,
                       labelspacing=0.3)

    # Grey the rows whose minutes fall short of KEY_PASS_MINUTES_QUALIFIED. Applied to the
    # TEXT rather than the marker, deliberately: the marker colour is the key that ties a
    # row to its dots on the pitch, so recolouring it would break that link. Legend texts
    # come back in the order the labelled artists were added, and the trailing "Other" row
    # (when present) has no entry in unqualified_labels, so zip stops before it.
    for text, unqualified in zip(legend.get_texts(), unqualified_labels):
        if unqualified:
            text.set_color('#6b6b6b')

    # bbox[1] is the requested box's bottom edge, not where the pitch actually ends —
    # mplsoccer's aspect lock shrinks and vertically centers the pitch within that box.
    # 0.05 is a fixed offset down from bbox[1] tuned to clear the pitch and its xlabel
    # (checked against a real render, same value used by the sibling turnover charts).
    # Anchored on the same TOP edge as before (old_y0 + old_height=0.06) and grown
    # downward from there — growing from y0 upward instead ate into the plot above it.
    #
    # Aligned to the chart itself rather than centred on the page's left half (x=0.25)
    # like the page-two charts: once this chart moved out to bbox[0]=0.03 the page-half
    # centre left the box visibly offset to the right of the pitch above it. Matching the
    # chart's own x-extent also avoids the NOTES_WIDTH_PAD overhang, which at this x
    # would have pushed the box's left edge to 0.01 — outside the page's inner border
    # at 0.018.
    notes_width = bbox[2]
    notes_x0 = bbox[0]
    notes_y0 = bbox[1] + 0.05 - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox

#%%
# bbox[0]=0.60 clears the offensive sibling's player legend, which extends to ~0.588 —
# at 0.53 the legend ran straight over this chart's pitch and title. The chart plus its
# colourbar and rotated label occupy ~0.345, so 0.60 still finishes inside the page's
# inner border at 0.982.
def generate_def_key_pass_map(leagues, team, team_matches, date, team_colors, page, bbox=[0.60, 0.50, 0.28, 0.42]):
    '''Where the scouted team CONCEDES key passes — the defensive mirror of
    generate_key_pass_map.'''

    # A team's own Player_Event_Data.csv holds only their own events, so the key passes
    # played *against* them live in their opponents' files — the same cross-reference
    # generate_pressing_turnover_map relies on for its denominator.
    opponent_matches = load_actual_opponent_events(team_matches, leagues, date, team)
    key_passes = extract_key_passes(opponent_matches)

    if key_passes.empty:
        print('No conceded key pass data available — skipping defensive key pass map.')
        return page, None

    valid_end = np.isfinite(key_passes['Pass End X Coord']) & np.isfinite(key_passes['Pass End Y Coord'])
    key_passes = key_passes[valid_end].copy()

    # Value FIRST, mirror second — the order matters. These coordinates are still in each
    # opponent's own frame (they attack toward their own x=100), so the delivery's threat
    # is the same attacking_toward_100=True lookup the offensive sibling makes. Only after
    # that are the coordinates mirrored for display. Mirroring first and compensating with
    # attacking_toward_100=False yields an identical number, but this ordering is much
    # harder to get subtly wrong later.
    key_passes['xT Delivered'] = lookup_xt(
        key_passes['Pass End X Coord'].to_numpy(),
        key_passes['Pass End Y Coord'].to_numpy(),
        attacking_toward_100=True,
    )

    # Mirror into the SCOUTED team's frame so this chart and its offensive sibling share
    # one coordinate system: the scouted team's own goal at x=0, the opposition attacking
    # leftward toward it. Chances they create then cluster right and chances they concede
    # cluster left, which is what makes the pair readable side by side. X only, never Y —
    # this dataset's width axis isn't flipped between two teams' own frames (see
    # mirrored_buildout_actions for the same convention).
    #
    # extract_key_passes' corner-kick filter keys off X > 99 in the passer's OWN frame, so
    # it has already run correctly above, before this mirror.
    for coord in ('Player X Coord', 'Pass End X Coord'):
        key_passes[coord] = 100 - key_passes[coord]

    ax = page.add_axes(bbox)
    # pitch_color='none' so the pitch's own background matches the page instead of
    # painting a separate fill behind it — same as every other chart in this file.
    pitch = Pitch(pitch_type='opta', pitch_color='none', line_color='#888888', linewidth=1)
    pitch.draw(ax=ax)

    # Binned by ORIGIN (where the opponent passer was), valued by DELIVERY (how dangerous
    # the position they found was) — the same construction as the offensive sibling, and
    # absolute rather than league-diffed for the same reason.
    xt_bins = pitch.bin_statistic(key_passes['Player X Coord'], key_passes['Player Y Coord'],
                                   values=key_passes['xT Delivered'], statistic='sum', bins=(6, 5))

    cmap = LinearSegmentedColormap.from_list('cmap', ['#ffffff', team_colors[0]])
    max_val = np.nanmax(xt_bins['statistic'])
    if not max_val:
        max_val = 1

    mesh = pitch.heatmap(xt_bins, ax=ax, cmap=cmap, vmin=0, vmax=max_val, zorder=1)

    # Redraw the pitch lines on top of the heatmap fill so they're visible — same
    # zorder-bump fix used by every other heatmap chart in this file.
    existing_artists = set(ax.get_children())
    pitch.draw(ax=ax)
    for artist in ax.get_children():
        if artist not in existing_artists:
            artist.set_zorder(mesh.get_zorder() + 1)

    # One flat colour and no legend, unlike the offensive sibling: these passes come from
    # whichever opponents the team happened to face (12 players across 3 different clubs
    # in the Pumas sample), so a per-player key would be listing strangers from unrelated
    # teams rather than a squad. team_colors[1] over a team_colors[0] heatmap is the same
    # contrast assumption the diverging maps elsewhere in this file already make.
    # alpha=0.45 rather than the sibling's 0.8: those dots are spread across a tab20
    # palette, but these are all one colour, so at high opacity they read as a solid mass
    # that swamps the heatmap underneath — which is the layer carrying the actual xT
    # magnitude. The lower opacity also makes overlapping dots legible as clusters.
    ax.scatter(key_passes['Player X Coord'], key_passes['Player Y Coord'],
               color=team_colors[1], s=45, edgecolors='black', linewidths=0.4,
               alpha=0.45, zorder=3)

    pitch_pos = ax.get_position()
    cbar_ax = page.add_axes([pitch_pos.x1 + 0.02, pitch_pos.y0, 0.015, pitch_pos.height])
    cbar = page.colorbar(mesh, cax=cbar_ax)
    cbar.set_label('ΣxT Conceded / Zone', labelpad=6, fontsize=8)

    ax.set_title('Key Passes Conceded', fontsize=10, pad=12)
    # Arrow points LEFT because the opposition attacks toward x=0 in this mirrored frame —
    # the visible cue that this is the defensive mirror of the chart beside it.
    ax.set_xlabel('←    Opposition Attack    ←', labelpad=8, fontsize=8)

    # Centered on the right half of the page (x=0.75), mirroring the left-side sibling —
    # the same convention the page-two charts use for their notes boxes.
    notes_width = bbox[2] + NOTES_WIDTH_PAD
    notes_x0 = 0.75 - notes_width / 2
    notes_y0 = bbox[1] + 0.05 - 0.06
    notes_bbox = draw_notes_box(page, [notes_x0, notes_y0, notes_width, 0.12])

    return page, notes_bbox
#%%
def main():


    return

if __name__ == "__main__":
    main()