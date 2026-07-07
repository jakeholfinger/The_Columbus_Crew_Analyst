# %%
import pandas as pd
import numpy as np
import os
from curl_cffi import requests
import time
from scipy.stats import linregress
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD, value
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import contextlib
from datetime import date, datetime, timezone, timedelta
# %%
def GetFiles(currentPath):
    '''Returns a list of all visible files inside the folder at currentPath'''
    return [f for f in os.listdir(currentPath) if not f.startswith('.')]
# %%
def getURLData(apiURL):
    #Declare a header dictionary, which makes the bot look human and prevent 403 errors
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'priority': 'u=1, i',
        'referer': 'https://www.sofascore.com/',
        'sec-ch-ua': '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
        'x-requested-with': '5d7735',
        #'If-Modified-Since': 'Sat, 14 Feb 2026 00:00:00 GMT'
    }

    response = None
    for attempt in range(3):
        try:
            response = requests.get(apiURL, headers=headers, impersonate="chrome120", timeout=30)
            break  # success, exit loop
        except requests.exceptions.Timeout:
            print(f"Timeout, retrying... attempt {attempt+1}")
            time.sleep(5)  # wait a bit before retrying

    if response is None:
        print(f"Failed to fetch {apiURL} after 3 attempts, skipping.")
        return
    
    #Convert raw data (which is in bytes) into JSON file
    dataJSON = response.json()

    return dataJSON
# %%
def scrapeNextMatches(teamID, targetDateTimestamp):
    teamID = int(teamID)
    targetTimestamp = int(targetDateTimestamp.timestamp())

    matchesData = []
    matchIndex = None
    count = 0
    done = False

    # Collect all upcoming matches up to and including the match after the target
    while not done:
        apiURL = f'https://www.sofascore.com/api/v1/team/{teamID}/events/next/{count}'
        data = getURLData(apiURL)
        events = data.get('events', [])

        for match in events:
            matchRow = {
                'matchID': match.get('id'),
                'matchTime': match.get('startTimestamp'),
                'team': match.get('homeTeam', {}).get('name') if match.get('homeTeam', {}).get('id') == teamID else match.get('awayTeam', {}).get('name'),
                'isHome': 'Home' if match.get('homeTeam', {}).get('id') == teamID else 'Away',
                'opposition': match.get('awayTeam', {}).get('name') if match.get('homeTeam', {}).get('id') == teamID else match.get('homeTeam', {}).get('name'),
                'oppositionID': match.get('awayTeam', {}).get('id') if match.get('homeTeam', {}).get('id') == teamID else match.get('homeTeam', {}).get('id'),
                'competition': match.get('tournament', {}).get('uniqueTournament', {}).get('name'),
            }
            matchesData.append(matchRow)

            if matchIndex is not None:
                # This is the match after the target — done
                done = True
                break

            if match.get('startTimestamp') > targetTimestamp:
                matchIndex = len(matchesData) - 1

        if not done:
            if not data.get('hasNextPage'):
                done = True
            else:
                count += 1

    if matchIndex is None:
        print(f'No future match found for team {teamID} after {targetDateTimestamp}')
        return pd.DataFrame(), {}

    # Scrape missing players for each match
    matchesMissingPlayers = []
    predictedMissingPlayers = {}
    for i, match in enumerate(matchesData):
        apiURL = f'https://www.sofascore.com/api/v1/event/{match["matchID"]}/lineups'
        lineupDataJSON = getURLData(apiURL)

        isHome = match['isHome'] == 'Home'
        teamData = lineupDataJSON.get('home', {}) if isHome else lineupDataJSON.get('away', {})

        missingPlayersDict = {}
        for player in teamData.get('missingPlayers', []):
            missingPlayersDict[player['player']['name']] = {
                'ID': player['player']['id'],
                'Availibility': player.get('type', ''),
                'Expected End Date': player.get('expectedEndDate', '').split('T')[0]
            }
        matchesMissingPlayers.append(missingPlayersDict)

        if i == matchIndex:
            predictedMissingPlayers = missingPlayersDict

    # Get full squad
    playersDataJSON = getURLData(f'https://www.sofascore.com/api/v1/team/{teamID}/players')
    squad = {}
    for playerEntry in playersDataJSON.get('players', []):
        player = playerEntry.get('player', {})
        squad[player['name']] = {
            'ID': player.get('id'),
            'Position': player.get('position', ''),
            'Detailed Positions': player.get('positionsDetailed', []),
            'Sofascore Market Value': player.get('proposedMarketValue')
        }

    # Build futureDF: one row per available player per match
    rows = []
    for matchRow, missingPlayers in zip(matchesData, matchesMissingPlayers):
        for playerName, playerInfo in squad.items():
            if playerName in missingPlayers:
                continue
            rows.append({
                'Player Name': playerName,
                'ID': playerInfo['ID'],
                'Position': playerInfo['Position'],
                'Detailed Positions': playerInfo['Detailed Positions'],
                'Sofascore Market Value': playerInfo['Sofascore Market Value'],
                'Match ID': matchRow['matchID'],
                'Start Timestamp': matchRow['matchTime'],
                'Team': matchRow['team'],
                'Team ID': teamID,
                'Home': matchRow['isHome'] == 'Home',
                'Opposition': matchRow['opposition'],
                'Opposition ID': matchRow['oppositionID'],
                'Competition': matchRow['competition'],
                'Is Future Match': True,
            })

    return pd.DataFrame(rows), predictedMissingPlayers, matchesData[matchIndex]['matchID']
#%%
def scrapeOppositionCompetitions(oppositionIDs):

    competitions = []
    competitionIDs = []

    for oppositionID in oppositionIDs:
        apiURL = f'https://www.sofascore.com/api/v1/team/{oppositionID}'

        #Convert raw data (which is in bytes) into JSON file
        dataJSON = getURLData(apiURL)

        data = dataJSON.get('team', {}).get('primaryUniqueTournament', {})

        competitionName = data.get('name')
        competitionID = data.get('id')

        competitions.append(competitionName)
        competitionIDs.append(competitionID)

    return competitions, competitionIDs
# %%
def normalizeFormation(formation):
    '''Converts any formation value to a plain dash-separated string.
    Handles actual lists and stringified lists e.g. "['3-4-3']" that
    result from CSV serialization of list-typed Formation columns.'''
    if isinstance(formation, list):
        formation = formation[0] if formation else ''
    s = str(formation).strip()
    if s.startswith('['):
        import ast
        try:
            lst = ast.literal_eval(s)
            s = str(lst[0]).strip() if lst else s
        except (ValueError, SyntaxError):
            pass
    return s
#%%
def getFormationSlots(formation):
    if formation == '4-3-3':
        return ['GK', 'DR', 'DCR', 'DCL', 'DL', 'MCR', 'MC', 'MCL', 'RW', 'ST', 'LW']
    elif formation == '4-4-2':
        return ['GK', 'DR', 'DCR', 'DCL', 'DL', 'MR', 'MCR', 'MCL', 'ML', 'STR', 'STL']
    elif formation == '4-2-3-1':
        return ['GK', 'DR', 'DCR', 'DCL', 'DL', 'DMR', 'DML', 'MR', 'AM', 'ML', 'ST']
    elif formation == '4-1-3-2':
        return ['GK', 'DR', 'DCR', 'DCL', 'DL', 'DM', 'MR', 'AM', 'ML', 'STR', 'STL']
    elif formation == '4-4-1-1':
        return ['GK', 'DR', 'DCR', 'DCL', 'DL', 'MR', 'MCR', 'MCL', 'ML', 'AM', 'ST']
    elif formation == '4-3-1-2':
        return ['GK', 'DR', 'DCR', 'DCL', 'DL', 'MCR', 'MC', 'MCL', 'AM', 'STR', 'STL']
    elif formation == '4-1-4-1':
        return ['GK', 'DR', 'DCR', 'DCL', 'DL', 'DM', 'MR', 'MCR', 'MCL', 'ML', 'ST']
    elif formation == '3-4-2-1' or formation == '3-4-3':
        return ['GK', 'DCR', 'DC', 'DCL', 'MR', 'MCR', 'MCL', 'ML', 'AMR', 'ST', 'AML']
    elif formation == '3-4-1-2':
        return ['GK', 'DCR', 'DC', 'DCL', 'MR', 'MCR', 'MCL', 'MR', 'AM', 'STR', 'STL']
    elif formation == '3-5-2':
        return ['GK', 'DCR', 'DC', 'DCL', 'MR', 'MCR', 'MC', 'MCL', 'MR', 'STR', 'STL']
    elif formation == '5-3-2':
        return ['GK', 'DR', 'DCR', 'DC', 'DCL', 'DL', 'MCR', 'MC', 'MCL', 'STR', 'STL']
    elif formation == '5-4-1':
        return ['GK', 'DR', 'DCR', 'DC', 'DCL', 'DL', 'MR', 'MCR', 'MCL', 'ML', 'ST']
    else:
        print(f'{formation} not found. Returning 4-3-3.')
        return ['GK', 'DR', 'DCR', 'DCL', 'DL', 'MCR', 'MC', 'MCL', 'RW', 'ST', 'LW']
#%%
SLOT_COORDS = {
                                        'GK':  (5,  50),
    'DR':  (25, 12),  'DCR': (25, 33),  'DC':  (25, 50),  'DCL': (25, 67),  'DL':  (25, 88),
                      'DMR': (38, 33),  'DM':  (38, 50),  'DML': (38, 67),  'ML':  (50, 88),
    'MR':  (50, 12),  'MCR': (50, 33),  'MC':  (50, 50),  'MCL': (50, 67),
    'RW':  (68, 12),  'AMR': (68, 35),  'AM':  (62, 50),  'AML': (68, 65),  'LW':  (68, 88),
                      'STR': (72, 35),  'ST':  (72, 50),  'STL': (72, 65),  
}

SLOT_CATEGORY = {
    'GK': 'G',
    'DR': 'D', 'DCR': 'D', 'DC': 'D', 'DCL': 'D', 'DL': 'D',
    'MR': 'M', 'DMR': 'M', 'DM': 'M', 'DML': 'M', 'ML': 'M',
    'MCR': 'M', 'MC': 'M', 'MCL': 'M', 'AM': 'M',
    'RW': 'F', 'AMR': 'F', 'STR': 'F', 'ST': 'F', 'STL': 'F', 'AML': 'F', 'LW': 'F',
}
#%%
def inferPlayerPositions(df):

    formation = df['Formation'].iloc[0]
    formationSlots = getFormationSlots(formation)

    df['Formation Slot'] = pd.Series(formationSlots)

    return df
# %%
def getData(leagues, team, date):
    matchAttributesList = []
    missingPlayersList = []
    playerStatisticsList = []
    subsList = []
    formationsList = []
    playerHeatmapsList = []

    month = int(date.split('-')[0])
    day = int(date.split('-')[1])
    year = int(date.split('-')[2])

    today = datetime.now(timezone.utc)
    currentYear = today.year
    currentMonth = today.month
    currentDay = today.day

    # Read data into lists of dataframes
    for league in leagues:
        teamPath = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data/{league.replace(' ', '_')}_Data/{team.replace(' ', '_')}_Data'
        for matchFolder in GetFiles(teamPath):
            matchPath = os.path.join(teamPath, matchFolder)
            for file in GetFiles(matchPath):
                filePath = os.path.join(matchPath, file)
                if file == 'Match_Attributes.csv':
                    matchAttributesList.append(pd.read_csv(filePath))
                elif file == 'Missing_Players.csv':
                    missingPlayersList.append(pd.read_csv(filePath))
                elif file == 'Player_Statistics.csv':
                    playerStatisticsList.append(pd.read_csv(filePath))
                elif file == 'Subs.csv':
                    subsList.append(pd.read_csv(filePath))
                elif file == 'Formations.csv':
                    formationsList.append(pd.read_csv(filePath))
                elif file == 'Player_Heatmaps.csv':
                    playerHeatmapsList.append(pd.read_csv(filePath))

    # Add relevant columns to playerStatistics dataframes
    index = 0
    for matchAttributes, missingPlayers, playerStatistics, subs, formations, playerHeatmaps in zip(matchAttributesList, missingPlayersList, playerStatisticsList, subsList, formationsList, playerHeatmapsList):
        # Add relevant columns in other dataframes to playerStatistics
        if 'ID' not in missingPlayers.columns or 'Availability' not in missingPlayers.columns:
            missingPlayers = pd.DataFrame(columns=['ID', 'Availability'])
        missingPlayers = missingPlayers[['ID', 'Availability']].copy()
        missingPlayers['Match ID'] = matchAttributes['id'].iloc[0]
        missingPlayersList[index] = missingPlayers
        playerStatistics = playerStatistics.merge(missingPlayers[['ID', 'Availability']], how='left', on='ID')
        playerStatistics['Match ID'] = matchAttributes['id'].iloc[0]
        playerStatistics['Start Timestamp'] = matchAttributes['startTimestamp'].iloc[0]
        playerStatistics['Competition'] = matchAttributes['tournament.uniqueTournament.name'].iloc[0]
        homeTeamName = matchAttributes['homeTeam.name'].iloc[0]
        awayTeamName = matchAttributes['awayTeam.name'].iloc[0]
        teamName = team.replace('_', ' ')
        playerStatistics['Home'] = teamName == homeTeamName
        playerStatistics['Team'] = teamName
        playerStatistics['Team ID'] = matchAttributes['homeTeam.id'].iloc[0] if teamName == homeTeamName else matchAttributes['awayTeam.id'].iloc[0]
        playerStatistics['Formation'] = normalizeFormation(formations['Formation'].iloc[0] if teamName == formations['Team Name'].iloc[0] else formations['Formation'].iloc[1])
        playerStatistics['Opposition'] = np.where(playerStatistics['Home'] == True, awayTeamName, homeTeamName)
        playerStatistics['Opposition ID'] = matchAttributes['awayTeam.id'].iloc[0] if teamName == homeTeamName else matchAttributes['homeTeam.id'].iloc[0]
        playerStatistics['Opposition Formation'] = normalizeFormation(formations['Formation'].iloc[0] if teamName == formations['Team Name'].iloc[1] else formations['Formation'].iloc[1])

        #Add started column before Substitute column; do it here so inferPlayerPosition() can determine the starting 11
        playerStatistics.insert(playerStatistics.columns.get_loc('Substitute'), 'Started', np.where(playerStatistics['Substitute'] == False, True, False))
        playerStatistics['Started'] = playerStatistics['Started'].astype(float)

         #Replace Substitute with Substituted column
        condition1 = playerStatistics['Substitute'] == True
        condition2 = playerStatistics['minutesPlayed'] > 0
        playerStatistics['Substituted'] = np.where(condition1 & condition2, True, False)

        # Infer specific player positions
        playerStatistics = inferPlayerPositions(playerStatistics)

        playerStatisticsList[index] = playerStatistics

        subs['Match ID'] = matchAttributes['id'].iloc[0]
        subsList[index] = subs

        index += 1

    #Create main dataframe by concatenating all playerStatistics dataframes, which have been edited above to include relevant columns from other dataframes
    mainDF = pd.concat(playerStatisticsList, ignore_index=True)

    #Rename columns to match naming convention
    mainDF.rename(columns={'minutesPlayed': 'Minutes Played'}, inplace=True)
    mainDF.rename(columns={'rating': 'Rating'}, inplace=True)

    #Clean columns
    mainDF['Minutes Played'] = mainDF['Minutes Played'].fillna(0)

    predictedMatchIDs = []
    mainDF['Is Future Match'] = False

    teamID = int(mainDF['Team ID'].iloc[0])
    targetDateTime = datetime(year, month, day, tzinfo=timezone.utc)

    # If target match is in the future, filter out any historical rows on or after the target date
    if not ((currentYear > year) or (currentYear == year and currentMonth > month) or (currentYear == year and currentMonth == month and currentDay >= day)):
        mainDF = mainDF[mainDF['Start Timestamp'] < int(targetDateTime.timestamp())]

    # Scrape upcoming matches in both cases:
    # - target in future: returns matches from now through the match after target
    # - target already played: returns the next upcoming match + one more for Next Match features
    futureDF, predictedMissingPlayers, targetMatchID = scrapeNextMatches(teamID, targetDateTime)
    predictedMatchIDs = [targetMatchID] if targetMatchID else []

    mainDF = pd.concat([mainDF, futureDF], ignore_index=True)

    # Detailed Positions is player-level; propagate from future rows to historical rows
    dpMap = (mainDF[mainDF['Detailed Positions'].apply(lambda v: isinstance(v, list))].drop_duplicates('ID').set_index('ID')['Detailed Positions'])
    mainDF['Detailed Positions'] = mainDF.apply(lambda row: dpMap[row['ID']] if row['ID'] in dpMap.index else row['Detailed Positions'], axis=1)

    mainDF.loc[mainDF['Is Future Match'] == True, 'Started'] = np.nan

    #Convert Start Timestamp from seconds to days
    mainDF.sort_values(by='Start Timestamp', inplace=True)
    mainDF['Days Since Season Start'] = ((mainDF['Start Timestamp'] - mainDF['Start Timestamp'].iloc[0]) / 86400).round(0)

    mainDF['Availability'] = mainDF['Availability'].fillna('Available')
    mainDF['Home'] = mainDF['Home'].astype(float)

    #Add column for the opposition team's primary competition
    uniqueOpps = mainDF[['Opposition', 'Opposition ID']].drop_duplicates('Opposition ID')
    oppositionCompetitions, oppositionCompetitionIDs = scrapeOppositionCompetitions(uniqueOpps['Opposition ID'].tolist())
    uniqueOpps['Opposition Primary Competition'] = oppositionCompetitions
    uniqueOpps['Opposition Primary Competition ID'] = oppositionCompetitionIDs
    mainDF = mainDF.merge(uniqueOpps[['Opposition ID', 'Opposition Primary Competition', 'Opposition Primary Competition ID']], how='left', on='Opposition ID')
    
    # Select only relevant columns
    mainDF = mainDF[['Player Name', 'ID', 'Number', 'Position', 'Detailed Positions', 'Formation Slot', 'Captain', 'Sofascore Market Value', 'Started', 'Substituted', 'Minutes Played', 'Rating', 'Availability', 'Match ID', 'Is Future Match', 'Days Since Season Start', 'Competition', 'Home', 'Team', 'Team ID', 'Formation', 'Opposition', 'Opposition ID', 'Opposition Formation', 'Opposition Primary Competition', 'Opposition Primary Competition ID']]

    subsDF = pd.concat(subsList, ignore_index=True) if subsList else pd.DataFrame(columns=['playerOut.id', 'playerIn.id', 'Match ID', 'time'])
    allMissingPlayersDF = pd.concat(missingPlayersList, ignore_index=True) if missingPlayersList else pd.DataFrame(columns=['ID', 'Availability', 'Match ID'])

    return mainDF, subsDF, predictedMatchIDs, allMissingPlayersDF, predictedMissingPlayers
# %%
def addPowerRankings(df, year):

    matchDF = df[['Match ID', 'Team', 'Opposition', 'Opposition Primary Competition', 'Days Since Season Start']].drop_duplicates(subset=['Match ID']).sort_values('Days Since Season Start').reset_index(drop=True)

    #matchDF = df[['Match ID', 'Opposition', 'Opposition Primary Competition']].drop_duplicates(subset=['Match ID']).sort_values('Days Since Season Start').reset_index(drop=True)

    #Run League_Power_Rankings to get power rankings dataframe
    import League_Power_Rankings
    oppositionPowerRatings = [None]
    teamPowerRatings = [None]
    for numMatches in range(1, len(matchDF)):
        league = matchDF['Opposition Primary Competition'].iloc[numMatches]
        if not league or (isinstance(league, float) and pd.isna(league)):
            teamPowerRatings.append(None)
            oppositionPowerRatings.append(None)
            continue
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            powerRankings = League_Power_Rankings.main(year, league, numMatches)

        if powerRankings is None:
            teamPowerRatings.append(None)
            oppositionPowerRatings.append(None)
        else:
            teamMatch = powerRankings.loc[powerRankings['Team Name'] == matchDF['Team'].iloc[numMatches], 'Team Rating']
            oppositionMatch = powerRankings.loc[powerRankings['Team Name'] == matchDF['Opposition'].iloc[numMatches], 'Team Rating']
            teamPowerRatings.append(teamMatch.iloc[0] if not teamMatch.empty else None)
            oppositionPowerRatings.append(oppositionMatch.iloc[0] if not oppositionMatch.empty else None)

    matchDF['Team Power Rating'] = teamPowerRatings
    matchDF['Opposition Power Rating'] = oppositionPowerRatings

    df = df.merge(matchDF[['Match ID', 'Team Power Rating', 'Opposition Power Rating']], how='left', on='Match ID')

    df['Power Rating Difference'] = df['Team Power Rating'] - df['Opposition Power Rating']

    return df
# %%
def rollingNDays(group, n):
    days = group['Days Since Season Start'].values
    mins = group['Minutes Played'].values

    result = np.zeros(len(group))

    for i in range(len(group)):
        currentDay = days[i]

        windowStart = currentDay - n

        # Filter for all matches within n days before current match
        mask = (days < currentDay) & (days >= windowStart)

        # Sum minutes in window
        result[i] = mins[mask].sum()

    return pd.Series(result, index=group.index)

#%%
def computeChanges(lineups):
    prev = None
    changes = []

    for lineup in lineups['Lineup']:
        if prev is None:
            changes.append(0)
        else:
            changes.append(len(lineup ^ prev))
        prev = lineup

    return pd.Series(changes, index=lineups.index)
#%%
def predictFormation(df, formationCol='Formation', groupByCol=None):
    '''Returns {matchID: predicted_formation} using only prior data (no leakage).
    formationCol: 'Formation' for team formation, 'Opposition Formation' for opposition.
    groupByCol: if provided (e.g. 'Opposition ID'), uses opponent-specific history first,
                falling back to global mode when no prior encounters exist.
    '''
    historicalDF = df[df['Is Future Match'] == False]
    cols = ['Match ID', 'Days Since Season Start', formationCol]
    if groupByCol:
        cols.append(groupByCol)
    matchDF = (historicalDF[cols].drop_duplicates('Match ID').sort_values('Days Since Season Start').reset_index(drop=True))

    predictions = {}
    for i, row in matchDF.iterrows():
        priorRows = matchDF.iloc[:i]
        if groupByCol and not priorRows.empty:
            groupPrior = priorRows[priorRows[groupByCol] == row[groupByCol]][formationCol].dropna()
            priorFormations = groupPrior if not groupPrior.empty else priorRows[formationCol].dropna()
        else:
            priorFormations = priorRows[formationCol].dropna()
        predictions[row['Match ID']] = (normalizeFormation(priorFormations.mode().iloc[0]) if not priorFormations.empty else '4-3-3')

    # Future matches: use all historical data (opponent-specific if available)
    allHistoricalFormations = historicalDF[formationCol].dropna()
    globalFallback = (normalizeFormation(allHistoricalFormations.mode().iloc[0]) if not allHistoricalFormations.empty else '4-3-3')
    for matchID in df.loc[df['Is Future Match'] == True, 'Match ID'].unique():
        if groupByCol:
            futureMask = (df['Is Future Match'] == True) & (df['Match ID'] == matchID)
            groupVal   = df.loc[futureMask, groupByCol].iloc[0] if futureMask.any() else None
            if groupVal is not None:
                groupPrior = historicalDF[historicalDF[groupByCol] == groupVal][formationCol].dropna()
                predictions[matchID] = (normalizeFormation(groupPrior.mode().iloc[0]) if not groupPrior.empty else globalFallback)
            else:
                predictions[matchID] = globalFallback
        else:
            predictions[matchID] = globalFallback

    return predictions
# %%
def engineerFeatures(df):

    # Market Value Log
    df['Sofascore Market Value'] = df['Sofascore Market Value'].fillna(df['Sofascore Market Value'].min())
    df['Market Value Log'] = np.log(df['Sofascore Market Value']+1)

    # Player Ratings and Usage
    df = df.sort_values(['ID', 'Days Since Season Start']).copy()
    historicalMask = ~df['Is Future Match']

    df['Minutes Weight'] = df['Minutes Played'] / 90.0
    df['Weighted Rating'] = df['Rating'].fillna(0) * df['Minutes Weight']

    df['Appeared'] = (df['Minutes Played']>0).astype(int)
    #df['Tenure Matches'] = (df.groupby('ID')['Appeared'].cumsum() / df.groupby('ID').cumcount().add(1)).astype(int)
    df['Tenure Matches In Squad'] = df.groupby('ID').cumcount()
    df['Played'] = (historicalMask & (df['Minutes Played'] > 0)).astype(int)
    df['Tenure Appearances'] = df.groupby('ID')['Played'].transform(lambda x: x.cumsum().shift(fill_value=0))

    for alpha, columnNameRating, columnNameMinutes, columnNameStarts, columnNameCaptain in zip([0.25, 0.05], ['Player Rating Form', 'Player Rating Overall'], ['Minutes Played Form', 'Minutes Played Overall'], ['Starts Form', 'Starts Overall'], ['Captain Rate Form', 'Captain Rate Overall']):
        # Player Ratings
        numerator = df['Weighted Rating'].where(historicalMask).groupby(df['ID']).transform(lambda x: x.shift().ewm(alpha=alpha, adjust=False).mean())
        denominator = df['Minutes Weight'].where(historicalMask).groupby(df['ID']).transform(lambda x: x.shift().ewm(alpha=alpha, adjust=False).mean())

        df[columnNameRating] = numerator / denominator

        # Player Usage
        df[columnNameMinutes] = df['Minutes Weight'].where(historicalMask).groupby(df['ID']).transform(lambda x: x.shift().ewm(alpha=alpha, adjust=False).mean())

        #df['Started Float'] = np.where(historicalMask, df['Started'].astype(float), np.nan)
        df[columnNameStarts] = df['Started'].astype(float).where(historicalMask).groupby(df['ID']).transform(lambda x: x.shift().ewm(alpha=alpha, adjust=False).mean())

        # Captain Rate: how often this player has been captain (proxy for team captain / guaranteed starter)
        captainHistorical = df['Captain'].map(lambda v: False if pd.isna(v) else bool(v)).astype(float).where(historicalMask)
        df[columnNameCaptain] = captainHistorical.groupby(df['ID']).transform(lambda x: x.shift().ewm(alpha=alpha, adjust=False).mean())

    # Fatigue
    df['Minutes Last Match'] = df.groupby('ID')['Minutes Played'].shift(1)
    df['Started Last Match'] = df.groupby('ID')['Started'].shift(1)
    df['Days Since Last Start'] = (df['Days Since Season Start'] - df['Days Since Season Start'].where(df['Started']==1).groupby(df['ID']).ffill().groupby(df['ID']).shift(1))
    df['Days Since Last Played'] = (df['Days Since Season Start'] - df['Days Since Season Start'].where(df['Minutes Played']>0).groupby(df['ID']).ffill().groupby(df['ID']).shift(1))
   
    for days in [4, 7, 10, 14]:
        df[f'Mins Last {days} Days'] = df.groupby('ID', group_keys=False).apply(rollingNDays, n=days, include_groups=False)

    # Player Availability
    df = df.sort_values(['ID', 'Days Since Season Start'])
    df['Matches Since Last Played'] = df.groupby('ID')['Played'].transform(lambda x: (~x.astype(bool)).groupby(x.cumsum()).cumcount())
    matchDF = df[['Match ID', 'Days Since Season Start']].drop_duplicates().sort_values('Days Since Season Start').reset_index(drop=True)
    matchDF['Global Match Number'] = range(len(matchDF))
    df = df.merge(matchDF[['Match ID', 'Global Match Number']], on='Match ID', how='left')
    df['Consecutive Played'] = df['Played'].where(historicalMask).groupby([df['ID'], df.where(historicalMask).groupby('ID')['Played'].transform(lambda x: x.eq(0).cumsum())]).cumsum()
    df['Consecutive Started'] = df['Started'].where(historicalMask).groupby([df['ID'], df.where(historicalMask).groupby('ID')['Started'].transform(lambda x: x.eq(0).cumsum())]).cumsum()
    df['Matches Since In Squad'] = df.groupby('ID')['Global Match Number'].diff().sub(1).clip(lower=0).fillna(0).astype(int)

    # Match Context
    df = df.sort_values(['ID', 'Days Since Season Start'])
    #matchDF = df[['Match ID', 'Days Since Season Start', 'Opposition Power Rating']].drop_duplicates().sort_values('Days Since Season Start')
    matchDF = df[['Match ID', 'Days Since Season Start', 'Opposition Power Rating', 'Competition', 'Opposition Primary Competition']].drop_duplicates(subset=['Match ID']).sort_values('Days Since Season Start').reset_index(drop=True)
    matchDF['Days Since Last Match'] = (matchDF['Days Since Season Start'] - matchDF['Days Since Season Start'].shift(1)).fillna(100).astype(int)
    matchDF['Days Until Next Match'] = matchDF['Days Since Last Match'].shift(-1).fillna(100).astype(int)
    _oppRating = pd.to_numeric(matchDF['Opposition Power Rating'], errors='coerce')
    matchDF['Next Match Opposition Power Rating'] = _oppRating.shift(-1).fillna(_oppRating.mean())
    matchDF['Next Match Competition'] = matchDF['Competition'].shift(-1)
    matchDF['Next Match Opposition Primary Competition'] = matchDF['Opposition Primary Competition'].shift(-1)

    df = df.merge(matchDF[['Match ID', 'Days Since Last Match', 'Days Until Next Match', 'Next Match Opposition Power Rating', 'Next Match Competition', 'Next Match Opposition Primary Competition']], on='Match ID', how='left')

    # Manager Behavior
    lineups = df[df['Started'] == 1].groupby(['Match ID', 'Days Since Season Start'])['ID'].apply(set).reset_index(name='Lineup').sort_values('Days Since Season Start')
    lineups['Lineup Changes'] = computeChanges(lineups)
    #df = df.merge(lineups[['Match ID', 'Lineup Changes']], how='left', on='Match ID')

    matchLineups = lineups[['Match ID', 'Days Since Season Start', 'Lineup Changes']].sort_values('Days Since Season Start')
    for alpha, columnName in zip([0.25, 0.05], ['Lineup Changes Form', 'Lineup Changes Overall']):
        matchLineups[columnName] = matchLineups['Lineup Changes'].shift().ewm(alpha=alpha, adjust=False).mean()

    #df['Competition Rotation Rate'] = df.groupby('Competition')['Lineup Changes'].transform(lambda x: x.shift().expanding().mean())
    matchCompetition = df[['Match ID', 'Competition']].drop_duplicates('Match ID')
    matchLineups = matchLineups.merge(matchCompetition, on='Match ID', how='left')
    matchLineups['Competition Rotation Rate'] = matchLineups.groupby('Competition')['Lineup Changes'].transform(lambda x: x.shift().expanding().mean())
    df = df.merge(matchLineups[['Match ID', 'Lineup Changes Form', 'Lineup Changes Overall', 'Competition Rotation Rate']], on='Match ID', how='left')

    #df['Rest Sensitivity'] = linregress(df['Days Since Season Start'], df['Lineup Changes']).slope
    #df['Opponent Sensitivity'] = linregress(df['Opposition Power Rating'], df['Lineup Changes']).slope
    
    # Predicted Formation (team and opposition) — leakage-free, per-match
    df['Predicted Formation']            = df['Match ID'].map(predictFormation(df))
    df['Predicted Opposition Formation'] = df['Match ID'].map(
        predictFormation(df, formationCol='Opposition Formation', groupByCol='Opposition ID')
    )

    # Predicted Captain — player per match with highest Captain Rate Form (already shift-corrected, no leakage)
    # Use a lambda that handles all-NaN groups (idxmax on all-NaN raises in older pandas, returns NaN in newer)
    captainIdx = df.groupby('Match ID')['Captain Rate Form'].transform(lambda s: s.idxmax() if s.notna().any() else np.nan)
    df['Predicted Captain'] = (df.index == captainIdx).astype(float)

    # Remove unnecessary columns
    df.drop(columns=['Sofascore Market Value', 'Minutes Weight', 'Weighted Rating', 'Appeared', 'Global Match Number', 'Availability'], inplace=True)

    return df

#%%
def computeStartingProbabilities(df, predictedMatchIDs, verbose=True):

    # Select relevant features for player probability model
    modelDF = df[['Player Name', 'ID', 'Days Since Season Start', 'Match ID', 'Is Future Match', 'Position', 'Predicted Formation', 'Predicted Opposition Formation', 'Started', 'Predicted Captain', 'Home', 'Competition', 'Next Match Competition', 'Opposition Primary Competition', 'Next Match Opposition Primary Competition', 'Team Power Rating', 'Opposition Power Rating', 'Power Rating Difference', 'Next Match Opposition Power Rating',
             'Market Value Log', 'Tenure Matches In Squad', 'Tenure Appearances', 'Player Rating Overall', 'Player Rating Form', 'Minutes Played Overall', 'Minutes Played Form', 'Starts Overall', 'Starts Form', 'Captain Rate Form', 'Captain Rate Overall', 'Minutes Last Match', 'Started Last Match', 'Days Since Last Start', 'Days Since Last Played', 'Mins Last 4 Days', 'Mins Last 7 Days', 'Mins Last 10 Days',
             'Mins Last 14 Days', 'Matches Since Last Played', 'Matches Since In Squad', 'Consecutive Played', 'Consecutive Started', 'Days Since Last Match', 'Days Until Next Match', 'Lineup Changes Form', 'Lineup Changes Overall', 'Competition Rotation Rate']].copy()

    # Remove features with zero importance
    zeroImportanceFeatures = ['Tenure Appearances', 'Next Match Competition', 'Next Match Opposition Primary Competition', 'Home', 'Started Last Match', 'Captain Rate Overall', 'Predicted Captain']
    #zeroImportanceFeatures = []
    modelDF.drop(columns=zeroImportanceFeatures, inplace=True)

    # Coerce power rating columns to float — they arrive as object when merged from CSV
    for col in ['Team Power Rating', 'Opposition Power Rating', 'Power Rating Difference', 'Next Match Opposition Power Rating']:
        if col in modelDF.columns:
            modelDF[col] = pd.to_numeric(modelDF[col], errors='coerce')

    # Encode Categorical Data
    categoricalColumns = ['Position', 'Predicted Formation', 'Predicted Opposition Formation', 'Competition', 'Opposition Primary Competition']
    #categoricalColumns = ['Position', 'Competition', 'Opposition Primary Competition', 'Next Match Competition', 'Next Match Opposition Primary Competition', 'Predicted Formation', 'Predicted Opposition Formation']
    for column in categoricalColumns:
        # Fill NaN before converting — XGBoost crashes if a categorical column has an empty
        # categories set (all NaN), which can happen for Formation Slot in early-season slices.
        modelDF[column] = modelDF[column].fillna('Unknown').astype('category')

    colsToDrop = ['Match ID', 'Is Future Match', 'Player Name', 'Days Since Season Start', 'ID']

    # Split data into training and testing
    trainingDF = modelDF[(modelDF['Is Future Match'] == False) & (modelDF['Started'].notna())].copy()
    testingDF = modelDF[modelDF['Is Future Match'] == True].copy()
    
    # Remove the last match as its now unneeded
    testingDF = testingDF[testingDF['Match ID'] == predictedMatchIDs[0]].copy()
    testPlayers = testingDF[['Player Name', 'ID']].copy()

    # Split into x and y
    xTrain = trainingDF.drop(columns=['Started'] + colsToDrop)
    yTrain = trainingDF['Started'].astype(int)
    xTest = testingDF.drop(columns=['Started'] + colsToDrop)

    # Split training data into training and validation
    trainingDF = trainingDF.sort_values('Days Since Season Start')
    matches = trainingDF['Match ID'].unique()
    splitIndex = int(len(matches) * 0.8)
    trainingMatches = matches[:splitIndex]
    validationMatches = matches[splitIndex:]

    xTrainingTest = trainingDF[trainingDF['Match ID'].isin(trainingMatches)].drop(columns=['Started'] + colsToDrop)
    yTrainingTest = trainingDF[trainingDF['Match ID'].isin(trainingMatches)]['Started'].astype(int)
    xValidation = trainingDF[trainingDF['Match ID'].isin(validationMatches)].drop(columns=['Started'] + colsToDrop)
    yValidation = trainingDF[trainingDF['Match ID'].isin(validationMatches)]['Started'].astype(int)

    # === DIAGNOSTIC: Feature column health (remove once dead features are identified) ===
    if verbose:
        print('=== DIAGNOSTIC: Feature column health (trainingDF) ===')
        for col in ['Captain Rate Form', 'Captain Rate Overall',
                    'Team Power Rating', 'Opposition Power Rating',
                    'Power Rating Difference', 'Next Match Opposition Power Rating',
                    'Predicted Captain']:
            if col in trainingDF.columns:
                s = trainingDF[col]
                print(f'  {col}: dtype={s.dtype} NaN={s.isna().sum()}/{len(s)} '
                      f'nunique={s.nunique()} min={s.min()} max={s.max()}')
        for col in categoricalColumns:
            if col in trainingDF.columns:
                cats = list(trainingDF[col].cat.categories)
                print(f'  {col}: ncats={len(cats)} sample={cats[:6]}')
        print('=== END DIAGNOSTIC ===')

    # Train Validation model
    valModel = XGBClassifier(
        n_estimators = 300,
        max_depth = 5,
        learning_rate = 0.03,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'binary:logistic',
        eval_metric = 'logloss',
        enable_categorical = True,
        tree_method = 'hist',
        random_state = 42
    )

    valModel.fit(xTrainingTest, yTrainingTest)

    validationProbabilities = valModel.predict_proba(xValidation)[:, 1]
    if verbose:
        print(f'Validation Log Loss: {log_loss(yValidation, validationProbabilities)}')
        featureImportance = pd.DataFrame({'Feature': xTrainingTest.columns, 'Importance': valModel.feature_importances_}).sort_values('Importance', ascending=False)
        print(featureImportance.to_string())

    # Train Final model
    finalModel = XGBClassifier(
        n_estimators = 300,
        max_depth = 5,
        learning_rate = 0.03,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'binary:logistic',
        eval_metric = 'logloss',
        enable_categorical = True,
        tree_method = 'hist',
        random_state = 42
    )

    finalModel.fit(xTrain, yTrain)

    starterProbabilities = finalModel.predict_proba(xTest)[:, 1]

    testPlayers['Starter Probability'] = starterProbabilities

    if verbose:
        print('Test Player Probabilities:')
        print(testPlayers[['Player Name', 'Starter Probability']].sort_values('Starter Probability', ascending=False).reset_index(drop=True).to_string())

    futureMask = (df['Is Future Match'] == True) & (df['Match ID'] == predictedMatchIDs[0])
    playerMeta = df.loc[futureMask, ['Player Name', 'ID', 'Position', 'Detailed Positions']].drop_duplicates('ID')
    probsDF = testPlayers.merge(playerMeta, on=['Player Name', 'ID'], how='left')

    return probsDF
#%%
# Maps Sofascore detailed position codes to formation slot types
#PRIMARY_TO_SLOTS = {'G': {'GK'}, 'D': {'DR', 'DC', 'DL'}, 'M': {'MR', 'DM', 'MC', 'AM', 'ML'}, 'F': {'RW', 'ST', 'LW'}}
#SLOT_ORDER = ['GK', 'DR', 'DC', 'DL', 'MR', 'DM', 'MC', 'ML', 'AM', 'RW', 'ST', 'LW']
COMPARABLE_SLOTS = {'DCL': ['DC', 'DCR'], 'DC': ['DCL', 'DCR'], 'DCR': ['DCL', 'DC'],
                    'DML': ['DM', 'DMR'], 'DM': ['DML', 'DMR'], 'DMR': ['DML', 'DM'], 
                    'MCL': ['MC', 'MCR'], 'MC': ['MCL', 'MCR'], 'MCR': ['MCL', 'MC'],
                    'AML': ['AM', 'AMR'], 'AM': ['AML', 'AMR'], 'AMR': ['AML', 'AM'], 
                    'STL': ['ST', 'STR'], 'ST': ['STL', 'STR'], 'STR': ['STL', 'ST']}
SIMILAR_SLOTS = {'DL': ['DR', 'ML'], 'DR': ['DL', 'MR'],
                    'DML': ['MCL', 'MC', 'MCR'], 'DM': ['MCL', 'MC', 'MCR'], 'DMR': ['MCL', 'MC', 'MCR'],
                    'MCL': ['DML', 'DM', 'DMR', 'AML', 'AM', 'AMR'], 'MC': ['DML', 'DM', 'DMR', 'AML', 'AM', 'AMR'], 'MCR': ['DML', 'DM', 'DMR', 'AML', 'AM', 'AMR'],
                    'ML': ['DL', 'MR', 'LW'], 'MR': ['DR', 'ML', 'RW'], 'AML': ['MCL', 'MC', 'MCR'], 'AM': ['MCL', 'MC', 'MCR'], 'AMR': ['MCL', 'MC', 'MCR'], 'RW': ['LW', 'MR'], 'LW': ['RW', 'ML']}
#%%
def getEligibleSlots(row, allPlayersPlayedSlots, formationSlots):
    import ast
    formationSlotsSet = set(formationSlots)
    primarySlots   = set()
    secondarySlots = set()
    tertiarySlots  = set()

    # Get played slots: primary is the most played slot, secondary is all other played slots
    playedSlots = allPlayersPlayedSlots['Formation Slot'].dropna()
    if not playedSlots.empty:
        primarySlots   = {playedSlots.value_counts().index[0]}     # most-played slot as a set
        secondarySlots = set(playedSlots.unique()) - primarySlots  # all other played slots

        # Add comparable slots of all played slots to secondary
        for playedSlot in primarySlots | secondarySlots:
            for comparableSlot in COMPARABLE_SLOTS.get(playedSlot, []):
                if comparableSlot not in primarySlots:
                    secondarySlots.add(comparableSlot)

        # Add similar slots of primary and secondary to tertiary
        for playedSlot in primarySlots | secondarySlots:
            for similarSlot in SIMILAR_SLOTS.get(playedSlot, []):
                if similarSlot not in primarySlots and similarSlot not in secondarySlots:
                    tertiarySlots.add(similarSlot)

    # Get the player's detailed positions listed by Sofascore
    detailedPositions = row['Detailed Positions']
    if isinstance(detailedPositions, str):
        try:
            detailedPositions = ast.literal_eval(detailedPositions)
        except (ValueError, SyntaxError):
            detailedPositions = []
    if not isinstance(detailedPositions, list):
        detailedPositions = []

    # Add each slot to tertiary if not already in primary or secondary
    for item in detailedPositions:
        slot = item.get('position') if isinstance(item, dict) else item
        if slot and slot not in primarySlots and slot not in secondarySlots:
            tertiarySlots.add(slot)
    # Expand tertiary by comparable slots (covers wide-vs-central variants of the same role)
    for slot in list(tertiarySlots):
        for comparableSlot in COMPARABLE_SLOTS.get(slot, []):
            if comparableSlot not in primarySlots and comparableSlot not in secondarySlots and comparableSlot not in tertiarySlots:
                tertiarySlots.add(comparableSlot)
    # Expand tertiary by similar slots (bridges e.g. MC → DML/DMR or AM/AML/AMR when the
    # formation uses different role layers than what the player has listed)
    for slot in list(tertiarySlots):
        for similarSlot in SIMILAR_SLOTS.get(slot, []):
            if similarSlot not in primarySlots and similarSlot not in secondarySlots and similarSlot not in tertiarySlots:
                tertiarySlots.add(similarSlot)

    # Reduce each tier to only slots present in the current formation
    primarySlots   &= formationSlotsSet
    secondarySlots &= formationSlotsSet
    tertiarySlots  &= formationSlotsSet

    # If primary is empty after reduction, promote secondary; if that's also empty, promote tertiary
    if not primarySlots:
        if secondarySlots:
            primarySlots   = secondarySlots
            secondarySlots = tertiarySlots
            tertiarySlots  = set()
        elif tertiarySlots:
            primarySlots = tertiarySlots
            tertiarySlots = set()

    '''
    print(f'{row["Player Name"]} eligibility:')
    print(f'  Primary:   {primarySlots}')
    print(f'  Secondary: {secondarySlots}')
    print(f'  Tertiary:  {tertiarySlots}')
    '''

    return primarySlots, secondarySlots, tertiarySlots
#%%
def predictLineup(mainDF, probsDF, formation, team, opposition, home, verbose=True):
    formationSlots = getFormationSlots(formation)
    probsDF = probsDF.reset_index(drop=True)
    players = probsDF.index.tolist()

    allPlayersPlayedSlots = mainDF[['Player Name', 'Formation', 'Formation Slot']]

    primaryEligibility = {}
    secondaryEligibility = {}
    tertiaryEligibility = {}
    for p in players:
        playerName = probsDF.loc[p, 'Player Name']
        primary, secondary, tertiary = getEligibleSlots(probsDF.loc[p], allPlayersPlayedSlots[allPlayersPlayedSlots['Player Name'] == playerName], formationSlots)
        primaryEligibility[p] = primary
        secondaryEligibility[p] = secondary
        tertiaryEligibility[p] = tertiary

    # Combined eligibility for the ILP: players can fill any natural or positional slot
    eligibility = {p: primaryEligibility[p] | secondaryEligibility[p] | tertiaryEligibility[p] for p in players}

    prob = LpProblem('LineupOptimizer', LpMaximize)

    # assign[p, s] = 1 if player p fills slot s
    assign = {
        (p, s): LpVariable(f'assign_{p}_{s}', cat=LpBinary)
        for p in players
        for s in eligibility[p]
    }

    # Objective: maximize starter probability weighted by position tier.
    # Weights are small enough to only break ties, not override probability differences.
    TIER_WEIGHTS = {'primary': 1.0, 'secondary': 0.85, 'tertiary': 0.60}
    prob += (
        lpSum(TIER_WEIGHTS['primary']   * probsDF.loc[p, 'Starter Probability'] * assign[p, s] for p in players for s in primaryEligibility[p]   if (p, s) in assign) +
        lpSum(TIER_WEIGHTS['secondary'] * probsDF.loc[p, 'Starter Probability'] * assign[p, s] for p in players for s in secondaryEligibility[p] if (p, s) in assign) +
        lpSum(TIER_WEIGHTS['tertiary']  * probsDF.loc[p, 'Starter Probability'] * assign[p, s] for p in players for s in tertiaryEligibility[p]  if (p, s) in assign)
    )

    # Each player fills at most one slot
    for p in players:
        if eligibility[p]:
            prob += lpSum(assign[p, s] for s in eligibility[p]) <= 1

    # Each slot must be filled by exactly one player
    for slot in formationSlots:
        prob += lpSum(assign[p, s] for (p, s) in assign if s == slot) == 1

    status = prob.solve(PULP_CBC_CMD(msg=0))
    if status != 1:
        print(f'Warning: ILP solver could not find a valid lineup (status {status}). Check that enough eligible players are available for the given formation.')
        return pd.DataFrame()

    # Collect results
    rows = []
    for (p, s), var in assign.items():
        v = value(var)
        if v is not None and v > 0.5:
            if s in primaryEligibility[p]:
                tier = 'Primary'
            elif s in secondaryEligibility[p]:
                tier = 'Secondary'
            else:
                tier = 'Tertiary'
            rows.append({
                'Player Name': probsDF.loc[p, 'Player Name'],
                'ID': probsDF.loc[p, 'ID'],
                'Position': probsDF.loc[p, 'Position'],
                'Slot': s,
                'Tier': tier,
                'Starter Probability': probsDF.loc[p, 'Starter Probability'],
            })

    resultDF = pd.DataFrame(rows).sort_values(
        'Slot', key=lambda col: col.map({s: i for i, s in enumerate(formationSlots)})
    )

    #if verbose:
    if home:
        print(f'\n{opposition} at {team}')
    else:
        print(f'\n{team} at {opposition}')
    print(f'{team} Predicted Starting XI:')
    print(f'Formation: {formation}')
    print(resultDF[['Player Name', 'ID', 'Position', 'Slot', 'Tier', 'Starter Probability']].to_string(index=False))

    #TODO: add average player rating to resultDF, weighted for minutes played

    resultDF = pd.merge(resultDF, mainDF.drop_duplicates('ID')[['ID', 'Number']], how='left', on='ID')

    return resultDF
#%%
def testLineupPredictions(df, missingPlayersDF=None, minTrainingMatches=0):
    historicalDF = df[df['Is Future Match'] == False].copy()

    matchDF = (historicalDF[['Match ID', 'Days Since Season Start', 'Team', 'Opposition', 'Home']]
               .drop_duplicates('Match ID')
               .dropna(subset=['Days Since Season Start'])
               .sort_values('Days Since Season Start')
               .reset_index(drop=True))

    results = []

    for i in range(minTrainingMatches, len(matchDF)):
        targetMatchID = matchDF.loc[i, 'Match ID']
        targetDays    = matchDF.loc[i, 'Days Since Season Start']
        team          = matchDF.loc[i, 'Team']
        opposition    = matchDF.loc[i, 'Opposition']
        home          = bool(matchDF.loc[i, 'Home'])

        # Slice to all data up to and including target match
        slicedDF = df[df['Days Since Season Start'] <= targetDays].copy()

        # Mark target match as pseudo-future to simulate a real prediction
        slicedDF.loc[slicedDF['Match ID'] == targetMatchID, 'Is Future Match'] = True
        slicedDF.loc[slicedDF['Match ID'] == targetMatchID, 'Started'] = np.nan

        # Add healthy scratch rows: players who were available but not in this squad.
        # Real predictions include the full roster; backtesting should match that pool.
        refRow = slicedDF[slicedDF['Match ID'] == targetMatchID].iloc[0]
        matchContextCols = [
            'Competition', 'Home', 'Team', 'Team ID', 'Formation',
            'Opposition', 'Opposition ID', 'Opposition Formation',
            'Opposition Primary Competition', 'Opposition Primary Competition ID',
            'Team Power Rating', 'Opposition Power Rating', 'Power Rating Difference',
            'Days Since Last Match', 'Days Until Next Match',
            'Next Match Opposition Power Rating', 'Next Match Competition',
            'Next Match Opposition Primary Competition',
            'Lineup Changes Form', 'Lineup Changes Overall', 'Competition Rotation Rate',
            'Predicted Formation', 'Predicted Opposition Formation'
        ]
        priorHistoricalDF   = slicedDF[slicedDF['Is Future Match'] == False]
        targetSquadIDs      = set(slicedDF[slicedDF['Match ID'] == targetMatchID]['ID'])
        scratchCandidateIDs = set(priorHistoricalDF['ID']) - targetSquadIDs
        if missingPlayersDF is not None and not missingPlayersDF.empty:
            injuredForMatch = set(missingPlayersDF[missingPlayersDF['Match ID'] == targetMatchID]['ID'])
            scratchCandidateIDs -= injuredForMatch

        scratchRows = []
        for scratchID in scratchCandidateIDs:
            playerPriorRows = priorHistoricalDF[priorHistoricalDF['ID'] == scratchID].sort_values('Days Since Season Start')
            if playerPriorRows.empty:
                continue

            lastPriorRow = playerPriorRows.iloc[-1]
            syntheticRow = lastPriorRow.copy()
            lastRowDays  = float(lastPriorRow['Days Since Season Start'])
            daysDiff     = targetDays - lastRowDays

            # Update match identity and context from the target match
            syntheticRow['Match ID']                = targetMatchID
            syntheticRow['Days Since Season Start'] = targetDays
            for col in matchContextCols:
                syntheticRow[col] = refRow[col]

            # shift(1)-based features: target row sees the player's most recent prior match
            syntheticRow['Minutes Last Match'] = float(lastPriorRow['Minutes Played'])
            syntheticRow['Started Last Match']  = float(lastPriorRow['Started']) if pd.notna(lastPriorRow['Started']) else np.nan

            # Days-since features: add elapsed days, anchoring on the prior row's last start/play
            if pd.notna(lastPriorRow['Started']) and lastPriorRow['Started'] == 1.0:
                syntheticRow['Days Since Last Start'] = float(daysDiff)
            elif pd.notna(syntheticRow['Days Since Last Start']):
                syntheticRow['Days Since Last Start'] = float(syntheticRow['Days Since Last Start']) + daysDiff

            if lastPriorRow['Played'] == 1:
                syntheticRow['Days Since Last Played'] = float(daysDiff)
            elif pd.notna(syntheticRow['Days Since Last Played']):
                syntheticRow['Days Since Last Played'] = float(syntheticRow['Days Since Last Played']) + daysDiff

            # Rolling fatigue windows: sum actual minutes within the window before the target
            for days in [4, 7, 10, 14]:
                windowRows = playerPriorRows[playerPriorRows['Days Since Season Start'] >= targetDays - days]
                syntheticRow[f'Mins Last {days} Days'] = float(windowRows['Minutes Played'].sum())

            # Matches missed since their last squad appearance
            matchesMissed = int(priorHistoricalDF[
                priorHistoricalDF['Days Since Season Start'] > lastRowDays
            ]['Match ID'].nunique())
            # +1: the target match itself is another non-played appearance in the run
            syntheticRow['Matches Since Last Played'] = float(syntheticRow['Matches Since Last Played']) + matchesMissed + 1
            syntheticRow['Matches Since In Squad']    = float(matchesMissed)
            syntheticRow['Tenure Matches In Squad']   = float(syntheticRow['Tenure Matches In Squad']) + matchesMissed

            # Mark as pseudo-future with unknown outcome
            syntheticRow['Is Future Match']   = True
            syntheticRow['Started']           = np.nan
            syntheticRow['Substituted']       = 0
            syntheticRow['Minutes Played']    = 0.0
            syntheticRow['Rating']            = np.nan
            syntheticRow['Captain']           = False
            syntheticRow['Played']            = 0
            # Scratch players aren't in the squad, so they're not the predicted captain
            syntheticRow['Predicted Captain'] = 0.0

            scratchRows.append(syntheticRow)

        if scratchRows:
            slicedDF = pd.concat([slicedDF, pd.DataFrame(scratchRows)], ignore_index=True)

        # Actual starters read from the original unmodified df
        actualStarterIDs = set(historicalDF.loc[
            (historicalDF['Match ID'] == targetMatchID) & (historicalDF['Started'] == 1.0), 'ID'
        ])
        if not actualStarterIDs:
            continue

        probsDF = computeStartingProbabilities(slicedDF, [targetMatchID], verbose=False)
        if probsDF is None or probsDF.empty:
            continue

        formation = predictFormation(slicedDF).get(targetMatchID, '4-3-3')

        resultDF = predictLineup(slicedDF, probsDF, formation, team, opposition, home, verbose=False)
        if resultDF is None or resultDF.empty:
            continue

        predictedIDs = set(probsDF.loc[probsDF['Player Name'].isin(resultDF['Player Name']), 'ID'])
        correct = len(predictedIDs & actualStarterIDs)
        accuracy = correct / 11

        results.append({'Match ID': targetMatchID, 'Opposition': opposition, 'Correct': correct, 'Accuracy': accuracy})
        print(f'Match {i + 1}/{len(matchDF)} ({team} vs {opposition}): {correct}/11 correct ({accuracy:.1%})')

    if results:
        resultsDF = pd.DataFrame(results)
        print(f'\nBacktest Results: {resultsDF["Correct"].mean():.2f}/11 avg ({resultsDF["Accuracy"].mean():.1%}) over {len(results)} matches')
        return resultsDF

    return pd.DataFrame()
#%%
def engineerSubstitutionFeatures(df):

    df = df.sort_values(['ID', 'Days Since Season Start'])
    historicalMask = ~df['Is Future Match']

    # Minute a bench player came on (NaN if didn't enter)
    df['Minute Subbed In'] = df['Minutes Played'].where((df['Substituted'] == 1) & historicalMask).rsub(90)
    # Minute a starter was subbed off (NaN if played full game); starters who play 90+ were not subbed out
    df['Minute Subbed Out'] = df['Minutes Played'].where((df['Started'] == 1) & (df['Minutes Played'] < 90) & historicalMask)

    # Per-player EWM of minutes and sub rates — must use groupby('ID') so each player gets their own window
    for alpha, startMinColumnName, subMinColumnName, subInColumnName, subOutColumnName in zip(
        [0.25, 0.05],
        ['Starting Mins Played Form', 'Starting Mins Played Overall'],
        ['Sub Mins Played Form', 'Sub Mins Played Overall'],
        ['Sub In Rate Form', 'Sub In Rate Overall'],
        ['Sub Out Rate Form', 'Sub Out Rate Overall']
    ):
        df[startMinColumnName] = df['Minutes Played'].where((df['Started'] == 1) & historicalMask).groupby(df['ID']).transform(
            lambda x: x.shift().ewm(alpha=alpha, adjust=False).mean()
        )
        df[subMinColumnName] = df['Minutes Played'].where((df['Substituted'] == 1) & historicalMask).groupby(df['ID']).transform(
            lambda x: x.shift().ewm(alpha=alpha, adjust=False).mean()
        )
        df[subInColumnName] = df['Substituted'].astype(float).where((df['Started'] == 0) & historicalMask).groupby(df['ID']).transform(
            lambda x: x.shift().ewm(alpha=alpha, adjust=False).mean()
        )
        subOutIndicator = ((df['Started'] == 1) & (df['Minutes Played'] < 90)).astype(float)
        df[subOutColumnName] = subOutIndicator.where(historicalMask).groupby(df['ID']).transform(
            lambda x: x.shift().ewm(alpha=alpha, adjust=False).mean()
        )

    # Team-level avg subs per match (EWM over match history, computed once for both alphas)
    matchSubCounts = df[historicalMask].groupby('Match ID')['Substituted'].sum().reset_index(name='Subs This Match')
    matchDayOrder = df[['Match ID', 'Days Since Season Start']].drop_duplicates('Match ID').sort_values('Days Since Season Start')
    matchSubCounts = matchSubCounts.merge(matchDayOrder, on='Match ID').sort_values('Days Since Season Start')
    matchSubCounts['Num Subs Form'] = matchSubCounts['Subs This Match'].shift().ewm(alpha=0.25, adjust=False).mean()
    matchSubCounts['Num Subs Overall'] = matchSubCounts['Subs This Match'].shift().ewm(alpha=0.05, adjust=False).mean()
    df = df.merge(matchSubCounts[['Match ID', 'Num Subs Form', 'Num Subs Overall']], on='Match ID', how='left')

    # Per-player rolling average of sub-out and sub-in minutes over last N appearances
    for window in [1, 5]:
        df[f'Avg Minute Subbed Out in Last {window} Matches'] = df.groupby('ID')['Minute Subbed Out'].transform(
            lambda x: x.shift().rolling(window, min_periods=1).mean()
        )
        df[f'Avg Minute Subbed In in Last {window} Matches'] = df.groupby('ID')['Minute Subbed In'].transform(
            lambda x: x.shift().rolling(window, min_periods=1).mean()
        )
    df['Avg Minute Subbed Out in All Matches'] = df.groupby('ID')['Minute Subbed Out'].transform(
        lambda x: x.shift().expanding().mean()
    )
    df['Avg Minute Subbed In in All Matches'] = df.groupby('ID')['Minute Subbed In'].transform(
        lambda x: x.shift().expanding().mean()
    )

    return df
#%%
def computeSubstitutionProbabilities(df, predictedMatchIDs, predictedLineup, verbose=True):

    featureCols = [
        'Position', 'Competition', 'Opposition Primary Competition',
        'Predicted Formation', 'Predicted Opposition Formation',
        'Team Power Rating', 'Opposition Power Rating', 'Power Rating Difference',
        'Next Match Opposition Power Rating',
        'Market Value Log', 'Tenure Matches In Squad', 'Tenure Appearances',
        'Player Rating Overall', 'Player Rating Form',
        'Minutes Played Overall', 'Minutes Played Form',
        'Starts Overall', 'Starts Form', 'Captain Rate Form',
        'Minutes Last Match', 'Started Last Match',
        'Days Since Last Start', 'Days Since Last Played',
        'Mins Last 4 Days', 'Mins Last 7 Days', 'Mins Last 10 Days', 'Mins Last 14 Days',
        'Matches Since Last Played', 'Matches Since In Squad',
        'Consecutive Played', 'Consecutive Started',
        'Days Since Last Match', 'Days Until Next Match',
        'Lineup Changes Form', 'Lineup Changes Overall', 'Competition Rotation Rate',
        'Sub Mins Played Form', 'Sub Mins Played Overall',
        'Sub In Rate Form', 'Sub In Rate Overall',
        'Num Subs Form', 'Num Subs Overall',
        'Avg Minute Subbed In in Last 1 Matches',
        'Avg Minute Subbed In in Last 5 Matches',
        'Avg Minute Subbed In in All Matches',
    ]
    categoricalCols = [
        'Position', 'Competition', 'Opposition Primary Competition',
        'Predicted Formation', 'Predicted Opposition Formation',
    ]

    for col in ['Team Power Rating', 'Opposition Power Rating', 'Power Rating Difference', 'Next Match Opposition Power Rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in categoricalCols:
        df[col] = df[col].fillna('Unknown').astype('category')

    # Training: historical bench players only
    benchHistoricalDF = df[(df['Is Future Match'] == False) & (df['Started'] == 0)].copy()
    if benchHistoricalDF.empty:
        print('No historical bench data found. Skipping substitution predictions.')
        return pd.DataFrame()

    matchOrder = (benchHistoricalDF[['Match ID', 'Days Since Season Start']].drop_duplicates('Match ID').sort_values('Days Since Season Start')['Match ID'].tolist())
    nTrain = int(len(matchOrder) * 0.8)
    trainMatchIDs = matchOrder[:nTrain]
    valMatchIDs = matchOrder[nTrain:]

    trainDF = benchHistoricalDF[benchHistoricalDF['Match ID'].isin(trainMatchIDs)]
    valDF = benchHistoricalDF[benchHistoricalDF['Match ID'].isin(valMatchIDs)]

    xTrain = trainDF[featureCols].copy()
    yTrain = trainDF['Substituted'].astype(int)
    xVal = valDF[featureCols].copy()
    yVal = valDF['Substituted'].astype(int)
    xFull = benchHistoricalDF[featureCols].copy()
    yFull = benchHistoricalDF['Substituted'].astype(int)

    modelParams = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
    )

    valModel = XGBClassifier(**modelParams)
    valModel.fit(xTrain, yTrain)

    if verbose and len(yVal) > 0:
        valPreds = valModel.predict_proba(xVal)[:, 1]
        print(f'Sub-In Classifier - Validation Log-Loss: {log_loss(yVal, valPreds):.3f}')
        featureImportance = pd.DataFrame({
            'Feature': xTrain.columns,
            'Importance': valModel.feature_importances_,
        }).sort_values('Importance', ascending=False)
        print(featureImportance.to_string())

    finalModel = XGBClassifier(**modelParams)
    finalModel.fit(xFull, yFull)

    # Test: future match bench players (not in predicted lineup)
    futureMask = (df['Is Future Match'] == True) & (df['Match ID'] == predictedMatchIDs[0])
    benchTestDF = df[futureMask & ~df['ID'].isin(predictedLineup['ID'])].copy()

    historicalIDs = set(df[df['Is Future Match'] == False]['ID'])
    benchTestDF = benchTestDF[benchTestDF['ID'].isin(historicalIDs)].copy()

    if benchTestDF.empty:
        print('No bench players with historical data found for predicted match.')
        return pd.DataFrame()

    rawProbs = finalModel.predict_proba(benchTestDF[featureCols].copy())[:, 1]
    # Normalize proportionally so probs sum to the historical average subs made per match.
    # Then scale the whole distribution down if the maximum exceeds 0.90 — even the most
    # certain sub-in candidate has some chance of not coming on (blowout, extra time, etc.).
    avgSubsPerMatch = benchHistoricalDF.groupby('Match ID')['Substituted'].sum().mean()
    expectedSubs = min(avgSubsPerMatch, len(rawProbs))
    total = rawProbs.sum()
    if total > 0:
        scaledProbs = rawProbs / total * expectedSubs
        maxScaled = scaledProbs.max()
        if maxScaled > 0.90:
            scaledProbs = scaledProbs * (0.90 / maxScaled)
        scaledProbs = scaledProbs.clip(0, 0.90)
    else:
        scaledProbs = rawProbs
    benchTestDF['Sub In Probability'] = scaledProbs.round(3)

    historicalDF = df[df['Is Future Match'] == False]
    mostLikelySlot = (historicalDF.groupby('Player Name')['Formation Slot']
                      .agg(lambda x: x.value_counts().index[0] if not x.value_counts().empty else None)
                      .reset_index()
                      .rename(columns={'Formation Slot': 'Slot'}))
    benchTestDF = benchTestDF.merge(mostLikelySlot, on='Player Name', how='left')
    benchTestDF['Slot'] = benchTestDF['Slot'].fillna(benchTestDF['Position'])

    return (benchTestDF[['Player Name', 'ID', 'Slot', 'Sub In Probability']]
            .sort_values('Sub In Probability', ascending=False)
            .reset_index(drop=True))
#%%
def predictSubstitutions(subCandidatesDF, team, opposition, home, verbose=True):

    if subCandidatesDF is None or subCandidatesDF.empty:
        return pd.DataFrame()

    top5 = subCandidatesDF.head(5).copy()
    top5 = top5.rename(columns={'Player Name': 'Player'})
    resultDF = top5[['Player', 'Slot', 'Sub In Probability']].reset_index(drop=True)

    if verbose and not resultDF.empty:
        if home:
            print(f'\n{opposition} at {team}')
        else:
            print(f'\n{team} at {opposition}')
        print(f'{team} Top Sub-In Candidates:')
        print(resultDF.to_string(index=False))

    return resultDF
# %%
def main(leagues=['MLS'], team='Columbus_Crew', date=date.today().strftime('%m-%d-%Y')):
    year = date.split('-')[2]

    mainDF, subsDF, predictedMatchIDs, allMissingPlayersDF, predictedMissingPlayers = getData(leagues, team, date)

    mainDF = addPowerRankings(mainDF, year)

    mainDF = engineerFeatures(mainDF)

    team = mainDF['Team'].iloc[0]
    home = mainDF['Home'].iloc[0]
    opposition = mainDF[mainDF['Match ID'] == predictedMatchIDs[0]]['Opposition'].iloc[0]
    
    probsDF = computeStartingProbabilities(mainDF.copy(), predictedMatchIDs)

    formation = predictFormation(mainDF).get(predictedMatchIDs[0], '4-3-3')

    predictedLineup = predictLineup(mainDF, probsDF, formation, team, opposition, home)

    testLineupPredictions(mainDF, allMissingPlayersDF.copy())

    mainDF = engineerSubstitutionFeatures(mainDF)

    subCandidatesDF = computeSubstitutionProbabilities(mainDF.copy(), predictedMatchIDs, predictedLineup)

    predictedSubstitutions = predictSubstitutions(subCandidatesDF, team, opposition, bool(home))

    if predictedMissingPlayers:
        historicalDF = mainDF[mainDF['Is Future Match'] == False]
        mostLikelySlot = (
            historicalDF.groupby('Player Name')['Formation Slot']
            .agg(lambda x: x.dropna().value_counts().index[0] if x.dropna().any() else None)
            .to_dict()
        )
        mostLikelyPosition = (
            historicalDF.groupby('Player Name')['Position']
            .agg(lambda x: x.dropna().value_counts().index[0] if x.dropna().any() else None)
            .to_dict()
        )
        for name, info in predictedMissingPlayers.items():
            info['Slot'] = mostLikelySlot.get(name) or mostLikelyPosition.get(name) or ''

    return predictedLineup, predictedSubstitutions, predictedMissingPlayers

if __name__ == "__main__":
    main()