# %%
import pandas as pd
import numpy as np
import os
import math

# %%
def GetFiles(currentPath):
    '''Returns a list of all visible files inside the folder at currentPath'''
    try:
        return [f for f in os.listdir(currentPath) if not f.startswith('.')]
    except:
        return []

# %%
def getData(year, league, numMatches):

    #--------------------------------
    #   LOAD DATA INTO DATAFRAMES
    #--------------------------------

    # Declare season dataframes
    attributesList = []
    statisticsList = []
    momentumList = []
    playerStatsList = []
    missingPlayersList = []

    seenIDs = set()

    # Load data into dataframes
    teams = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data/{league.replace(' ', '_')}_Data'
    for team in GetFiles(teams):
        teamPath = os.path.join(teams, team)

        # Sort match folders by their startTimestamp so `numMatches` truncates the
        # OLDEST matches (chronological order), not whatever order the OS happens
        # to return from listdir. Previously filesystem order made the truncation
        # essentially random.
        matchFolders = []
        for match in GetFiles(teamPath):
            matchAttributesPath = os.path.join(teamPath, match, 'Match_Attributes.csv')
            try:
                attrs = pd.read_csv(matchAttributesPath)
                ts = attrs['startTimestamp'].iloc[0]
                mid = attrs['id'].iloc[0]
            except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, IndexError) as e:
                print(f'Skipped {teamPath}/{match}: cannot read Match_Attributes.csv ({type(e).__name__}: {e})')
                continue
            matchFolders.append((ts, match, mid))
        matchFolders.sort(key=lambda x: x[0])

        for i, (ts, match, matchID) in enumerate(matchFolders):
            if i >= numMatches:
                break

            matchPath = os.path.join(teamPath, match)

            for dataFile in GetFiles(matchPath):
                filePath = os.path.join(matchPath, dataFile)

                try:
                    data = pd.read_csv(filePath)
                except pd.errors.EmptyDataError:
                    print(f"Empty file skipped: {filePath}")
                    continue

                data['Match ID'] = matchID
                # To prevent repeats, don't add game attributes, stats, or momentum if the match was already scraped
                if matchID not in seenIDs:
                    # Load data file and add to a list of dataframes, then add after loop for efficiency
                    if dataFile == 'Match_Attributes.csv':
                        attributesList.append(data)
                    elif dataFile == 'Full_Team_Statistics.csv':
                        statisticsList.append(data)
                    elif dataFile == 'Match_Momentum.csv':
                        momentumList.append(data)

                # Always add player statistics and missing players as they only have data about their players
                if dataFile == 'Player_Statistics.csv':
                    playerStatsList.append(data)
                elif dataFile == 'Missing_Players.csv':
                    missingPlayersList.append(data)

            if matchID not in seenIDs:
                seenIDs.add(matchID)

    # createPowerRankings only consumes attributes/statistics/momentum — playerStats and
    # missingPlayers are returned for callers' convenience but are NOT required for ranking.
    # Don't refuse to build rankings just because nobody had a missing-players entry.
    listCounts = {
        'attributesList': len(attributesList),
        'statisticsList': len(statisticsList),
        'momentumList': len(momentumList),
        'playerStatsList': len(playerStatsList),
        'missingPlayersList': len(missingPlayersList),
    }
    requiredLists = ['attributesList', 'statisticsList', 'momentumList']
    emptyRequired = [name for name in requiredLists if listCounts[name] == 0]
    if emptyRequired:
        print(f'getData: cannot build power rankings — required lists are empty: {emptyRequired}')
        print(f'  (loaded counts: {listCounts}) — check that the expected CSV files exist for the loaded matches.')
        return

    try:
        matchesAttributes = pd.concat(attributesList, ignore_index=True)
        teamsStatistics = pd.concat(statisticsList, ignore_index=True)
        matchesMomentum = pd.concat(momentumList, ignore_index=True)
        # Optional lists: use empty DataFrames when nothing was loaded so callers don't have to special-case None.
        playersMatchStats = pd.concat(playerStatsList, ignore_index=True) if playerStatsList else pd.DataFrame()
        missingPlayers = pd.concat(missingPlayersList, ignore_index=True) if missingPlayersList else pd.DataFrame()
    except Exception as e:
        print(f'getData: concat failed with {type(e).__name__}: {e}')
        return

    dataframes = [matchesAttributes, teamsStatistics, matchesMomentum, playersMatchStats, missingPlayers]

    '''
    print('')
    print('Matches Attributes')
    print(matchesAttributes)
    print('')
    print('Teams Statistics')
    print(teamsStatistics)
    print('')
    print('Matches Momentum')
    print(matchesMomentum)
    print('')
    print('Players Match Statistics')
    print(playersMatchStats)
    print('')
    print('Missing Players')
    print(missingPlayers)
    '''

    return dataframes

# %%
def GetGD(matchRow):
    '''Returns the goal differential for the given match.'''
    
    homeGD = matchRow['homeScore.display']
    awayGD = matchRow['awayScore.display']

    return homeGD - awayGD

# %%
def GetXGD(matchStatsDF, homeTeam):
    '''Returns the expected goal differential for the given match.'''
    
    homeRow = matchStatsDF.loc[matchStatsDF['Team Name'] == homeTeam]
    awayRow = matchStatsDF.loc[matchStatsDF['Team Name'] != homeTeam]
    
    homeXG = homeRow['Expected goals'].values[0]
    awayXG = awayRow['Expected goals'].values[0]

    return homeXG - awayXG

# %%
def createPowerRankings(year, league, matchesAttributes, teamsStatistics, matchesMomentum, table):  
    #------------------------------------
    #  Create xGD and GD for each team
    #------------------------------------

    # NOTE: Could create a new dataframe before calculating the score that contains one match for each row, which includes the match ID, home team, away team, GD, xGD, and momentum. 
    #       This would make it very easy to compute scores for each game.

    # Sort matches attributes dataframe so its sorted by oldest game at top and newest on bottom
    matchesAttributes = matchesAttributes.sort_values(by='startTimestamp')

    # Create new dataframe with one match per row, including match ID, home team, away team, GD, xGD, and momentum 
    # to make calculating averages of these stats easier for computing the z-scores of each stat for the match score.
    matchRows = []
    for index, matchAttributes in matchesAttributes.iterrows():
        #Check that matchAttributes is not empty
        if matchAttributes.empty:
            print('Skipped match as it has an empty match attributes dataframe.')
            continue

        # Match ID
        matchID = matchAttributes['id']

        # Home Team
        homeTeam = matchAttributes['homeTeam.name']

        # Away Team
        awayTeam = matchAttributes['awayTeam.name']

        # GD
        gd = GetGD(matchAttributes)

        # xGD
        matchStatsRows = teamsStatistics[teamsStatistics['Match ID'].isin([matchID])]
        matchStatsDF = pd.DataFrame(matchStatsRows)
        if matchStatsDF.empty:
            print(f'Skipped match {matchID} ({awayTeam} at {homeTeam}) as it has an empty match stats dataframe.')
            continue
        xGD = GetXGD(matchStatsDF, homeTeam)
        if np.isnan(xGD):
            print(f'Skipped match {matchID} ({awayTeam} at {homeTeam}) as its xGD was NaN.')
            continue

        # Momentum
        matchMomentumRows = matchesMomentum[matchesMomentum['Match ID'].isin([matchID])]
        matchMomentumDF = pd.DataFrame(matchMomentumRows)
        if matchMomentumDF.empty:
            momentum = 0.0
            print(f'{matchID} ({awayTeam} at {homeTeam}) as it has an empty match momentum dataframe. Assigned it momentum of 0.0.')
        else:
            momentum = matchMomentumDF['value'].sum()

        matchRows.append({'Match ID': matchID, 'Home Team': homeTeam, 'Away Team': awayTeam, 'GD': gd, 'xGD': xGD, 'Momentum': momentum})

    matchesDF = pd.DataFrame(matchRows)

    # Compute mean and standard deviation for GD, xGD, and momentum to calculate z-scores for each match
    meanGD = matchesDF['GD'].mean()
    stdGD = matchesDF['GD'].std()
    meanXGD = matchesDF['xGD'].mean()
    stdXGD = matchesDF['xGD'].std()
    meanMomentum = matchesDF['Momentum'].mean()
    stdMomentum = matchesDF['Momentum'].std()

    # Compute z-scores for stats to scale stats equally
    matchesDF['zGD'] = (matchesDF['GD'] - meanGD) / stdGD
    matchesDF['zXGD'] = (matchesDF['xGD'] - meanXGD) / stdXGD
    matchesDF['zMomentum'] = (matchesDF['Momentum'] - meanMomentum) / stdMomentum

    # Compute the match score and add to list for dataframe column
    matchesDF['Match Score'] = (0.5 * matchesDF['zGD']) + (0.75 * matchesDF['zXGD']) + (0.15 * matchesDF['zMomentum'])

    # Get all team names
    teamNames = matchesDF['Home Team'].unique()
    teamNames = pd.concat([matchesDF['Home Team'], matchesDF['Away Team']]).unique().tolist()

    #Adjust the match scores by adjusting for home team advantage
    print(f'Average Home Match Score: {matchesDF['Match Score'].mean()}')
    matchesDF['Home Adjusted Match Score'] = matchesDF['Match Score'] - matchesDF['Match Score'].mean()

    #print(matchesDF.to_string())

    #Initialize home and away team 
    matchesDF['Home Team Strength'] = 0.0
    matchesDF['Away Team Strength'] = 0.0

    for i in range(0,5):
        # Create dictionary for total team score
        totalTeamScores = {}

        #Update team strength columns
        meanHomeTeamStrength = matchesDF['Home Team Strength'].mean()
        stdHomeTeamStrength = matchesDF['Home Team Strength'].std()
        if meanHomeTeamStrength == 0.0 and stdHomeTeamStrength == 0.0:
            matchesDF['zHomeTeamStrength'] = 0.0
        else:
            matchesDF['zHomeTeamStrength'] = (matchesDF['Home Team Strength'] - meanHomeTeamStrength) / stdHomeTeamStrength
        meanAwayTeamStrength = matchesDF['Away Team Strength'].mean()
        stdAwayTeamStrength = matchesDF['Away Team Strength'].std()
        if meanAwayTeamStrength == 0.0 and stdAwayTeamStrength == 0.0:
            matchesDF['zAwayTeamStrength'] = 0.0
        else:
            matchesDF['zAwayTeamStrength'] = (matchesDF['Away Team Strength'] - meanAwayTeamStrength) / stdAwayTeamStrength

        for team in teamNames:
            totalTeamScores[team] = 0.0
            teamDF = matchesDF[matchesDF['Home Team'].isin([team]) | matchesDF['Away Team'].isin([team])]
            matchesPlayed = len(teamDF)
            matchNumber = 0
            for index, match in teamDF.iterrows():
                matchesAgo = matchesPlayed - matchNumber
                formWeight = math.exp(-0.2 * matchesAgo)
                if match['Home Team'] == team:
                    totalTeamScores[team] += ((match['Home Adjusted Match Score'] + (0.25 * match['zAwayTeamStrength'])) * formWeight)
                    #totalTeamScores[team] += ((match['Home Adjusted Match Score']) * formWeight)
                elif match['Away Team'] == team:
                    # Symmetry: the strength term should give a bonus to whichever team faced a stronger opponent.
                    # Previously the away formula had `+ 0.25*zHomeStrength` inside a `-=`, which penalized the away
                    # team MORE for losing to a strong home team (the opposite of correct).
                    totalTeamScores[team] -= ((match['Home Adjusted Match Score'] - (0.25 * match['zHomeTeamStrength'])) * formWeight)
                    #totalTeamScores[team] -= ((match['Home Adjusted Match Score']) * formWeight)
                matchNumber += 1

        #Update team strength columns in matchesDF
        for index, match in matchesDF.iterrows():
            homeTeam = match['Home Team']
            awayTeam = match['Away Team']
            matchesDF.at[index, 'Home Team Strength'] = totalTeamScores[homeTeam]
            matchesDF.at[index, 'Away Team Strength'] = totalTeamScores[awayTeam]

    # Create Power rankings
    powerRankings = pd.DataFrame.from_dict(totalTeamScores, orient='index').reset_index()
    powerRankings.columns = ['Team Name', 'Team Rating']
    powerRankings = powerRankings.sort_values(by='Team Rating', ascending=False)
    powerRankings.index = np.arange(1, len(powerRankings) + 1)
    powerRankings['Team Rating'] = powerRankings['Team Rating'].round(2)

    print('')
    print(f'{year} {league} POWER RANKINGS:')
    print(powerRankings)

    if table:
        matchesDF['Home Points'] = matchesDF['GD'].apply(lambda x: 3 if x > 0 else (1 if x == 0 else 0))
        matchesDF['Away Points'] = matchesDF['GD'].apply(lambda x: 3 if x < 0 else (1 if x == 0 else 0))

        homeStandings = matchesDF.groupby('Home Team')['Home Points'].sum()
        awayStandings = matchesDF.groupby('Away Team')['Away Points'].sum()

        standingsDF = (homeStandings.add(awayStandings, fill_value=0)
                       .reset_index()
                       .rename(columns={'Home Team': 'Team Name', 0: 'Points'})
                       .sort_values('Points', ascending=False))
        standingsDF.columns = ['Team Name', 'Points']
        standingsDF.index = np.arange(1, len(standingsDF) + 1)

        print('')
        print(f'{year} {league} STANDINGS:')
        print(standingsDF)

        return powerRankings, standingsDF

    return powerRankings

# %%
def main(year='2026', league='MLS', numMatches=100000000, table=False):

    dataframes = getData(year, league, numMatches)

    if dataframes is None:
        print('Data Not Found. Skipping Power Rankings.')
        return

    powerRankings = createPowerRankings(year, league, dataframes[0], dataframes[1], dataframes[2], table)

    return powerRankings

if __name__ == "__main__":

    main()