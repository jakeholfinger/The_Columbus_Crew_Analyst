# %%
#pip install curl_cffi

# %%
# Use 'curl_cffi' instead of standard 'requests' to prevent 403 error
from curl_cffi import requests
import pandas as pd
import numpy as np
import os
import time
import random
from datetime import date, datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor
import re
import json
import urllib.request
import urllib.parse

# %%
def CreateFolderPath(previousFolderPath, folderName):
    """Creates a folder with a defined name and path if it doesn't exist, then returns the folder path"""
    #Set up folder path
    folderPath = os.path.join(previousFolderPath, folderName)

    #Create folder if it doesn't exist
    os.makedirs(folderPath, exist_ok=True)

    return folderPath

# %%
def safe_request(url, headers, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                impersonate='chrome120',
                timeout=10
            )
            return response
        except Exception as e:
            print(f"Request failed (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return None

# %%
def ScrapeStatistics(data, fullDataframe, firstHalfDataframe, secondHalfDataframe, count):
    """Adds name, homeValue, and awayValue data to dataframe"""
    #If data is a dictionary
    if isinstance(data, dict):
        #If the dictionary has 'name', 'homeValue', and 'awayValue' as keys
        if 'name' in data.keys() and 'homeValue' in data.keys() and 'awayValue' in data.keys():
            #Add the data as column to the dataframe
            if data['name'] == 'Ball possession':
                count[0] += 1;
            if count[0] == 1:
                fullDataframe[data['name']] = [data['homeValue'], data['awayValue']]
            elif count[0] == 2:
                firstHalfDataframe[data['name']] = [data['homeValue'], data['awayValue']]
            elif count[0] == 3:
                secondHalfDataframe[data['name']] = [data['homeValue'], data['awayValue']]
        #If the dictionary isn't what we're looking for
        else:
            #Keep looking through every value
            for value in data.values():
                ScrapeStatistics(value, fullDataframe, firstHalfDataframe, secondHalfDataframe, count)
    #If data is a list
    elif isinstance(data, list):
        #Look through each item in the list
        for item in data:
            ScrapeStatistics(item, fullDataframe, firstHalfDataframe, secondHalfDataframe, count)

# %%
def ScrapePlayerStats(data, playerRows, missingRows, formation):
    """Adds the team's player statistics to dataframe and return the team's formation"""
    #If data is a dictionary
    if isinstance(data, dict):
        #If data holds player data
        if 'formation' in data.keys():
            formation[0] = data['formation']
        #If dictionary contains data of player who played
        if 'player' in data.keys() and 'statistics' in data.keys() and 'substitute' in data.keys():
            #Add all pairs (relevant statistics) in statistics dictionary to stats dictionary
            stats = {}
            for key, value in data['statistics'].items():
                if not isinstance(value,dict):
                    stats[key] = value
            #Add data as row to correct dataframe
            playerData = {'ID': data.get('player', {}).get('id'),
                          'Country': data.get('player', {}).get('country', {}).get('name', ''),
                          'Birth Time': data.get('player', {}).get('dateOfBirthTimestamp'),
                          'Position': data.get('player', {}).get('position', ''),
                          'Captain': data.get('captain'),
                          'Height': data.get('player', {}).get('height'),
                          'Number': data.get('player', {}).get('jerseyNumber'),
                          'Sofascore Market Value': data.get('player', {}).get('proposedMarketValueRaw', {}).get('value'),
                          'Substitute': data.get('substitute')
                         }
            playerData.update(stats)
            playerName = data.get('player', {}).get('name')
            playerRows[playerName] = playerData
        #If data contains data of missing player
        elif 'player' in data.keys() and 'description' in data.keys() and 'type' in data.keys():
            #Add data to a row to correct dataframe
            playerData = {'ID': data.get('player', {}).get('id'),
                          'Country': data.get('player', {}).get('country', {}).get('name', ''),
                          'Birth Time': data.get('player', {}).get('dateOfBirthTimestamp'),
                          'Position': data.get('player', {}).get('position', ''),
                          'Height': data.get('player', {}).get('height'),
                          'Number': data.get('player', {}).get('jerseyNumber'),
                          'Sofascore Market Value': data.get('player', {}).get('proposedMarketValueRaw', {}).get('value'),
                          'Injury': data.get('description', ''),
                          'Availability': data.get('type', '')
                         }
            #If awayFormation is empty
            playerName = data.get('player', {}).get('name')
            missingRows[playerName] = playerData
        #If the dictionary isn't what we're looking for
        else:
            #Keep looking through every value
            for value in data.values():
                ScrapePlayerStats(value, playerRows, missingRows, formation)
    #If data is a list
    elif isinstance(data, list):
        #Look through each item in the list
        for item in data:
            ScrapePlayerStats(item, playerRows, missingRows, formation)

# %%
def ScrapeShotMap(data, homeDataframe, awayDataframe):
    if isinstance(data, dict):
        if 'player' in data.keys() and 'playerCoordinates' in data.keys() and 'shotType' in data.keys():
            shotData = {'Added Time': data.get('addedTime'),
                        'Body Part': data.get('bodyPart', ''),
                        'X Start of Shot': data.get('draw', {}).get('start', {}).get('x'),
                        'Y Start of Shot': data.get('draw', {}).get('start', {}).get('y'),
                        'X End of Shot': data.get('draw', {}).get('end', {}).get('x'),
                        'Y End of Shot': data.get('draw', {}).get('end', {}).get('y'),
                        'X Goal Shot': data.get('draw', {}).get('goal', {}).get('x'),
                        'Y Goal Shot': data.get('draw', {}).get('goal', {}).get('y'),
                        'X Block': data.get('draw', {}).get('block', {}).get('x'),
                        'Y Block': data.get('draw', {}).get('block', {}).get('y')
                       }
            shotData.update(stats)
        else:
            #Keep looking through every value
            for value in data.values():
                ScrapeShotMap(value, homeDataframe, awayDataframe)
    elif isinstance(data,list):
        #Keep looking through every item in the list
        for item in data:
            ScrapeShotMap(item, homeDataframe, awayDataframe)
#%%
_venue_surface_cache = {}

# %%
def ScrapeVenueSurface(venueName):
    """Returns the playing surface for a venue by searching Wikipedia's stadium infobox."""
    if venueName in _venue_surface_cache:
        return _venue_surface_cache[venueName]

    try:
        # Search Wikipedia for the venue
        searchParams = {
            'action': 'query',
            'list': 'search',
            'srsearch': venueName,
            'srnamespace': '0',
            'srlimit': '1',
            'format': 'json'
        }
        wikiHeaders = {'User-Agent': 'CCAnalystBot/1.0 (thecolumbuscrewanalyst@gmail.com) Python-urllib/3'}
        searchURL = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(searchParams)
        with urllib.request.urlopen(urllib.request.Request(searchURL, headers=wikiHeaders)) as r:
            results = json.loads(r.read()).get('query', {}).get('search', [])
        if not results:
            _venue_surface_cache[venueName] = 'Unknown'
            return 'Unknown'

        pageTitle = results[0]['title']

        # Get the article's wikitext
        parseParams = {
            'action': 'parse',
            'page': pageTitle,
            'prop': 'wikitext',
            'format': 'json'
        }
        parseURL = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(parseParams)
        with urllib.request.urlopen(urllib.request.Request(parseURL, headers=wikiHeaders)) as r:
            wikitext = json.loads(r.read()).get('parse', {}).get('wikitext', {}).get('*', '')

        # Parse surface field from infobox — stop at newline only, not | (| appears inside [[link|display]] syntax)
        match = re.search(r'\|\s*surface\s*=\s*([^\n{}]+)', wikitext, re.IGNORECASE)
        if not match:
            _venue_surface_cache[venueName] = 'Unknown'
            return 'Unknown'

        surface = match.group(1).strip()
        # Strip wikitext link markup: [[Target|Display]] → Display, [[Target]] → Target
        surface = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', surface)
        # Strip bold/italic markup and leading bullet points
        surface = re.sub(r"'{2,3}", '', surface)
        surface = surface.lstrip('* ').strip()

        # Normalize to one of: Grass, Turf, Hybrid, Unknown
        s = surface.lower()
        grass_terms = ('grass', 'natural', 'lawn', 'bermuda', 'bluegrass', 'ryegrass',
                       'fescue', 'cynodon', 'festuca', 'lolium', 'zoysia', 'poa ')
        if 'hybrid' in s or 'grassmaster' in s or 'sisgrass' in s:
            category = 'Hybrid'
        elif 'artificial' in s or 'synthetic' in s:
            category = 'Turf'
        elif any(t in s for t in grass_terms):
            category = 'Grass'
        elif 'turf' in s:
            category = 'Turf'
        else:
            category = 'Unknown'
            print('Unknown surface:', surface)

        _venue_surface_cache[venueName] = category
        return category

    except Exception as e:
        print(f"Error scraping surface for '{venueName}': {e}")
        _venue_surface_cache[venueName] = 'Unknown'
        return 'Unknown'
# %%
def ScrapeDataType(homeTeamMatchFolderPath, awayTeamMatchFolderPath, dataType, matchID, homeTeamName, awayTeamName, homeTeamID, headers, overwriteMatchFiles):
    """Scrapes a defined type of data from a defined match on sofascore."""

    fileNames = {'/shotmap': ['Shot_Map.csv'],
                 '/heatmap': ['Player_Spatial_Points.csv', 'Goalkeeper_Spatial_Points.csv'],
                 '/graph': ['Match_Momentum.csv'],
                 '': ['Match_Attributes.csv'],
                 '/lineups': ['Player_Statistics.csv', 'Missing_Players.csv', 'Formations.csv'],
                 '/average-positions': ['Average_Positions.csv', 'Subs.csv'],
                 '/statistics': ['Full_Team_Statistics.csv', 'First_Half_Team_Statistics.csv', 'Second_Half_Team_Statistics.csv'],
                }

    if not overwriteMatchFiles:
        #Rename the heatmap datatype to match with dictionary
        checkDataType = dataType
        if '/heatmap' in dataType:
            checkDataType = '/heatmap'
        #Loop through all file names for the data type
        for fileName in fileNames[checkDataType]:
            homeDataFilePath = os.path.join(homeTeamMatchFolderPath, fileName)
            awayDataFilePath = os.path.join(awayTeamMatchFolderPath, fileName)
            #If both data files exist, don't scrape them
            if os.path.exists(homeDataFilePath) and os.path.exists(awayDataFilePath):
                print(f'Skipped Scraping {fileName}')
                if dataType == '/lineups':
                    homePlayersIDs = {}
                    awayPlayersIDs = {}
                    return homePlayersIDs, awayPlayersIDs
                else:
                    return

    #Construct api url
    apiURL = f'https://www.sofascore.com/api/v1/event/{matchID}{dataType}'

    response = safe_request(apiURL, headers)

    #Convert raw data (which is in bytes) into JSON file
    dataJSON = response.json()

    if dataJSON is None or response.status_code == 404:
        print(f"Failed to get data from {apiURL}, skipping.")
        return {}, {}

    #Declare lists
    homeDataframes = []
    awayDataframes = []
    homeFileNames = []
    awayFileNames = []

    #Convert list to table
    if dataType == '/shotmap':
        #Convert JSON file to list for formatting. The brackets are so it returns an empty list if there's no shots instead of crashing
        dataList = dataJSON.get('shotmap', [])

        #Convert to dataframe
        dataframe = pd.json_normalize(dataList)

        #Filter for relevant columns
        dataframe = dataframe.reindex(columns=['player.name', 'player.id', 'isHome', 'shotType', 'situation', 'bodyPart', 'goalMouthLocation', 'xg', 'xgot', 'id', 'time', 'timeSeconds', 'addedTime', 'player.position', 'playerCoordinates.x', 'playerCoordinates.y', 'playerCoordinates.z', 'goalMouthCoordinates.x', 'goalMouthCoordinates.y', 'goalMouthCoordinates.z', 'goalkeeper.name', 'goalkeeper.id', 'draw.start.x', 'draw.start.y', 'draw.end.x', 'draw.end.y', 'draw.goal.x', 'draw.goal.y', 'blockCoordinates.x', 'blockCoordinates.y', 'blockCoordinates.z', 'draw.block.x', 'draw.block.y', 'goalType'])

        #Add dataframe to list
        homeDataframes.append(dataframe)
        awayDataframes.append(dataframe)

        #Set up file names
        homeFileNames.append('Shot_Map.csv')
        awayFileNames.append('Shot_Map.csv')
    elif '/heatmap' in dataType:
        #Convert JSON files to list for formatting
        dataPlayerList = dataJSON.get('playerPoints', [])
        dataGoalkeeperList = dataJSON.get('goalkeeperPoints', [])

        if str(homeTeamID) in dataType:
            #Convert to dataframes
            homeDataframes.append(pd.DataFrame(dataPlayerList))
            homeDataframes.append(pd.DataFrame(dataGoalkeeperList))

            #Name files and add them to files lists
            homeFileNames.append('Player_Spatial_Points.csv')
            homeFileNames.append('Goalkeeper_Spatial_Points.csv')
        else:
            #Convert to dataframes
            awayDataframes.append(pd.DataFrame(dataPlayerList))
            awayDataframes.append(pd.DataFrame(dataGoalkeeperList))

            #Name files and add them to files lists
            awayFileNames.append('Player_Spatial_Points.csv')
            awayFileNames.append('Goalkeeper_Spatial_Points.csv')
    elif dataType == '/graph':
        #Convert JSON file to list for formatting. The brackets are so it returns an empty list if there's no shots instead of crashing
        dataList = dataJSON.get('graphPoints', [])

        #Convert to dataframe
        dataframe = pd.json_normalize(dataList)

        #Add dataframe to list
        homeDataframes.append(dataframe)
        awayDataframes.append(dataframe)

        #Set up file names
        homeFileNames.append('Match_Momentum.csv')
        awayFileNames.append('Match_Momentum.csv')
    elif dataType == '':
        #Convert JSON file to list for formatting. The brackets are so it returns an empty list if there's no shots instead of crashing
        dataList = dataJSON.get('event', [])

        #Convert to dataframe
        dataframe = pd.json_normalize(dataList)

        #Filter relevant columns
        dataframe = dataframe.reindex(columns=['id', 'customID', 'attendence', 'hasGlobalHighlights', 'hasXg', 'hasEventPlayerStatistics', 'hasEventPlayerHeatMap', 'crowadsourcingDataDisplayEnabled', 'awayRedCards', 'slug', 'startTimestamp', 'finalResultOnly', 'cupMatchesInRound', 'seasonStatisticsType', 'roundInfo.name', 'roundInfo.round', 'roundInfo.cupRoundType', 'tournament.name', 'tournament.id', 'tournament.slug', 'tournament.category.name', 'tournament.category.id', 'tournament.uniqueTournament.name', 'tournament.uniqueTournament.id', 'tournament.uniqueTournament.primaryColorHex', 'tournament.uniqueTournament.secondaryColorHex', 'tournament.uniqueTournament.hasRounds', 'tournament.uniqueTournament.hasPerformanceGraphFeature', 'tournament.uniqueTournament.hasEventPlayerStatistics', 'tournament.uniqueTournament.displayInverseHomeAwayTeams', 'tournament.competitionType', 'tournament.isGroup', 'season.name', 'season.year', 'season.id', 'status.type', 'venue.venueCoordinates.latitude', 'venue.venueCoordinates.longitude', 'venue.name', 'venue.capacity', 'venue.country.name', 'venue.id', 'venue.city.name', 'referee.name', 'referee.yellowCards', 'referee.redCards', 'referee.yellowRedCards', 'referee.games', 'referee.country.name', 'referee.id', 'homeTeam.name', 'homeTeam.manager.name', 'homeTeam.manager.country.name', 'homeTeam.manager.id', 'homeTeam.national', 'homeTeam.id', 'homeTeam.fullName', 'homeTeam.nameCode', 'homeTeam.teamColors.primary', 'homeTeam.teamColors.secondary', 'homeTeam.teamColors.text','homeTeam.foundationDateTimestamp', 'awayTeam.name', 'awayTeam.manager.name', 'awayTeam.manager.country.slug', 'awayTeam.manager.id', 'awayTeam.venue.venueCoordinates.latitude', 'awayTeam.venueCoordinates.longitude', 'awayTeam.venue.name', 'awayTeam.venue.capacity', 'awayTeam.venue.country.name', 'awayTeam.venue.id', 'awayTeam.venue.city.name', 'awayTeam.nameCode', 'awayTeam.national', 'awayTeam.country.name', 'awayTeam.id', 'awayTeam.fullName', 'awayTeam.teamColors.primary', 'awayTeam.teamColors.secondary', 'awayTeam.teamColors.text', 'awayTeam.foundationDateTimestamp', 'homeScore.display', 'homeScore.period1', 'homeScore.period2', 'homeScore.normalTime', 'homeScore.extra1', 'homeScore.extra2', 'homeScore.overtime', 'homeScore.penalties', 'awayScore.display', 'awayScore.period1', 'awayScore.period2', 'awayScore.normalTime', 'awayScore.extra1', 'awayScore.extra2', 'awayScore.overtime', 'awayScore.penalties', 'aggregatedWinnerCode', 'winnerCode', 'time.injuryTime1', 'time.injuryTime2'])

        if dataframe['venue.name'].iloc[0] is None:
            venueData = dataJSON.get('homeTeam', {}).get('venue', {})
            dataframe['venue.name'].iloc[0] = venueData.get('stadium', {}).get('name')
            dataframe['venue.id'].iloc[0] = venueData.get('id')
            dataframe['venue.venueCoordinates.latitude'].iloc[0] = venueData.get('venueCoordinates', {}).get('latitude')
            dataframe['venue.venueCoordinates.longitude'].iloc[0] = venueData.get('venueCoordinates', {}).get('longitude')
            dataframe['venue.capacity'].iloc[0] = venueData.get('capacity')
            dataframe['venue.city.name'].iloc[0] = venueData.get('country', {}).get('name')
            dataframe['venue.country.name'].iloc[0] = venueData.get('city', {}).get('name')
        dataframe['venue.surface'] = ScrapeVenueSurface(dataframe['venue.name'].iloc[0])

        #Add dataframe to list
        homeDataframes.append(dataframe)
        awayDataframes.append(dataframe)

        #Set up file names
        homeFileNames.append('Match_Attributes.csv')
        awayFileNames.append('Match_Attributes.csv')
    elif dataType == '/lineups':
        #Declare variables to store scraped data in
        homeRows = {}
        awayRows = {}
        homeMissingRows = {}
        awayMissingRows = {}
        homeFormation = ['']
        awayFormation = ['']

        #Scrape player data
        ScrapePlayerStats(dataJSON['home'], homeRows, homeMissingRows, homeFormation)
        ScrapePlayerStats(dataJSON['away'], awayRows, awayMissingRows, awayFormation)

        #Convert dictionaries to dataframes
        homePlayersDataframe = pd.DataFrame.from_dict(homeRows, orient='index').reset_index(names='Player Name')
        awayPlayersDataframe = pd.DataFrame.from_dict(awayRows, orient='index').reset_index(names='Player Name')
        homeMissingDataframe = pd.DataFrame.from_dict(homeMissingRows, orient='index').reset_index(names='Player Name')
        awayMissingDataframe = pd.DataFrame.from_dict(awayMissingRows, orient='index').reset_index(names='Player Name')

        #Create formation dataframes
        formationData = {'Team Name': [homeTeamName, awayTeamName], 'Formation': [homeFormation, awayFormation]}
        formationDataframe = pd.DataFrame(formationData)

        #Add dataframe to list
        homeDataframes.append(homePlayersDataframe)
        homeDataframes.append(homeMissingDataframe)
        homeDataframes.append(formationDataframe)
        awayDataframes.append(awayPlayersDataframe)
        awayDataframes.append(awayMissingDataframe)
        awayDataframes.append(formationDataframe)

        #Set up file names
        homeFileNames.append('Player_Statistics.csv')
        homeFileNames.append('Missing_Players.csv')
        homeFileNames.append('Formations.csv')
        awayFileNames.append('Player_Statistics.csv')
        awayFileNames.append('Missing_Players.csv')
        awayFileNames.append('Formations.csv')
    elif dataType == '/average-positions':
        #Convert JSON file to list for formatting. The brackets are so it returns an empty list if there's no shots instead of crashing
        homeDataList = dataJSON.get('home', [])
        awayDataList = dataJSON.get('away', [])
        subsDataList = dataJSON.get('substitutions', [])

        #Convert to dataframe
        homeDataframe = pd.json_normalize(homeDataList)
        awayDataframe = pd.json_normalize(awayDataList)
        subsDataframe = pd.json_normalize(subsDataList)
        #homeTeam = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)

        #Filter subs only for relevant team
        if subsDataframe.empty == False:
            homeSubsDataframe = subsDataframe[subsDataframe['isHome'] == True]
            awaySubsDataframe = subsDataframe[subsDataframe['isHome'] == False]
        else:
            homeSubsDataframe = pd.DataFrame()
            awaySubsDataframe = pd.DataFrame()

        #Filter only for relevant columns
        homeDataframe = homeDataframe.reindex(columns=['player.id', 'player.name', 'player.position', 'averageX', 'averageY', 'pointsCount'])
        awayDataframe = awayDataframe.reindex(columns=['player.id', 'player.name', 'player.position', 'averageX', 'averageY', 'pointsCount'])
        homeSubsDataframe = homeSubsDataframe.reindex(columns=['id', 'time', 'reversedPeriodTime', 'incidentClass', 'injury', 'playerIn.name', 'playerIn.position', 'playerIn.id', 'playerOut.name', 'playerOut.position', 'playerOut.id', 'addedTime'])
        awaySubsDataframe = awaySubsDataframe.reindex(columns=['id', 'time', 'reversedPeriodTime', 'incidentClass', 'injury', 'playerIn.name', 'playerIn.position', 'playerIn.id', 'playerOut.name', 'playerOut.position', 'playerOut.id', 'addedTime'])

        #Add dataframe to list
        homeDataframes.append(homeDataframe)
        homeDataframes.append(homeSubsDataframe)
        awayDataframes.append(awayDataframe)
        awayDataframes.append(awaySubsDataframe)

        #Set up file names
        homeFileNames.append('Average_Positions.csv')
        homeFileNames.append('Subs.csv')
        awayFileNames.append('Average_Positions.csv')
        awayFileNames.append('Subs.csv')
    elif dataType == '/statistics':
        #Convert statistics JSON file to dataframe
        teamNames = [homeTeamName, awayTeamName]
        fullDataframe = pd.DataFrame(teamNames, columns=['Team Name'])
        firstHalfDataframe = pd.DataFrame(teamNames, columns=['Team Name'])
        secondHalfDataframe = pd.DataFrame(teamNames, columns=['Team Name'])

        scrape = True
        count = 1
        while scrape and count <= 2:
            ScrapeStatistics(dataJSON, fullDataframe, firstHalfDataframe, secondHalfDataframe, [0])

            count += 1
            if 'Expected goals' in fullDataframe.columns:
                scrape = False
            else:
                if count <= 2:
                    print(f'Scraping did not work, retrying attempt {count}. Match: {awayTeamName} at {homeTeamName}.')

        #Add dataframe to list
        homeDataframes.append(fullDataframe)
        homeDataframes.append(firstHalfDataframe)
        homeDataframes.append(secondHalfDataframe)
        awayDataframes.append(fullDataframe)
        awayDataframes.append(firstHalfDataframe)
        awayDataframes.append(secondHalfDataframe)

        #Set up file names
        homeFileNames.append('Full_Team_Statistics.csv')
        homeFileNames.append('First_Half_Team_Statistics.csv')
        homeFileNames.append('Second_Half_Team_Statistics.csv')
        awayFileNames.append('Full_Team_Statistics.csv')
        awayFileNames.append('First_Half_Team_Statistics.csv')
        awayFileNames.append('Second_Half_Team_Statistics.csv')

    #Home loop
    for name, dataframe in zip(homeFileNames, homeDataframes):
        #Create file path
        teamDataFilePath = os.path.join(homeTeamMatchFolderPath, name)

        #Convert table to csv files. index=False prevents numbering of columns
        dataframe.to_csv(teamDataFilePath, index=False)

        print(f'File saved to {teamDataFilePath}')

    #Away loop
    for name, dataframe in zip(awayFileNames, awayDataframes):
        #Create file path
        teamDataFilePath = os.path.join(awayTeamMatchFolderPath, name)

        #Convert table to csv files. index=False prevents numbering of columns
        dataframe.to_csv(teamDataFilePath, index=False)

        print(f'File saved to {teamDataFilePath}')

    #If the data type is lineups, return two dictionaries of player names mapped to their id
    if dataType == '/lineups':
        homePlayersIDs = dict(zip(homePlayersDataframe['Player Name'], homePlayersDataframe['ID']))
        awayPlayersIDs = dict(zip(awayPlayersDataframe['Player Name'], awayPlayersDataframe['ID']))
        return homePlayersIDs, awayPlayersIDs

# %%
def MatchHasEventData(matchID, headers, homePlayersIDs, awayPlayersIDs):
    """Returns whether the match has event data"""
    '''
    #Construct api url
    apiURL = f'https://www.sofascore.com/api/v1/event/{matchID}'
    #Get data from api request
    response = requests.get(apiURL, headers=headers, impersonate='chrome120')
    #Convert raw data (which is in bytes) into JSON file
    data = response.json()
    if data.get('event', {}).get('hasGlobalHighlights'):
        return True
    '''
    count = 0
    for playersIDs, playersNames in zip([homePlayersIDs.values(), awayPlayersIDs.values()], [homePlayersIDs.keys(), awayPlayersIDs.keys()]):
        for playerID, playerName in zip(playersIDs, playersNames):
            #Construct API URL
            apiURL = f'https://www.sofascore.com/api/v1/event/{matchID}/player/{playerID}/rating-breakdown'
            #Get data from API request
            response = requests.get(apiURL, headers=headers, impersonate='chrome120')
            if response.status_code != 200:
                count += 1
                if count >= 5:
                    return False
            else:
                return True

# %%
def scrapeOnePlayer(matchID, playerID, playerName, homePlayersIDs, headers, dataTypes, hasEventData):
    """Returns (home_rating_rows, away_rating_rows, home_heatmap_rows, away_heatmap_rows) for one player."""
    home_rating_rows, away_rating_rows, home_heatmap_rows, away_heatmap_rows = [], [], [], []
    is_home = playerID in homePlayersIDs.values()
    try:
        if hasEventData and '/player/rating-breakdown' in dataTypes:
            apiURL = f'https://www.sofascore.com/api/v1/event/{matchID}/player/{playerID}/rating-breakdown'
            response = safe_request(apiURL, headers)
            if response is not None and response.status_code == 200:
                data = response.json()
                time.sleep(random.uniform(0.3, 0.8))
                if data is not None:
                    for ratingType in data:
                        if ratingType == 'ball-carries':
                            for ballCarry in data[ratingType]:
                                eventData = {'Player': playerName,
                                             'Event Type': ballCarry.get('eventActionType'),
                                             'Player X Coord': ballCarry.get('playerCoordinates', {}).get('x'),
                                             'Player Y Coord': ballCarry.get('playerCoordinates', {}).get('y'),
                                             'Pass End X Coord': ballCarry.get('passEndCoordinates', {}).get('x'),
                                             'Pass End Y Coord': ballCarry.get('passEndCoordinates', {}).get('y')}
                                (home_rating_rows if is_home else away_rating_rows).append(eventData)
                        elif ratingType == 'defensive':
                            for defAction in data[ratingType]:
                                eventData = {'Player': playerName,
                                             'Event Type': 'Defensive Action',
                                             'Event Sub-Type': defAction.get('eventActionType'),
                                             'Key Pass': defAction.get('keypass'),
                                             'Outcome': defAction.get('outcome'),
                                             'Player X Coord': defAction.get('playerCoordinates', {}).get('x'),
                                             'Player Y Coord': defAction.get('playerCoordinates', {}).get('y'),
                                             'Pass End X Coord': defAction.get('passEndCoordinates', {}).get('x'),
                                             'Pass End Y Coord': defAction.get('passEndCoordinates', {}).get('y')}
                                (home_rating_rows if is_home else away_rating_rows).append(eventData)
                        elif ratingType == 'dribbles':
                            for dribble in data[ratingType]:
                                eventData = {'Player': playerName,
                                             'Event Type': dribble.get('eventActionType'),
                                             'Key Pass': dribble.get('keypass'),
                                             'Outcome': dribble.get('outcome'),
                                             'Player X Coord': dribble.get('playerCoordinates', {}).get('x'),
                                             'Player Y Coord': dribble.get('playerCoordinates', {}).get('y')}
                                (home_rating_rows if is_home else away_rating_rows).append(eventData)
                        elif ratingType == 'passes':
                            for passed in data[ratingType]:
                                eventData = {'Player': playerName,
                                             'Event Type': passed.get('eventActionType'),
                                             'Key Pass': passed.get('keypass'),
                                             'Outcome': passed.get('outcome'),
                                             'Player X Coord': passed.get('playerCoordinates', {}).get('x'),
                                             'Player Y Coord': passed.get('playerCoordinates', {}).get('y'),
                                             'Pass End X Coord': passed.get('passEndCoordinates', {}).get('x'),
                                             'Pass End Y Coord': passed.get('passEndCoordinates', {}).get('y')}
                                (home_rating_rows if is_home else away_rating_rows).append(eventData)
                    time.sleep(random.uniform(0.5, 1.0))

        if '/player/heatmap' in dataTypes:
            apiURL = f'https://www.sofascore.com/api/v1/event/{matchID}/player/{playerID}/heatmap'
            response = safe_request(apiURL, headers)
            if response is not None and response.status_code == 200:
                data = response.json()
                time.sleep(random.uniform(0.3, 0.8))
                for coords in data.get('heatmap', {}):
                    playerCoords = {'Player': playerName, 'X': coords.get('x'), 'Y': coords.get('y')}
                    (home_heatmap_rows if is_home else away_heatmap_rows).append(playerCoords)
    except Exception as e:
        print(f"Error with player {playerID}: {e}")
    return home_rating_rows, away_rating_rows, home_heatmap_rows, away_heatmap_rows

# %%
def ScrapeEventData(homeRatingBreakdowns, awayRatingBreakdowns, homeHeatmaps, awayHeatmaps, homeTeamMatchFolderPath, awayTeamMatchFolderPath, matchID, homePlayersIDs, awayPlayersIDs, headers, dataTypes, hasEventData, overwriteMatchFiles):
    """Scrapes each player's event data for the match."""

    fileNames = {'/player/rating-breakdown': ['Player_Event_Data.csv'],
                 '/player/heatmap': ['Player_Heatmaps.csv']
                }

    checkDataTypes = []

    if not overwriteMatchFiles:
        #Rename the datatype to the list if it should be scraped
        if '/player/rating-breakdown' in dataTypes:
            checkDataTypes.append('/player/rating-breakdown')
        elif '/player/heatmap' in dataTypes:
            checkDataTypes.append('/player/heatmap')
        #Loop through all file names for each data type
        for checkDataType in checkDataTypes:
            for fileName in fileNames[checkDataType]:
                homeDataFilePath = os.path.join(homeTeamMatchFolderPath, fileName)
                awayDataFilePath = os.path.join(awayTeamMatchFolderPath, fileName)
                #If both data files exist, don't scrape them
                if os.path.exists(homeDataFilePath) and os.path.exists(awayDataFilePath):
                    print(f'Skipped Scraping {fileName}')
                    return

    all_players = (
        [(pid, pname) for pname, pid in homePlayersIDs.items()] +
        [(pid, pname) for pname, pid in awayPlayersIDs.items()]
    )

    homeRatingRows, awayRatingRows, homeHeatmapRows, awayHeatmapRows = [], [], [], []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(scrapeOnePlayer, matchID, playerID, playerName,
                            homePlayersIDs, headers, dataTypes, hasEventData)
            for playerID, playerName in all_players
        ]
        for future in futures:
            hr, ar, hh, ah = future.result()
            homeRatingRows.extend(hr)
            awayRatingRows.extend(ar)
            homeHeatmapRows.extend(hh)
            awayHeatmapRows.extend(ah)

    #Add dataframes to list if they have data
    fileNames = []
    homeDataframes = []
    awayDataframes = []
    if hasEventData and '/player/rating-breakdown' in dataTypes:
        homeDataframes.append(pd.DataFrame(homeRatingRows))
        awayDataframes.append(pd.DataFrame(awayRatingRows))
        fileNames.append('Player_Event_Data.csv')
    if '/player/heatmap' in dataTypes:
        homeDataframes.append(pd.DataFrame(homeHeatmapRows))
        awayDataframes.append(pd.DataFrame(awayHeatmapRows))
        fileNames.append('Player_Heatmaps.csv')

    #Home loop
    for name, dataframe in zip(fileNames, homeDataframes):
        #Create file path
        teamDataFilePath = os.path.join(homeTeamMatchFolderPath, name)

        #Convert table to csv files. index=False prevents numbering of columns
        dataframe.to_csv(teamDataFilePath, index=False)

        print(f'File saved to {teamDataFilePath}')

    #Away loop
    for name, dataframe in zip(fileNames, awayDataframes):
        #Create file path
        teamDataFilePath = os.path.join(awayTeamMatchFolderPath, name)

        #Convert table to csv files. index=False prevents numbering of columns
        dataframe.to_csv(teamDataFilePath, index=False)

        print(f'File saved to {teamDataFilePath}')

# %%
def GetPlayersIDs(homePlayersIDs, awayPlayersIDs, matchID, headers):
    """Scrapes all players' IDs who played in the match and stores them in homePlayersIDs and awayPlayers IDs."""

    #Construct api url
    apiURL = f'https://www.sofascore.com/api/v1/event/{matchID}/lineups'

    #Get data from api request
    #response = requests.get(apiURL, headers=headers, impersonate='chrome120')
    response = safe_request(apiURL, headers)

    #Convert raw data (which is in bytes) into JSON file
    data = response.json()

    #Get player lists
    try:
        homePlayersData = data.get('home').get('players', {})
        awayPlayersData = data.get('away').get('players', {})
    except:
        return

    #Get home players' IDs and Names
    for player in homePlayersData:
        homePlayersIDs[player.get('player', {}).get('name')] = player.get('player', {}).get('id')

    #Get away players' IDs
    for player in awayPlayersData:
        awayPlayersIDs[player.get('player', {}).get('name')] = player.get('player', {}).get('id')

# %%
def ScrapeMatchData(leagueFolderPath, matchURL, matchID, matchDate, homeTeamName, awayTeamName, homeTeamID, awayTeamID, dataTypes, booleans, headers):

    #Set up home team folder
    homeTeamFolderName = f'{homeTeamName.replace(" ", "_")}_Data'
    homeTeamFolderPath = CreateFolderPath(leagueFolderPath, homeTeamFolderName)

    #Set up away team folder
    awayTeamFolderName = f'{awayTeamName.replace(" ", "_")}_Data'
    awayTeamFolderPath = CreateFolderPath(leagueFolderPath, awayTeamFolderName)

    #Set up home team match folder
    homeTeamMatchFolderName = f'{matchDate}_vs_{awayTeamName}'.replace(' ', '_')
    homeTeamMatchFolderPath = os.path.join(homeTeamFolderPath, homeTeamMatchFolderName)

    #Set up away team match folder
    awayTeamMatchFolderName = f'{matchDate}_vs_{homeTeamName}'.replace(' ', '_')
    awayTeamMatchFolderPath = os.path.join(awayTeamFolderPath, awayTeamMatchFolderName)

    #Declare player id dictionaries
    homePlayersIDs = {}
    awayPlayersIDs = {}

    #Get the player ids of the players in the match
    GetPlayersIDs(homePlayersIDs, awayPlayersIDs, matchID, headers)

    #Find whether match has event data
    hasEventData = MatchHasEventData(matchID, headers, homePlayersIDs, awayPlayersIDs)
    print(f'Match Has Event Data: {hasEventData}')

    #Declare all potential file names
    fileNames = ['Player_Spatial_Points.csv', 'Goalkeeper_Spatial_Points.csv', 'Match_Momentum.csv', 'Match_Attributes.csv', 'Player_Statistics.csv', 'Missing_Players.csv', 'Formations.csv', 'Average_Positions.csv', 'Subs.csv', 'Full_Team_Statistics.csv', 'First_Half_Team_Statistics.csv', 'Second_Half_Team_Statistics.csv', 'Player_Event_Data.csv', 'Player_Heatmaps.csv']

    matchFoldersAreIncomplete = False
    #Check if both home and away match folders exist
    if os.path.exists(homeTeamMatchFolderPath) and os.path.exists(awayTeamMatchFolderPath):
        homeFiles = os.listdir(homeTeamMatchFolderPath)
        awayFiles = os.listdir(awayTeamMatchFolderPath)
        #Loop through every potential file and check if its in each match folder
        for fileName in fileNames:
            if fileName not in homeFiles or fileName not in awayFiles:
                #If the match is missing the event data file, and if it should have the event data file
                if fileName == 'Player_Event_Data.csv':
                    if hasEventData:
                        print('Match is incomplete - event data')
                        matchFoldersAreIncomplete = True
                        break;
                else:
                    print(f'Match is incomplete - missing {fileName} file')
                    matchFoldersAreIncomplete = True
                    break;

    else:
        print('Match is incomplete - match folders dont exist')
        matchFoldersAreIncomplete = True

    #Scrape the match if not only incomplete matches should be scraped or only incomplete matches should be scraped and the match folder is incomplete (doesn't have all files)
    if (not booleans['onlyScrapeIncompleteMatches']) or (booleans['onlyScrapeIncompleteMatches'] and matchFoldersAreIncomplete):

        #Create home and away match folders
        homeTeamMatchFolderPath = CreateFolderPath(homeTeamFolderPath, homeTeamMatchFolderName)
        awayTeamMatchFolderPath = CreateFolderPath(awayTeamFolderPath, awayTeamMatchFolderName)

        #Scrape the match if any match should be scraped or only new matches should be scraped and one of the paths does not exist
        matchFoldersExists = os.path.exists(homeTeamMatchFolderPath) and os.path.exists(awayTeamMatchFolderPath)
        if not booleans['onlyScrapeNewMatches'] or not matchFoldersExists:

            #If scraper is scraping heatmap data, add the data type for each team's heatmap
            matchDataTypes = dataTypes.copy()
            if '/heatmap' in matchDataTypes:
                matchDataTypes.remove('/heatmap')
                matchDataTypes.append(f'/heatmap/{homeTeamID}')
                matchDataTypes.append(f'/heatmap/{awayTeamID}')

            #Scrape data from match
            for dataType in matchDataTypes:
                if not dataType == '/player/rating-breakdown' and not dataType == '/player/heatmap':
                    #Only store a return value if the dataType is '/lineups'
                    if dataType == '/lineups':
                        homePlayersIDs, awayPlayersIDs = ScrapeDataType(homeTeamMatchFolderPath, awayTeamMatchFolderPath, dataType, matchID, homeTeamName, awayTeamName, homeTeamID, headers, booleans['overwriteMatchFiles'])
                    else:
                        ScrapeDataType(homeTeamMatchFolderPath, awayTeamMatchFolderPath, dataType, matchID, homeTeamName, awayTeamName, homeTeamID, headers, booleans['overwriteMatchFiles'])

            # Get players IDs if lineups datatype wasn't called
            if homePlayersIDs == {} and awayPlayersIDs == {}:
                GetPlayersIDs(homePlayersIDs, awayPlayersIDs, matchID, headers)
            homeRatingBreakdowns = pd.DataFrame(columns=['Player', 'Event Type', 'Event Sub-Type', 'Key Pass', 'Outcome', 'Player X Coord', 'Player Y Coord', 'Pass End X Coord', 'Pass End Y Coord'])
            awayRatingBreakdowns = pd.DataFrame(columns=['Player', 'Event Type', 'Event Sub-Type', 'Key Pass', 'Outcome', 'Player X Coord', 'Player Y Coord', 'Pass End X Coord', 'Pass End Y Coord'])
            homeHeatmaps = pd.DataFrame(columns=['Player', 'X', 'Y'])
            awayHeatmaps = pd.DataFrame(columns=['Player', 'X', 'Y'])
            ScrapeEventData(homeRatingBreakdowns, awayRatingBreakdowns, homeHeatmaps, awayHeatmaps, homeTeamMatchFolderPath, awayTeamMatchFolderPath, matchID, homePlayersIDs, awayPlayersIDs, headers, matchDataTypes, hasEventData, booleans['overwriteMatchFiles'])
            time.sleep(1)
    else:
        print(f'Skipped Match Because Match Folders Are Complete')

# %%
def GetMatches(data, matchesDictionary, leagueID, seasonID, count):
    print('GetMatches')
    print(count)
    matchPageApiURL = f'https://www.sofascore.com/api/v1/unique-tournament/{leagueID}/season/{seasonID}/events/last/{count}'
    for match in data.get('events'):
        matchData = {'Away Team Name': match.get('awayTeam', {}).get('name', ''),
                     'Away Team ID': match.get('awayTeam', {}).get('id', ''),
                     #Move to game characteristic data scraping?
                     'Away Team Primary Color': match.get('awayTeam', {}).get('teamColors', {}).get('primary', ''),
                     'Away Team Secondary Color': match.get('awayTeam', {}).get('teamColors', {}).get('secondary', ''),
                     'Custom ID': match.get('customID'),
                     'Home Team Name': match.get('homeTeam', {}).get('name', ''),
                     'Home Team ID': match.get('homeTeam', {}).get('id', ''),
                     #Move to game characteristic data scraping?
                     'Home Team Primary Color': match.get('homeTeam', {}).get('teamColors', {}).get('primary', ''),
                     'Home Team Secondary Color': match.get('homeTeam', {}).get('teamColors', {}).get('secondary', ''),
                     'Match ID': match.get('id'),
                     'Match Slug': match.get('slug'),
                     'Match Date': match.get('startTimestamp'),
                     'Match Status Code': match.get('status', {}).get('code'),
                     'Match Status Description:': match.get('status', {}).get('description')
                    }
        matchesDictionary[matchData.get('Match ID')] = matchData

# %%
def ScrapeSeasonData(leagueName, leagueID, seasonID, yearFolderPath, dataTypes, totalMatchesScraped, booleans, headers):
    print('ScrapeSeasonData')

    #Set up league folder
    leagueFolderName = f'{leagueName}_Data'.replace(' ', '_')
    leagueFolderPath = CreateFolderPath(yearFolderPath, leagueFolderName)

    #if overwriteGameFiles or not os.path.exists(gameFilePath):
    #else: print(f'Skipping {gameFilePath} because it already exists')

    #teamIDs = {dataJSON.get('homeTeam', {}).get('id'): dataJSON.get('homeTeam', {}).get('slug'), dataJSON.get('awayTeam', {}).get('id'): dataJSON.get('awayTeam', {}).get('slug')}

    #Construct api url
    #matchesApiURL = f'https://www.sofascore.com/api/v1/unique-tournament/{leagueID}/season/{seasonID}/team-events/total'
    matchesApiURL = f'https://www.sofascore.com/api/v1/unique-tournament/{leagueID}/season/{seasonID}/events/last/0'

    #Get data from api request
    response = requests.get(matchesApiURL, headers=headers, impersonate='chrome120')

    #Convert raw data (which is in bytes) into JSON file
    dataJSON = response.json()

    #Get all match api urls
    matchesDictionary = {}
    count = 0
    if isinstance(dataJSON, dict):
        while dataJSON.get('hasNextPage'):
            GetMatches(dataJSON, matchesDictionary, leagueID, seasonID, count)
            count += 1
            #Construct api url
            matchesApiURL = f'https://www.sofascore.com/api/v1/unique-tournament/{leagueID}/season/{seasonID}/events/last/{count}'
            #Get data from api request
            response = requests.get(matchesApiURL, headers=headers, impersonate='chrome120')
            #Convert raw data (which is in bytes) into JSON file
            dataJSON = response.json()

        GetMatches(dataJSON, matchesDictionary, leagueID, seasonID, count)

    totalMatchesScraped[0] = len(matchesDictionary)
    numMatchesScraped = 0

    #Loop through matches:
    for match in matchesDictionary.values():
        #Get match characteristics
        matchID = match.get('Match ID')
        matchSlug = match.get('Match Slug')
        matchTimestamp = match.get('Match Date')
        matchDate = datetime.fromtimestamp(matchTimestamp)
        matchDate = matchDate.strftime('%m-%d-%Y').replace('-0', '-').lstrip('0')

        #If the match occurred or ended
        if match.get('Match Status Code') == 100 or match.get('Match Status Code') == 120:
            customID = match.get('Custom ID')
            matchURL= f'https://www.sofascore.com/football/match/{matchSlug}/{customID}'
            homeTeamName = match.get('Home Team Name')
            awayTeamName = match.get('Away Team Name')
            homeTeamID = match.get('Home Team ID')
            awayTeamID = match.get('Away Team ID')
            print('-----------------------------------------------------')
            print(f'SCRAPING MATCH: {matchSlug} {matchDate}  {numMatchesScraped}/{totalMatchesScraped[0]}')
            print('-----------------------------------------------------')
            ScrapeMatchData(leagueFolderPath, matchURL, matchID, matchDate, homeTeamName, awayTeamName, homeTeamID, awayTeamID, dataTypes, booleans, headers)
        else:
            print('-----------------------------------------------------')
            print(f'SKIPPING MATCH: {matchSlug} {matchDate} because match was {match.get("Match Status Description")}  {numMatchesScraped}/{totalMatchesScraped[0]}')
            print('-----------------------------------------------------')
        numMatchesScraped += 1

# %%
def ScrapeLeaguesInfo(leagueDatabase, data, headers, numLeaguesInfoScraped):
    """Scrapes league info of all leagues that have matches on the given page and updates leagueDatabase"""
    leagues = data.get('scheduled', [])
    for leagueInitial in leagues:
        #Get unique tournament data
        league = leagueInitial.get('tournament', {}).get('uniqueTournament', {})

        #If league isn't already in database
        if league.get('id') not in leagueDatabase['League ID'].values:

            print(f'{numLeaguesInfoScraped}: {league.get("name")}, {league.get("category", {}).get("name")}, {league.get("id")}')

            leagueInfo = {
                          'League Name': league.get('name'),
                          'League ID': league.get('id'),
                          'Country Name': league.get('category', {}).get('name'),
                          'Country ID': league.get('category', {}).get('id')
                          #'Has Player Stats': league.get('hasEventPlayerStatistics'),
                          #'Has Performance Graph': league.get('hasPerformanceGraphFeature')
                         }
            seasonsApiURL = f'https://www.sofascore.com/api/v1/unique-tournament/{leagueInfo.get("League ID")}/seasons'

            #Get data from api request
            response = requests.get(seasonsApiURL, headers=headers, impersonate='chrome120')

            #Convert raw data (which is in bytes) into JSON file
            seasonsData = response.json()

            #Get seasons dictionary
            seasons = seasonsData.get('seasons', {})

            #Loop through seasons
            for season in seasons:
                #Get the season's year
                year = season.get('year')
                if not year:
                    continue
                year = str(year)
                #If the year has a '/' in it, split it on the '/' and use the secondary year and convert it to a 4-digit year
                if '/' in year:
                    part = year.split('/')[1]
                    if len(part) == 2:
                        partInt = int(part)
                        if partInt <= 30:
                            year = 2000 + partInt
                        else:
                            year = 1900 + partInt
                    else:
                        try:
                            year = int(part)
                        except:
                            print(f'Skipped {numLeaguesInfoScraped}: {league.get("name")}, {league.get("category", {}).get("name")}, {league.get("id")}')
                            print(f'Failed to parse \'{part}\'')
                            continue
                #If the year has a space in it, split the string at the space and pick the string that's a number
                elif ' ' in year:
                    year = year.split(' ')
                    if year[0].isdecimal():
                        year = year[0]
                    else:
                        year = year[1]

                #If the year is after 2010, get the season's id, else stop getting the seasons
                try:
                    year = int(year)
                except:
                    print(f'Skipped {numLeaguesInfoScraped}: {league.get("name")}, {league.get("category", {}).get("name")}, {league.get("id")}')
                    print(f'Failed to parse \'{year}\'')
                    continue
                if year >= 2010:
                    #If the Season ID already exists, make the year one year older
                    if f'{year} Season ID' in leagueInfo:
                        year -= 1
                    leagueInfo[f'{year} Season ID']= season.get('id')

            #Loop through all keys in the dictionary
            for key in leagueInfo:
                #If the key isn't a column
                if key not in leagueDatabase.columns:
                    #Create a new column with the key
                    leagueDatabase[key] = None

            #Add dictionary to dataframe as a row
            leagueDatabase.loc[len(leagueDatabase)] = leagueInfo

        else:
            print(f'Skipped {numLeaguesInfoScraped}: {league.get("name")}, {league.get("category", {}).get("name")}, {league.get("id")}')

        #Increment league counter
        numLeaguesInfoScraped[0] += 1

# %%
def RunScraper(booleans, leagueNames, years, dataTypes):
    #MAIN SCRIPT
    startTime = time.perf_counter()

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

    #Initialize league and match counter
    numLeaguesInfoScraped = [0]
    totalMatchesScraped = [0]

    #Define database path
    databasePath = '/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/League_Database.csv'

    if booleans['scrapeLeaguesInfo'] or not os.path.exists(databasePath):

        #Define dates to scrape (Example: 2026-03-20)
        today = date.today()
        dates = ['2026-04-29', today.strftime('%Y-%m-%d')]

        #If the league database should be cleared or it doesn't exist, create an empty file there
        if booleans['clearLeaguesDatabase'] or not os.path.exists(databasePath):
            #dataframe = pd.DataFrame(columns=['League Name', 'League ID', 'Year', 'Season ID', 'Country Name', 'Country ID', 'Has Player Stats', 'Has Performance Graph'])
            dataframe = pd.DataFrame(columns=['League Name', 'League ID', 'Country Name', 'Country ID', '2026 Season ID', '2025 Season ID'])
            dataframe.to_csv(databasePath, index=False)

        #Extract database
        leagueDatabase = pd.read_csv(databasePath)

        #Loop through dates
        for currentDate in dates:
            count = 1

            #Construct api url
            matchesApiURL = f'https://www.sofascore.com/api/v1/sport/football/scheduled-tournaments/{currentDate}/page/{count}'

            #Get data from api request
            response = requests.get(matchesApiURL, headers=headers, impersonate='chrome120')

            #Convert raw data (which is in bytes) into JSON file
            data = response.json()

            #Loop through pages
            if isinstance(data, dict):
                while data.get('hasNextPage'):
                    #Scrape league info
                    ScrapeLeaguesInfo(leagueDatabase, data, headers, numLeaguesInfoScraped)
                    count += 1
                    #Construct api url
                    matchesApiURL = f'https://www.sofascore.com/api/v1/sport/football/scheduled-tournaments/{currentDate}/page/{count}'
                    #Get data from api request
                    response = requests.get(matchesApiURL, headers=headers, impersonate='chrome120')
                    #Convert raw data (which is in bytes) into JSON file
                    data = response.json()

            #Scrape last page
            ScrapeLeaguesInfo(leagueDatabase, data, headers, numLeaguesInfoScraped)

        # Convert all Season ID columns to Int64
        for col in leagueDatabase.columns:
            if col.endswith("Season ID"):
                leagueDatabase[col] = leagueDatabase[col].astype("Int64")

        #Store league database to path
        leagueDatabase.to_csv(databasePath, index=False)

    #Get league info database
    leagueDatabase = pd.read_csv(databasePath)

    if booleans['scrapeMatches']:
        #if deleteAllExtraFiles:
            #function which deletes all files that aren't one of the file names

        #Set up and create sofascore folder
        dataFolderDirectory = '/Users/jakeholfinger/Desktop/CC Analyst/Data'
        sofaScoreFolderName = f'SofaScore_Data'
        sofaScoreFolderPath = CreateFolderPath(dataFolderDirectory, sofaScoreFolderName)

        #Loop through each year
        for year in years:
            print('-----------------------')
            print(f'SCRAPING YEAR: {year}')
            print('-----------------------')
            #Set up and create year folder
            yearFolderName = f'{year}_Data'
            yearFolderPath = CreateFolderPath(sofaScoreFolderPath, yearFolderName)

            #Loop through each league
            for leagueName in leagueNames:
                if leagueName in leagueDatabase['League Name'].values:
                    print('------------------------------------')
                    print(f'SCRAPING LEAGUE: {leagueName}')
                    print('------------------------------------')
                    #Find league's row in league info database and get the league and season id
                    row = leagueDatabase[leagueDatabase['League Name'] == leagueName]
                    leagueID = int(row['League ID'].values[0])
                    seasonID = row[f'{year} Season ID'].values[0]
                    if pd.isna(seasonID):
                        print(f'No {year} season found for {leagueName}, skipping.')
                        continue
                    seasonID = int(seasonID)

                    ScrapeSeasonData(leagueName, leagueID, seasonID, yearFolderPath, dataTypes, totalMatchesScraped, booleans, headers)
                    print(f'{totalMatchesScraped}: {leagueName}, {leagueID}')
                else:
                    print(f'{leagueName} is not in league database.')

    endTime = time.perf_counter()
    executionTime = endTime - startTime

    print(f'Execution Time: {executionTime} seconds')
    print(f'Number of League\'s Info Scraped: {numLeaguesInfoScraped[0]}')
    print(f'Number of Matches Scraped: {totalMatchesScraped[0]}')
    print(f'Number of Data Types Scraped: {len(dataTypes) if totalMatchesScraped[0] != 0 else 0}')
    print(f'Execution Time Per League (Info): {executionTime/numLeaguesInfoScraped[0] if numLeaguesInfoScraped[0] != 0 else 0} seconds')
    print(f'Execution Time Per Year: {executionTime/len(years) if len(years) != 0 else 0} seconds')
    print(f'Execution Time Per League: {executionTime/len(leagueNames) if len(leagueNames) != 0 else 0} seconds')
    print(f'Execution Time Per Match: {executionTime/totalMatchesScraped[0] if totalMatchesScraped[0] != 0 else 0} seconds')
    print(f'Execution Time Per Data File: {executionTime/(totalMatchesScraped[0]*len(dataTypes)) if totalMatchesScraped[0] != 0 and len(dataTypes) != 0 else 0} seconds')

# %%
#DEFINE BOOLEANS

booleans = {}

#Determines whether existing files should be replaced
booleans['overwriteMatchFiles'] = False

#Determines whether only matches without all files should be scraped
booleans['onlyScrapeIncompleteMatches'] = True

#Determines whether only new matches should be scraped (only scrapes games that don't have a folder)
booleans['onlyScrapeNewMatches'] = False

#Determines whether extra files should be deleted
booleans['deleteAllExtraFiles'] = False

#Determines whether league info should be updated
booleans['scrapeLeaguesInfo'] = False

#Determines whether the leagues database should be cleared
booleans['clearLeaguesDatabase'] = False

#Determines whether matches should be scraped
booleans['scrapeMatches'] = True

#DEFINE VARIABLES

#AllFileNames = ['Average_Positions.csv', 'First_Half_Team_Statistics.csv', 'Formations.csv', 'Full_Team_Statistics.csv', 'Match_Attributes.csv', 'Goalkeeper_Spatial_Points.csv', 'Match_Momentum.csv', 'Missing_Players.csv', 'Player_Event_Data.csv', 'Player_Heatmaps.csv', 'Player_Spatial_Points.csv', 'Player_Statistics.csv', 'Second_Half_Team_Statistics.csv', 'Shot_Map.csv', 'Subs_Average_Positions.csv']
#leagueNames = ['MLS']
leagueNames = ['MLS', 'US Open Cup']
#leagueNames = ['USL Championship', 'USL League One']

#AllYears = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017, '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026']
#years = ['2026', '2025']
years = ['2026']

#List of data types to scrape
#AllDataTypes = ['/shotmap', '/heatmap', '', '/graph', '/average-positions', '/statistics', '/lineups', '/player/rating-breakdown', '/player/heatmap']
dataTypes = ['/shotmap', '/heatmap', '', '/graph', '/average-positions', '/statistics', '/lineups', '/player/rating-breakdown', '/player/heatmap']

#RUN SCRAPER
subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
RunScraper(booleans, leagueNames, years, dataTypes)

#TO-DO
# - Add Enums to variable definitions to prevent wrong inputted values
# - Add scraping capabilties for runnning/walking statistics for leagues that have it
# - Add league description file to store what type of data the league has, whether the season is complete, and any files that aren't filled.
