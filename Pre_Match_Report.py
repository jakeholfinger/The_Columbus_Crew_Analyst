#%%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import os
from curl_cffi import requests
import contextlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import time
import re
import urllib.request
import urllib.parse
import json
import io
from collections import Counter
# %%
def getFiles(currentPath):
    '''Returns a list of all visible files inside the folder at currentPath'''
    return [f for f in os.listdir(currentPath) if not f.startswith('.')]
# %%
def scrapeURLData(apiURL):
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
    }

    response = None
    for attempt in range(3):
        try:
            response = requests.get(apiURL, headers=headers, impersonate="chrome120", timeout=30)
            break
        except requests.exceptions.Timeout:
            print(f"Timeout, retrying... attempt {attempt+1}")
            time.sleep(5)

    if response is None:
        print(f"Failed to fetch {apiURL} after 3 attempts, skipping.")
        return

    dataJSON = response.json()
    return dataJSON
#%%
def loadMatchData(leagues, team, date):
    year = date.split('-')[2]
    matches = {}

    for league in leagues:
        teamPath = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data/{league.replace(" ", "_")}_Data/{team.replace(" ", "_")}_Data'
        if not os.path.exists(teamPath):
            continue
        for matchFolder in getFiles(teamPath):
            matchPath = os.path.join(teamPath, matchFolder)
            if not os.path.isdir(matchPath):
                continue
            scrapedFiles = {}
            # Fixed: was getFiles(matchFolder) which passed the folder NAME not full path
            for file in getFiles(matchPath):
                filePath = os.path.join(matchPath, file)
                try:
                    scrapedFiles[file] = pd.read_csv(filePath)
                except Exception:
                    pass
            matches[matchFolder] = scrapedFiles

    # Folders are named "M-D-YYYY_vs_Opponent" (e.g. "2-21-2026_vs_Portland_Timbers").
    # Parse the date prefix with strptime so we sort chronologically within a year.
    def folderDate(folderName):
        return datetime.strptime(folderName.split('_')[0], '%m-%d-%Y')

    sortedMatches = dict(sorted(matches.items(), key=lambda item: folderDate(item[0])))

    # Keep only matches that kicked off strictly before the target date.
    month, day, yr = date.split('-')
    targetDate = datetime(int(yr), int(month), int(day)).date()
    filteredMatches = {
        folder: data for folder, data in sortedMatches.items()
        if folderDate(folder).date() < targetDate
    }

    return filteredMatches
#%%
def scrapeTargetMatchData(opposition, date, matches):
    # Added `date` parameter — was previously read from outer scope (a scoping bug)
    oppositionName = opposition.replace('_', ' ')

    # Search historical match data to find the opposition's SofaScore team ID.
    # We loop through all loaded matches because the first match may not involve
    # the opposition.
    oppositionID = None
    for matchFolder, matchFiles in matches.items():
        matchAttributes = matchFiles.get('Match_Attributes.csv')
        if matchAttributes is None:
            continue
        homeTeamName = matchAttributes['homeTeam.name'].iloc[0]
        awayTeamName = matchAttributes['awayTeam.name'].iloc[0]
        if homeTeamName == oppositionName:
            oppositionID = int(matchAttributes['homeTeam.id'].iloc[0])
            break
        elif awayTeamName == oppositionName:
            oppositionID = int(matchAttributes['awayTeam.id'].iloc[0])
            break

    if oppositionID is None:
        print(f'Could not find team ID for {oppositionName} in loaded match data.')
        return None

    # Build target date as a UTC datetime and compare to now to decide which
    # SofaScore endpoint (next vs last) is more likely to contain the match.
    month, day, yr = date.split('-')
    targetDate = datetime(int(yr), int(month), int(day), tzinfo=timezone.utc)
    currentTimestamp = int(datetime.now(timezone.utc).timestamp())

    # If match date is in the future (or within the past day) try upcoming first
    if int(targetDate.timestamp()) > currentTimestamp - 86400:
        matchTimeframes = ['next', 'last']
    else:
        matchTimeframes = ['last', 'next']

    foundMatch = False
    matchData = None
    for matchTimeframe in matchTimeframes:
        count = 0
        hasNextPage = True
        while hasNextPage:
            apiURL = f'https://www.sofascore.com/api/v1/team/{oppositionID}/events/{matchTimeframe}/{count}'
            data = scrapeURLData(apiURL)
            if data is None:
                break
            # API returns {"events": [...], "hasNextPage": bool}
            for match in data.get('events', []):
                matchDate = datetime.fromtimestamp(match.get('startTimestamp', 0), tz=timezone.utc).date()
                if matchDate == targetDate.date():
                    matchID = match.get('id', None)
                    if matchID:
                        matchAPIURL = f'https://www.sofascore.com/api/v1/event/{matchID}'
                        matchData = scrapeURLData(matchAPIURL).get('event', {})
                    foundMatch = True
                    break
            if foundMatch:
                break
            hasNextPage = data.get('hasNextPage', False)
            count += 1
        if foundMatch:
            break

    if not foundMatch:
        print(f'No match found for {oppositionName} on {date}')
        return None

    return matchData
#%%
_venue_coords_cache = {}
#%%
def scrapeCoordinatesAndTimezone(venue):
    '''Returns (lat, lng, timezone_str, city, state, country) for the venue.
    Uses OpenStreetMap Nominatim for geocoding and timeapi.io for timezone lookup.'''
    if venue in _venue_coords_cache:
        return _venue_coords_cache[venue]

    fallback = (None, None, None, '', '', '')

    try:
        geoResp = requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={'q': venue, 'format': 'json', 'addressdetails': '1', 'limit': '1'},
            headers={'User-Agent': 'CCAnalystBot/1.0 (thecolumbuscrewanalyst@gmail.com)'},
        )
        results = geoResp.json()
    except Exception as e:
        print(f'Nominatim geocoding failed for "{venue}": {e}')
        _venue_coords_cache[venue] = fallback
        return fallback

    if not results:
        print(f'No geocoding results for "{venue}".')
        _venue_coords_cache[venue] = fallback
        return fallback

    r = results[0]
    lat = float(r.get('lat', 0))
    lng = float(r.get('lon', 0))
    addr = r.get('address', {})
    city = addr.get('city') or addr.get('town') or addr.get('village') or ''
    state = addr.get('state', '')
    countryCode = addr.get('country_code', '').upper()
    country = 'USA' if countryCode == 'US' else addr.get('country', countryCode)

    try:
        tzResp = requests.get(
            f'https://timeapi.io/api/timezone/coordinate?latitude={lat}&longitude={lng}'
        )
        venueTimezone = tzResp.json().get('timeZone', 'UTC')
    except Exception as e:
        print(f'Timezone lookup failed for "{venue}": {e}')
        venueTimezone = 'UTC'

    result = (lat, lng, venueTimezone, city, state, country)
    _venue_coords_cache[venue] = result
    return result
#%%
WEATHER_CODES = {
    0: 'Clear',
    1: 'Mainly Clear',
    2: 'Partly Cloudy',
    3: 'Overcast',
    45: 'Fog',
    48: 'Fog',
    51: 'Light Rain',
    53: 'Rain',
    55: 'Heavy Rain',
    56: 'Freezing Rain',
    57: 'Freezing Rain',
    61: 'Light Rain',
    63: 'Rain',
    65: 'Heavy Rain',
    66: 'Freezing Rain',
    67: 'Freezing Rain',
    71: 'Light Snow',
    73: 'Snow',
    75: 'Heavy Snow',
    77: 'Snow',
    80: 'Light Rain Showers',
    81: 'Rain Showers',
    82: 'Heavy Rain Showers',
    85: 'Light Snow Showers',
    86: 'Heavy Snow Showers',
    95: 'Thunderstorm',
    96: 'Thunderstorm with Light Hail',
    99: 'Thunderstorm with Heavy Hail'
}

CARDINAL_DIRECTIONS = {
    0: 'N',
    45: 'NE',
    90: 'E',
    135: 'SE',
    180: 'S',
    225: 'SW',
    270: 'W',
    315: 'NW',
    360: 'N'
}

#%%
def scrapeWeather(venueLatitude, venueLongitude, startTimestamp):
    '''Returns a single-row DataFrame with weather at kickoff, or None if unavailable.

    Returns None when:
    - The match is more than 16 days in the future (open-meteo forecast limit)
    - The archive API hasn't yet processed a very recent past match (~5-day delay)
    - Any network or parsing error occurs
    '''
    kickoffTime = pd.to_datetime(startTimestamp, unit='s', utc=True)
    currentTimestamp = int(datetime.now(timezone.utc).timestamp())
    daysFromNow = (startTimestamp - currentTimestamp) / 86400

    if daysFromNow > 16:
        print(f'Match is {daysFromNow:.0f} days away — beyond open-meteo 16-day forecast limit.')
        return None

    if startTimestamp > currentTimestamp:
        urlStart = 'https://api.open-meteo.com/v1/forecast'
        extraParams = '&forecast_days=16'
    else:
        urlStart = 'https://archive-api.open-meteo.com/v1/archive'
        extraParams = ''

    # temperature_unit=fahrenheit and wind_speed_unit=mph give US-friendly values
    url = (
        f'{urlStart}'
        f'?latitude={venueLatitude}'
        f'&longitude={venueLongitude}'
        '&hourly='
        'apparent_temperature,'
        'weather_code,'
        'wind_speed_10m,'
        'wind_direction_10m,'
        'precipitation_probability'
        f'&temperature_unit=fahrenheit'
        f'&wind_speed_unit=mph'
        f'&start_date={kickoffTime.date()}'
        f'&end_date={kickoffTime.date()}'
        f'&timezone=UTC'
        f'{extraParams}'
    )

    try:
        data = requests.get(url).json()
        if 'hourly' not in data:
            print(f'Weather API returned no hourly data: {data.get("reason", "unknown error")}')
            return None

        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'], utc=True)

        # Select the row matching kickoff hour
        kickoffHour = kickoffTime.floor('h')
        weatherRow = df[df['time'].dt.floor('h') == kickoffHour].copy()

        if weatherRow.empty:
            print(f'No hourly weather row found for kickoff hour {kickoffHour}.')
            return None

        # Map numeric weather code to a human-readable description
        code = int(weatherRow['weather_code'].iloc[0])
        if code not in WEATHER_CODES:
            print(f'Weather code {code} not found in WEATHER_CODES dictionary.')
        weatherRow['weather_code'] = WEATHER_CODES.get(code, 'Unknown')

        # Round raw wind direction (0-360°) to nearest 45° before cardinal lookup
        rawAngle = weatherRow['wind_direction_10m'].iloc[0]
        roundedAngle = round(rawAngle / 45) * 45 % 360
        weatherRow['wind_direction_10m'] = CARDINAL_DIRECTIONS.get(roundedAngle, '')

        return weatherRow

    except Exception as e:
        print(f'Weather scrape failed: {e}')
        return None
#%%
_venue_surface_cache = {}
#%%
def scrapeVenueSurface(venueName):
    """Returns the playing surface for a venue by searching Wikipedia's stadium infobox."""
    if venueName in _venue_surface_cache:
        return _venue_surface_cache[venueName]

    try:
        wikiHeaders = {'User-Agent': 'CCAnalystBot/1.0 (thecolumbuscrewanalyst@gmail.com) Python-urllib/3'}

        searchParams = {
            'action': 'query',
            'list': 'search',
            'srsearch': venueName,
            'srnamespace': '0',
            'srlimit': '1',
            'format': 'json'
        }
        searchURL = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(searchParams)
        with urllib.request.urlopen(urllib.request.Request(searchURL, headers=wikiHeaders)) as r:
            results = json.loads(r.read()).get('query', {}).get('search', [])
        if not results:
            _venue_surface_cache[venueName] = 'Unknown'
            return 'Unknown'

        pageTitle = results[0]['title']

        parseParams = {
            'action': 'parse',
            'page': pageTitle,
            'prop': 'wikitext',
            'format': 'json'
        }
        parseURL = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(parseParams)
        with urllib.request.urlopen(urllib.request.Request(parseURL, headers=wikiHeaders)) as r:
            wikitext = json.loads(r.read()).get('parse', {}).get('wikitext', {}).get('*', '')

        match = re.search(r'\|\s*surface\s*=\s*([^\n{}]+)', wikitext, re.IGNORECASE)
        if not match:
            _venue_surface_cache[venueName] = 'Unknown'
            return 'Unknown'

        surface = match.group(1).strip()
        surface = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', surface)
        surface = re.sub(r"'{2,3}", '', surface)
        surface = surface.lstrip('* ').strip()

        s = surface.lower()
        grass_terms = ('grass', 'natural', 'lawn', 'bermuda', 'bluegrass', 'ryegrass',
                       'fescue', 'cynodon', 'festuca', 'lolium', 'zoysia', 'poa ')
        if 'hybrid' in s or 'grassmaster' in s or 'sisgrass' in s:
            category = 'Hybrid'
        elif 'artificial' in s or 'synthetic' in s or 'turf' in s:
            category = 'Turf'
        elif any(t in s for t in grass_terms):
            category = 'Grass'
        else:
            category = 'Unknown'
            print('Unknown surface:', surface)

        _venue_surface_cache[venueName] = category
        return category

    except Exception as e:
        print(f"Error scraping surface for '{venueName}': {e}")
        _venue_surface_cache[venueName] = 'Unknown'
        return 'Unknown'
#%%
_team_logo_cache = {}
#%%
def scrapeLogo(teamName):
    """Returns the team's logo as a matplotlib image array by searching Wikipedia's club infobox."""
    if teamName in _team_logo_cache:
        return _team_logo_cache[teamName]

    try:
        wikiHeaders = {'User-Agent': 'CCAnalystBot/1.0 (thecolumbuscrewanalyst@gmail.com) Python-urllib/3'}

        searchParams = {
            'action': 'query',
            'list': 'search',
            'srsearch': teamName + ' soccer club',
            'srnamespace': '0',
            'srlimit': '1',
            'format': 'json'
        }
        searchURL = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(searchParams)
        with urllib.request.urlopen(urllib.request.Request(searchURL, headers=wikiHeaders)) as r:
            results = json.loads(r.read()).get('query', {}).get('search', [])
        if not results:
            _team_logo_cache[teamName] = None
            return None

        pageTitle = results[0]['title']

        parseParams = {
            'action': 'parse',
            'page': pageTitle,
            'prop': 'wikitext',
            'format': 'json'
        }
        parseURL = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(parseParams)
        with urllib.request.urlopen(urllib.request.Request(parseURL, headers=wikiHeaders)) as r:
            wikitext = json.loads(r.read()).get('parse', {}).get('wikitext', {}).get('*', '')

        imageMatch = re.search(
            r'\|\s*(?:image|logo|logo_file|image_file)\s*=\s*([^\n{}]+)',
            wikitext, re.IGNORECASE
        )
        if not imageMatch:
            _team_logo_cache[teamName] = None
            return None

        rawField = imageMatch.group(1).strip()
        fileMatch = re.search(r'(?:File:|Image:)?([^\[\]|]+\.(?:svg|png|jpg|jpeg))', rawField, re.IGNORECASE)
        if not fileMatch:
            _team_logo_cache[teamName] = None
            return None
        fileName = fileMatch.group(1).strip()

        infoParams = {
            'action': 'query',
            'titles': f'File:{fileName}',
            'prop': 'imageinfo',
            'iiprop': 'url',
            'iiurlwidth': '200',
            'format': 'json'
        }
        infoURL = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(infoParams)
        with urllib.request.urlopen(urllib.request.Request(infoURL, headers=wikiHeaders)) as r:
            pages = json.loads(r.read()).get('query', {}).get('pages', {})

        imageURL = None
        for page in pages.values():
            infos = page.get('imageinfo', [])
            if infos:
                imageURL = infos[0].get('thumburl') or infos[0].get('url')
                break

        if not imageURL:
            _team_logo_cache[teamName] = None
            return None

        with urllib.request.urlopen(urllib.request.Request(imageURL, headers=wikiHeaders)) as r:
            imageBytes = r.read()

        img = mpimg.imread(io.BytesIO(imageBytes), format='png')
        _team_logo_cache[teamName] = img
        return img

    except Exception as e:
        print(f"Error scraping logo for '{teamName}': {e}")
        _team_logo_cache[teamName] = None
        return None
#%%
def drawFigureLine(page, color, y, x0=0.018, x1=0.982):
    """Draw a horizontal rule at figure coordinate y, spanning inside the primary border."""
    line = mlines.Line2D([x0, x1], [y, y], transform=page.transFigure, color=color, linewidth=3.0)
    page.add_artist(line)
#%%
def generatePageTemplate(team, opposition, targetMatch, frontPage=False):

    teamName = team.replace('_', ' ')
    teamLogo = scrapeLogo(teamName)
    oppositionName = opposition.replace('_', ' ')
    competitionName = targetMatch.get('tournament', {}).get('uniqueTournament', {}).get('name', 'Unknown Competition')
    competitionRound = (targetMatch.get('roundInfo', {}).get('name') or targetMatch.get('tournament', {}).get('groupName') or ('Group Stage' if targetMatch.get('tournament', {}).get('uniqueTournament', {}).get('hasRounds') else 'Regular Season'))
    kickoffTimestamp = targetMatch.get('startTimestamp')
    venueName = targetMatch.get('venue', {}).get('name')
    venueTimezone = 'UTC'
    if venueName:
        _, _, tz, _, _, _ = scrapeCoordinatesAndTimezone(venueName)
        if tz:
            venueTimezone = tz

    # Format kickoff time in the venue's local timezone and append the timezone abbreviation
    if kickoffTimestamp is None:
        kickoffTime = 'TBD'
    else:
        kickoffDt = datetime.fromtimestamp(kickoffTimestamp, tz=ZoneInfo('UTC')).astimezone(ZoneInfo(venueTimezone))
        tzAbbrev = kickoffDt.strftime('%Z')     # e.g. "EST", "EDT", "CST"
        kickoffTime = kickoffDt.strftime('%-m/%-d/%Y @ %-I:%M %p') + f' {tzAbbrev}'

    # Team colors live under homeTeam.teamColors / awayTeam.teamColors in the API response
    isHome = targetMatch.get('homeTeam', {}).get('name') == teamName
    # NOTE: Don't use sofascore's colors as they're 
    #teamKey = 'homeTeam' if isHome else 'awayTeam'
    #primaryColor   = '#' + (targetMatch.get(teamKey, {}).get('teamColors', {}).get('primary', '000000') or '000000').lstrip('#')
    #secondaryColor = '#' + (targetMatch.get(teamKey, {}).get('teamColors', {}).get('secondary', 'ffffff') or 'ffffff').lstrip('#')

    teamID = targetMatch.get('homeTeam', {}).get('id') if isHome else targetMatch.get('awayTeam', {}).get('id')
    teamColorDatabasePath = '/Users/jakeholfinger/Desktop/CC Analyst/Data/Team Colors.json'

    if os.path.exists(teamColorDatabasePath):
        # Load database
        with open(teamColorDatabasePath, 'r') as f:
            teamColorDatabase = json.load(f)
    else:
        # Create database
        teamColorDatabase = {}

    try:
        teamColors = teamColorDatabase[str(teamID)]
        primaryColor = teamColors[0]
        secondaryColor = teamColors[1]
    except:
        # Get team color information from user as the database does not contain the team's colors
        primaryColor = input(f'Enter {teamName}\'s primary color in hexcode (Ex: #f24a96). ')
        secondaryColor = input(f'Enter {teamName}\'s secondary color in hexcode (Ex: #f24a96). ')
        teamColors = [primaryColor, secondaryColor]

        # Add user-provided team colors to database
        teamColorDatabase[teamID] = teamColors
        with open(teamColorDatabasePath, 'w') as f:
            json.dump(teamColorDatabase, f)
        
    # All coordinates below are standard matplotlib figure coordinates:
    #   x: 0 = left edge, 1 = right edge
    #   y: 0 = bottom edge, 1 = top edge
    # The original code had x and y arguments swapped; those are corrected here.

    page = plt.figure(figsize=(8.5, 11), dpi=300)

    # Double border using filled rectangles layered on top of each other.
    # Secondary color fills the entire figure background (flush with page edge by definition).
    # Primary color fills inside that, leaving a 6pt band of secondary visible.
    # White fills inside that, leaving a 5pt band of primary visible.
    # All measurements in pts; 1 pt = 1/72 inch.
    SEC_W = 6   # secondary border thickness in pts
    PRI_W = 5   # primary border thickness in pts
    sx = SEC_W / (72 * 8.5)              # ≈ 0.00980 fig units
    sy = SEC_W / (72 * 11)              # ≈ 0.00758 fig units
    px = (SEC_W + PRI_W) / (72 * 8.5)  # ≈ 0.01797 fig units
    py = (SEC_W + PRI_W) / (72 * 11)   # ≈ 0.01389 fig units

    page.patch.set_facecolor(secondaryColor)   # layer 1: secondary fills entire page
    page.add_artist(Rectangle((sx, sy), 1 - 2*sx, 1 - 2*sy,
                               edgecolor='none', facecolor=primaryColor,
                               transform=page.transFigure, zorder=-2))
    page.add_artist(Rectangle((px, py), 1 - 2*px, 1 - 2*py,
                               edgecolor='none', facecolor='white',
                               transform=page.transFigure, zorder=-1))

    # --- HEADER ---
    # Large centered title
    page.text(0.5, 0.950, 'PRE-MATCH REPORT', ha='center', fontsize=30, fontweight='bold')
    # Competition name and round (e.g. "MLS Regular Season")
    page.text(0.5, 0.9125, f'{competitionName.upper()} {competitionRound.upper()}', ha='center', fontsize=19)#, fontweight='bold')
    # "VS" for home games, "AT" for away games, followed by all-caps opponent name
    homeIndicator = 'VS' if isHome else 'AT'
    page.text(0.5, 0.8725, f'{homeIndicator} {oppositionName.upper()}', ha='center', fontsize=25, fontweight='bold')
    # Kickoff date/time with local timezone abbreviation
    page.text(0.5, 0.835, kickoffTime, ha='center', fontsize=19)#, fontweight='bold')

    # Team logos: placed using add_axes([left, bottom, width, height]) so we can
    # call imshow() on them. axis('off') removes the axes border/ticks.
    if teamLogo is not None:
        logoWidth = 0.09
        logoHeight = 0.13

        #axLeft = page.add_axes([0.07, 0.840, 0.09, 0.13])
        leftTargetX = 0.11
        leftTargetY = 0.875
        axLeft = page.add_axes([(leftTargetX-(logoWidth/2)), (leftTargetY-(logoHeight/2)), logoWidth, logoHeight])
        axLeft.imshow(teamLogo)
        axLeft.axis('off')

        #axRight = page.add_axes([0.82, 0.840, 0.09, 0.13])
        rightTargetX = 0.89
        rightTargetY = 0.875
        axRight = page.add_axes([(rightTargetX-(logoWidth/2)), (rightTargetY-(logoHeight/2)), logoWidth, logoHeight])
        axRight.imshow(teamLogo)
        axRight.axis('off')

    # Separator line below header (not drawn on the front page — see frontPage flag)
    if not frontPage:
        drawFigureLine(page, secondaryColor, y=0.820)

    return page, teamColors
#%%
def generateMatchInfo(leagues, opposition, date, matches, targetMatch, page, teamColors):

    kickoffTimestamp = targetMatch.get('startTimestamp')
    venueName = targetMatch.get('venue', {}).get('name')
    venueLatitude = targetMatch.get('venue', {}).get('venueCoordinates', {}).get('latitude')
    venueLongitude = targetMatch.get('venue', {}).get('venueCoordinates', {}).get('longitude')
    venueCity, venueState, venueCountry = '', '', ''

    if venueName:
        geoLat, geoLng, _, venueCity, venueState, venueCountry = scrapeCoordinatesAndTimezone(venueName)
        if venueLatitude is None:
            venueLatitude = geoLat
        if venueLongitude is None:
            venueLongitude = geoLng

    venueLocation = f'{venueCity}, {venueState}' if venueCountry == 'USA' else f'{venueCity}, {venueCountry}'

    weatherRow = scrapeWeather(venueLatitude, venueLongitude, kickoffTimestamp) if (venueLatitude and venueLongitude) else None

    if weatherRow is None:
        weather = 'WEATHER UNAVAILABLE'
    else:
        # scrapeWeather already mapped weather_code and wind_direction_10m to
        # human-readable strings; values come from a DataFrame so use .iloc[0].
        weatherCondition = str(weatherRow['weather_code'].iloc[0]).upper()
        feelsLike = int(round(weatherRow['apparent_temperature'].iloc[0]))
        rawPrecip = weatherRow['precipitation_probability'].iloc[0]
        precipStr = f'{int(rawPrecip)}% PRECIP' if rawPrecip is not None else 'PRECIP N/A'
        windSpeed = int(round(weatherRow['wind_speed_10m'].iloc[0]))
        windDir = str(weatherRow['wind_direction_10m'].iloc[0])
        weather = f'{weatherCondition}, FEELS LIKE {feelsLike}°F, {precipStr}, {windSpeed} MPH WIND {windDir}'

    venueSurface = scrapeVenueSurface(venueName) if venueName else 'Unknown'
    venueDisplay = venueName.upper() if venueName else 'VENUE TBD'
    venueLocationDisplay = venueLocation.upper() if venueLocation.strip(', ') else 'LOCATION TBD'

    # Line 1: venue name and playing surface (e.g. "SCOTTS MIRACLE-GRO FIELD - GRASS")
    page.text(0.5, 0.7925, f'{venueDisplay} - {venueSurface.upper()}', ha='center', fontsize=19)#, fontweight='bold')
    # Line 2: city/state and weather summary (e.g. "COLUMBUS, OHIO - SUNNY, FEELS LIKE 78°F, ...")
    page.text(0.5, 0.760, f'{venueLocationDisplay} - {weather}', ha='center', fontsize=19)#, fontweight='bold')

    drawFigureLine(page, teamColors[1], y=0.742)

    return page
#%%
def getStyleOfPlay():

    #TODO: Implement Method
    
    return
#%%
def generateMatchOverview(leagues, opposition, date, matches, page, teamColors):
    #TODO: Finish Implementing Method

    oppositionName = opposition.replace('_', ' ')

    # Folders are named "M-D-YYYY_vs_Opponent" — sort chronologically by the date prefix
    matches = dict(sorted(matches.items(),
        key=lambda item: datetime.strptime(item[0].split('_')[0], '%m-%d-%Y')))

    # Get the last 5 results: W/D/L from the opposition's perspective
    form = []
    for matchFolder, matchFiles in list(matches.items())[-5:]:
        attrs = matchFiles.get('Match_Attributes.csv')
        if attrs is None:
            continue
        winnerCode = attrs['winnerCode'].iloc[0]
        homeTeamName = attrs['homeTeam.name'].iloc[0]
        # winnerCode 1 = home win, 2 = away win, 0 = draw
        if winnerCode == 1:
            form.append('W' if homeTeamName == oppositionName else 'L')
        elif winnerCode == 2:
            form.append('L' if homeTeamName == oppositionName else 'W')
        else:
            form.append('D')

    styleOfPlay = getStyleOfPlay()

    # Count which formations the opposition has used; take the top two
    oppositionFormations = []
    for matchFolder, matchFiles in matches.items():
        formations = matchFiles.get('Formations.csv')
        if formations is None:
            continue
        # Formations.csv has columns 'Team Name' and 'Formation', one row per team
        rows = formations.loc[formations['Team Name'] == oppositionName, 'Formation']
        if not rows.empty:
            # Formations are stored as Python list reprs e.g. "['4-2-3-1']" — strip to bare string
            formStr = str(rows.iloc[0]).strip("[]'\"")
            oppositionFormations.append(formStr)

    counts = Counter(oppositionFormations)
    topFormations = counts.most_common(2)
    primaryFormation = topFormations[0][0] if len(topFormations) >= 1 else 'N/A'
    secondaryFormation = topFormations[1][0] if len(topFormations) >= 2 else None

    import League_Power_Rankings

    year = date.split('-')[2]
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        result = League_Power_Rankings.main(year, leagues[0], len(matches), table=True)

    if result is None:
        powerRanking = 'N/A'
        standing = 'N/A'
    else:
        powerRankings, standingsDF = result
        rankRow = powerRankings[powerRankings['Team Name'] == oppositionName]
        powerRanking = rankRow.index[0] if not rankRow.empty else 'N/A'
        # Guard against a missing column from League_Power_Rankings
        try:
            standRow = standingsDF[standingsDF['Team Name'] == oppositionName]
            standing = standRow.index[0] if not standRow.empty else 'N/A'
        except KeyError:
            standing = 'N/A'

    # --- MATCH OVERVIEW SECTION ---
    # Section title — centered
    page.text(0.5, 0.700, 'TEAM OVERVIEW', ha='center', fontsize=27.5, fontweight='bold')

    # Left column (x=0.048): league table, power ranking, form, expected points
    page.text(0.048, 0.660, f'Table Position: {standing}', fontsize=19)
    page.text(0.048, 0.620, f'Power Ranking: {powerRanking}', fontsize=19)
    page.text(0.048, 0.580, f'Form: {" ".join(form)}', fontsize=19)
    #page.text(0.048, 0.579, 'Expected Points: N/A', fontsize=22.5)

    # Right column (x=0.500): tactical info
    page.text(0.480, 0.660, f'Primary Formation: {primaryFormation}', fontsize=19)
    page.text(0.480, 0.620, f'Secondary Formation: {secondaryFormation or "N/A"}', fontsize=19)
    page.text(0.480, 0.580, f'Style Of Play: {styleOfPlay or "N/A"}', fontsize=19)
    #page.text(0.500, 0.579, 'Win Probability: N/A', fontsize=22.5)

    drawFigureLine(page, teamColors[1], y=0.560)

    return page

# (x, y) where x = field WIDTH (0=left touchline, 100=right touchline)
#             y = field DEPTH (0=GK goal line, 100=halfway line)
# This maps directly onto fieldAxis data coordinates: x is horizontal, y is vertical.
# Right-sided positions (DR, MR, RW) are at high x; left-sided (DL, ML, LW) at low x.
POSITION_COORDINATES = {
    'GK':  (50,  6),
    'DR':  (88, 32), 'DCR': (67, 23), 'DC':  (50, 23), 'DCL': (33, 23), 'DL':  (12, 32),
    'DMR': (67, 43), 'DM':  (50, 43), 'DML': (33, 43),
    'MR':  (88, 55), 'MCR': (67, 55), 'MC':  (50, 55), 'MCL': (33, 55), 'ML':  (12, 55),
    'RW':  (88, 68), 'AMR': (67, 68), 'AM':  (50, 68), 'AML': (33, 68), 'LW':  (12, 68),
    'STR': (67, 82), 'ST':  (50, 82), 'STL': (33, 82)
}
#%%
def generatePredictedLineup(leagues, opposition, date, matches, page):

    import Expected_Lineup_Temp_V2

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            expectedLineup, expectedSubs, missingPlayersDict = Expected_Lineup_Temp_V2.main(leagues, opposition, date)

    oppositionMatches = loadMatchData(leagues, opposition, date)
    ratingAccum = {}
    ratingCount = {}
    for matchFolder, matchFiles in oppositionMatches.items():
        playerStats = matchFiles.get('Player_Statistics.csv')
        if playerStats is None or playerStats.empty:
            continue
        for _, row in playerStats.iterrows():
            name = row.get('Player Name')
            rat = row.get('rating')
            if name and rat is not None and not (isinstance(rat, float) and pd.isna(rat)):
                ratingAccum[name] = ratingAccum.get(name, 0.0) + float(rat)
                ratingCount[name] = ratingCount.get(name, 0) + 1
    avgPlayerRatings = {name: round(ratingAccum[name] / ratingCount[name], 1) for name in ratingAccum}

    if missingPlayersDict:
        missingPlayers = (pd.DataFrame.from_dict(missingPlayersDict, orient='index')
                          .reset_index()
                          [['index', 'Availibility', 'Slot']]
                          .rename(columns={'index': 'Player', 'Slot': 'Position'}))
        missingPlayers['Avg Rating'] = missingPlayers['Player'].map(avgPlayerRatings).fillna(6.0)
    else:
        missingPlayers = pd.DataFrame()

    # --- PERSONNEL SECTION ---
    # Section title — centered, y=0.522 places it just below the Match Overview separator
    page.text(0.5, 0.522, 'PERSONNEL', ha='center', fontsize=27.5, fontweight='bold')

    # Sub-section headers: Expected Lineup (left), Expected Subs (centre-right),
    # Potential Absences (far right). x positions chosen to align with their content areas.
    #page.text(0.09, 0.485, 'Expected Lineup', s='Perfectly Centered', fontsize=22.5, fontweight='bold')
    page.text(0.26, 0.48, 'Expected Lineup', ha='center', fontsize=22.5, fontweight='bold')


    # Field axis: left=4.7%, bottom=1%, width=43%, height=46% of the figure.
    # xlim 0-100 = left touchline → right touchline (68 m wide)
    # ylim -5 to 100 = behind goal → halfway line (52.5 m deep); -5 makes room for the goal frame
    #fieldAxis = page.add_axes([0.047, 0.01, 0.43, 0.46])
    fieldWidth = 0.425
    fieldHeight = 0.435
    targetX = 0.26
    targetY = 0.245
    fieldAxis = page.add_axes([(targetX-(fieldWidth/2)), (targetY-(fieldHeight/2)), fieldWidth, fieldHeight])
    fieldAxis.set_xlim(0, 100)
    fieldAxis.set_ylim(-5, 100)
    fieldAxis.set_xticks([])
    fieldAxis.set_yticks([])
    for spine in fieldAxis.spines.values():
        spine.set_color('#1b5e20')

    fieldBgPath = '/Users/jakeholfinger/Desktop/CC Analyst/Pictures/Field Background.png'
    if os.path.exists(fieldBgPath):
        fieldAxis.imshow(mpimg.imread(fieldBgPath), extent=[0, 100, -5, 100], aspect='auto', zorder=0)
    else:
        fieldAxis.set_facecolor('#256729')

    # Each player card is CW units wide × CH units tall, centred on (xCoord, yCoord).
    # Card layout (y increases upward in the axes):
    #   Top half  [yCoord → yCoord+HH]: [number NUM_W] [rating RAT_W] [position POS_W]
    #   Bot half  [yCoord-HH → yCoord]: black name bar, full width
    # All box widths are fractions of CW so text centers stay correct when CW changes.
    CW = 18     # card width  — change this to resize all cards uniformly
    CH = 7.5    # card height
    HW = CW / 2
    HH = CH / 2
    NUM_W = 0.30 * CW   # number box  (left 25 %)
    RAT_W = 0.40 * CW   # rating box  (centre 50 %)
    POS_W = 0.30 * CW   # position box (right 25 %)

    def getRatingColor(rating):
        if rating >= 8.0: return 'royalblue'
        elif rating >= 7.0: return 'darkgreen'
        elif rating >= 6.0: return '#4CAF50'
        elif rating >= 5.0: return 'gold'
        elif rating >= 4.0: return 'orange'
        else: return 'crimson'

    for _, player in expectedLineup.iterrows():
        slot = player['Slot'] if player['Slot'] in POSITION_COORDINATES else 'MC'
        xCoord = POSITION_COORDINATES[slot][0]
        yCoord = POSITION_COORDINATES[slot][1]
        rating = avgPlayerRatings.get(player['Player Name'], 6.0)
        rc = getRatingColor(rating)
        lastName = player['Player Name'].split()[-1]

        # Number box (top-left)
        fieldAxis.add_patch(Rectangle((xCoord - HW, yCoord), width=NUM_W, height=HH,
                                      linewidth=0.5, edgecolor='dimgray', facecolor='dimgray', zorder=3))
        fieldAxis.text(xCoord - HW + NUM_W / 2, yCoord + HH / 2, str(int(player['Number'])),
                       fontsize=6, color='white', fontweight='bold', ha='center', va='center', zorder=4)

        # Rating box (top-centre)
        fieldAxis.add_patch(Rectangle((xCoord - HW + NUM_W, yCoord), width=RAT_W, height=HH,
                                      linewidth=0.5, edgecolor=rc, facecolor=rc, zorder=3))
        fieldAxis.text(xCoord, yCoord + HH / 2, str(rating),
                       fontsize=6, color='white', fontweight='bold', ha='center', va='center', zorder=4)

        # Position box (top-right)
        fieldAxis.add_patch(Rectangle((xCoord + HW - POS_W, yCoord), width=POS_W, height=HH,
                                      linewidth=0.5, edgecolor='dimgray', facecolor='dimgray', zorder=3))
        fieldAxis.text(xCoord + HW - POS_W / 2, yCoord + HH / 2, slot,
                       fontsize=5, color='white', fontweight='bold', ha='center', va='center', zorder=4)

        # Name bar (bottom, full card width)
        fieldAxis.add_patch(Rectangle((xCoord - HW, yCoord - HH), width=CW, height=HH,
                                      linewidth=0.5, edgecolor='black', facecolor='black', zorder=3))
        fieldAxis.text(xCoord, yCoord - HH / 2, lastName,
                       fontsize=7, color='white', fontweight='bold', ha='center', va='center', zorder=4)

    page.text(0.75,  0.48, 'Expected Subs', ha='center', fontsize=22.5, fontweight='bold')

    # Subs table: axes occupies the right half of the Personnel section, below the header.
    if expectedSubs is not None and hasattr(expectedSubs, 'values') and len(expectedSubs) > 0:
        expectedSubs['Avg Match Rating'] = expectedSubs['Player'].map(avgPlayerRatings)#.fillna(6.0)
        expectedSubs['Sub %'] = expectedSubs['Sub In Probability'].apply(lambda x: f'{x * 100:.1f}')
        reducedExpectedSubs = expectedSubs[['Player', 'Slot', 'Sub %', 'Avg Match Rating']].rename(columns={'Slot': 'Position', 'Avg Match Rating': 'Avg Rating'})
        #subsAx = page.add_axes([0.55, 0.14, 0.40, 0.45])
        tableWidth = 0.40
        tableHeight = 0.45
        targetX = 0.725
        targetY = 0.37
        subsAx = page.add_axes([(targetX-(tableWidth/2)), (targetY-(tableHeight/2)), tableWidth, tableHeight])

        subsAx.axis('off')
        subsTable = subsAx.table(
            cellText=reducedExpectedSubs.values.tolist(),
            colLabels=reducedExpectedSubs.columns.tolist(),
            loc='center', cellLoc='center'
        )
        subsTable.auto_set_font_size(False)
        subsTable.set_fontsize(10)
        subsTable.auto_set_column_width(col=list(range(len(reducedExpectedSubs.columns))))
        subsTable.scale(1, 2)
        # Make the cell borders transparent
        for (row, col), cell in subsTable.get_celld().items():
            cell.set_linewidth(0)        
            cell.set_edgecolor('none')

    page.text(0.75,  0.22, 'Potential Absences', ha='center', fontsize=22.5, fontweight='bold')

    # Absences table: narrow axes in the bottom-right of the Personnel section.
    if isinstance(missingPlayers, pd.DataFrame) and not missingPlayers.empty:
        #absAx = page.add_axes([0.50, -0.05, 0.45, 0.35])
        tableWidth = 0.40
        tableHeight = 0.35
        targetX = 0.74
        targetY = 0.12
        absAx = page.add_axes([(targetX-(tableWidth/2)), (targetY-(tableHeight/2)), tableWidth, tableHeight])

        absAx.axis('off')
        absTable = absAx.table(
            cellText=missingPlayers.values.tolist(),
            colLabels=missingPlayers.columns.tolist(),
            loc='center', cellLoc='center'
        )
        absTable.auto_set_font_size(False)
        absTable.set_fontsize(10)
        absTable.auto_set_column_width(col=list(range(len(missingPlayers.columns))))
        absTable.scale(1, 2)
        # Make the cell borders transparent
        for (row, col), cell in absTable.get_celld().items():
            cell.set_linewidth(0)        
            cell.set_edgecolor('none') 

    return page
#%%
def generateFirstPage(leagues, team, opposition, date, matches, targetMatch):

    page, teamColors = generatePageTemplate(team, opposition, targetMatch, frontPage=True)

    # Fixed: was passing (leagues, team, opposition, date, matches, page) which
    # swapped 'team' in as 'opposition' and omitted targetMatch entirely.
    page = generateMatchInfo(leagues, opposition, date, matches, targetMatch, page, teamColors)

    page = generateMatchOverview(leagues, opposition, date, matches, page, teamColors)

    page = generatePredictedLineup(leagues, opposition, date, matches, page)

    return page

#%%
def main(leagues=['MLS'], team='Columbus_Crew', opposition='New_York_City_FC', date='7-22-2026'):
    # Note: first value of leagues must be the opposition's primary league
    # Note: date format is 'M-D-YYYY'

    matches = loadMatchData(leagues, team, date)

    targetMatch = scrapeTargetMatchData(opposition, date, matches)
    if targetMatch is None:
        print(f'Could not retrieve match data for {opposition} on {date}. Aborting report.')
        return

    outputPath = f'/Users/jakeholfinger/Desktop/CC Analyst/Prematch Reports/Pre_Match_Report_{opposition}_{date}.pdf'
    with PdfPages(outputPath) as pdf:
        pageOne = generateFirstPage(leagues, team, opposition, date, matches, targetMatch)
        pdf.savefig(pageOne)
        plt.close(pageOne)

    print(f'Report saved to {outputPath}')


    return

if __name__ == "__main__":
    main()
