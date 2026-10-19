#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import os
import shutil
from curl_cffi import requests
import Request_Utils
import contextlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import time
import re
import unicodedata
import urllib.request
import urllib.parse
import json
import io
import math
from collections import Counter
# %%
def getFiles(currentPath):
    '''Returns a list of all visible files inside the folder at currentPath'''
    return [f for f in os.listdir(currentPath) if not f.startswith('.')]
# %%
def scrapeURLData(apiURL):
    #Declare a header dictionary, which makes the bot look human and prevent 403 errors
    headers = {
        #No 'user-agent' or 'sec-ch-ua' here on purpose. SofaScore rejects stale Chrome
        #UA strings outright: measured 2026-09-27 on an otherwise identical request,
        #Chrome/149 got 200 while Chrome/144, /142 and /120 all got 403. Pinning a
        #version just restarts that clock, and overriding the UA while curl_cffi
        #impersonates a different browser is self-contradictory anyway. Leaving these
        #out lets curl_cffi send the UA and client hints matching IMPERSONATE.
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'priority': 'u=1, i',
        'referer': 'https://www.sofascore.com/',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'x-requested-with': '5d7735',
    }

    for attempt in range(3):
        try:
            response = requests.get(apiURL, headers=headers, impersonate=Request_Utils.CurrentProfile(headers), timeout=30)
        except Exception as e:
            print(f"Request failed ({e}), retrying... attempt {attempt+1}")
            time.sleep(5 * 2 ** attempt)
            continue

        if response.status_code == 200:
            try:
                return response.json()
            except Exception as e:
                print(f"Failed to parse JSON from {apiURL}: {e}")
                return None

        # THE STATUS CODE HAS TO BE CHECKED BEFORE .json() IS TRUSTED. SofaScore's rejections come
        # back as valid JSON — a 403 body is {"error": {"code": 403, "reason": "challenge"}} — so
        # parsing without looking at the status returns a perfectly well-formed dict that carries
        # no events. Callers then read that as "the fixture doesn't exist" and say so, which is how
        # a blocked API surfaced as "No match found for Columbus Crew on 8-26-2026" several layers
        # up, sending the reader looking for a scheduling problem that wasn't there.
        reason = _responseReason(response)

        # 403 is NOT retried, deliberately, even though it is the one most likely to be temporary.
        # SofaScore serves it as reason="challenge" when it has decided a client looks automated,
        # and the thing that provokes that is request volume — so retrying a challenge is pouring
        # petrol on the fire. Fail fast, say why, and let the caller back off. 429 and 5xx ARE
        # retried, since those are the server asking for patience rather than refusing outright.
        retriable = response.status_code == 429 or response.status_code >= 500
        if retriable and attempt < 2:
            # Honour Retry-After when the server sends a usable one, otherwise back off
            # exponentially. SofaScore has been observed sending a literal 0, which is not a
            # sensible wait, so anything below the floor falls back to the computed delay.
            computed = 5 * 2 ** attempt
            try:
                wait = max(float(response.headers.get('retry-after', 0)), computed)
            except (TypeError, ValueError):
                wait = computed
            print(f"SofaScore returned {response.status_code} ({reason}) for {apiURL} — "
                  f"retrying in {wait:.0f}s (attempt {attempt + 1}/3)")
            time.sleep(min(wait, 60))
            continue

        note = ''
        if response.status_code == 403:
            note = ("  This is SofaScore refusing the request, not a missing fixture. reason="
                    f"'{reason}' usually means their bot check has flagged this IP; it is commonly "
                    "provoked by too many requests in a short window and commonly clears on its "
                    "own after a quiet period.")
        print(f"SofaScore returned {response.status_code} ({reason}) for {apiURL}, giving up.{note}")
        return None

    print(f"Failed to fetch {apiURL} after 3 attempts, skipping.")
    return None

def _responseReason(response):
    '''The server's own words for why it refused, pulled out of the error body when there is one.

    SofaScore answers a rejection with {"error": {"code": 403, "reason": "challenge"}}, and that
    reason is the single most useful thing in the response — "challenge" and "Forbidden" mean
    different things and point at different fixes.'''
    try:
        body = response.json()
    except Exception:
        return (response.text or '').strip()[:80] or 'no body'
    if isinstance(body, dict) and isinstance(body.get('error'), dict):
        return body['error'].get('reason', 'unknown')
    return 'unknown'
#%%
def loadMatchData(leagues, team, date, extraSeasons=0):
    '''Every scraped match for `team` before `date`, keyed by folder name.

    extraSeasons reaches back that many prior calendar years as well. It defaults to 0, so every
    existing caller keeps loading exactly the one season it always did — widening the window
    silently would corrupt every per-90 in the report, which all divide by minutes drawn from
    whatever this returns. Only the penalty charts opt in (see generateFourthPage): their keys are
    pure counts and sums, so they are the one place a longer window costs nothing.

    Matches are still de-duplicated by SofaScore match ID across the whole window, and the
    chronological sort works across years because folder names carry a full M-D-YYYY date.'''
    year = int(date.split('-')[2])
    years = [str(year - offset) for offset in range(extraSeasons + 1)]
    matches = {}

    seenMatchIDs = set()
    for season, league in ((s, l) for s in years for l in leagues):
        teamPath = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{season}_Data/{league.replace(" ", "_")}_Data/{team.replace(" ", "_")}_Data'
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
            # The same match can exist under two folder names (folder dates follow the
            # machine's timezone at scrape time) — keep only the first copy so form,
            # formation counts, and rating averages don't double-count it
            attrs = scrapedFiles.get('Match_Attributes.csv')
            if attrs is not None and 'id' in attrs.columns:
                matchID = attrs['id'].iloc[0]
                if matchID in seenMatchIDs:
                    print(f'Skipping duplicate match folder {matchFolder} (match ID {matchID})')
                    continue
                seenMatchIDs.add(matchID)
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
def findOppositionTeamID(leagueFolders, opposition, year):
    '''Scans each given league's locally saved data for `opposition`'s own team folder
    and returns their SofaScore team ID from any match inside it (a team's ID doesn't
    change match-to-match, so the first one found is enough). Returns None if not found
    in any of the given leagues.'''
    oppositionName = opposition.replace('_', ' ')
    for league in leagueFolders:
        leaguePath = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data/{league.replace(" ", "_")}_Data'
        if not os.path.exists(leaguePath):
            continue
        for teamFolder in getFiles(leaguePath):
            if teamFolder.removesuffix('_Data') != opposition:
                continue
            teamPath = os.path.join(leaguePath, teamFolder)
            if not os.path.isdir(teamPath):
                continue
            for matchFolder in getFiles(teamPath):
                attrsPath = os.path.join(teamPath, matchFolder, 'Match_Attributes.csv')
                if not os.path.exists(attrsPath):
                    continue
                matchAttributes = pd.read_csv(attrsPath)
                homeTeamName = matchAttributes['homeTeam.name'].iloc[0]
                awayTeamName = matchAttributes['awayTeam.name'].iloc[0]
                if homeTeamName == oppositionName:
                    return int(matchAttributes['homeTeam.id'].iloc[0])
                elif awayTeamName == oppositionName:
                    return int(matchAttributes['awayTeam.id'].iloc[0])
            break  # found the opposition's folder for this league; no need to keep scanning
    return None
#%%
def namesLikelySameTeam(localName, apiName):
    '''Whether two team names plausibly refer to the same club.

    Local folder names and SofaScore's names agree often but not always ("Columbus Crew" vs
    "Columbus Crew SC", accents, suffixes like CF/FC/SC). An exact comparison would reject real
    fixtures, so fall back to significant-word overlap after stripping the generic club suffixes
    that carry no identifying information.'''
    if not localName or not apiName:
        return False

    NOISE = {'fc', 'sc', 'cf', 'afc', 'cd', 'club', 'de', 'the', 'ii'}

    def tokens(name):
        cleaned = re.sub(r'[^\w\s]', ' ', unicodedata.normalize('NFKD', str(name))
                         .encode('ascii', 'ignore').decode()).lower()
        return {word for word in cleaned.split() if word and word not in NOISE}

    localTokens, apiTokens = tokens(localName), tokens(apiName)
    if not localTokens or not apiTokens:
        return False

    #Either name's distinctive words being wholly contained in the other's is a match: this
    #accepts "Columbus Crew" vs "Columbus Crew SC" while still rejecting "Inter Miami" vs
    #"San Diego", which share no distinctive word at all.
    return localTokens <= apiTokens or apiTokens <= localTokens

#%%
def scrapeTargetMatchData(leagues, opposition, date, team=None):
    # Added `date` parameter — was previously read from outer scope (a scoping bug)
    oppositionName = opposition.replace('_', ' ')
    teamName = team.replace('_', ' ') if team else None

    # Find the opposition's SofaScore team ID by locating their own folder
    # somewhere in the league's locally saved data (not the focus team's match
    # history) — the focus team may not have played the opposition yet this
    # season, but the opposition's own matches are typically scraped independently.
    year = date.split('-')[2]
    oppositionID = findOppositionTeamID(leagues, opposition, year)

    if oppositionID is None:
        print(f'Could not find team ID for {oppositionName} in local league data.')
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
    # Tracks whether any request came back empty-handed, so the "not found" message below can tell
    # "this fixture isn't scheduled" apart from "we never got to ask". Those need opposite
    # responses from a reader and used to print identically.
    apiUnreachable = False
    for matchTimeframe in matchTimeframes:
        count = 0
        hasNextPage = True
        while hasNextPage:
            apiURL = f'https://www.sofascore.com/api/v1/team/{oppositionID}/events/{matchTimeframe}/{count}'
            data = scrapeURLData(apiURL)
            if data is None:
                apiUnreachable = True
                break
            # API returns {"events": [...], "hasNextPage": bool}
            for match in data.get('events', []):
                # Match on a +/-1 day window: the user passes the LOCAL calendar date but
                # kickoff timestamps are UTC, so night games (e.g. >= 8pm ET) land on the
                # next UTC day and an exact UTC-date comparison misses them. Teams never
                # play two matches within a day of each other, so the window is unambiguous.
                matchDate = datetime.fromtimestamp(match.get('startTimestamp', 0), tz=timezone.utc).date()
                if abs((matchDate - targetDate.date()).days) <= 1:
                    # The window finds a match on roughly the right DATE, which is not the same as
                    # finding the right FIXTURE. This endpoint lists every match the opposition
                    # plays, so a wrong date silently returns whoever they happened to face that
                    # week and the whole report is built about the wrong game — measured
                    # 2026-09-27, asking for Columbus Crew vs Inter Miami on 9-19 returned
                    # 'Inter Miami CF vs San Diego FC' on 9-20 and produced a complete report
                    # carrying that match's date, kickoff time and venue. It also poisons the team
                    # colour database, because generatePageTemplate then resolves teamID to a side
                    # that isn't either of the two teams the caller asked about.
                    #
                    # So confirm the focus team is actually in it. Names are compared loosely
                    # because local folder names don't always match SofaScore's exactly.
                    if teamName is not None:
                        sides = [match.get('homeTeam', {}).get('name', ''),
                                 match.get('awayTeam', {}).get('name', '')]
                        if not any(namesLikelySameTeam(teamName, side) for side in sides):
                            print(f'Skipping {sides[0]} vs {sides[1]} on {matchDate}: within a day of '
                                  f'{date} but {teamName} is not playing in it.')
                            continue

                    matchID = match.get('id', None)
                    if matchID:
                        matchAPIURL = f'https://www.sofascore.com/api/v1/event/{matchID}'
                        eventJSON = scrapeURLData(matchAPIURL)
                        matchData = eventJSON.get('event', {}) if eventJSON else None
                    foundMatch = True
                    break
            if foundMatch:
                break
            hasNextPage = data.get('hasNextPage', False)
            count += 1
        if foundMatch:
            break

    if not foundMatch:
        if apiUnreachable:
            print(f'Could not reach the SofaScore API while looking for {oppositionName} on {date} '
                  f'— see the error above. The fixture may well exist; we never got a list to '
                  f'search. Locally cached match data is unaffected.')
        else:
            if teamName is not None:
                print(f'No match found between {teamName} and {oppositionName} within a day of {date}. '
                      f'Check the date — a fixture on a different date will NOT be used.')
            else:
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
        dateParams = f'&forecast_days=16'
    else:
        urlStart = 'https://archive-api.open-meteo.com/v1/archive'
        dateParams = f'&start_date={kickoffTime.date()}&end_date={kickoffTime.date()}'

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
        f'&timezone=UTC'
        f'{dateParams}'
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
def cropToContent(img, padding=0.04, backgroundTolerance=0.05):
    '''Crops an RGBA/RGB float image (as returned by mpimg.imread) to the tight bounding
    box of its non-background content, then re-pads by `padding` (a fraction of the larger
    cropped dimension) so the content doesn't sit pixel-flush against its box. Falls back
    to the original image if no content is detected.'''
    arr = np.asarray(img)
    if arr.ndim != 3:
        return img

    height, width = arr.shape[:2]

    if arr.shape[2] == 4:
        mask = arr[:, :, 3] > backgroundTolerance
    else:
        corners = np.concatenate([
            arr[0, :5, :3].reshape(-1, 3), arr[0, -5:, :3].reshape(-1, 3),
            arr[-1, :5, :3].reshape(-1, 3), arr[-1, -5:, :3].reshape(-1, 3),
        ])
        bgColor = np.median(corners, axis=0)
        mask = np.abs(arr[:, :, :3] - bgColor).sum(axis=2) > backgroundTolerance

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img

    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    padPx = int(round(padding * max(bottom - top, right - left)))
    top    = max(0, top - padPx)
    bottom = min(height - 1, bottom + padPx)
    left   = max(0, left - padPx)
    right  = min(width - 1, right + padPx)

    return arr[top:bottom+1, left:right+1]
#%%
_team_logo_cache = {}
#%%
def scrapeLogo(teamName):
    """Returns the team's logo as a matplotlib image array by searching Wikipedia's club infobox."""
    if teamName in _team_logo_cache:
        return _team_logo_cache[teamName]

    wikiHeaders = {'User-Agent': 'CCAnalystBot/1.0 (thecolumbuscrewanalyst@gmail.com) Python-urllib/3'}

    # Appending a club-type suffix disambiguates a bare club name from other Wikipedia
    # articles that happen to share it (players, other sports) — "soccer club" works for
    # MLS teams, whose own Wikipedia coverage uses American phrasing. But for clubs whose
    # coverage uses international phrasing instead, "soccer club" can rank a lower-
    # relevance page above the club itself: confirmed live, "Pumas UNAM soccer club"
    # surfaces a player's bio ahead of the "Pumas UNAM" club page, so no infobox logo was
    # ever found and scrapeLogo silently returned None. Try each suffix in turn, only
    # falling through to the next if the current one doesn't resolve to an actual logo.
    querySuffixes = [' soccer club', ' football club', '']

    try:
        for suffix in querySuffixes:
            searchParams = {
                'action': 'query',
                'list': 'search',
                'srsearch': teamName + suffix,
                'srnamespace': '0',
                'srlimit': '1',
                'format': 'json'
            }
            searchURL = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(searchParams)
            with urllib.request.urlopen(urllib.request.Request(searchURL, headers=wikiHeaders)) as r:
                results = json.loads(r.read()).get('query', {}).get('search', [])
            if not results:
                continue

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
                continue

            rawField = imageMatch.group(1).strip()
            fileMatch = re.search(r'(?:File:|Image:)?([^\[\]|]+\.(?:svg|png|jpg|jpeg))', rawField, re.IGNORECASE)
            if not fileMatch:
                continue
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
                continue

            with urllib.request.urlopen(urllib.request.Request(imageURL, headers=wikiHeaders)) as r:
                imageBytes = r.read()

            img = mpimg.imread(io.BytesIO(imageBytes), format='png')
            img = cropToContent(img)
            _team_logo_cache[teamName] = img
            return img

        _team_logo_cache[teamName] = None
        return None

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
def fitTextToWidth(page, textArtist, maxWidthFraction=0.94, minFontSize=11):
    """Shrinks textArtist's font size until it fits within maxWidthFraction of the page
    width, so a long value can't run past the page border.

    Needed because these header lines are centered single lines built from scraped venue
    names, which vary hugely in length: measured at fontsize 18, "LOWER.COM FIELD - GRASS
    - COLUMBUS, OHIO" occupies 0.70 of the page width, but "ESTADIO OLIMPICO
    UNIVERSITARIO - GRASS - MEXICO CITY, MEXICO" occupies 1.00 and overflows both edges.

    Widths are measured against page.bbox.width rather than figsize*dpi: on a Retina
    display the canvas carries a device_pixel_ratio of 2, so the raw window extent is
    double what figsize*dpi predicts, and only the ratio against page.bbox is correct on
    both kinds of display.

    minFontSize is a floor — below it the text is left overflowing rather than shrunk to
    something unreadable, on the grounds that a visibly wrong layout is easier to notice
    and fix than silently illegible text.
    """
    try:
        renderer = page.canvas.get_renderer()
    except AttributeError:
        # Not every matplotlib backend exposes a renderer before the figure is saved.
        # Fail open: leave the text at its requested size rather than crashing the report.
        return textArtist

    fontSize = textArtist.get_fontsize()
    while fontSize > minFontSize:
        if textArtist.get_window_extent(renderer).width / page.bbox.width <= maxWidthFraction:
            break
        fontSize -= 0.5
        textArtist.set_fontsize(fontSize)

    return textArtist
#%%
# Rightmost a line of body text may reach. drawFigureLine already spans 0.018..0.982, which is the
# inside of the printed border, so this keeps a value clear of it by 0.012 — about 7pt.
PAGE_TEXT_RIGHT_LIMIT = 0.970

def drawLabelledValue(page, x, y, label, value, fontsize, rightLimit=PAGE_TEXT_RIGHT_LIMIT,
                      minFontSize=11):
    """Draw "Label: value" as TWO artists so only the value can shrink.

    A single text artist would have to shrink whole, which drops the label out of step with the
    others in its column — "Style Of Play" set smaller than "Primary Formation" directly above it
    reads as a mistake rather than as a fit. Splitting them keeps every label on the page at one
    size and spends the space on the part that actually varies.

    The value starts where the label ends, measured rather than assumed: at fontsize 18 the label
    widths differ by several characters between rows, and a fixed offset would either overlap the
    label or leave a visible gap.

    Returns (labelArtist, valueArtist).
    """
    labelArtist = page.text(x, y, label, fontsize=fontsize)
    try:
        renderer = page.canvas.get_renderer()
    except AttributeError:
        # No renderer before the figure is saved on some backends. Fail open the way
        # fitTextToWidth does: draw the value at the requested size, unmeasured.
        return labelArtist, page.text(x, y, f'{" " * len(label)}{value}', fontsize=fontsize)

    valueX = x + labelArtist.get_window_extent(renderer).width / page.bbox.width
    valueArtist = page.text(valueX, y, value, fontsize=fontsize)
    # fitTextToWidth measures the artist's own width against a fraction of the page, so the
    # fraction to hand it is what is left between where the value starts and the border.
    fitTextToWidth(page, valueArtist, maxWidthFraction=max(rightLimit - valueX, 0.0),
                   minFontSize=minFontSize)
    return labelArtist, valueArtist
def fitTextToDataWidth(page, textArtist, axes, dataWidth, minFontSize=3.5):
    """Shrinks textArtist until it fits within `dataWidth` units of `axes`' own x scale.

    The sibling fitTextToWidth measures against the FIGURE, which is the right frame for a header
    line running across the page. This one measures against a box drawn in an axes' data
    coordinates, which is what the lineup cards need: a name bar is 20 units of a 0-100 pitch, and
    what it can hold depends on how big that pitch happens to be drawn.

    Resolution-independent despite mixing units. A text extent in pixels scales with the figure's
    dpi, and so does the axes' data-to-pixel transform, so the ratio between the two is the same
    whatever dpi the page is finally saved at.

    minFontSize is a floor, and below it the name is left overflowing rather than shrunk to
    something nobody can read — the same call fitTextToWidth makes, on the grounds that a visibly
    wrong layout gets fixed and silently illegible text does not.
    """
    try:
        renderer = page.canvas.get_renderer()
    except AttributeError:
        # Not every backend exposes a renderer before the figure is saved; leave the text alone.
        return textArtist

    (left, __), (right, __) = axes.transData.transform([(0, 0), (dataWidth, 0)])
    available = abs(right - left)

    fontSize = textArtist.get_fontsize()
    # Quarter-point steps rather than the header fitter's half: these cards start at 7pt, so a
    # half-point is already a 7% jump and lands visibly short of the room actually available.
    while fontSize > minFontSize:
        if textArtist.get_window_extent(renderer).width <= available:
            break
        fontSize -= 0.25
        textArtist.set_fontsize(fontSize)

    return textArtist
#%%
def fitBoxToAspect(imgShape, maxWidth, maxHeight):
    '''Returns (width, height) in figure-fraction units that preserve imgShape's aspect
    ratio while maximizing size within the (maxWidth, maxHeight) ceiling.'''
    imgHeight, imgWidth = imgShape[:2]
    imgAspect = imgWidth / imgHeight
    maxAspect = maxWidth / maxHeight
    if imgAspect > maxAspect:
        return maxWidth, maxWidth / imgAspect
    return maxHeight * imgAspect, maxHeight
#%%
# The furthest out a crest may be pushed before the header type has to give way instead. The white
# area of the page begins at (SEC_W + PRI_W)/(72*8.5) = 0.018, so this leaves a gap of 0.012 —
# about 7pt — between a crest and the printed border. Past that a crest reads as falling off the
# page, which is a worse fault than a line of type half a point smaller.
HEADER_LOGO_MIN_X = 0.030

# Kept between the inner edge of a crest and the end of a header line. Purely visual: at zero the
# two touch and read as one object rather than as a crest beside a title.
#
# 0.024 of page width is about 15pt, roughly 0.6 of the 25pt the opponent's name is set in, which
# is what makes it read as a deliberate gap rather than as a near miss. It is nearly free: swept
# against every MLS opponent, three competitions and both real crest shapes, the tall Columbus
# Crew crest (446x250) still needs no shrinking anywhere up to a gap of 0.032, and the square
# crest — which has almost no travel and so shrinks whatever this is set to — costs one point of
# type against the old 0.008 (22.5pt to 21.5pt). Past 0.036 the tall crest runs out of page too
# and every report starts paying.
HEADER_LOGO_TEXT_GAP = 0.024

def clearHeaderLogos(page, headerLines, logoCentreX, logoWidth, logoCentreY, logoHeight):
    '''Keep the header lines clear of the two club crests. Returns the LEFT crest's centre x; the
    right one mirrors it at 1 - x.

    Both crests sat at a fixed 0.11 and 0.89 before this, which a long opponent name simply ran
    into: "VS NEW ENGLAND REVOLUTION" measures 0.727 of page width at fontsize 25, spanning
    0.137-0.863, against crests that reach 0.175 and 0.825 at full size. That is an overlap of
    0.038 a side, and it is not a rare case — "AT CONCACAF CHAMPIONS CUP" and
    "VS VANCOUVER WHITECAPS FC" are as wide, and the kickoff line clears by only 0.005.

    Two moves, in the order that costs the reader least. FIRST the crests slide outward, which
    costs nothing — they are decoration and the page has margin there. Only once they reach
    HEADER_LOGO_MIN_X and a line still will not fit does the TYPE give way, shrunk by
    fitTextToWidth until it clears. Which of the two is enough depends on the crest: a narrow one
    has 0.045 of travel and never needs the second move, while a full-width 0.13 crest has 0.015
    and leaves the longest names about two points to lose.

    The crests move TOGETHER and symmetrically. The header is centred, so one crest nearer its
    edge than the other reads as a mistake rather than as a fit.

    Every header line level with the crests is considered, not just the opponent's name, because
    moving a crest clears all of them at once and the widest is the one that binds. Level is
    measured from the drawn extents rather than assumed from the y positions: at fontsize 30
    "PRE-MATCH REPORT" rises well into the crests' band even though its anchor sits above it.

    Crests never move INWARD. Where nothing conflicts the header keeps the layout it always had.'''
    try:
        renderer = page.canvas.get_renderer()
    except AttributeError:
        # Not every backend exposes a renderer before the figure is saved. Fail open and leave the
        # header alone, exactly as fitTextToWidth does.
        return logoCentreX

    logoTop, logoBottom = logoCentreY + logoHeight / 2, logoCentreY - logoHeight / 2

    def linesLevelWithLogos():
        level = []
        for line in headerLines:
            box = line.get_window_extent(renderer)
            if (box.y0 / page.bbox.height) < logoTop and (box.y1 / page.bbox.height) > logoBottom:
                level.append((box.width / page.bbox.width, line))
        return sorted(level, key=lambda item: -item[0])

    level = linesLevelWithLogos()
    if not level:
        return logoCentreX

    # Where the crest's centre would have to sit for the widest line to clear it, and the furthest
    # out it is allowed to go.
    wanted = 0.5 - level[0][0] / 2 - HEADER_LOGO_TEXT_GAP - logoWidth / 2
    limit = HEADER_LOGO_MIN_X + logoWidth / 2
    logoCentreX = max(limit, min(logoCentreX, wanted))

    if wanted < limit:
        # Out of travel, so the type gives way. Measured against the gap the crests actually leave,
        # and applied per line, so a short line at the same height is never shrunk to match a long
        # one it does not share a problem with.
        allowed = 2 * (0.5 - (logoCentreX + logoWidth / 2) - HEADER_LOGO_TEXT_GAP)
        for width, line in level:
            if width > allowed:
                fitTextToWidth(page, line, maxWidthFraction=allowed)

    return logoCentreX
#%%
def generatePageTemplate(team, opposition, targetMatch, frontPage=False):

    teamName = team.replace('_', ' ')
    teamLogo = scrapeLogo(teamName)
    oppositionName = opposition.replace('_', ' ')
    oppositionLogo = scrapeLogo(oppositionName)
    competitionName = targetMatch.get('tournament', {}).get('uniqueTournament', {}).get('name', 'Unknown Competition')
    competitionRound = (targetMatch.get('roundInfo', {}).get('name') or targetMatch.get('tournament', {}).get('groupName') or ('Group Stage' if targetMatch.get('tournament', {}).get('uniqueTournament', {}).get('hasRounds') else 'Regular Season'))
    # Team colors live under homeTeam.teamColors / awayTeam.teamColors in the API response
    # Resolve which side the team is on. A one-way name check silently assumed 'away'
    # whenever the local name didn't exactly match SofaScore's, pulling the wrong team ID
    # and colors. Check both sides, then fall back to the opposition's side (its name
    # comes from the same SofaScore data, so if it matches one side the team is the other).
    #
    # Resolved BEFORE the kickoff time below, which needs it to pick out the team's own ground.
    homeAPITeamName = targetMatch.get('homeTeam', {}).get('name')
    awayAPITeamName = targetMatch.get('awayTeam', {}).get('name')
    if (teamName == homeAPITeamName) or (oppositionName == awayAPITeamName):
        isHome = True
    elif (teamName == awayAPITeamName) or (oppositionName == homeAPITeamName):
        isHome = False
    else:
        print(f'Warning: neither "{teamName}" nor "{oppositionName}" matches the API team names '
              f'("{homeAPITeamName}" vs "{awayAPITeamName}"). Assuming {teamName} is the away team.')
        isHome = False

    kickoffTimestamp = targetMatch.get('startTimestamp')
    matchVenueName = targetMatch.get('venue', {}).get('name')
    # The team's OWN ground, which is what the header's primary time is expressed in — the report
    # is read from the team's city, so "when does this kick off for us" is the question it should
    # answer first. The API carries both sides' home grounds (homeTeam.venue / awayTeam.venue), so
    # this is a lookup rather than a guess; it falls back to the match venue if the side's ground
    # is missing, which reproduces the old behaviour rather than failing.
    teamVenueName = (targetMatch.get('homeTeam' if isHome else 'awayTeam', {})
                     .get('venue', {}).get('name')) or matchVenueName

    def resolveTimezone(venue):
        '''IANA zone for a venue, or None. Cached inside scrapeCoordinatesAndTimezone, so the
        two lookups here cost one geocode each per distinct ground per session.'''
        if not venue:
            return None
        _, _, tz, _, _, _ = scrapeCoordinatesAndTimezone(venue)
        return tz

    teamTimezone = resolveTimezone(teamVenueName) or 'UTC'
    matchTimezone = resolveTimezone(matchVenueName) or teamTimezone

    # Kickoff in the TEAM's timezone, with the venue's in brackets whenever the two differ — an
    # away trip across a zone boundary is exactly when both numbers matter, and showing only one
    # leaves a reader to work out which it was. The zone abbreviations distinguish them.
    #
    # The bracketed half is kept terse because this line runs BETWEEN the two club logos, and at
    # 0.641 of page width it is one of the widest things in the header. A spelled-out
    # "(7:30 PM CDT at venue)" measured 0.788. clearHeaderLogos now guards the collision rather
    # than leaving it to luck, but every character kept off this line is a character the crests do
    # not have to be pushed out of the way to make room for.
    if kickoffTimestamp is None:
        kickoffTime = 'TBD'
    else:
        kickoffUTC = datetime.fromtimestamp(kickoffTimestamp, tz=ZoneInfo('UTC'))
        teamKickoff = kickoffUTC.astimezone(ZoneInfo(teamTimezone))
        # %Z gives the abbreviation with daylight saving already applied — "EDT" not "EST".
        kickoffTime = teamKickoff.strftime('%-m/%-d/%Y @ %-I:%M %p %Z')
        venueKickoff = kickoffUTC.astimezone(ZoneInfo(matchTimezone))
        if venueKickoff.strftime('%Z') != teamKickoff.strftime('%Z'):
            kickoffTime += venueKickoff.strftime(' (%-I:%M %p %Z)')

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
    sx = SEC_W / (72 * 8.5)            # ≈ 0.00980 fig units
    sy = SEC_W / (72 * 11)             # ≈ 0.00758 fig units
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
    # Kept as artists rather than drawn and forgotten: clearHeaderLogos measures them to decide
    # where the crests can go, and shrinks them if the crests run out of room.
    headerLines = [
        # Large centered title
        page.text(0.5, 0.950, 'PRE-MATCH REPORT', ha='center', fontsize=30, fontweight='bold'),
        # Competition name and round (e.g. "MLS Regular Season")
        page.text(0.5, 0.9125, f'{competitionName.upper()} {competitionRound.upper()}', ha='center', fontsize=19),#, fontweight='bold')
        # "VS" for home games, "AT" for away games, followed by all-caps opponent name
        page.text(0.5, 0.8725, f'{"VS" if isHome else "AT"} {oppositionName.upper()}', ha='center', fontsize=25, fontweight='bold'),
        # Kickoff date/time with local timezone abbreviation
        page.text(0.5, 0.835, kickoffTime, ha='center', fontsize=19),#, fontweight='bold')
    ]

    # Team logos: placed using add_axes([left, bottom, width, height]) so we can
    # call imshow() on them. axis('off') removes the axes border/ticks.
    if teamLogo is not None:
        maxLogoWidth = 0.13
        maxLogoHeight = 0.13
        logoWidth, logoHeight = fitBoxToAspect(teamLogo.shape, maxLogoWidth, maxLogoHeight)

        #axLeft = page.add_axes([0.07, 0.840, 0.09, 0.13])
        logoTargetY = 0.9
        # 0.11 is where the crests sit when nothing is in their way; a long header line pushes them
        # outward from there, and shrinks itself if they run out of page. See clearHeaderLogos.
        leftTargetX = clearHeaderLogos(page, headerLines, 0.11, logoWidth, logoTargetY, logoHeight)
        axLeft = page.add_axes([(leftTargetX-(logoWidth/2)), (logoTargetY-(logoHeight/2)), logoWidth, logoHeight])
        axLeft.imshow(teamLogo)
        axLeft.axis('off')

        #axRight = page.add_axes([0.82, 0.840, 0.09, 0.13])
        rightTargetX = 1 - leftTargetX
        axRight = page.add_axes([(rightTargetX-(logoWidth/2)), (logoTargetY-(logoHeight/2)), logoWidth, logoHeight])
        # TODO: Pick which logo to show
        axRight.imshow(teamLogo)
        #axRight.imshow(oppositionLogo)
        axRight.axis('off')

    # Separator line below header (not drawn on the front page — see frontPage flag)
    if not frontPage:
        drawFigureLine(page, secondaryColor, y=0.820)

    return page, teamColors
#%%
def generateMatchInfo(targetMatch, page, teamColors):

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
        # pd.notna also catches NaN — the archive API returns nulls for
        # precipitation_probability, which pandas can surface as float NaN
        precipStr = f'{int(rawPrecip)}% PRECIP' if pd.notna(rawPrecip) else 'PRECIP N/A'
        windSpeed = int(round(weatherRow['wind_speed_10m'].iloc[0]))
        windDir = str(weatherRow['wind_direction_10m'].iloc[0])
        weather = f'{weatherCondition}, FEELS LIKE {feelsLike}°F, {precipStr}, {windSpeed} MPH WIND {windDir}'

    venueSurface = scrapeVenueSurface(venueName) if venueName else 'Unknown'
    venueDisplay = venueName.upper() if venueName else 'VENUE TBD'
    venueLocationDisplay = venueLocation.upper() if venueLocation.strip(', ') else 'LOCATION TBD'

    # Line 1: venue name and playing surface (e.g. "SCOTTS MIRACLE-GRO FIELD - GRASS")
    venueLine = page.text(0.5, 0.7925, f'{venueDisplay} - {venueSurface.upper()} - {venueLocationDisplay}', ha='center', fontsize=18)
    fitTextToWidth(page, venueLine)#, fontweight='bold')
    # Line 2: city/state and weather summary (e.g. "COLUMBUS, OHIO - SUNNY, FEELS LIKE 78°F, ...")
    weatherLine = page.text(0.5, 0.760, f'{weather}', ha='center', fontsize=18)#, fontweight='bold')
    fitTextToWidth(page, weatherLine)

    drawFigureLine(page, teamColors[1], y=0.742)

    return page
#%%
def getStyleOfPlay(matches, team, date):

    import Indentify_Style_Of_Play

    styleOfPlay = Indentify_Style_Of_Play.main(matches, team=team, date=date)

    print(f'Style of Play: {styleOfPlay}')

    return styleOfPlay
#%%
def generateMatchOverview(leagues, opposition, date, matches, page, teamColors):
    #TODO: Finish Implementing Method

    oppositionName = opposition.replace('_', ' ')

    # Load the opposition's own match history — the passed-in `matches` is the
    # team's data and would give wrong form/formation results.
    matches = loadMatchData(leagues, opposition, date)

    # Folders are named "M-D-YYYY_vs_Opponent" — sort chronologically by the date prefix
    matches = dict(sorted(matches.items(),
        key=lambda item: datetime.strptime(item[0].split('_')[0], '%m-%d-%Y')))

    # Get the last 5 results: W/D/L from the opposition's perspective.
    #
    # Filter for Match_Attributes.csv BEFORE taking the last five, not after. Slicing first and
    # skipping incomplete folders inside the loop silently returned fewer than five results —
    # measured 2026-09-27, the Crew's form read as four matches because their most recent folder
    # (9-19-2026_vs_CF_Montréal) was scraped without Match_Attributes.csv, so it was sliced in and
    # then dropped. A short form line looks like a team that has only played four times rather
    # than like missing data, which is the worst way for this to fail.
    playedMatches = [(folder, files) for folder, files in matches.items()
                     if files.get('Match_Attributes.csv') is not None]
    if len(playedMatches) < len(matches):
        incomplete = [f for f, files in matches.items() if files.get('Match_Attributes.csv') is None]
        print(f'Note: {len(incomplete)} {oppositionName} match folder(s) have no Match_Attributes.csv '
              f'and are excluded from form/standings: {incomplete}. Re-scrape them for accurate results.')

    form = []
    for matchFolder, matchFiles in playedMatches[-5:]:
        attrs = matchFiles.get('Match_Attributes.csv')
        winnerCode = attrs['winnerCode'].iloc[0]
        homeTeamName = attrs['homeTeam.name'].iloc[0]
        # winnerCode 1 = home win, 2 = away win, 0 = draw
        if winnerCode == 1:
            form.append('W' if homeTeamName == oppositionName else 'L')
        elif winnerCode == 2:
            form.append('L' if homeTeamName == oppositionName else 'W')
        else:
            form.append('D')

    styleOfPlay = getStyleOfPlay(matches, opposition, date)

    # Count which formations the opposition has used; take the top two
    oppositionFormations = []
    for matchFolder, matchFiles in matches.items():
        formations = matchFiles.get('Formations.csv')
        if formations is None:
            continue
        # Formations.csv has columns 'Team Name' and 'Formation', one row per team
        rows = formations.loc[formations['Team Name'] == oppositionName, 'Formation']
        if not rows.empty:
            val = rows.iloc[0]
            if pd.notna(val):
                # Formations are stored as Python list reprs e.g. "['4-2-3-1']" — strip to bare string
                formStr = str(val).strip("[]'\"")
                if formStr:
                    oppositionFormations.append(formStr)

    counts = Counter(oppositionFormations)
    topFormations = counts.most_common(2)
    primaryFormation = topFormations[0][0] if len(topFormations) >= 1 else 'N/A'
    secondaryFormation = topFormations[1][0] if len(topFormations) >= 2 else None

    import League_Power_Rankings
    import Expected_Lineup_Temp_V2

    year = date.split('-')[2]

    # leagues[0] is the FOCUS team's own league — for an opponent from a different
    # competition (e.g. a Leagues Cup opponent from Liga MX), that league's table never
    # contains them, so Table Position/Power Ranking always fell through to N/A. Resolve
    # the opponent's own real competition instead. `matches` above is scoped to whatever
    # `leagues` the caller passed, which for a genuinely cross-competition opponent may
    # not include their real league at all (that's the whole problem), so scan every
    # locally scraped league directly for the opponent's own folder instead of relying
    # on `matches` — same helper `scrapeTargetMatchData` uses, just given every local
    # league instead of only the caller-provided ones.
    leaguesRoot = f'/Users/jakeholfinger/Desktop/CC Analyst/Data/SofaScore_Data/{year}_Data'
    allLocalLeagues = []
    if os.path.exists(leaguesRoot):
        allLocalLeagues = [f.removesuffix('_Data').replace('_', ' ') for f in getFiles(leaguesRoot)
                            if os.path.isdir(os.path.join(leaguesRoot, f))]
    oppositionID = findOppositionTeamID(allLocalLeagues, opposition, year)

    leagueForRankings = leagues[0]
    if oppositionID is not None:
        oppositionCompetitions, _ = Expected_Lineup_Temp_V2.scrapeOppositionCompetitions([oppositionID])
        liveCompetition = oppositionCompetitions[0] if oppositionCompetitions else None
        if liveCompetition:
            # SofaScore's primaryUniqueTournament can lag behind what's actually been
            # scraped locally — confirmed live for Atlas FC: it reports "Liga MX,
            # Clausura" (last season) while only "Liga MX, Apertura" (current season)
            # exists on disk. Match on the league name before any comma-separated
            # season/stage suffix against what's actually there, rather than requiring
            # an exact string match that a stale live value would never satisfy.
            liveLeaguePrefix = liveCompetition.split(',')[0].strip()
            for localLeagueName in allLocalLeagues:
                if localLeagueName.split(',')[0].strip() == liveLeaguePrefix:
                    leagueForRankings = localLeagueName
                    break

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        result = League_Power_Rankings.main(year, leagueForRankings, len(matches), table=True)

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
    page.text(0.5, 0.700, 'OPPOSITION OVERVIEW', ha='center', fontsize=27.5, fontweight='bold')

    # Left column (x=0.048): league table, power ranking, form, expected points
    page.text(0.048, 0.660, f'Table Position: {standing}', fontsize=18)
    page.text(0.048, 0.620, f'Power Ranking: {powerRanking}', fontsize=18)
    page.text(0.048, 0.580, f'Form: {" ".join(form)}', fontsize=18)
    #page.text(0.048, 0.579, 'Expected Points: N/A', fontsize=22.5)

    # Right column (x=0.500): tactical info
    page.text(0.450, 0.660, f'Primary Formation: {primaryFormation}', fontsize=18)
    page.text(0.450, 0.620, f'Secondary Formation: {secondaryFormation or "N/A"}', fontsize=18)
    # styleOfPlay is None when the classifier had nothing to work with at all — a competition
    # missing even the core files. When it DID classify but on partial data, it says so via
    # 'limited', and that has to reach the page: an archetype built from 38% of the possession
    # weight and one match should not read identically to one built on a full MLS season.
    # See Indentify_Style_Of_Play.main for what goes into that flag.
    archetype = styleOfPlay.get('primary_archetype') if styleOfPlay else None
    styleLabel = archetype or 'N/A'
    # BOTH names when the two are close. Nearest-centroid always returns something, so a team with
    # no distinguishable style otherwise reads exactly like one with a pronounced one — see
    # Indentify_Style_Of_Play.STYLE_AMBIGUOUS_MARGIN for where the line is drawn and why. Nearest
    # first, so the pair still says which way the team leans.
    #
    # Joined with " or ", NOT with a slash: one of the archetypes is itself called Wide/Crossing,
    # so a slash separator printed New England as "Wide/Crossing / Counter-Attack" — which reads
    # as three styles, with the two slashes spaced differently to compound it. "or" also says
    # plainly what the pair means, which a punctuation mark leaves the reader to infer.
    if archetype and styleOfPlay.get('archetype_ambiguous'):
        runnerUp = styleOfPlay.get('secondary_archetype')
        if runnerUp:
            styleLabel = f'{archetype} or {runnerUp}'
    if archetype and styleOfPlay.get('limited'):
        styleLabel += ' (limited data)'
    # A floor of 7.5 rather than fitTextToWidth's usual 11, measured against the worst string this
    # line can actually be asked to hold. Every archetype, every ordered pair of them and every one
    # of those with the suffix is 99 strings; the longest is
    # "Counter-Attack / Vertical Possession (limited data)" at 51 characters, and from where the
    # label leaves it that needs 7.5pt to clear the border. At 11 it ran off the page ENTIRELY,
    # which is a worse fault than small type, and it is a real combination — a competition thin
    # enough to set 'limited' is no less likely to produce a close pair than a full one.
    drawLabelledValue(page, 0.450, 0.580, 'Style Of Play: ', styleLabel, fontsize=18,
                      minFontSize=7.5)
    #page.text(0.500, 0.579, 'Win Probability: N/A', fontsize=22.5)

    drawFigureLine(page, teamColors[1], y=0.550)

    return page

# (x, y) where x = field WIDTH (0=left touchline, 100=right touchline)
#             y = field DEPTH (0=GK goal line, 100=halfway line)
# This maps directly onto fieldAxis data coordinates: x is horizontal, y is vertical.
# Right-sided positions (DR, MR, RW) are at high x; left-sided (DL, ML, LW) at low x.
#
# The same slot NAME means a different tactical role depending on how many players form
# the back line, so a single fixed table renders it wrong — e.g. a back-three's DCR/DCL
# have no fullback outside them and should sit far wider than a back-four's half-space
# center-backs, and a back-three/five's holding midfield line typically sits a touch
# deeper (lower y) in front of the extra covering defender. Three tables, chosen by
# back-line shape, cover this without needing one table per individual formation.
POSITION_COORDINATES_4BACK = {
    'GK':  (50,  6),
    'DR':  (88, 30), 'DCR': (63, 30), 'DC':  (50, 30), 'DCL': (37, 30), 'DL':  (12, 30),
                     'DMR': (63, 43), 'DM':  (50, 43), 'DML': (37, 43),
    'MR':  (88, 55), 'MCR': (63, 55), 'MC':  (50, 55), 'MCL': (37, 55), 'ML':  (12, 55),
    'RW':  (88, 68), 'AMR': (63, 68), 'AM':  (50, 68), 'AML': (37, 68), 'LW':  (12, 68),
                     'STR': (63, 82), 'ST':  (50, 82), 'STL': (37, 82)
}

POSITION_COORDINATES_3BACK = {
    'GK':  (50,  6),
    'DR':  (88, 30), 'DCR': (75, 30), 'DC':  (50, 30), 'DCL': (25, 30), 'DL':  (12, 30),
                     'DMR': (63, 41), 'DM':  (50, 41), 'DML': (37, 41),
    'MR':  (88, 52), 'MCR': (63, 52), 'MC':  (50, 52), 'MCL': (37, 52), 'ML':  (12, 52),
    'RW':  (88, 68), 'AMR': (63, 68), 'AM':  (50, 68), 'AML': (37, 68), 'LW':  (12, 68),
                     'STR': (63, 82), 'ST':  (50, 82), 'STL': (37, 82)
}

POSITION_COORDINATES_5BACK = {
    'GK':  (50,  6),
    'DR':  (84, 30), 'DCR': (66, 30), 'DC':  (50, 30), 'DCL': (34, 30), 'DL':  (16, 30),
                     'DMR': (63, 41), 'DM':  (50, 41), 'DML': (37, 41),
    'MR':  (88, 52), 'MCR': (63, 52), 'MC':  (50, 52), 'MCL': (37, 52), 'ML':  (12, 52),
    'RW':  (88, 68), 'AMR': (63, 68), 'AM':  (50, 68), 'AML': (37, 68), 'LW':  (12, 68),
                     'STR': (63, 82), 'ST':  (50, 82), 'STL': (37, 82)
}

def getShapeCoordinates(slots):
    '''Choose the back-line-shape coordinate table from a lineup's assigned Formation
    Slots. DC only appears in back-3/back-5 formations (getFormationSlots); DR/DL only
    appear in back-4/back-5. Presence of both distinguishes all three shapes without
    needing the raw formation string.'''
    defSlots = {s for s in slots if s in ('DR', 'DCR', 'DC', 'DCL', 'DL')}
    if 'DC' in defSlots and 'DR' in defSlots and 'DL' in defSlots:
        table = POSITION_COORDINATES_5BACK
    elif 'DC' in defSlots:
        table = POSITION_COORDINATES_3BACK
    else:
        table = POSITION_COORDINATES_4BACK

    # A genuine central trio (MC present) needs wider spacing than the shared table
    # gives it — MC/MCR/MCL are only 13 units apart there, but each card is CW=20 units
    # wide (see below), so they overlap by 7 units on each side. The safe spread depends
    # on whether wide ML/MR are also in the same row:
    if 'MC' in slots:
        table = dict(table)
        _, y = table['MCR']
        if 'ML' in slots or 'MR' in slots:
            # Trio + full wide mid (3-5-2): 5 cards need to share the row, so all five
            # get spaced exactly CW=20 apart, centered on 50.
            table['ML'], table['MCL'], table['MC'] = (10, y), (30, y), (50, y)
            table['MCR'], table['MR'] = (70, y), (90, y)
        else:
            # Trio only, no wide mid competing for the row (4-3-3, 4-3-1-2, 5-3-2) —
            # spread into the unused wide space instead of the tight shared-table gap.
            table['MCR'], table['MCL'] = (75, y), (25, y)
    return table
#%%
def generatePredictedLineup(leagues, opposition, date, page, formationOverride=False):

    import Expected_Lineup_Temp_V2

    if formationOverride:
        # formationOverride triggers an input() prompt inside Expected_Lineup_Temp_V2.main
        # — input()'s prompt text goes to stdout, so it must NOT be suppressed here or
        # the user would see a blank terminal while the process silently blocks on stdin.
        expectedLineup, expectedSubs, missingPlayersDict = Expected_Lineup_Temp_V2.main(leagues, opposition, date, formationOverride=formationOverride)
    else:
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                expectedLineup, expectedSubs, missingPlayersDict = Expected_Lineup_Temp_V2.main(leagues, opposition, date, formationOverride=formationOverride)

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
        missingPlayers['Avg Rating'] = missingPlayers['Player'].map(avgPlayerRatings).fillna('N/A')
        # Cap the displayed list so it can't outgrow the table's allocated space — sort by
        # Avg Rating first so a truncation drops the least-impactful absences, not whoever
        # happened to come last from the API.
        missingPlayers = missingPlayers.sort_values(
            by='Avg Rating', key=lambda s: pd.to_numeric(s, errors='coerce'), ascending=False, na_position='last'
        ).head(8)
    else:
        missingPlayers = pd.DataFrame()

    # --- PERSONNEL SECTION ---
    # Section title — centered, y=0.522 places it just below the Match Overview separator
    page.text(0.5, 0.505, 'PERSONNEL', ha='center', fontsize=27.5, fontweight='bold')

    # Sub-section headers: Expected Lineup (left), Expected Subs (centre-right),
    # Potential Absences (far right). x positions chosen to align with their content areas.
    #page.text(0.09, 0.485, 'Expected Lineup', s='Perfectly Centered', fontsize=22.5, fontweight='bold')
    page.text(0.26, 0.45, 'Expected Lineup', ha='center', fontsize=22.5, fontweight='bold')


    # Field axis: left=4.7%, bottom=1%, width=43%, height=46% of the figure.
    # xlim 0-100 = left touchline → right touchline (68 m wide)
    # ylim -5 to 100 = behind goal → halfway line (52.5 m deep); -5 makes room for the goal frame
    #fieldAxis = page.add_axes([0.047, 0.01, 0.43, 0.46])
    fieldWidth = 0.4
    fieldHeight = 0.4
    targetX = 0.26
    targetY = 0.225
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
    CW = 20     # card width  — change this to resize all cards uniformly
    CH = 8    # card height
    HW = CW / 2
    HH = CH / 2
    NUM_W = 0.30 * CW   # number box  (left 25 %)
    RAT_W = 0.40 * CW   # rating box  (centre 50 %)
    POS_W = 0.30 * CW   # position box (right 25 %)
    # Share of the name bar a surname may occupy. The remainder is breathing room at each end: a
    # name fitted to the bar's exact width touches the black on both sides and reads as clipped
    # even when every letter is there.
    NAME_BAR_FILL = 0.92

    def getRatingColor(rating, sofascore=True):
        if sofascore:
            if rating == 'N/A': return 'dimgray'
            elif rating >= 8.0: return 'royalblue'
            elif rating >= 7.0: return 'darkgreen'
            elif rating >= 6.5: return '#4CAF50'
            elif rating >= 6.0: return 'gold'
            elif rating >= 5.0: return 'orange'
            else: return 'crimson'
        else:
            if rating == 'N/A': return 'dimgray'
            elif rating >= 8.0: return 'royalblue'
            elif rating >= 7.0: return 'darkgreen'
            elif rating >= 6.0: return '#4CAF50'
            elif rating >= 5.0: return 'gold'
            elif rating >= 4.0: return 'orange'
            else: return 'crimson'

    positionCoordinates = getShapeCoordinates(expectedLineup['Slot'].tolist()) if not expectedLineup.empty else POSITION_COORDINATES_4BACK

    for _, player in expectedLineup.iterrows():
        slot = player['Slot'] if player['Slot'] in positionCoordinates else 'MC'
        xCoord = positionCoordinates[slot][0]
        yCoord = positionCoordinates[slot][1]
        rating = avgPlayerRatings.get(player['Player Name'], 'N/A')
        rc = getRatingColor(rating)
        lastName = player['Player Name'].split()[-1]

        # Number box (top-left)
        fieldAxis.add_patch(Rectangle((xCoord - HW, yCoord), width=NUM_W, height=HH,
                                      linewidth=0.5, edgecolor='dimgray', facecolor='dimgray', zorder=3))
        playerNumber = '' if (player['Number'] is None or (isinstance(player['Number'], float) and math.isnan(player['Number']))) else str(int(player['Number']))
        fieldAxis.text(xCoord - HW + NUM_W / 2, yCoord + HH / 2, playerNumber,
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
        nameText = fieldAxis.text(xCoord, yCoord - HH / 2, lastName,
                       fontsize=7, color='white', fontweight='bold', ha='center', va='center', zorder=4)
        # The bar is CW units wide but only about 49pt on the page, which holds roughly eleven
        # characters at 7pt bold — so a long surname ran out past the black on both sides and sat
        # on the grass. Shrink to fit rather than truncate: a cut-off surname is a different
        # player's name, which is worse than a small one. NAME_BAR_FILL keeps it off the edges.
        fitTextToDataWidth(page, nameText, fieldAxis, CW * NAME_BAR_FILL)

    page.text(0.75,  0.45, 'Expected Subs', ha='center', fontsize=22.5, fontweight='bold')

    # Subs table: axes occupies the right half of the Personnel section, below the header.
    if expectedSubs is not None and hasattr(expectedSubs, 'values') and len(expectedSubs) > 0:
        expectedSubs['Avg Match Rating'] = expectedSubs['Player'].map(avgPlayerRatings).fillna('N/A')
        expectedSubs['Sub %'] = expectedSubs['Sub In Probability'].apply(lambda x: f'{x * 100:.1f}')
        reducedExpectedSubs = expectedSubs[['Player', 'Slot', 'Sub %', 'Avg Match Rating']].rename(columns={'Slot': 'Position', 'Avg Match Rating': 'Avg Rating'})
        #subsAx = page.add_axes([0.55, 0.14, 0.40, 0.45])
        tableWidth = 0.425
        tableHeight = 0.16
        targetX = 0.725
        targetY = 0.369
        subsAx = page.add_axes([(targetX-(tableWidth/2)), (targetY-(tableHeight/2)), tableWidth, tableHeight])

        subsAx.axis('off')
        subsTable = subsAx.table(
            cellText=reducedExpectedSubs.values.tolist(),
            colLabels=reducedExpectedSubs.columns.tolist(),
            loc='center', cellLoc='center'
        )
        subsTable.auto_set_font_size(False)
        subsTable.set_fontsize(9)
        subsTable.auto_set_column_width(col=list(range(len(reducedExpectedSubs.columns))))
        subsTable.scale(1, 1.5)
        # Make the cell borders transparent
        for (row, col), cell in subsTable.get_celld().items():
            cell.set_linewidth(0)
            cell.set_edgecolor('none')

    page.text(0.75,  0.257, 'Potential Absences', ha='center', fontsize=22.5, fontweight='bold')

    # Absences table: narrow axes in the bottom-right of the Personnel section. missingPlayers
    # is already capped (see above) so this table's rendered height (row-count-driven, not
    # tableHeight below) can't outgrow the space reserved for it here.
    if isinstance(missingPlayers, pd.DataFrame) and not missingPlayers.empty:
        #absAx = page.add_axes([0.50, -0.05, 0.45, 0.35])
        tableWidth = 0.40
        tableHeight = 0.22
        targetX = 0.74
        # ax.table(loc='center') centers itself on the axes regardless of tableHeight above,
        # and its actual rendered height is row-count-driven (~0.02273 figure-fraction per
        # row at fontsize=9/scale(1,1.5)) — centering on a fixed targetY would make the
        # table's TOP edge drift depending on how many players are missing. Anchor the TOP
        # instead (fixed distance below the header, matching the tuned 8-row-cap case) and
        # work backward to the center targetY this specific row count needs.
        absencesTableTop = 0.2439
        perRowHeight = 0.02273
        actualTableHeight = (len(missingPlayers) + 1) * perRowHeight
        targetY = absencesTableTop - actualTableHeight / 2
        absAx = page.add_axes([(targetX-(tableWidth/2)), (targetY-(tableHeight/2)), tableWidth, tableHeight])

        absAx.axis('off')
        absTable = absAx.table(
            cellText=missingPlayers.values.tolist(),
            colLabels=missingPlayers.columns.tolist(),
            loc='center', cellLoc='center'
        )
        absTable.auto_set_font_size(False)
        absTable.set_fontsize(9)
        absTable.auto_set_column_width(col=list(range(len(missingPlayers.columns))))
        absTable.scale(1, 1.5)
        # Make the cell borders transparent
        for (row, col), cell in absTable.get_celld().items():
            cell.set_linewidth(0)        
            cell.set_edgecolor('none') 

    # expectedLineup/expectedSubs are returned, not just used locally, so page three's key
    # pass legend can grey players who aren't expected to feature. Passing them down is
    # deliberate rather than letting that chart call Expected_Lineup_Temp_V2.main again:
    # that call is a network scrape plus an ILP solve, and with formationOverride=True it
    # blocks on input() — a second call would prompt for the formation twice per report.
    return page, expectedLineup, expectedSubs
#%%
def generateFirstPage(leagues, team, opposition, date, matches, targetMatch, formationOverride=False):

    page, teamColors = generatePageTemplate(team, opposition, targetMatch, frontPage=True)

    # Fixed: was passing (leagues, team, opposition, date, matches, page) which
    # swapped 'team' in as 'opposition' and omitted targetMatch entirely.
    page = generateMatchInfo(targetMatch, page, teamColors)

    page = generateMatchOverview(leagues, opposition, date, matches, page, teamColors)

    page, expectedLineup, expectedSubs = generatePredictedLineup(leagues, opposition, date, page, formationOverride=formationOverride)

    return page, expectedLineup, expectedSubs

#%%
def generateSecondPage(leagues, team, opposition, targetMatch, date):

    page, teamColors = generatePageTemplate(team, opposition, targetMatch)

    import Tactical_Visualizations

    # The report scouts the opposition (like the first page). The passed-in
    # `matches` is the focus team's data, so reload the opposition's own history
    # and pass `opposition` — otherwise the build-out map shows the wrong team's
    # goalkeeper (and excludes the wrong team from the league baseline).
    oppositionMatches = loadMatchData(leagues, opposition, date)

    page, gkNotesBbox = Tactical_Visualizations.generate_relative_gk_pass_map(leagues, opposition, oppositionMatches, date, teamColors, page)

    page, defActionNotesBbox = Tactical_Visualizations.generate_defensive_action_map(leagues, opposition, oppositionMatches, date, teamColors, page)

    page, buildupNotesBbox = Tactical_Visualizations.generate_buildup_turnover_map(leagues, opposition, oppositionMatches, date, teamColors, page)

    page, pressingNotesBbox = Tactical_Visualizations.generate_pressing_turnover_map(leagues, opposition, oppositionMatches, date, teamColors, page)

    # Each chart function draws an empty bordered placeholder box for its notes field but
    # can't make it interactive itself (matplotlib has no notion of a fillable PDF form
    # field) — the (name, bbox) pairs collected here get turned into real AcroForm fields
    # in a post-processing pass over the finished PDF, in main(). A chart with no data
    # (empty league/team data) skips its own box and returns None instead of a bbox.
    notesFields = [
        ('gk_pass_notes', gkNotesBbox),
        ('def_action_notes', defActionNotesBbox),
        ('buildup_turnover_notes', buildupNotesBbox),
        ('pressing_turnover_notes', pressingNotesBbox)
    ]
    notesFields = [(name, bbox) for name, bbox in notesFields if bbox is not None]

    return page, notesFields

#%%
def generateThirdPage(leagues, team, opposition, targetMatch, date, expectedLineup=None, expectedSubs=None):

    page, teamColors = generatePageTemplate(team, opposition, targetMatch)
    
    import Tactical_Visualizations

    # The report scouts the opposition (like the first page). The passed-in
    # `matches` is the focus team's data, so reload the opposition's own history
    # and pass `opposition` — otherwise the build-out map shows the wrong team's
    # goalkeeper (and excludes the wrong team from the league baseline).
    oppositionMatches = loadMatchData(leagues, opposition, date)

    page, keyPassNotesBbox, keyPassVmax = Tactical_Visualizations.generate_key_pass_map(
        leagues, opposition, oppositionMatches, date, teamColors, page,
        expected_lineup=expectedLineup, expected_subs=expectedSubs)

    # Distinct variable from the call above — reusing keyPassNotesBbox would silently
    # discard the offensive chart's notes field, leaving its box drawn on the page but
    # never turned into a real fillable form field.
    page, defKeyPassNotesBbox = Tactical_Visualizations.generate_def_key_pass_map(
        leagues, opposition, oppositionMatches, date, teamColors, page, shared_vmax=keyPassVmax)

    # expectedLineup/expectedSubs are forwarded for the same reason generate_key_pass_map takes
    # them: a player the model predicts to feature stays un-greyed in the player key however
    # few minutes they have banked so far.
    # shotVmax is the maximum ΣxG across BOTH directions, which only the offensive chart can see
    # (it extracts the conceded frame for its own situation key). Handing it to the conceded chart
    # puts the pair on one colour scale, so a solid cell means the same number on both.
    page, shotNotesBbox, shotVmax = Tactical_Visualizations.generate_shot_map(
        leagues, opposition, oppositionMatches, date, teamColors, page,
        expected_lineup=expectedLineup, expected_subs=expectedSubs)

    # Takes neither leagues/date nor the lineup prediction, unlike every call above it: those
    # feed the situation key's league percentiles and the player key's greying rule, and the
    # conceded chart has neither key. Its own notes variable for the same reason
    # defKeyPassNotesBbox is distinct — reusing shotNotesBbox would leave one of the two boxes
    # drawn on the page but never turned into a real fillable form field.
    page, defShotNotesBbox = Tactical_Visualizations.generate_def_shot_map(
        opposition, oppositionMatches, teamColors, page, shared_vmax=shotVmax)

    notesFields = [
        ('key_pass_notes', keyPassNotesBbox),
        ('def_key_pass_notes', defKeyPassNotesBbox),
        ('shot_notes', shotNotesBbox),
        ('def_shot_notes', defShotNotesBbox)
    ]
    notesFields = [(name, bbox) for name, bbox in notesFields if bbox is not None]

    return page, notesFields

#%%
# How many prior seasons the penalty charts reach back over. One: only 2025 and 2026 are scraped,
# so this is everything there is, and it roughly doubles-to-quintuples a squad's penalty count on
# MLS. Deliberately NOT applied to any other chart — see the note at the call site.
PENALTY_EXTRA_SEASONS = 1

def generateFourthPage(leagues, team, opposition, targetMatch, date,
                       expectedLineup=None, expectedSubs=None):
    '''Set pieces. Two pairs, each offence on the left and defence on the right: corners taken and
    conceded across the top, penalties taken and conceded beneath them.'''

    page, teamColors = generatePageTemplate(team, opposition, targetMatch)

    import Tactical_Visualizations

    # The report scouts the opposition, as pages one and three do, so reload their own history.
    oppositionMatches = loadMatchData(leagues, opposition, date)

    # The two charts share ONE key column between them: there is no room for a text list to the
    # right of the conceded chart, whose colourbar label already runs to x~0.95. The taken chart
    # therefore hands back where its own stack ended, and the conceded chart hangs its defender
    # list underneath — which is also why it must be drawn second.
    # expectedLineup/expectedSubs are forwarded for the same reason page three's charts take
    # them: a player the model predicts to feature stays un-greyed in the Taker and Corner Shot
    # Threat keys however few minutes they have banked so far. See player_qualifier.
    page, cornerNotesBbox, cornerKeyColumn, cornerVmax = Tactical_Visualizations.generate_corner_map(
        leagues, opposition, oppositionMatches, date, teamColors, page,
        expected_lineup=expectedLineup, expected_subs=expectedSubs)

    page, defCornerNotesBbox = Tactical_Visualizations.generate_def_corner_map(
        leagues, opposition, oppositionMatches, date, teamColors, page,
        key_column=cornerKeyColumn, shared_vmax=cornerVmax,
        expected_lineup=expectedLineup, expected_subs=expectedSubs)

    # Second pair, same arrangement: the taken chart is drawn first and owns the whole shared key
    # column, including the keeper table describing the conceded chart's marks. It hands back the
    # colours that table used, so the conceded chart's marks match it, and whether this competition
    # carries xGOT — judged over both directions rather than from the conceded side's few penalties.
    #
    # These two get a WIDER window than every other chart on the report: penalties are rare enough
    # that one season barely fills a chart — Nashville take three in 2026 and seventeen across two,
    # Columbus four and ten. They are also the only charts that can safely take it, because their
    # keys are counts and sums with no per-90 anywhere; every other chart divides by minutes drawn
    # from `oppositionMatches` and would be corrupted by a longer window. Players who have since
    # left still get greyed by player_qualifier's current-squad check, which is the case it exists
    # for. Where no prior season is scraped — Liga MX has none — this simply returns the same
    # matches and the charts say nothing about it.
    penaltyMatches = loadMatchData(leagues, opposition, date,
                                   extraSeasons=PENALTY_EXTRA_SEASONS)

    page, penaltyNotesBbox, penaltyXgAvailable, penaltyKeeperColors = (
        Tactical_Visualizations.generate_penalty_map(
            leagues, opposition, penaltyMatches, date, teamColors, page,
            expected_lineup=expectedLineup, expected_subs=expectedSubs))

    page, defPenaltyNotesBbox = Tactical_Visualizations.generate_def_penalty_map(
        leagues, opposition, penaltyMatches, date, teamColors, page,
        keeper_colors=penaltyKeeperColors, xg_available=penaltyXgAvailable,
        expected_lineup=expectedLineup, expected_subs=expectedSubs)

    notesFields = [
        ('corner_notes', cornerNotesBbox),
        ('def_corner_notes', defCornerNotesBbox),
        ('penalty_notes', penaltyNotesBbox),
        ('def_penalty_notes', defPenaltyNotesBbox)
    ]
    notesFields = [(name, bbox) for name, bbox in notesFields if bbox is not None]

    return page, notesFields

#%%
def moveExistingReportToTrash(path):
    """Move a report that is about to be replaced to the Trash. True if it is no longer at `path`.

    Overwriting was the previous behaviour and it is unrecoverable, which matters more here than
    it looks: pages two to four carry real AcroForm notes fields (see PDF_Form_Fields), so the
    copy being replaced can hold typed match notes that exist nowhere else. Regenerating a report
    for the same opponent and date — to pick up a lineup change, or a scrape that failed the
    first time — silently destroyed them.

    send2trash is preferred because it goes through the platform's own API rather than moving the
    file by hand. On macOS that is NSFileManager's trashItemAtURL:, which records the move so the
    file keeps its Finder "Put Back" entry and returns to this folder in one step.

    SUCCESS IS CONFIRMED BY THE SOURCE BEING GONE, not by send2trash returning quietly — it
    returns None either way — and not by looking for the file in the Trash, which is not something
    this process can rely on doing: macOS gates even listing ~/.Trash behind Full Disk Access, so
    os.listdir on it raises PermissionError from an ordinary Python run. Checking the path we
    started from needs no such permission and is the thing actually being asserted.

    For the same reason no destination is returned. The OS de-duplicates a name already in the
    Trash, so any path built here would be a guess, and printing a guessed path to a user trying
    to recover a file is worse than telling them plainly that it is in the Trash.

    The manual fallback exists so a machine without the package still gets a recoverable copy
    instead of quietly reverting to a destructive overwrite. It lands the file in the same Trash
    but registers nothing, so it has no Put Back and has to be dragged out. It de-duplicates its
    own name the way Finder does, by appending the time: two reports trashed for one fixture would
    otherwise have the second overwrite the first, which is the exact loss this exists to prevent.
    """
    if not os.path.exists(path):
        return False

    try:
        from send2trash import send2trash
        send2trash(path)
    except Exception:
        pass
    if not os.path.exists(path):
        return True

    trashDir = os.path.expanduser('~/.Trash')
    if not os.path.isdir(trashDir):
        return False

    stem, extension = os.path.splitext(os.path.basename(path))
    destination = os.path.join(trashDir, stem + extension)
    # os.path.exists on a known path is allowed where listing the directory is not, so this
    # collision check still works under the restriction described above.
    if os.path.exists(destination):
        destination = os.path.join(
            trashDir, f'{stem} {datetime.now().strftime("%H.%M.%S")}{extension}')
    try:
        shutil.move(path, destination)
    except Exception:
        return False
    return not os.path.exists(path)
#%%
def main(leagues=['MLS', 'Leagues_Cup', 'US_Open_Cup', 'CONCACAF_Champions_Cup'], team='Columbus_Crew', opposition='Inter_Miami_CF', date='9-27-2026', formationOverride=False):
                #'Liga_MX,_Apertura', 
#def main(leagues=['NCAA'], team='Ohio_State_Buckeyes', opposition='Washington_Huskies', date='9-21-2026', formationOverride=False):
    # Note: first value of leagues must be the opposition's primary league
    # formationOverride: pass a boolean that determines whether the deduced formation should be used or the user should input it
    # Note: date format is 'M-D-YYYY'
    
    
    matches = loadMatchData(leagues, team, date)

    targetMatch = scrapeTargetMatchData(leagues, opposition, date, team=team)
    if targetMatch is None:
        print(f'Could not retrieve match data for {opposition} on {date}. Aborting report.')
        return

    outputDir = '/Users/jakeholfinger/Desktop/CC Analyst/Prematch Reports'
    os.makedirs(outputDir, exist_ok=True)
    outputPath = os.path.join(outputDir, f'Pre_Match_Report_{opposition}_{date}.pdf')
    # BEFORE PdfPages, which truncates the path the moment it opens it. Anything already there is
    # a previous run for this same fixture, and it goes to the Trash rather than being written
    # over — see moveExistingReportToTrash for why that is worth doing.
    if os.path.exists(outputPath):
        if moveExistingReportToTrash(outputPath):
            print(f'Existing report moved to the Trash: {os.path.basename(outputPath)}')
        else:
            print(f'WARNING: could not move the existing report to the Trash — '
                  f'{outputPath} is about to be OVERWRITTEN and any notes in it lost.')
    with PdfPages(outputPath) as pdf:
        pageOne, expectedLineup, expectedSubs = generateFirstPage(leagues, team, opposition, date, matches, targetMatch, formationOverride=formationOverride)
        pdf.savefig(pageOne)
        plt.close(pageOne)

        pageTwo, notesFieldsTwo = generateSecondPage(leagues, team, opposition, targetMatch, date)
        pdf.savefig(pageTwo)
        plt.close(pageTwo)

        pageThree, notesFieldsThree = generateThirdPage(leagues, team, opposition, targetMatch, date,
                                                        expectedLineup=expectedLineup, expectedSubs=expectedSubs)
        pdf.savefig(pageThree)
        plt.close(pageThree)

        pageFour, notesFieldsFour = generateFourthPage(leagues, team, opposition, targetMatch, date,
                                                       expectedLineup=expectedLineup, expectedSubs=expectedSubs)
        pdf.savefig(pageFour)
        plt.close(pageFour)

    # matplotlib/PdfPages only produces flattened, static PDF content — turning the
    # notes placeholder boxes into real clickable/editable fields requires a separate
    # post-processing pass over the finished file (see PDF_Form_Fields.py). Each page's
    # fields are applied in their own call, since add_fillable_text_fields only targets
    # one page_index at a time — page two's fields would otherwise get silently dropped
    # (and page three's misapplied onto page two) if both were collapsed into one
    # notesFields variable.
    import PDF_Form_Fields
    if notesFieldsTwo:
        PDF_Form_Fields.add_fillable_text_fields(outputPath, page_index=1, fields=notesFieldsTwo)
    if notesFieldsThree:
        PDF_Form_Fields.add_fillable_text_fields(outputPath, page_index=2, fields=notesFieldsThree)
    if notesFieldsFour:
        PDF_Form_Fields.add_fillable_text_fields(outputPath, page_index=3, fields=notesFieldsFour)

    print(f'Report saved to {outputPath}')


    return

if __name__ == "__main__":
    main()
