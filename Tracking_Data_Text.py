from curl_cffi import requests
import pandas as pd
import time
import Request_Utils

# %%
def safe_request(url, headers, retries=3, delay=2):
    """Delegates to the shared helper so this script picks up the negotiated fingerprint too.

    Note this module talks to Sportradar, not SofaScore, and Sportradar gates on the signed
    token in the URL rather than on Cloudflare Turnstile — so the fingerprint has never been the
    thing blocking it here. It shares the helper anyway so that if Sportradar does start
    fingerprint-scoring, the fallback is already in place. The negotiation probe is a SofaScore
    endpoint; if that ever proves to be the wrong proxy for this host, pass an explicit
    `impersonate=` here instead.
    """
    return Request_Utils.safe_request(url, headers, retries=retries, delay=delay)
# %%
#Token from the SofaScore page request. acl=/* means it is not path-bound,
#so the same token works for every gismo endpoint. exp is a unix timestamp -
#once it passes, grab a fresh URL from the browser network tab.
TOKEN = 'T=exp=1789072135~acl=/*~data=eyJvIjoiaHR0cHM6Ly93d3cuc29mYXNjb3JlLmNvbSIsImEiOiJhY2Y1MWVkZWVlYzEyNjQzMjcwN2MwYmYwNzY3M2Q4NiIsImFjdCI6Im9yaWdpbmNoZWNrIiwib3NyYyI6Im9yaWdpbiJ9~hmac=fd2caaf58c47e34a4eb52ccbba29c55804a5867bccfec3092d2dd2ace92fca74'

# %%
def fetch_events(match_id, headers, endpoint='match_timeline'):
    """Return the event list for a match.

    'match_timeline'      -> every event of the match so far (what you want)
    'match_timelinedelta' -> only the handful of events since the last poll
    """
    api_url = f'https://lmt.fn.sportradar.com/common/en/Etc:UTC/gismo/{endpoint}/{match_id}?{TOKEN}'

    response = safe_request(api_url, headers)
    if response is None:
        return None

    try:
        data = response.json()
    except Exception:
        print('Error converting raw data to JSON file')
        return None

    docs = data.get('doc') or []
    if not docs:
        return None

    return docs[0].get('data', {}).get('events') or []
# %%
def events_to_frame(events):
    """Flatten the event dicts into one row per event.

    The feed already arrives in match order - 'seconds' is monotonic across the
    events that carry a real clock, and the ones marked -1 (kickoff markers,
    period boundaries, ballcoordinates) sit in their correct slots. Do not sort:
    'uts' is not monotonic, and sorting on 'seconds' strands the -1 rows.
    """
    df = pd.json_normalize(events)

    #Keep the feed order recoverable after a CSV round-trip.
    df.insert(0, 'event_index', range(len(df)))

    return df
# %%
def poll_live(match_id, headers, duration=90 * 60, interval=10):
    """Accumulate the live-only event types by repeatedly polling the delta feed.

    'match_timeline' keeps the full history of discrete events (shots, corners,
    throw-ins...) but only ONE current sample of 'ballcoordinates', 'possession'
    and 'matchsituation'. Those three are state snapshots, not history - the feed
    overwrites them, so the ball-position trail only exists if you capture it as
    the match happens. Run this alongside the match; it cannot be backfilled.

    Dedupes on '_id', so overlapping polls are harmless.
    """
    seen = {}
    deadline = time.time() + duration

    while time.time() < deadline:
        events = fetch_events(match_id, headers, endpoint='match_timelinedelta')

        if events:
            new = [e for e in events if e.get('_id') not in seen]
            for event in new:
                seen[event['_id']] = event
            if new:
                print(f'+{len(new):2d} new ({len(seen)} total)')

        time.sleep(interval)

    return list(seen.values())
# %%
def main():
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


    match_id = 74165884

    events = fetch_events(match_id, headers)

    if not events:
        return 'No data'

    df = events_to_frame(events)

    print(f'{len(df)} events, minutes {df["time"].max()} covered')
    print('Dataframe:')
    print(df)

    file_path = '/Users/jakeholfinger/Desktop/Tracking_Data_Text.csv'

    df.to_csv(file_path, index=False)

    return df.to_string()

if __name__ == "__main__":
    main()


'https://lmt.fn.sportradar.com/common/en/Etc:UTC/gismo/match_timelinedelta/74165884?T=exp=1789072135~acl=/*~data=eyJvIjoiaHR0cHM6Ly93d3cuc29mYXNjb3JlLmNvbSIsImEiOiJhY2Y1MWVkZWVlYzEyNjQzMjcwN2MwYmYwNzY3M2Q4NiIsImFjdCI6Im9yaWdpbmNoZWNrIiwib3NyYyI6Im9yaWdpbiJ9~hmac=fd2caaf58c47e34a4eb52ccbba29c55804a5867bccfec3092d2dd2ace92fca74',
'https://lmt.fn.sportradar.com/common/en/Etc:UTC/gismo/match_timelinedelta/74165884?T=exp=1789072135~acl=/*~data=eyJvIjoiaHR0cHM6Ly93d3cuc29mYXNjb3JlLmNvbSIsImEiOiJhY2Y1MWVkZWVlYzEyNjQzMjcwN2MwYmYwNzY3M2Q4NiIsImFjdCI6Im9yaWdpbmNoZWNrIiwib3NyYyI6Im9yaWdpbiJ9~hmac=fd2caaf58c47e34a4eb52ccbba29c55804a5867bccfec3092d2dd2ace92fca74',


[{'_doc': 'event', '_doctype': 'possibleevent', '_id': 'e5530ef1-6b13-4396-9421-d3d95e2a4616', '_scoutid': None, '_sid': 1, '_rcid': 393, '_tid': 151665, '_dc': False, '_typeid': '1016', 'uts': 1788986186, 'updated_uts': 1788986188, 'type': 'possible_event', 'matchid': 74165884, 'disabled': 0, 'time': 78, 'seconds': 4652, 'injurytime': 0, 'team': 'home', 'name': 'Possible corner'}, {'_doc': 'event', '_doctype': 'matchsituation', '_id': '35e6e1b0-3dd1-4560-bdb2-d603cf188110', '_scoutid': None, '_sid': 1, '_rcid': 393, '_tid': 151665, '_dc': False, '_typeid': '1103', 'uts': 1788986188, 'updated_uts': 1788986188, 'type': 'matchsituation', 'matchid': 74165884, 'disabled': 0, 'time': 78, 'seconds': 4653, 'injurytime': 0, 'X': 100, 'Y': 0, 'team': 'home', 'name': 'matchsituation', 'situation': 'dangerous'}, {'_doc': 'event', '_doctype': 'ballcoordinates', '_id': '35e6e1b0-3dd1-4560-bdb2-d603cf188110-1140', '_scoutid': None, '_sid': 1, '_rcid': 393, '_tid': 151665, '_dc': False, '_typeid': '1140', 'uts': 1788986188, 'updated_uts': 1788986188, 'type': 'ballcoordinates', 'matchid': 74165884, 'disabled': 0, 'time': -1, 'seconds': -1, 'injurytime': 0, 'team': '', 'name': 'Pitch coordinates', 'coordinates': [{'team': 'home', 'X': 100, 'Y': 0}, {'team': 'home', 'X': 92, 'Y': 39}, {'team': 'home', 'X': 66, 'Y': 59}, {'team': 'away', 'X': 66, 'Y': 59}]}]
[{'_doc': 'event', '_doctype': 'corner', '_id': '155aaad1-742e-4624-a18b-b033cdde7273', '_scoutid': None, '_sid': 1, '_rcid': 393, '_tid': 151665, '_dc': False, '_typeid': '154', 'uts': 1788986219, 'updated_uts': 1788986219, 'type': 'corner', 'matchid': 74165884, 'disabled': 0, 'time': 79, 'seconds': 4685, 'injurytime': 0, 'X': 100, 'Y': 0, 'team': 'home', 'name': 'Corner kick', 'side': 'left', 'sideid': 'Left', 'player': {'_doc': 'player', '_id': 1064588, 'name': 'Szoboszlai, Dominik', 'birthdate': {'_doc': 'time', 'time': '00:00', 'date': '25/10/00', 'tz': 'UTC', 'tzoffset': 0, 'uts': 972432000}, 'nationality': {'_doc': 'countrycode', '_id': 97, 'a2': 'hu', 'name': 'Hungary', 'a3': 'HUN', 'ioc': 'HUN', 'continentid': 1, 'continent': 'Europe', 'population': 9797000}, 'position': {'_id': '6', '_type': 'M', 'name': 'Midfielder', 'shortname': 'MID', 'abbr': 'M'}, 'primarypositiontype': None, 'haslogo': False}}, {'_doc': 'event', '_doctype': 'throwin', '_id': '49efd246-24d4-4806-8936-55eda9ac6086', '_scoutid': None, '_sid': 1, '_rcid': 393, '_tid': 151665, '_dc': False, '_typeid': '152', 'uts': 1788986226, 'updated_uts': 1788986226, 'type': 'throwin', 'matchid': 74165884, 'disabled': 0, 'time': 79, 'seconds': 4692, 'injurytime': 0, 'X': 79, 'Y': 0, 'team': 'home', 'name': 'Throw-in', 'player': {'name': ''}}, {'_doc': 'event', '_doctype': 'matchsituation', '_id': '87208763-84f2-42e4-9b42-e6c42c57c500', '_scoutid': None, '_sid': 1, '_rcid': 393, '_tid': 151665, '_dc': False, '_typeid': '1103', 'uts': 1788986231, 'updated_uts': 1788986231, 'type': 'matchsituation', 'matchid': 74165884, 'disabled': 0, 'time': 79, 'seconds': 4697, 'injurytime': 0, 'X': 84, 'Y': 5, 'team': 'home', 'name': 'matchsituation', 'situation': 'dangerous'}, {'_doc': 'event', '_doctype': 'ballcoordinates', '_id': '87208763-84f2-42e4-9b42-e6c42c57c500-1140', '_scoutid': None, '_sid': 1, '_rcid': 393, '_tid': 151665, '_dc': False, '_typeid': '1140', 'uts': 1788986231, 'updated_uts': 1788986231, 'type': 'ballcoordinates', 'matchid': 74165884, 'disabled': 0, 'time': -1, 'seconds': -1, 'injurytime': 0, 'team': '', 'name': 'Pitch coordinates', 'coordinates': [{'team': 'home', 'X': 84, 'Y': 5}, {'team': 'home', 'X': 79, 'Y': 0}, {'team': 'home', 'X': 84, 'Y': 4}, {'team': 'home', 'X': 85, 'Y': 19}]}]