# %%
"""Shared request plumbing for every script that talks to SofaScore.

The one thing worth centralising is the TLS/HTTP2 fingerprint curl_cffi presents, because that
is what decides whether SofaScore answers at all. Their edge challenges clients it scores as
bots, replying {'error': {'code': 403, 'reason': 'challenge'}} on every endpoint, and which
fingerprints it scores changes over time — so the profile is negotiated at runtime rather than
hardcoded in five different files.

Call sites stay as they are and just ask for the profile:

    from curl_cffi import requests
    import Request_Utils
    requests.get(url, headers=headers, impersonate=Request_Utils.CurrentProfile(headers))

CurrentProfile negotiates once per process on first use and caches the answer, so scripts that
never make a request pay nothing and scripts that make thousands pay for one negotiation.
"""

from curl_cffi import requests
import threading
import time

#Fingerprints to try, in preference order. Chrome first because it is the honest description of
#these scripts. Measured 2026-09-27: chrome mostly loses — chrome142/136/120 and the 'chrome'
#alias were challenged on most attempts, while safari18_4/18_0 answered 200 on every attempt,
#from the same machine, network and headers. Chrome impersonation is the common scraping
#signature, so it is what their edge scores against. It is intermittent rather than absolute
#(chrome142 answered 200, then 403, then 200 within ten minutes), which is why this is a
#negotiated list and why NegotiateImpersonation insists on repeated successes before committing.
IMPERSONATE_PROFILES = ['chrome142', 'chrome120', 'safari18_4', 'safari18_0', 'firefox144']

#The profile currently in use. Read it through CurrentProfile() or as Request_Utils.IMPERSONATE —
#never `from Request_Utils import IMPERSONATE`, which would copy the value and miss the update.
IMPERSONATE = IMPERSONATE_PROFILES[0]

#A cheap, always-present endpoint to probe against.
PROBE_URL = 'https://www.sofascore.com/api/v1/sport/football/events/live'

_negotiated = False
_negotiateLock = threading.Lock()


# %%
def safe_request(url, headers, retries=3, delay=2, impersonate=None):
    """GET with retries, using the negotiated fingerprint unless one is named."""
    profile = CurrentProfile(headers) if impersonate is None else impersonate

    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                impersonate=profile,
                timeout=10
            )
            return response
        except Exception as e:
            print(f"Request failed (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return None


# %%
def NegotiateImpersonation(headers, probeURL=PROBE_URL, confirmations=2, spacing=2.0):
    """Set IMPERSONATE to the first profile SofaScore answers `confirmations` times running.

    Returns the winning profile name, or None if every profile was refused.

    More than one probe per profile is required because the challenge is not a clean per-profile
    verdict — chrome142 was seen answering 200 and then 403 minutes apart. Accepting a single
    lucky 200 would pin a whole run to a fingerprint that starts failing once the real work
    begins, which is the exact failure this is meant to prevent.
    """
    global IMPERSONATE

    for index, profile in enumerate(IMPERSONATE_PROFILES):
        #Space the probes out. Unthrottled this walks the whole list in about five seconds, and
        #that burst is itself enough to trip the rate limiter — measured 2026-09-27, a lone
        #probe returned 200 while the same profile inside a back-to-back negotiation returned
        #403. A negotiation that provokes the block it is testing for is worse than useless.
        if index:
            time.sleep(spacing)

        codes = []
        for confirmation in range(confirmations):
            if confirmation:
                time.sleep(spacing)
            #Name the profile explicitly: asking CurrentProfile here would recurse.
            response = safe_request(probeURL, headers, retries=1, impersonate=profile)
            codes.append('none' if response is None else str(response.status_code))
            if response is None or response.status_code != 200:
                break

        if codes.count('200') == confirmations:
            IMPERSONATE = profile
            print(f'  {profile}: {"/".join(codes)} <- using this')
            return profile

        print(f'  {profile}: {"/".join(codes)}')

    return None


# %%
def CurrentProfile(headers=None):
    """Return the working fingerprint, negotiating once per process on first use.

    Falls back to the first profile in the list if negotiation fails outright, so a caller still
    gets a usable value and surfaces the failure through its own error handling rather than
    crashing here. The lock matters because SofaScore_Scraper negotiates from a ThreadPoolExecutor.
    """
    global _negotiated

    if _negotiated:
        return IMPERSONATE

    with _negotiateLock:
        #Re-check inside the lock: another thread may have negotiated while this one waited.
        if _negotiated:
            return IMPERSONATE

        print('Negotiating TLS fingerprint...')
        NegotiateImpersonation(headers or {}, PROBE_URL)
        _negotiated = True

    return IMPERSONATE


# %%
def CheckSiteAccess(headers, probeURL=PROBE_URL):
    """Find a working fingerprint up front, or explain why the API is unreachable.

    Worth calling explicitly from long-running scripts so they fail in a handful of requests
    rather than thousands of matches in. Short scripts can skip it and let CurrentProfile
    negotiate lazily on their first request.
    """
    global _negotiated

    print('Preflight: negotiating TLS fingerprint...')
    profile = NegotiateImpersonation(headers, probeURL)
    _negotiated = True

    if profile is not None:
        print(f'Preflight OK (impersonate={profile}).')
        return True

    print('Preflight failed: every fingerprint was refused.')
    print('  If the errors above are HTTP 403, this client is being challenged as a bot — the')
    print('  fix is a fingerprint their edge is not scoring, so try adding newer curl_cffi')
    print('  profiles to IMPERSONATE_PROFILES (safari17_0 and safari18_4_ios also worked once).')
    print('  If instead there was no response and the error mentions an expired certificate, a')
    print('  captive portal is intercepting TLS — open any http site, accept the portal, retry.')
    return False


# %%
def CheckedJSON(response):
    """Parse a SofaScore response, returning None if it is an error payload.

    A Cloudflare challenge answers with {'error': {'code': 403, 'reason': 'challenge'}}. That is
    valid JSON and still a dict, so without this it flows downstream untouched and only surfaces
    much later as a missing key, far from the request that actually went wrong.
    """
    if response is None:
        return None

    try:
        data = response.json()
    except Exception:
        print(f'Non-JSON response (HTTP {response.status_code}) from {response.url}')
        return None

    if isinstance(data, dict) and 'error' in data:
        error = data.get('error') or {}
        print(f'API error {error.get("code")} ({error.get("reason")}) from {response.url}')
        return None

    return data