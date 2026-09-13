#!/usr/bin/env python3
"""
diagnose_ici_prnewswire_mirror.py  v1.0
=============================================================
DIAGNOSTIC ONLY -- not wired into any production data path.

Purpose: ICI's own site (ici.org) is confirmed permanently blocked from
GitHub Actions/Cloudflare Workers runner IPs (Akamai bot-management --
see GUIDELINES.md's "Confirmed blocked from GH Actions/CF Workers"
list). ICI's weekly "Estimated Long-Term Mutual Fund Flows" release is
distributed publicly via PRNewswire and re-published verbatim, with the
full data table intact, by financial news aggregators. This script
checks, from the REAL GitHub Actions runner IP (not a residential IP --
a manual curl test from a residential connection already came back
clean and is not evidence about this runner's own IP reputation), two
independent things:

  1. Whether StreetInsider's own PRNewswire category feed (an RSS feed
     advertised in the page's own <head>, not guessed) reliably surfaces
     the latest ICI weekly release URL without needing to hardcode or
     re-discover an article ID by hand every week.
  2. Whether fetching that live article URL returns the real data table
     (verified by matching known real-content markers -- the PRNewswire
     syndication markup itself: class="prnbcc", <chron>/<money> tags,
     and the literal phrase "Estimated Flows") or a bot-challenge page
     (verified by matching known block-signature markers), from THIS
     runner's actual IP -- per the standing "a WAF/bot-block confirmed
     from one IP class is not evidence about a different IP class"
     discipline (GUIDELINES.md).

Same secondary probe run against MarketsMedia's mirror of the same
release, since the earlier manual test there was inconclusive (a plain
curl without redirect-following only saw the 301 wrapper, never the
real page).

What this script does NOT do:
  - It does not attempt any stealth/evasion technique (no
    playwright-stealth, no fingerprint spoofing, no proxy rotation).
    Per the standing "no code path defeats an anti-bot challenge on a
    site's own protected assets" discipline (see the RBNZ Cloudflare
    Turnstile / Myfxbook login precedents in GUIDELINES.md), this
    script only checks whether a PLAIN request with a realistic but
    honest browser User-Agent succeeds against a page these
    aggregators already redistribute for public consumption -- it is
    not trying to defeat a protection ICI itself put up on ici.org.
  - It does not write to any committed data file. Zero production
    impact. It only prints a structured, human-readable report to the
    job log.
=============================================================
"""

import re
import sys
import json
from datetime import datetime, timezone

import requests

SCRIPT_VERSION = "1.0"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 20

SI_RSS_URL = "https://www.streetinsider.com/freefeed.php?cid=81"  # PRNewswire category feed, advertised in the page's own <link rel="alternate"> -- not guessed
MM_FALLBACK_URL = "https://www.marketsmedia.com/ici-reports-estimated-long-term-mutual-fund-flows-12"  # known-old article, used only as a reachability probe since MarketsMedia has no discovered feed yet

BLOCK_MARKERS = [
    "just a moment", "checking your browser", "cf-mitigated", "captcha",
    "access denied", "forbidden", "__cf_chl", "challenge-platform",
    "attention required", "please enable javascript and cookies",
    "akamai", "bot detection", "verify you are human",
]
CONTENT_MARKERS_ICI = [
    "estimated flows", "prnbcc", "investment company institute",
    "long-term mutual fund",
]


def _report(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _scan(html_lower, markers):
    return [m for m in markers if m in html_lower]


def check_url(url, label):
    _report(f"[{label}] GET {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")
        return {"label": label, "url": url, "ok": False, "reason": f"exception:{type(e).__name__}"}

    body = resp.text or ""
    body_lower = body.lower()
    hits_block = _scan(body_lower, BLOCK_MARKERS)
    hits_content = _scan(body_lower, CONTENT_MARKERS_ICI)

    print(f"Final URL after redirects : {resp.url}")
    print(f"HTTP status               : {resp.status_code}")
    print(f"Body length (bytes)       : {len(body)}")
    print(f"Block-signature hits      : {hits_block or 'none'}")
    print(f"Real-content-marker hits  : {hits_content or 'none'}")

    ok = resp.status_code == 200 and not hits_block and bool(hits_content)
    verdict = "REACHABLE — real content, no block signature" if ok else "NOT CONFIRMED — see hits above"
    print(f"VERDICT: {verdict}")

    return {
        "label": label, "url": url, "final_url": resp.url,
        "status": resp.status_code, "body_len": len(body),
        "block_hits": hits_block, "content_hits": hits_content, "ok": ok,
    }


def discover_latest_ici_url_from_si_rss():
    _report(f"[StreetInsider RSS] GET {SI_RSS_URL}")
    try:
        resp = requests.get(SI_RSS_URL, headers=HEADERS, timeout=TIMEOUT)
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")
        return None

    print(f"HTTP status : {resp.status_code}")
    if resp.status_code != 200:
        print("RSS feed did not return 200 -- cannot discover latest ICI URL this way.")
        return None

    # Look for <item> entries whose <title> mentions ICI's known release title,
    # without assuming any fixed item ordering or exact phrasing.
    items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL | re.IGNORECASE)
    print(f"RSS items found: {len(items)}")
    for item in items:
        title_m = re.search(r"<title>(.*?)</title>", item, re.DOTALL | re.IGNORECASE)
        link_m = re.search(r"<link>(.*?)</link>", item, re.DOTALL | re.IGNORECASE)
        if not title_m or not link_m:
            continue
        title = title_m.group(1).strip()
        link = link_m.group(1).strip()
        if "mutual fund flows" in title.lower() and "ici" in title.lower():
            print(f"Matched RSS item -> title={title!r} link={link}")
            return link

    print("No ICI 'mutual fund flows' item found in the current RSS window "
          "(this is expected outside release weeks/days -- the feed is "
          "recency-windowed, not a permanent archive).")
    return None


def main():
    print(f"diagnose_ici_prnewswire_mirror.py v{SCRIPT_VERSION}")
    print(f"Run time (UTC): {datetime.now(timezone.utc).isoformat()}")

    results = []

    # 1. Can we discover the CURRENT week's ICI release automatically via RSS?
    discovered_url = discover_latest_ici_url_from_si_rss()

    # 2. Reachability check against a known real StreetInsider ICI article
    #    (used as the IP-reputation/bot-signature probe regardless of
    #    whether discovery above found this week's release yet).
    known_si_url = "https://www.streetinsider.com/PRNewswire/ICI+Reports+Estimated+Long-Term+Mutual+Fund+Flows/25848980.html"
    results.append(check_url(known_si_url, "StreetInsider (known article)"))

    if discovered_url and discovered_url != known_si_url:
        results.append(check_url(discovered_url, "StreetInsider (RSS-discovered, current)"))

    # 3. MarketsMedia reachability (inconclusive in the manual test --
    #    that test never followed the 301; this one does).
    results.append(check_url(MM_FALLBACK_URL, "MarketsMedia (known article)"))

    _report("SUMMARY")
    for r in results:
        status = "OK" if r.get("ok") else "NOT CONFIRMED"
        print(f"- {r['label']}: {status} (status={r.get('status')}, "
              f"block_hits={r.get('block_hits')}, content_hits={r.get('content_hits')})")

    print(f"\nRSS auto-discovery of the CURRENT week's release: "
          f"{'WORKED' if discovered_url else 'no match this run (see note above)'}")

    all_ok = any(r.get("ok") for r in results if "StreetInsider" in r["label"])
    print(f"\nOVERALL: StreetInsider mirror path is "
          f"{'VIABLE from this runner IP' if all_ok else 'NOT CONFIRMED from this runner IP'} "
          f"-- this report is the evidence, not a guess.")

    sys.exit(0)


if __name__ == "__main__":
    main()
