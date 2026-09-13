#!/usr/bin/env python3
"""
diagnose_ici_prnewswire_mirror.py  v1.2
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

v1.1 UPDATE (this run's live evidence): StreetInsider is confirmed
BLOCKED from the real GitHub Actions runner IP (HTTP 403, Cloudflare
challenge-platform markers) -- the same failure class already
documented for ici.org itself, just a different CDN (Cloudflare, not
Akamai) applying the same datacenter-IP-reputation logic. MarketsMedia,
by contrast, is CONFIRMED REACHABLE from the same runner IP (HTTP 200,
zero block-signature hits, all three real-content markers present).
Since the only article tested there was a known, hardcoded, old URL,
this version adds THREE independent discovery probes -- MarketsMedia
runs WordPress (confirmed via its own response headers: `X-Powered-By`,
`X-Redirect-By: WordPress`) -- to find out whether the CURRENT week's
release URL can be discovered automatically, without which this source
cannot be wired into a real weekly fetcher regardless of how clean the
reachability check is:
  (a) the default WordPress REST API search endpoint
      (`/wp-json/wp/v2/posts?search=...`), which most WordPress sites
      expose without any auth;
  (b) the default WordPress site-wide RSS feed (`/feed/`), filtered for
      ICI-titled items;
  (c) `sitemap.xml` (or a post-type sub-sitemap linked from it), as a
      fallback if neither of the above is exposed.
None of these is assumed to work -- each is checked independently.

v1.2 UPDATE: adds a third, previously-untested distribution channel --
PRNewswire's OWN site, not just the two aggregators that redistribute
it. ICI's release is issued as a genuine PRNewswire wire release (the
"/PRNewswire/" byline is on the release text itself), which means
PRNewswire's own article pages are the original syndication point, not
a third-party mirror -- worth testing independently since a block
confirmed on an aggregator (StreetInsider, Cloudflare) says nothing
about whether PRNewswire's own domain carries the same or a different
WAF posture. Two probes added:
  (d) direct reachability of a known PRNewswire article URL (same
      block-signature / content-marker method as every other probe in
      this script -- no assumption of success);
  (e) PRNewswire's own all-releases RSS feed
      (`/rss/news-releases-list.rss`) -- NOTE this is the firehose feed
      (every release from every company PRNewswire distributes, not
      scoped to ICI), so it is tested purely for domain-level
      reachability from this runner's IP, not as a discovery mechanism;
      no per-organization PRNewswire feed for ICI was found to exist,
      so this script does not claim one and does not fabricate a URL
      for it.
Also extends the MarketsMedia sitemap probe (c) from v1.1: rather than
reporting sitemap.xml's own top-level structure only, this version
descends into the highest-numbered post-sitemap (empirically the most
recent one, per v1.1's live run showing post-sitemap1.xml..20.xml in
ascending order) and greps its entries for "ici-reports" slugs, since
the wp-json search endpoint's own relevance ranking proved too weak in
v1.1's live run (returned two multi-year-old, topically-unrelated
articles instead of the current week's release).

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

SCRIPT_VERSION = "1.2"

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

# Known real PRNewswire article URLs for this exact release family, found via
# web search of PRNewswire's own domain -- used only as reachability probes
# (this script does not claim these specific old releases are current).
PN_KNOWN_URLS = [
    "https://www.prnewswire.com/news-releases/ici-reports-estimated-long-term-mutual-fund-flows-302661462.html",
    "https://www.prnewswire.com/news-releases/ici-reports-estimated-etf-net-issuance-302654178.html",
]
PN_ALL_RELEASES_RSS = "https://www.prnewswire.com/rss/news-releases-list.rss"  # PRNewswire's firehose feed -- every company's releases, NOT scoped to ICI; tested for domain reachability only, not discovery
MM_SITEMAP_INDEX_URL = "https://www.marketsmedia.com/sitemap.xml"

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


MM_BASE = "https://www.marketsmedia.com"
MM_SEARCH_TERMS = ["ICI Reports Estimated Long-Term Mutual Fund Flows"]


def discover_latest_mm_url_wpjson():
    _report("[MarketsMedia discovery A] WordPress REST API search")
    url = f"{MM_BASE}/wp-json/wp/v2/posts"
    params = {"search": "ICI Estimated Long-Term Mutual Fund Flows", "per_page": 5, "orderby": "date", "order": "desc"}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")
        return None

    print(f"GET {resp.url}")
    print(f"HTTP status : {resp.status_code}")
    if resp.status_code != 200:
        print("wp-json endpoint not reachable/enabled with this status.")
        return None

    try:
        posts = resp.json()
    except Exception as e:
        print(f"Response was not valid JSON ({e}) -- wp-json likely disabled/rewritten.")
        return None

    if not isinstance(posts, list) or not posts:
        print("wp-json reachable but returned no matching posts.")
        return None

    print(f"Posts returned: {len(posts)}")
    for p in posts:
        title = (p.get("title") or {}).get("rendered", "")
        link = p.get("link")
        date = p.get("date")
        print(f"  - [{date}] {title!r} -> {link}")

    top = posts[0]
    link = top.get("link")
    title = (top.get("title") or {}).get("rendered", "")
    if link and "mutual fund flow" in title.lower():
        print(f"SELECTED: {link}")
        return link
    print("Top result did not look like an ICI mutual-fund-flows release -- not selecting it.")
    return None


def discover_latest_mm_url_rss():
    _report("[MarketsMedia discovery B] default WordPress /feed/")
    url = f"{MM_BASE}/feed/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")
        return None

    print(f"HTTP status : {resp.status_code}")
    if resp.status_code != 200:
        print("Site-wide feed not reachable with this status.")
        return None

    items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL | re.IGNORECASE)
    print(f"RSS items found: {len(items)}")
    for item in items:
        title_m = re.search(r"<title>(.*?)</title>", item, re.DOTALL | re.IGNORECASE)
        link_m = re.search(r"<link>(.*?)</link>", item, re.DOTALL | re.IGNORECASE)
        if not title_m or not link_m:
            continue
        title = title_m.group(1).strip()
        link = link_m.group(1).strip()
        if "mutual fund flow" in title.lower() and "ici" in title.lower():
            print(f"Matched RSS item -> title={title!r} link={link}")
            return link

    print("No ICI mutual-fund-flows item in the current site-wide feed window "
          "(expected outside release weeks -- this feed is recency-windowed, "
          "not a permanent archive, and may also simply be diluted by MarketsMedia's "
          "other daily content).")
    return None


def discover_latest_mm_url_sitemap():
    _report("[MarketsMedia discovery C] sitemap.xml -> highest-numbered post-sitemap")
    try:
        resp = requests.get(MM_SITEMAP_INDEX_URL, headers=HEADERS, timeout=TIMEOUT)
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")
        return None

    print(f"HTTP status : {resp.status_code}")
    if resp.status_code != 200:
        print("sitemap.xml not reachable with this status.")
        return None

    sub_sitemaps = re.findall(r"<loc>(.*?)</loc>", resp.text)
    print(f"Top-level sitemap.xml entries: {len(sub_sitemaps)}")

    post_sitemaps = [s for s in sub_sitemaps if re.search(r"post-sitemap\d+\.xml", s)]
    if not post_sitemaps:
        print("No post-sitemapN.xml entries found -- cannot descend. "
              "Structure only:")
        for s in sub_sitemaps[:20]:
            print(f"  - {s}")
        return None

    def _num(u):
        m = re.search(r"post-sitemap(\d+)\.xml", u)
        return int(m.group(1)) if m else -1

    post_sitemaps.sort(key=_num, reverse=True)
    latest_sitemap_url = post_sitemaps[0]
    print(f"{len(post_sitemaps)} post-sitemaps found. Descending into the "
          f"highest-numbered one (assumed most recent, per v1.1's observed "
          f"ascending numbering): {latest_sitemap_url}")

    try:
        resp2 = requests.get(latest_sitemap_url, headers=HEADERS, timeout=TIMEOUT)
    except Exception as e:
        print(f"REQUEST FAILED fetching {latest_sitemap_url}: {type(e).__name__}: {e}")
        return None

    print(f"HTTP status (post-sitemap) : {resp2.status_code}")
    if resp2.status_code != 200:
        print("Highest-numbered post-sitemap not reachable with this status.")
        return None

    urls = re.findall(r"<loc>(.*?)</loc>", resp2.text)
    print(f"URLs in this post-sitemap: {len(urls)}")

    ici_urls = [u for u in urls if "ici-reports" in u.lower()]
    print(f"URLs matching 'ici-reports': {len(ici_urls)}")
    for u in ici_urls:
        print(f"  - {u}")

    if not ici_urls:
        print("No 'ici-reports' slug found in the most recent post-sitemap "
              "-- either MarketsMedia's slug convention differs from what "
              "was assumed, or this sitemap's date range doesn't cover the "
              "latest release yet (sitemaps can lag a live publish).")
        return None

    # Sitemaps list <loc> entries; without a reliable <lastmod> per URL in
    # every case, take the last one in document order as the probable most
    # recent (WordPress sitemap generators typically emit in publish order).
    selected = ici_urls[-1]
    print(f"SELECTED (last 'ici-reports' entry in document order): {selected}")
    return selected


def check_prnewswire_direct():
    _report("[PRNewswire direct] known-article reachability probes")
    results = []
    for url in PN_KNOWN_URLS:
        results.append(check_url(url, "PRNewswire (known article)"))
    return results


def check_prnewswire_rss_domain_reachability():
    _report("[PRNewswire RSS] domain-reachability probe only (firehose feed, not ICI-scoped)")
    try:
        resp = requests.get(PN_ALL_RELEASES_RSS, headers=HEADERS, timeout=TIMEOUT)
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")
        return {"label": "PRNewswire RSS (firehose)", "ok": False, "reason": f"exception:{type(e).__name__}"}

    body_lower = (resp.text or "").lower()
    hits_block = _scan(body_lower, BLOCK_MARKERS)
    item_count = len(re.findall(r"<item>", resp.text, re.IGNORECASE))

    print(f"HTTP status : {resp.status_code}")
    print(f"Block-signature hits : {hits_block or 'none'}")
    print(f"<item> entries found : {item_count}")
    ok = resp.status_code == 200 and not hits_block and item_count > 0
    print(f"VERDICT: {'REACHABLE (domain-level only -- this feed is NOT ICI-scoped)' if ok else 'NOT CONFIRMED'}")
    return {"label": "PRNewswire RSS (firehose)", "ok": ok, "status": resp.status_code, "block_hits": hits_block}


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

    # 3. MarketsMedia reachability (confirmed clean in v1.0's run --
    #    re-checked here for a fresh timestamped log alongside discovery).
    results.append(check_url(MM_FALLBACK_URL, "MarketsMedia (known article)"))

    # 4. MarketsMedia auto-discovery of the CURRENT week's release --
    #    the actual remaining blocker before this source can be wired
    #    into a real weekly fetcher, since v1.0 only proved reachability
    #    against a hardcoded old URL.
    mm_wpjson_url = discover_latest_mm_url_wpjson()
    mm_rss_url = discover_latest_mm_url_rss()
    discover_latest_mm_url_sitemap()  # structure-only report, no URL returned

    if mm_wpjson_url:
        results.append(check_url(mm_wpjson_url, "MarketsMedia (wp-json discovered)"))
    if mm_rss_url and mm_rss_url != mm_wpjson_url:
        results.append(check_url(mm_rss_url, "MarketsMedia (RSS discovered)"))

    mm_sitemap_url = discover_latest_mm_url_sitemap()
    if mm_sitemap_url and mm_sitemap_url not in (mm_wpjson_url, mm_rss_url):
        results.append(check_url(mm_sitemap_url, "MarketsMedia (sitemap discovered)"))

    # 5. PRNewswire's own site -- the original syndication point, not an
    #    aggregator/mirror. Never tested before this version.
    results.extend(check_prnewswire_direct())
    pn_rss_result = check_prnewswire_rss_domain_reachability()

    _report("SUMMARY")
    for r in results:
        status = "OK" if r.get("ok") else "NOT CONFIRMED"
        print(f"- {r['label']}: {status} (status={r.get('status')}, "
              f"block_hits={r.get('block_hits')}, content_hits={r.get('content_hits')})")

    print(f"\nStreetInsider RSS auto-discovery of the CURRENT week's release: "
          f"{'WORKED' if discovered_url else 'no match this run (see note above)'}")
    print(f"MarketsMedia wp-json auto-discovery: "
          f"{'WORKED -> ' + mm_wpjson_url if mm_wpjson_url else 'no match this run'}")
    print(f"MarketsMedia RSS auto-discovery: "
          f"{'WORKED -> ' + mm_rss_url if mm_rss_url else 'no match this run'}")
    print(f"MarketsMedia sitemap auto-discovery: "
          f"{'WORKED -> ' + mm_sitemap_url if mm_sitemap_url else 'no match this run'}")

    si_ok = any(r.get("ok") for r in results if "StreetInsider" in r["label"])
    mm_ok = any(r.get("ok") for r in results if "MarketsMedia" in r["label"])
    pn_direct_ok = any(r.get("ok") for r in results if "PRNewswire (known article)" in r["label"])
    mm_discovery_worked = bool(mm_wpjson_url or mm_rss_url or mm_sitemap_url)

    print(f"\nOVERALL — StreetInsider: "
          f"{'VIABLE from this runner IP' if si_ok else 'NOT CONFIRMED from this runner IP'}")
    print(f"OVERALL — MarketsMedia:  "
          f"{'VIABLE from this runner IP' if mm_ok else 'NOT CONFIRMED from this runner IP'}"
          f"{' -- but still needs a working CURRENT-week discovery method before it can be wired into production' if mm_ok and not mm_discovery_worked else ''}"
          f"{' -- AND a discovery method now confirmed working' if mm_ok and mm_discovery_worked else ''}")
    print(f"OVERALL — PRNewswire direct: "
          f"{'VIABLE from this runner IP (article pages)' if pn_direct_ok else 'NOT CONFIRMED from this runner IP'}"
          f"; firehose RSS domain reachability: "
          f"{'OK' if pn_rss_result.get('ok') else 'NOT CONFIRMED'} "
          f"(this feed cannot discover ICI's specific release -- it is not "
          f"organization-scoped; no ICI-specific PRNewswire feed was found "
          f"to exist)")
    print("This report is the evidence, not a guess.")

    sys.exit(0)


if __name__ == "__main__":
    main()
