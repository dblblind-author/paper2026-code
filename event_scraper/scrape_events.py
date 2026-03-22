#!/usr/bin/env python3
"""
Standalone raw event scraper for Milanotoday.

This script keeps only the scraping stages from the notebook:
1. discover event URLs from the listing pages with Playwright
2. fetch each event page and extract raw fields with requests + BeautifulSoup

The output JSON is intentionally shaped to be consumed by `elaborate_events.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

BASE_URL = "https://www.milanotoday.it"
DEFAULT_OUTPUT = "raw_events.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
        "Gecko/20100101 Firefox/124.0"
    )
}

BANNED_LOCATION = ["dove", "indirizzo non disponibile"]
BANNED_DATA_ORA = ["quando", "orario non disponibile"]
BANNED_PREZZO = ["prezzo"]

PATTERN_LOCATION = re.compile("|".join(re.escape(word) for word in BANNED_LOCATION), re.IGNORECASE)
PATTERN_DATA_ORA = re.compile("|".join(re.escape(word) for word in BANNED_DATA_ORA), re.IGNORECASE)
PATTERN_PREZZO = re.compile("|".join(re.escape(word) for word in BANNED_PREZZO), re.IGNORECASE)


def month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return (
        f"{year:04d}-{month:02d}-01",
        f"{year:04d}-{month:02d}-{last_day:02d}",
    )


async def scroll_to_bottom(page, pause_ms: int = 500, max_iters: int = 30) -> None:
    last_height = 0
    for _ in range(max_iters):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(pause_ms)
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


async def get_max_page(page) -> int:
    locator = page.locator(
        "a.u-label-04.o-link-text.u-none.u-block\\@sm, "
        "a.u-label-04.o-link-text.u-none.u.block\\@sm"
    )
    max_page = 1
    try:
        count = await locator.count()
        for idx in range(count):
            href = (await locator.nth(idx).get_attribute("href")) or ""
            match = re.search(r"/pag/(\d+)/", href)
            if match:
                max_page = max(max_page, int(match.group(1)))
    except Exception:
        return max_page
    return max_page


async def extract_events_from_page(page) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    cards = page.locator(".c-card__pull-down")
    count = await cards.count()

    for idx in range(count):
        card = cards.nth(idx)
        title = ""
        href = ""

        h2_link = card.locator("h2 a")
        if await h2_link.count() > 0:
            title = (await h2_link.first.inner_text() or "").strip()
            href = (await h2_link.first.get_attribute("href") or "").strip()
        else:
            h2 = card.locator("h2")
            if await h2.count() > 0:
                title = (await h2.first.inner_text() or "").strip()
            link = card.locator("a")
            if await link.count() > 0:
                href = (await link.first.get_attribute("href") or "").strip()

        if not title or not href:
            continue

        if href.startswith("/"):
            href = f"{BASE_URL}{href}"

        results.append({"title": title, "url": href})

    return results


async def scrape_event_links(
    years: Iterable[int],
    *,
    headless: bool,
    slow_mo: int,
    listing_pause_ms: int,
    navigation_timeout_ms: int,
) -> List[Dict[str, Any]]:
    seen_urls: set[str] = set()
    all_rows: List[Dict[str, Any]] = []

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless, slow_mo=slow_mo)
        context = await browser.new_context(
            viewport={"width": 1200, "height": 900},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        for year in years:
            for month in range(1, 13):
                month_start, month_end = month_bounds(year, month)
                base_url = f"{BASE_URL}/eventi/dal/{month_start}/al/{month_end}/"

                await page.goto(base_url, wait_until="domcontentloaded", timeout=navigation_timeout_ms)
                await scroll_to_bottom(page)
                try:
                    await page.wait_for_load_state("networkidle", timeout=3000)
                except PlaywrightTimeoutError:
                    pass

                max_page = await get_max_page(page)

                for page_num in range(1, max_page + 1):
                    url = base_url if page_num == 1 else f"{base_url}pag/{page_num}/"
                    await page.wait_for_timeout(listing_pause_ms)
                    await page.goto(url, wait_until="domcontentloaded", timeout=navigation_timeout_ms)
                    await scroll_to_bottom(page)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=3000)
                    except PlaywrightTimeoutError:
                        pass

                    rows = await extract_events_from_page(page)
                    for row in rows:
                        if row["url"] in seen_urls:
                            continue
                        seen_urls.add(row["url"])
                        row.update(
                            {
                                "year": year,
                                "month": month,
                                "month_start": month_start,
                                "month_end": month_end,
                                "page": page_num,
                                "source_page_url": url,
                            }
                        )
                        all_rows.append(row)

                    print(
                        f"[links] year={year} month={month:02d} page={page_num}/{max_page} "
                        f"total_urls={len(all_rows)}"
                    )

        await browser.close()

    return all_rows


def get_page_content(session: requests.Session, url: str, timeout_seconds: int) -> bytes:
    response = session.get(url, headers=HEADERS, timeout=timeout_seconds)
    response.raise_for_status()
    return response.content


def _extract_info_blocks(soup: BeautifulSoup) -> List[Any]:
    return soup.select(
        "div.l-grid__item.u-p-xxsmal, "
        "div.l-grid__item.u-p-xxsmall"
    )


def _clean_field(pattern: re.Pattern[str], value: str) -> str:
    cleaned = pattern.sub("", value).strip()
    return cleaned or "N/A"


def get_event_details(
    session: requests.Session,
    url: str,
    *,
    fallback_title: str = "",
    timeout_seconds: int,
) -> Dict[str, Any]:
    soup = BeautifulSoup(get_page_content(session, url, timeout_seconds), "html.parser")

    title_tag = soup.find("h1")
    titolo = title_tag.get_text(" ", strip=True) if title_tag else (fallback_title or "N/A")

    info_blocks = _extract_info_blocks(soup)
    first = info_blocks[0] if len(info_blocks) >= 1 else None
    second = info_blocks[1] if len(info_blocks) >= 2 else None
    third = info_blocks[2] if len(info_blocks) >= 3 else None

    first_text = first.get_text(" ", strip=True) if first else "N/A"
    second_text = second.get_text(" ", strip=True) if second else "N/A"
    third_text = third.get_text(" ", strip=True) if third else "N/A"

    paragraphs = soup.find_all("p")
    description = " ".join(p.get_text(" ", strip=True) for p in paragraphs) if paragraphs else "N/A"

    return {
        "titolo": titolo,
        "data_ora": _clean_field(PATTERN_DATA_ORA, second_text),
        "prezzo": _clean_field(PATTERN_PREZZO, third_text),
        "location": _clean_field(PATTERN_LOCATION, first_text),
        "event_info": description or "N/A",
        "url": url,
    }


def scrape_event_details(
    links: List[Dict[str, Any]],
    *,
    timeout_seconds: int,
) -> List[Dict[str, Any]]:
    session = requests.Session()
    events: List[Dict[str, Any]] = []

    try:
        for idx, entry in enumerate(links, start=1):
            url = entry["url"]
            try:
                event = get_event_details(
                    session,
                    url,
                    fallback_title=entry.get("title", ""),
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                print(f"[details] failed {idx}/{len(links)} url={url} error={exc}")
                continue

            events.append(event)
            print(f"[details] scraped {idx}/{len(links)} title={event['titolo']}")
    finally:
        session.close()

    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape raw Milanotoday events into the schema expected by elaborate_events.py."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024],
        help="Years to scrape from Milanotoday listings (default: 2023 2024)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--links-output",
        default=None,
        help="Optional path to also save the intermediate discovered event links JSON",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run Playwright in headed mode instead of headless mode",
    )
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=200,
        help="Playwright slow motion in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--listing-pause-ms",
        type=int,
        default=2000,
        help="Pause before each listing page navigation in milliseconds (default: 2000)",
    )
    parser.add_argument(
        "--navigation-timeout-ms",
        type=int,
        default=60000,
        help="Playwright navigation timeout in milliseconds (default: 60000)",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout for event detail pages in seconds (default: 60)",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    links = await scrape_event_links(
        args.years,
        headless=not args.show_browser,
        slow_mo=args.slow_mo,
        listing_pause_ms=args.listing_pause_ms,
        navigation_timeout_ms=args.navigation_timeout_ms,
    )

    if args.links_output:
        links_path = Path(args.links_output)
        links_path.write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(links)} discovered URLs to {links_path}")

    events = scrape_event_details(links, timeout_seconds=args.request_timeout_seconds)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(events)} raw events to {out_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
