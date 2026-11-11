#!/usr/bin/env python3
"""
Download NYISO grid data and EV-related datasets for a New York EV-grid study.

What this script supports
-------------------------
1) NYISO public market/grid data (no API key required), via NYISO MIS list pages:
   - Integrated real-time actual load (P-58C)
   - Day-ahead zonal LBMP (P-2A)
   - Real-time fuel mix (P-63)

2) New York EV charging-station infrastructure data:
   - NY Open Data / NYSERDA alternative fuel stations dataset (CSV; no key required)
   - Optional NREL / NLR Alternative Fuel Stations API (API key required)

3) Optional ACN-Data charging-session data (token required)
   - Useful for EV behavior modeling when local New York session-level data are unavailable.

Notes
-----
- The script intentionally downloads raw files first. Parsing/feature engineering should be a separate step.
- ACN-Data is not New York specific, but it is a widely used public charging-session dataset for learning arrival,
  departure, and energy-demand distributions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


NYISO_BASE = "https://mis.nyiso.com"
NYS_ALT_FUEL_CSV = "https://data.ny.gov/api/views/bpkx-gmh7/rows.csv?accessType=DOWNLOAD"
NREL_ALT_FUEL_API = "https://developer.nlr.gov/api/alt-fuel-stations/v1.csv"
ACN_API_BASE = "https://ev.caltech.edu/api/v1"


@dataclass(frozen=True)
class NyisoDataset:
    name: str
    list_url: str
    out_subdir: str
    # File links on the list page can be daily CSVs or monthly archive ZIPs.
    # We use a filename regex to filter files after scraping anchors.
    filename_regex: str


NYISO_DATASETS: Dict[str, NyisoDataset] = {
    "load": NyisoDataset(
        name="integrated_real_time_actual_load",
        list_url="https://mis.nyiso.com/public/P-58Clist.htm",
        out_subdir="nyiso/load",
        filename_regex=r"(\d{8}pal(?:Integrated)?\.csv|\d{8}pal\.csv|\d{8}isolf\.csv|\d{8}.*\.csv)$",
    ),
    "lbmp": NyisoDataset(
        name="day_ahead_zonal_lbmp",
        list_url="https://mis.nyiso.com/public/P-2Alist.htm",
        out_subdir="nyiso/lbmp",
        filename_regex=r"\d{8}damlbmp_zone\.csv$",
    ),
    "fuelmix": NyisoDataset(
        name="real_time_fuel_mix",
        list_url="https://mis.nyiso.com/public/P-63list.htm",
        out_subdir="nyiso/fuelmix",
        filename_regex=r"(\d{8}rtfuelmix\.csv|\d{6}01rtfuelmix_csv\.zip|\d{6}rtfuelmix_csv\.zip)$",
    ),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
    )
    return s


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def download_file(session: requests.Session, url: str, dst: Path, overwrite: bool = False) -> None:
    if dst.exists() and not overwrite:
        print(f"[skip] {dst}")
        return
    ensure_dir(dst.parent)
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[ok]   {dst}")


# -----------------------------
# NYISO helpers
# -----------------------------

def scrape_links(session: requests.Session, page_url: str) -> List[str]:
    r = session.get(page_url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(page_url, href)
        links.append(full)
    # Preserve order while deduplicating
    seen = set()
    deduped = []
    for x in links:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped


def filename_from_url(url: str) -> str:
    return url.split("?")[0].rstrip("/").split("/")[-1]


def matches_any_date(filename: str, dates: Iterable[dt.date]) -> bool:
    date_tokens = {d.strftime("%Y%m%d") for d in dates}
    month_tokens = {d.strftime("%Y%m") for d in dates}
    for tok in date_tokens.union(month_tokens):
        if tok in filename:
            return True
    return False


def select_nyiso_urls(
    all_links: List[str], dataset: NyisoDataset, start: dt.date, end: dt.date
) -> List[str]:
    pat = re.compile(dataset.filename_regex, flags=re.IGNORECASE)
    days = list(daterange(start, end))
    selected = []
    for link in all_links:
        fname = filename_from_url(link)
        if not pat.search(fname):
            continue
        if matches_any_date(fname, days):
            selected.append(link)
    return selected


def download_nyiso_dataset(
    session: requests.Session,
    dataset_key: str,
    start: dt.date,
    end: dt.date,
    out_root: Path,
    overwrite: bool = False,
) -> None:
    if dataset_key not in NYISO_DATASETS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    dataset = NYISO_DATASETS[dataset_key]
    links = scrape_links(session, dataset.list_url)
    selected = select_nyiso_urls(links, dataset, start, end)

    if not selected:
        print(f"[warn] No files found for {dataset_key} in {start}..{end}")
        return

    out_dir = out_root / dataset.out_subdir
    ensure_dir(out_dir)
    for url in selected:
        fname = filename_from_url(url)
        download_file(session, url, out_dir / fname, overwrite=overwrite)
        time.sleep(0.2)


# -----------------------------
# EV infrastructure datasets
# -----------------------------

def download_ny_open_data_stations(session: requests.Session, out_root: Path, overwrite: bool = False) -> None:
    out_path = out_root / "ev" / "ny_alternative_fuel_stations.csv"
    download_file(session, NYS_ALT_FUEL_CSV, out_path, overwrite=overwrite)


def download_nrel_alt_fuel_stations(
    session: requests.Session,
    api_key: str,
    out_root: Path,
    overwrite: bool = False,
    state: str = "NY",
    fuel_type: str = "ELEC",
    access: str = "public",
    status: str = "E",
) -> None:
    params = {
        "api_key": api_key,
        "state": state,
        "fuel_type": fuel_type,
        "access": access,
        "status": status,
        "limit": "all",
    }
    out_path = out_root / "ev" / f"nrel_alt_fuel_stations_{state}_{fuel_type}.csv"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path}")
        return
    ensure_dir(out_path.parent)
    r = session.get(NREL_ALT_FUEL_API, params=params, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"[ok]   {out_path}")


# -----------------------------
# ACN-Data charging sessions
# -----------------------------

def _acn_get_json(
    session: requests.Session,
    endpoint: str,
    token: str,
    params: Optional[dict] = None,
) -> dict:
    auth = (token, "")
    r = session.get(endpoint, params=params, auth=auth, timeout=120)
    r.raise_for_status()
    return r.json()


def download_acn_sessions(
    session: requests.Session,
    token: str,
    out_root: Path,
    site_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    with_timeseries: bool = False,
    pretty: bool = False,
    max_pages: Optional[int] = None,
    sleep_seconds: float = 0.2,
    overwrite: bool = False,
) -> None:
    suffix = "ts" if with_timeseries else "sessions"
    out_path = out_root / "ev" / f"acn_{site_id}_{suffix}.jsonl"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path}")
        return
    ensure_dir(out_path.parent)

    endpoint = f"{ACN_API_BASE}/sessions/{site_id}"
    if with_timeseries:
        endpoint += "/ts"

    params = {}
    if pretty:
        params["pretty"] = ""

    if start or end:
        clauses = []
        if start:
            start_dt = dt.datetime.fromisoformat(start).strftime("%a, %-d %b %Y %H:%M:%S GMT")
            clauses.append(f'connectionTime>="{start_dt}"')
        if end:
            end_dt = dt.datetime.fromisoformat(end).strftime("%a, %-d %b %Y %H:%M:%S GMT")
            clauses.append(f'connectionTime<="{end_dt}"')
        if clauses:
            params["where"] = " and ".join(clauses)

    page = 1
    total_written = 0
    next_href = None

    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            if next_href:
                payload = _acn_get_json(session, urljoin(f"{ACN_API_BASE}/", next_href), token)
            else:
                payload = _acn_get_json(session, endpoint, token, params=params)

            items = payload.get("_items", [])
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_written += 1

            links = payload.get("_links", {})
            next_link = links.get("next", {}).get("href")
            if not next_link:
                break

            page += 1
            if max_pages is not None and page > max_pages:
                break
            next_href = next_link
            time.sleep(sleep_seconds)

    print(f"[ok]   {out_path} ({total_written} sessions)")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download NYISO and EV datasets for EV-grid modeling.")
    p.add_argument("--out", type=str, default="data/raw", help="Output root directory.")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD for NYISO daily pulls.")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD for NYISO daily pulls.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    p.add_argument(
        "--nyiso",
        nargs="*",
        default=[],
        choices=list(NYISO_DATASETS.keys()) + ["all"],
        help="NYISO datasets to download: load lbmp fuelmix all",
    )

    p.add_argument("--ny-open-ev-stations", action="store_true", help="Download NY Open Data EV station CSV.")
    p.add_argument("--nrel-ev-stations", action="store_true", help="Download NREL/NLR EV station CSV (requires API key).")
    p.add_argument("--nrel-api-key", type=str, default=os.getenv("NREL_API_KEY"))

    p.add_argument("--acn", action="store_true", help="Download ACN-Data sessions (requires token).")
    p.add_argument("--acn-token", type=str, default=os.getenv("ACN_TOKEN"))
    p.add_argument(
        "--acn-sites",
        nargs="*",
        default=["caltech", "jpl", "office001"],
        help="ACN site IDs to download.",
    )
    p.add_argument("--acn-start", type=str, default=None, help="ACN start datetime, ISO format, e.g. 2024-01-01T00:00:00")
    p.add_argument("--acn-end", type=str, default=None, help="ACN end datetime, ISO format, e.g. 2024-12-31T23:59:59")
    p.add_argument("--acn-ts", action="store_true", help="Use ACN /ts endpoint (slow, heavy).")
    p.add_argument("--acn-max-pages", type=int, default=None, help="Max pages per ACN site.")

    return p


def main() -> int:
    args = build_parser().parse_args()
    out_root = Path(args.out)
    ensure_dir(out_root)
    session = get_session()

    # NYISO
    if args.nyiso:
        if args.start is None or args.end is None:
            raise SystemExit("When using --nyiso, please provide --start and --end in YYYY-MM-DD format.")
        start = parse_date(args.start)
        end = parse_date(args.end)
        keys = list(NYISO_DATASETS.keys()) if "all" in args.nyiso else args.nyiso
        for key in keys:
            download_nyiso_dataset(session, key, start, end, out_root, overwrite=args.overwrite)

    # New York open EV stations
    if args.ny_open_ev_stations:
        download_ny_open_data_stations(session, out_root, overwrite=args.overwrite)

    # NREL/NLR EV stations API
    if args.nrel_ev_stations:
        if not args.nrel_api_key:
            raise SystemExit("--nrel-ev-stations requires --nrel-api-key or NREL_API_KEY environment variable.")
        download_nrel_alt_fuel_stations(
            session,
            api_key=args.nrel_api_key,
            out_root=out_root,
            overwrite=args.overwrite,
        )

    # ACN-Data
    if args.acn:
        if not args.acn_token:
            raise SystemExit("--acn requires --acn-token or ACN_TOKEN environment variable.")
        for site in args.acn_sites:
            download_acn_sessions(
                session=session,
                token=args.acn_token,
                out_root=out_root,
                site_id=site,
                start=args.acn_start,
                end=args.acn_end,
                with_timeseries=args.acn_ts,
                max_pages=args.acn_max_pages,
                overwrite=args.overwrite,
            )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
