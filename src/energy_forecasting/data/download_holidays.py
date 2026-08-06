"""Download UK bank holidays from the official GOV.UK endpoint.

Source
------
``https://www.gov.uk/bank-holidays.json`` — the GOV.UK Bank Holidays API. It holds
three separate calendars: ``england-and-wales``, ``scotland`` and
``northern-ireland`` (their dates and even holiday *names* differ, e.g. Scotland's
"2nd January" and Northern Ireland's "St Patrick's Day").

Output columns
--------------
``date``, ``holiday_name``, ``is_england_wales_holiday``, ``is_scotland_holiday``,
``is_northern_ireland_holiday``, ``is_any_uk_holiday`` and a population-weighted
``gb_holiday_weight`` (see the note below). Saved to
``data/external/uk_bank_holidays.parquet``.

Population-weighting assumption
-------------------------------
The target is **Great Britain** National Demand (NESO ND), which covers England,
Scotland and Wales — **not** Northern Ireland (NI sits on the all-island SEM grid,
separate from the GB transmission system). We therefore weight a holiday by the
share of *GB* population it affects:

* England & Wales ≈ 0.916 of GB population
* Scotland        ≈ 0.084 of GB population
* Northern Ireland → 0.0 (outside GB; its holidays do not move GB demand)

So a UK-wide holiday scores ~1.0, an England-&-Wales-only holiday ~0.916, a
Scotland-only holiday ~0.084, and an NI-only holiday 0.0. These shares are
approximate (mid-decade ONS estimates) and are a modelling convenience, not an
exact demand elasticity — the raw per-nation flags are kept so any other weighting
can be applied downstream.
"""

from __future__ import annotations

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..logging_config import get_logger
from ..settings import EXTERNAL_DIR

logger = get_logger(__name__)

BANK_HOLIDAYS_URL = "https://www.gov.uk/bank-holidays.json"

# Division key -> flag column name. Order sets holiday_name priority.
DIVISIONS: dict[str, str] = {
    "england-and-wales": "is_england_wales_holiday",
    "scotland": "is_scotland_holiday",
    "northern-ireland": "is_northern_ireland_holiday",
}

# GB population shares (approximate; NI excluded — see module docstring).
ENGLAND_WALES_SHARE = 0.916
SCOTLAND_SHARE = 0.084


def _make_session(retries: int = 4, backoff: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "uk-energy-demand-forecasting/0.1"})
    return session


def fetch_bank_holidays(session: requests.Session | None = None) -> dict:
    """GET the raw GOV.UK bank-holidays JSON payload."""
    sess = session or _make_session()
    resp = sess.get(BANK_HOLIDAYS_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_bank_holidays(
    payload: dict, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Build the tidy per-date holiday table from the GOV.UK payload.

    One row per calendar date on which *any* division has a bank holiday.
    """
    # division -> {date_str: title}
    by_division: dict[str, dict[str, str]] = {}
    for key in DIVISIONS:
        events = payload.get(key, {}).get("events", [])
        by_division[key] = {ev["date"]: ev["title"] for ev in events}

    all_dates = sorted(set().union(*(d.keys() for d in by_division.values())))

    rows = []
    for date_str in all_dates:
        flags = {col: int(date_str in by_division[key]) for key, col in DIVISIONS.items()}
        # holiday_name: first available in division priority order.
        name = next(
            (by_division[key][date_str] for key in DIVISIONS if date_str in by_division[key]),
            None,
        )
        weight = (
            ENGLAND_WALES_SHARE * flags["is_england_wales_holiday"]
            + SCOTLAND_SHARE * flags["is_scotland_holiday"]
        )
        rows.append(
            {
                "date": pd.Timestamp(date_str),
                "holiday_name": name,
                **flags,
                "is_any_uk_holiday": int(any(flags.values())),
                "gb_holiday_weight": round(weight, 4),
            }
        )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df = df[
        [
            "date",
            "holiday_name",
            "is_england_wales_holiday",
            "is_scotland_holiday",
            "is_northern_ireland_holiday",
            "is_any_uk_holiday",
            "gb_holiday_weight",
        ]
    ]

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def download_holidays(
    start: str | None = "2020-01-01",
    end: str | None = "2025-12-31",
    save: bool = True,
) -> pd.DataFrame:
    """Fetch, parse and (optionally) save the UK bank-holiday table."""
    payload = fetch_bank_holidays()
    holidays = parse_bank_holidays(payload, start=start, end=end)
    logger.info(
        "Bank holidays: %d dates from %s to %s.",
        len(holidays),
        holidays["date"].min().date() if len(holidays) else None,
        holidays["date"].max().date() if len(holidays) else None,
    )

    if save:
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EXTERNAL_DIR / "uk_bank_holidays.parquet"
        holidays.to_parquet(out_path, index=False)
        logger.info("Saved -> %s", out_path)

    return holidays
