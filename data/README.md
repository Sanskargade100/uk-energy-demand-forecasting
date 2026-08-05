# Data

Data files are **not** committed (see `.gitignore`). Only `.gitkeep` markers are tracked.
Run `make data` to populate these folders.

## Layers

| Folder       | Contents                                                                 |
|--------------|--------------------------------------------------------------------------|
| `raw/`       | Untouched downloads exactly as received from each source.                |
| `interim/`   | Parsed, type-cast, timezone-localized, but not yet feature-engineered.   |
| `processed/` | Model-ready feature matrix (and the SQLite DB).                          |
| `external/`  | Reference tables that rarely change (e.g. holiday calendars).           |

## Target variable

**National Demand (ND)**, in **MW**, at 30-minute settlement-period resolution.
NESO defines ND as the national demand met by the GB transmission network.

## Primary source

NESO (National Energy System Operator) — **Historic Demand Data**. Half-hourly GB
demand plus embedded wind/solar generation. Used under the **NESO Open Data Licence**.
Because of the licence terms this repo does not redistribute the data; it is fetched
at build time via the CKAN datastore API (see `configs/data.yaml`).

Key columns typically include: `SETTLEMENT_DATE`, `SETTLEMENT_PERIOD`, `ND`,
`ENGLAND_WALES_DEMAND`, `EMBEDDED_WIND_GENERATION`, `EMBEDDED_SOLAR_GENERATION`.

## Supplementary sources

- **Weather** — temperature, wind speed, solar radiation, humidity for major GB cities,
  combined into a population-weighted GB series (`configs/data.yaml`).
- **Holidays** — GB bank holidays (England, Scotland, Wales) via the `holidays` package.
