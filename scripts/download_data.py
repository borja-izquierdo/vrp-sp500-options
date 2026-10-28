"""Download the raw data into ``data/raw/`` (requires your own API access).

    cp .env.example .env        # then fill in THETADATA_API_KEY and FRED_API_KEY
    python scripts/download_data.py

Writes ``underlying.parquet`` (SPX OHLC), ``macro.parquet`` (VIX, UST 3M/10Y,
SKEW) and ``chain.parquet`` (cleaned SPX monthly option chains). The chain
download is resumable: each expiration is checkpointed in
``data/raw/chain_parts/``; re-run the script after an interruption.

Notes
-----
* Needs a ThetaData *Options Standard* subscription (or higher). The full
  2018-2025 download took ~19 h in the thesis run.
* The thesis run retrieved no usable 2021 chains; this script was refactored
  after the thesis and has not been re-run end to end against the live API.
* None of the downloaded files may be redistributed; ``data/`` is git-ignored.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrp import data  # noqa: E402

log = logging.getLogger("download_data")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--skip-options", action="store_true",
                   help="only download SPX OHLC and macro series (no ThetaData needed)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The thetadata client logs account details at login; keep them out of the console/logs.
    for name in ("thetadata", "httpx", "httpcore", "grpc"):
        logging.getLogger(name).setLevel(logging.WARNING)

    present = data.load_credentials(args.env_file)
    log.info("Credentials present: %s", present)
    raw = args.data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cfg = data.DataConfig(start_date=args.start, end_date=args.end)

    underlying = data.fetch_underlying(cfg)
    underlying.to_parquet(raw / "underlying.parquet")
    log.info("SPX OHLC: %d sessions", len(underlying))

    macro = data.fetch_macro(cfg)
    macro.to_parquet(raw / "macro.parquet")
    log.info("Macro: %s", [c for c in macro.columns if c != "date"])

    if args.skip_options:
        return
    first, last = int(args.start[:4]), int(args.end[:4])
    chain = data.download_chains(cfg, underlying, data.third_fridays(first, last),
                                 raw / "chain_parts")
    if chain.empty:
        raise SystemExit("No option data downloaded: check THETADATA_API_KEY and your plan.")
    chain.to_parquet(raw / "chain.parquet")
    log.info("Cleaned chain: %d rows, %s -> %s", len(chain),
             chain["date"].min().date(), chain["date"].max().date())


if __name__ == "__main__":
    main()
