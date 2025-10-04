"""
Inspect available NASA datasets and summarise the study period.

This script is a stub; in a real deployment it would query the NASA CMR
(Common Metadata Repository) and Harmony APIs to discover dataset coverage
within the region of interest and time range specified in the configuration.
Since network access is disabled in the competition environment, the current
implementation simply echoes the configured dates and lists the datasets that
would normally be fetched.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from typing import Any, Dict

try:
    from .utils import load_config, date_range
except ImportError:
    from utils import load_config, date_range


DATASETS = {
    "MUR-JPL-L4-GLOB-v4.1": {
        "description": "Multi-scale Ultra-high Resolution sea surface temperature (SST)",
        "variables": ["sst"],
    },
    "MEaSUREs-GRAV-SSH": {
        "description": "MEaSUREs gridded sea surface height anomaly",
        "variables": ["ssh_anom"],
    },
    "OSCAR-5day": {
        "description": "OSCAR surface currents (u, v)",
        "variables": ["u_current", "v_current"],
    },
    "PACE-L2": {
        "description": "PACE ocean colour (chlorophyll-a)",
        "variables": ["chlorophyll"],
    },
    "SMAP-L3-Salinity": {
        "description": "SMAP sea surface salinity",
        "variables": ["sss"],
    },
    "GPM-IMERG": {
        "description": "GPM IMERG precipitation accumulation",
        "variables": ["rain_7d"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scout available satellite datasets for the sharks‑from‑space pipeline.")
    parser.add_argument("--config", default="config/params.yaml", help="Path to YAML configuration file")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no network calls)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Error reading config: {exc}")
        sys.exit(1)
    roi = config.get("roi", {})
    time_cfg = config.get("time", {})
    start = time_cfg.get("start")
    end = time_cfg.get("end")
    dates = date_range(start, end)
    print("\n[Data Scout]")
    print(f"Region of interest: lon {roi.get('lon_min')}…{roi.get('lon_max')}, "
          f"lat {roi.get('lat_min')}…{roi.get('lat_max')}")
    print(f"Study period: {start} to {end} ({len(dates)} days)")
    print("Planned datasets and variables:")
    for name, info in DATASETS.items():
        vars_str = ", ".join(info["variables"])
        print(f"  • {name}: {info['description']} [variables: {vars_str}]")
    # Check for Earthdata credentials
    token = os.environ.get("EARTHDATA_TOKEN")
    user = os.environ.get("EARTHDATA_USERNAME")
    pwd = os.environ.get("EARTHDATA_PASSWORD")
    if not token and not (user and pwd):
        print("\nWARNING: No Earthdata credentials found in the environment. "
              "Real data retrieval will fail.  Set EARTHDATA_TOKEN or "
              "EARTHDATA_USERNAME/EARTHDATA_PASSWORD in your .env file.")
    if args.demo:
        print("\nDemo mode: no network requests will be made.  Downstream steps "
              "will synthesise data for demonstration purposes.")


if __name__ == "__main__":
    main()