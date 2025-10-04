"""
Fetch remote NASA satellite data using earthaccess library.

This script iterates over the configured date range and downloads NetCDF
subsets for each dataset using NASA's earthaccess library with proper authentication.
It requires valid Earthdata credentials and network access.

The raw data are not committed to git; `.gitignore` should exclude `data/raw`.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import xarray as xr
import earthaccess
from dotenv import load_dotenv

try:
    from .utils import load_config, date_range, ensure_dir, setup_logging
except ImportError:
    from utils import load_config, date_range, ensure_dir, setup_logging

# Load environment variables
load_dotenv()

# Dataset configurations with their short names and variable mappings
DATASET_CONFIGS = {
    "mur_sst": {
        "short_name": "MUR-JPL-L4-GLOB-v4.1",
        "variables": ["analysed_sst"],
        "spatial_resolution": 0.01,
        "temporal_resolution": "daily"
    },
    "measures_ssh": {
        "short_name": "SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL2205",
        "variables": ["sla"],
        "spatial_resolution": 0.25,
        "temporal_resolution": "5-day"
    },
    "oscar_currents": {
        "short_name": "OSCAR_L4_OC_third-deg",
        "variables": ["u", "v"],
        "spatial_resolution": 0.33,
        "temporal_resolution": "5-day"
    },
    "pace_chl": {
        "short_name": "PACE_OCI_L2_OCEAN",
        "variables": ["chlor_a"],
        "spatial_resolution": 0.04,
        "temporal_resolution": "daily"
    },
    "smap_salinity": {
        "short_name": "SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5",
        "variables": ["sss_smap"],
        "spatial_resolution": 0.25,
        "temporal_resolution": "8-day"
    },
    "gpm_precipitation": {
        "short_name": "GPM_3IMERGDF",
        "variables": ["precipitation"],
        "spatial_resolution": 0.1,
        "temporal_resolution": "daily"
    }
}


def authenticate_earthaccess() -> bool:
    """Authenticate with NASA Earthdata using earthaccess library."""
    try:
        # Check if credentials are available
        username = os.getenv('EARTHDATA_USERNAME')
        password = os.getenv('EARTHDATA_PASSWORD')
        token = os.getenv('EARTHDATA_TOKEN')
        
        if username and password:
            print("[Fetch Data] Using Earthdata username/password for authentication")
            auth = earthaccess.login(strategy="environment")
        elif token:
            print("[Fetch Data] Using Earthdata token for authentication")
            # Set token in environment for earthaccess
            os.environ['EARTHDATA_TOKEN'] = token
            auth = earthaccess.login(strategy="environment")
        else:
            print("[Fetch Data] No Earthdata credentials found, attempting interactive login")
            auth = earthaccess.login(strategy="interactive")
        
        if auth:
            print("[Fetch Data] Successfully authenticated with NASA Earthdata")
            return True
        else:
            print("[Fetch Data] Authentication failed")
            return False
            
    except Exception as e:
        print(f"[Fetch Data] Authentication error: {e}")
        return False


def fetch_dataset_data(dataset_key: str, dataset_config: Dict[str, Any], 
                      bbox: List[float], start_date: str, end_date: str, 
                      output_dir: str, batch_size: int = 30) -> bool:
    """Fetch data for a specific dataset using earthaccess library."""
    
    short_name = dataset_config["short_name"]
    
    # Create output directory
    dataset_dir = os.path.join(output_dir, dataset_key)
    ensure_dir(dataset_dir)
    
    # Parse dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    print(f"[Fetch Data] Processing {dataset_key} ({short_name}): {start_date} to {end_date}")
    
    try:
        # Search for datasets
        print(f"[Fetch Data] Searching for {dataset_key} data...")
        
        results = earthaccess.search_data(
            short_name=short_name,
            bounding_box=bbox,  # [west, south, east, north]
            temporal=(start_date, end_date),
            count=1000  # Adjust based on expected number of files
        )
        
        if not results:
            print(f"[Fetch Data] No data found for {dataset_key}")
            return False
        
        print(f"[Fetch Data] Found {len(results)} files for {dataset_key}")
        
        # Download files
        print(f"[Fetch Data] Downloading {dataset_key} data...")
        
        downloaded_files = earthaccess.download(
            results,
            local_path=dataset_dir
        )
        
        if downloaded_files:
            print(f"[Fetch Data] Successfully downloaded {len(downloaded_files)} files for {dataset_key}")
            return True
        else:
            print(f"[Fetch Data] Failed to download files for {dataset_key}")
            return False
            
    except Exception as e:
        print(f"[Fetch Data] Error fetching {dataset_key}: {str(e)}")
        return False


def fetch_real_data(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Fetch real NASA data using earthaccess library."""
    
    # Authenticate first
    if not authenticate_earthaccess():
        print("[Fetch Data] Authentication failed. Cannot proceed with data download.")
        return
    
    roi = config.get('roi', {})
    bbox = [roi['lon_min'], roi['lat_min'], roi['lon_max'], roi['lat_max']]
    
    time_cfg = config.get('time', {})
    start_date = time_cfg.get('start')
    end_date = time_cfg.get('end')
    
    data_sources = config.get('data_sources', {})
    performance_cfg = config.get('performance', {})
    batch_size = performance_cfg.get('batch_size', 30)
    
    output_dir = 'data/raw'
    ensure_dir(output_dir)
    
    print(f"[Fetch Data] Fetching NASA data from {start_date} to {end_date}")
    print(f"[Fetch Data] Region: {bbox}")
    print(f"[Fetch Data] Datasets: {list(data_sources.keys())}")
    print(f"[Fetch Data] Output directory: {output_dir}")
    
    # Process each dataset
    successful_datasets = 0
    total_datasets = len(data_sources)
    
    for dataset_key, dataset_config in data_sources.items():
        if dataset_key in DATASET_CONFIGS:
            print(f"\n[Fetch Data] Processing dataset: {dataset_key}")
            
            success = fetch_dataset_data(
                dataset_key, 
                DATASET_CONFIGS[dataset_key], 
                bbox, 
                start_date, 
                end_date, 
                output_dir, 
                batch_size
            )
            
            if success:
                successful_datasets += 1
                print(f"[Fetch Data] ✓ {dataset_key} completed successfully")
            else:
                print(f"[Fetch Data] ✗ {dataset_key} failed")
        else:
            print(f"[Fetch Data] ⚠ Unknown dataset: {dataset_key}")
    
    print(f"\n[Fetch Data] Summary: {successful_datasets}/{total_datasets} datasets downloaded successfully")
    
    if successful_datasets == total_datasets:
        print("[Fetch Data] All datasets downloaded successfully!")
    elif successful_datasets > 0:
        print("[Fetch Data] Some datasets failed to download. Check logs above for details.")
    else:
        print("[Fetch Data] No datasets were downloaded successfully.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch NASA satellite data using earthaccess library")
    parser.add_argument("--config", type=str, default="config/params_enhanced.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Maximum number of parallel downloads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Always run real data fetching
    fetch_real_data(config, args)


if __name__ == "__main__":
    main()