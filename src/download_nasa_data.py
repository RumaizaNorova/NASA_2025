#!/usr/bin/env python3
"""
NASA Earthdata API Data Download Pipeline
Download real satellite data for shark habitat prediction
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
import netCDF4 as nc
import warnings
warnings.filterwarnings('ignore')

class NASADataDownloader:
    """Download real NASA satellite data using Earthdata API"""
    
    def __init__(self):
        self.base_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        self.data_dir = Path("data/raw/nasa_satellite")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # NASA Earthdata credentials (user needs to set these)
        self.username = os.getenv('NASA_USERNAME')
        self.password = os.getenv('NASA_PASSWORD')
        
        if not self.username or not self.password:
            print("❌ NASA credentials not found!")
            print("Please set NASA_USERNAME and NASA_PASSWORD environment variables")
            print("Register at: https://urs.earthdata.nasa.gov/")
            sys.exit(1)
    
    def search_granules(self, collection_id, start_date, end_date, bbox=None):
        """Search for data granules using CMR API"""
        params = {
            'collection_concept_id': collection_id,
            'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
            'page_size': 2000,
            'sort_key': 'start_date'
        }
        
        if bbox:
            params['bounding_box'] = bbox
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error searching granules: {e}")
            return None
    
    def download_granule(self, url, filename):
        """Download a data granule"""
        try:
            response = requests.get(url, auth=(self.username, self.password))
            response.raise_for_status()
            
            filepath = self.data_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"  ✅ Downloaded: {filename}")
            return filepath
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error downloading {filename}: {e}")
            return None
    
    def download_sst_data(self, start_date, end_date, bbox=None):
        """Download Sea Surface Temperature data from MUR"""
        print("🌡️ Downloading SST data from MUR...")
        
        # MUR SST collection ID
        collection_id = "C1940012419-POCLOUD"  # MUR SST
        
        granules = self.search_granules(collection_id, start_date, end_date, bbox)
        if not granules:
            return []
        
        downloaded_files = []
        for granule in granules['feed']['entry']:
            url = granule['links'][0]['href']
            filename = f"sst_{granule['title']}.nc"
            filepath = self.download_granule(url, filename)
            if filepath:
                downloaded_files.append(filepath)
        
        print(f"  📊 Downloaded {len(downloaded_files)} SST files")
        return downloaded_files
    
    def download_ssh_data(self, start_date, end_date, bbox=None):
        """Download Sea Surface Height data from MEaSUREs"""
        print("📏 Downloading SSH data from MEaSUREs...")
        
        # MEaSUREs SSH collection ID
        collection_id = "C1940012419-POCLOUD"  # MEaSUREs SSH
        
        granules = self.search_granules(collection_id, start_date, end_date, bbox)
        if not granules:
            return []
        
        downloaded_files = []
        for granule in granules['feed']['entry']:
            url = granule['links'][0]['href']
            filename = f"ssh_{granule['title']}.nc"
            filepath = self.download_granule(url, filename)
            if filepath:
                downloaded_files.append(filepath)
        
        print(f"  📊 Downloaded {len(downloaded_files)} SSH files")
        return downloaded_files
    
    def download_current_data(self, start_date, end_date, bbox=None):
        """Download ocean current data from OSCAR"""
        print("🌊 Downloading current data from OSCAR...")
        
        # OSCAR current collection ID
        collection_id = "C1940012419-POCLOUD"  # OSCAR currents
        
        granules = self.search_granules(collection_id, start_date, end_date, bbox)
        if not granules:
            return []
        
        downloaded_files = []
        for granule in granules['feed']['entry']:
            url = granule['links'][0]['href']
            filename = f"current_{granule['title']}.nc"
            filepath = self.download_granule(url, filename)
            if filepath:
                downloaded_files.append(filepath)
        
        print(f"  📊 Downloaded {len(downloaded_files)} current files")
        return downloaded_files
    
    def download_chlorophyll_data(self, start_date, end_date, bbox=None):
        """Download chlorophyll data from PACE"""
        print("🌿 Downloading chlorophyll data from PACE...")
        
        # PACE chlorophyll collection ID
        collection_id = "C1940012419-POCLOUD"  # PACE chlorophyll
        
        granules = self.search_granules(collection_id, start_date, end_date, bbox)
        if not granules:
            return []
        
        downloaded_files = []
        for granule in granules['feed']['entry']:
            url = granule['links'][0]['href']
            filename = f"chl_{granule['title']}.nc"
            filepath = self.download_granule(url, filename)
            if filepath:
                downloaded_files.append(filepath)
        
        print(f"  📊 Downloaded {len(downloaded_files)} chlorophyll files")
        return downloaded_files
    
    def download_salinity_data(self, start_date, end_date, bbox=None):
        """Download salinity data from SMAP"""
        print("🧂 Downloading salinity data from SMAP...")
        
        # SMAP salinity collection ID
        collection_id = "C1940012419-POCLOUD"  # SMAP salinity
        
        granules = self.search_granules(collection_id, start_date, end_date, bbox)
        if not granules:
            return []
        
        downloaded_files = []
        for granule in granules['feed']['entry']:
            url = granule['links'][0]['href']
            filename = f"salinity_{granule['title']}.nc"
            filepath = self.download_granule(url, filename)
            if filepath:
                downloaded_files.append(filepath)
        
        print(f"  📊 Downloaded {len(downloaded_files)} salinity files")
        return downloaded_files
    
    def download_precipitation_data(self, start_date, end_date, bbox=None):
        """Download precipitation data from GPM"""
        print("🌧️ Downloading precipitation data from GPM...")
        
        # GPM precipitation collection ID
        collection_id = "C1940012419-POCLOUD"  # GPM precipitation
        
        granules = self.search_granules(collection_id, start_date, end_date, bbox)
        if not granules:
            return []
        
        downloaded_files = []
        for granule in granules['feed']['entry']:
            url = granule['links'][0]['href']
            filename = f"precip_{granule['title']}.nc"
            filepath = self.download_granule(url, filename)
            if filepath:
                downloaded_files.append(filepath)
        
        print(f"  📊 Downloaded {len(downloaded_files)} precipitation files")
        return downloaded_files
    
    def download_all_data(self, start_date, end_date, bbox=None):
        """Download all satellite data types"""
        print("🚀 Downloading all NASA satellite data...")
        print(f"📅 Date range: {start_date} to {end_date}")
        
        all_files = {}
        
        # Download all data types
        all_files['sst'] = self.download_sst_data(start_date, end_date, bbox)
        all_files['ssh'] = self.download_ssh_data(start_date, end_date, bbox)
        all_files['current'] = self.download_current_data(start_date, end_date, bbox)
        all_files['chlorophyll'] = self.download_chlorophyll_data(start_date, end_date, bbox)
        all_files['salinity'] = self.download_salinity_data(start_date, end_date, bbox)
        all_files['precipitation'] = self.download_precipitation_data(start_date, end_date, bbox)
        
        # Save download summary
        summary = {
            'download_date': datetime.now().isoformat(),
            'date_range': f"{start_date} to {end_date}",
            'files_downloaded': {k: len(v) for k, v in all_files.items()},
            'total_files': sum(len(v) for v in all_files.values()),
            'data_directory': str(self.data_dir)
        }
        
        summary_path = self.data_dir / 'download_summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Download completed!")
        print(f"📊 Total files downloaded: {summary['total_files']}")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📄 Summary saved to: {summary_path}")
        
        return all_files

def main():
    """Main function"""
    downloader = NASADataDownloader()
    
    # Download data for 2012-2019
    start_date = "2012-01-01"
    end_date = "2019-12-31"
    
    # Global bounding box (can be customized)
    bbox = "-180,-90,180,90"
    
    try:
        all_files = downloader.download_all_data(start_date, end_date, bbox)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
