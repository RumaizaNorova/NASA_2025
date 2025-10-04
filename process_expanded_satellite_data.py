#!/usr/bin/env python3
"""
Feature Engineering for Expanded Satellite Data
Processes multiple years of satellite data into training features
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def process_satellite_data():
    """Process expanded satellite data into features"""
    print("Processing expanded satellite data...")
    
    data_dir = Path("data/raw")
    output_dir = Path("data/interim")
    output_dir.mkdir(exist_ok=True)
    
    # Process each dataset
    datasets = {
        'mur_sst': 'Sea Surface Temperature',
        'measures_ssh': 'Sea Surface Height', 
        'oscar_currents': 'Ocean Currents',
        'pace_chl': 'Chlorophyll',
        'smap_salinity': 'Sea Surface Salinity',
        'gpm_precipitation': 'Precipitation'
    }
    
    all_features = []
    
    for dataset_name, description in datasets.items():
        print(f"  Processing {dataset_name}: {description}")
        
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            print(f"    Directory not found: {dataset_dir}")
            continue
        
        # Get all NetCDF files
        nc_files = list(dataset_dir.glob('*.nc'))
        print(f"    Found {len(nc_files)} files")
        
        # Process each file (simplified)
        for nc_file in nc_files[:5]:  # Limit for demo
            try:
                # Load NetCDF data
                ds = xr.open_dataset(nc_file)
                
                # Extract features (simplified)
                features = {
                    'dataset': dataset_name,
                    'file': nc_file.name,
                    'mean_value': float(ds[list(ds.data_vars)[0]].mean().values) if ds.data_vars else 0.0,
                    'std_value': float(ds[list(ds.data_vars)[0]].std().values) if ds.data_vars else 0.0,
                    'min_value': float(ds[list(ds.data_vars)[0]].min().values) if ds.data_vars else 0.0,
                    'max_value': float(ds[list(ds.data_vars)[0]].max().values) if ds.data_vars else 0.0,
                    'processed_at': datetime.now().isoformat()
                }
                
                all_features.append(features)
                ds.close()
                
            except Exception as e:
                print(f"    Error processing {nc_file}: {e}")
    
    # Save features
    if all_features:
        features_df = pd.DataFrame(all_features)
        output_path = output_dir / 'satellite_features_expanded.csv'
        features_df.to_csv(output_path, index=False)
        print(f"  Features saved to: {output_path}")
        print(f"  Total features: {len(features_df)}")
    
    return len(all_features)

if __name__ == "__main__":
    print("Feature Engineering for Expanded Satellite Data")
    print("=" * 50)
    
    try:
        n_features = process_satellite_data()
        print(f"\nFeature engineering completed!")
        print(f"Processed {n_features} feature sets")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
