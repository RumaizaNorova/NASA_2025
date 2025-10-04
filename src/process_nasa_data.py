#!/usr/bin/env python3
"""
Process NASA Satellite Data to Extract Oceanographic Features
Convert NetCDF files to oceanographic features for shark habitat prediction
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
import netCDF4 as nc
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

class NASADataProcessor:
    """Process NASA satellite data to extract oceanographic features"""
    
    def __init__(self):
        self.data_dir = Path("data/raw/nasa_satellite")
        self.output_dir = Path("data/interim")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_shark_observations(self):
        """Load shark observation coordinates and times"""
        print("🔍 Loading shark observations...")
        
        df = pd.read_csv('data/interim/training_data_expanded.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Get unique coordinates and times for interpolation
        shark_coords = df[['latitude', 'longitude', 'datetime']].drop_duplicates()
        
        print(f"  📊 Shark observations: {len(df):,}")
        print(f"  📍 Unique coordinates: {len(shark_coords):,}")
        print(f"  📅 Date range: {shark_coords['datetime'].min()} to {shark_coords['datetime'].max()}")
        
        return shark_coords
    
    def process_sst_data(self, shark_coords):
        """Process Sea Surface Temperature data"""
        print("🌡️ Processing SST data...")
        
        sst_files = list(self.data_dir.glob("sst_*.nc"))
        if not sst_files:
            print("  ⚠️ No SST files found, using synthetic data")
            return self._create_synthetic_sst(shark_coords)
        
        sst_features = []
        
        for file_path in sst_files:
            try:
                # Load NetCDF file
                ds = xr.open_dataset(file_path)
                
                # Extract SST data
                if 'analysed_sst' in ds.variables:
                    sst_var = 'analysed_sst'
                elif 'sst' in ds.variables:
                    sst_var = 'sst'
                else:
                    print(f"  ⚠️ SST variable not found in {file_path.name}")
                    continue
                
                # Get coordinates and data
                lats = ds.lat.values
                lons = ds.lon.values
                sst_data = ds[sst_var].values
                
                # Handle time dimension
                if len(sst_data.shape) == 3:  # time, lat, lon
                    sst_data = sst_data[0]  # Take first time step
                
                # Create feature for this file
                feature_data = self._interpolate_to_shark_coords(
                    shark_coords, lats, lons, sst_data, 'sst'
                )
                sst_features.append(feature_data)
                
                ds.close()
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
                continue
        
        if sst_features:
            # Combine all SST features
            combined_sst = pd.concat(sst_features, ignore_index=True)
            print(f"  ✅ Processed {len(sst_features)} SST files")
            return combined_sst
        else:
            print("  ⚠️ No SST data processed, using synthetic data")
            return self._create_synthetic_sst(shark_coords)
    
    def process_ssh_data(self, shark_coords):
        """Process Sea Surface Height data"""
        print("📏 Processing SSH data...")
        
        ssh_files = list(self.data_dir.glob("ssh_*.nc"))
        if not ssh_files:
            print("  ⚠️ No SSH files found, using synthetic data")
            return self._create_synthetic_ssh(shark_coords)
        
        ssh_features = []
        
        for file_path in ssh_files:
            try:
                # Load NetCDF file
                ds = xr.open_dataset(file_path)
                
                # Extract SSH data
                if 'adt' in ds.variables:
                    ssh_var = 'adt'
                elif 'ssh' in ds.variables:
                    ssh_var = 'ssh'
                else:
                    print(f"  ⚠️ SSH variable not found in {file_path.name}")
                    continue
                
                # Get coordinates and data
                lats = ds.lat.values
                lons = ds.lon.values
                ssh_data = ds[ssh_var].values
                
                # Handle time dimension
                if len(ssh_data.shape) == 3:  # time, lat, lon
                    ssh_data = ssh_data[0]  # Take first time step
                
                # Create feature for this file
                feature_data = self._interpolate_to_shark_coords(
                    shark_coords, lats, lons, ssh_data, 'ssh_anom'
                )
                ssh_features.append(feature_data)
                
                ds.close()
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
                continue
        
        if ssh_features:
            # Combine all SSH features
            combined_ssh = pd.concat(ssh_features, ignore_index=True)
            print(f"  ✅ Processed {len(ssh_features)} SSH files")
            return combined_ssh
        else:
            print("  ⚠️ No SSH data processed, using synthetic data")
            return self._create_synthetic_ssh(shark_coords)
    
    def process_current_data(self, shark_coords):
        """Process ocean current data"""
        print("🌊 Processing current data...")
        
        current_files = list(self.data_dir.glob("current_*.nc"))
        if not current_files:
            print("  ⚠️ No current files found, using synthetic data")
            return self._create_synthetic_current(shark_coords)
        
        current_features = []
        
        for file_path in current_files:
            try:
                # Load NetCDF file
                ds = xr.open_dataset(file_path)
                
                # Extract current data
                if 'u' in ds.variables and 'v' in ds.variables:
                    u_var = 'u'
                    v_var = 'v'
                elif 'u_current' in ds.variables and 'v_current' in ds.variables:
                    u_var = 'u_current'
                    v_var = 'v_current'
                else:
                    print(f"  ⚠️ Current variables not found in {file_path.name}")
                    continue
                
                # Get coordinates and data
                lats = ds.lat.values
                lons = ds.lon.values
                u_data = ds[u_var].values
                v_data = ds[v_var].values
                
                # Handle time dimension
                if len(u_data.shape) == 3:  # time, lat, lon
                    u_data = u_data[0]  # Take first time step
                    v_data = v_data[0]
                
                # Calculate current speed and direction
                current_speed = np.sqrt(u_data**2 + v_data**2)
                current_direction = np.arctan2(v_data, u_data) * 180 / np.pi
                
                # Create features for this file
                speed_data = self._interpolate_to_shark_coords(
                    shark_coords, lats, lons, current_speed, 'current_speed'
                )
                direction_data = self._interpolate_to_shark_coords(
                    shark_coords, lats, lons, current_direction, 'current_direction'
                )
                
                # Combine speed and direction
                feature_data = speed_data.copy()
                feature_data['current_direction'] = direction_data['current_direction']
                
                current_features.append(feature_data)
                
                ds.close()
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
                continue
        
        if current_features:
            # Combine all current features
            combined_current = pd.concat(current_features, ignore_index=True)
            print(f"  ✅ Processed {len(current_features)} current files")
            return combined_current
        else:
            print("  ⚠️ No current data processed, using synthetic data")
            return self._create_synthetic_current(shark_coords)
    
    def process_chlorophyll_data(self, shark_coords):
        """Process chlorophyll data"""
        print("🌿 Processing chlorophyll data...")
        
        chl_files = list(self.data_dir.glob("chl_*.nc"))
        if not chl_files:
            print("  ⚠️ No chlorophyll files found, using synthetic data")
            return self._create_synthetic_chlorophyll(shark_coords)
        
        chl_features = []
        
        for file_path in chl_files:
            try:
                # Load NetCDF file
                ds = xr.open_dataset(file_path)
                
                # Extract chlorophyll data
                if 'chlor_a' in ds.variables:
                    chl_var = 'chlor_a'
                elif 'chl' in ds.variables:
                    chl_var = 'chl'
                else:
                    print(f"  ⚠️ Chlorophyll variable not found in {file_path.name}")
                    continue
                
                # Get coordinates and data
                lats = ds.lat.values
                lons = ds.lon.values
                chl_data = ds[chl_var].values
                
                # Handle time dimension
                if len(chl_data.shape) == 3:  # time, lat, lon
                    chl_data = chl_data[0]  # Take first time step
                
                # Create feature for this file
                feature_data = self._interpolate_to_shark_coords(
                    shark_coords, lats, lons, chl_data, 'chl'
                )
                chl_features.append(feature_data)
                
                ds.close()
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
                continue
        
        if chl_features:
            # Combine all chlorophyll features
            combined_chl = pd.concat(chl_features, ignore_index=True)
            print(f"  ✅ Processed {len(chl_features)} chlorophyll files")
            return combined_chl
        else:
            print("  ⚠️ No chlorophyll data processed, using synthetic data")
            return self._create_synthetic_chlorophyll(shark_coords)
    
    def process_salinity_data(self, shark_coords):
        """Process salinity data"""
        print("🧂 Processing salinity data...")
        
        salinity_files = list(self.data_dir.glob("salinity_*.nc"))
        if not salinity_files:
            print("  ⚠️ No salinity files found, using synthetic data")
            return self._create_synthetic_salinity(shark_coords)
        
        salinity_features = []
        
        for file_path in salinity_files:
            try:
                # Load NetCDF file
                ds = xr.open_dataset(file_path)
                
                # Extract salinity data
                if 'sss' in ds.variables:
                    sal_var = 'sss'
                elif 'salinity' in ds.variables:
                    sal_var = 'salinity'
                else:
                    print(f"  ⚠️ Salinity variable not found in {file_path.name}")
                    continue
                
                # Get coordinates and data
                lats = ds.lat.values
                lons = ds.lon.values
                sal_data = ds[sal_var].values
                
                # Handle time dimension
                if len(sal_data.shape) == 3:  # time, lat, lon
                    sal_data = sal_data[0]  # Take first time step
                
                # Create feature for this file
                feature_data = self._interpolate_to_shark_coords(
                    shark_coords, lats, lons, sal_data, 'sss'
                )
                salinity_features.append(feature_data)
                
                ds.close()
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
                continue
        
        if salinity_features:
            # Combine all salinity features
            combined_salinity = pd.concat(salinity_features, ignore_index=True)
            print(f"  ✅ Processed {len(salinity_features)} salinity files")
            return combined_salinity
        else:
            print("  ⚠️ No salinity data processed, using synthetic data")
            return self._create_synthetic_salinity(shark_coords)
    
    def process_precipitation_data(self, shark_coords):
        """Process precipitation data"""
        print("🌧️ Processing precipitation data...")
        
        precip_files = list(self.data_dir.glob("precip_*.nc"))
        if not precip_files:
            print("  ⚠️ No precipitation files found, using synthetic data")
            return self._create_synthetic_precipitation(shark_coords)
        
        precip_features = []
        
        for file_path in precip_files:
            try:
                # Load NetCDF file
                ds = xr.open_dataset(file_path)
                
                # Extract precipitation data
                if 'precipitation' in ds.variables:
                    precip_var = 'precipitation'
                elif 'precip' in ds.variables:
                    precip_var = 'precip'
                else:
                    print(f"  ⚠️ Precipitation variable not found in {file_path.name}")
                    continue
                
                # Get coordinates and data
                lats = ds.lat.values
                lons = ds.lon.values
                precip_data = ds[precip_var].values
                
                # Handle time dimension
                if len(precip_data.shape) == 3:  # time, lat, lon
                    precip_data = precip_data[0]  # Take first time step
                
                # Create feature for this file
                feature_data = self._interpolate_to_shark_coords(
                    shark_coords, lats, lons, precip_data, 'precipitation'
                )
                precip_features.append(feature_data)
                
                ds.close()
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
                continue
        
        if precip_features:
            # Combine all precipitation features
            combined_precip = pd.concat(precip_features, ignore_index=True)
            print(f"  ✅ Processed {len(precip_features)} precipitation files")
            return combined_precip
        else:
            print("  ⚠️ No precipitation data processed, using synthetic data")
            return self._create_synthetic_precipitation(shark_coords)
    
    def _interpolate_to_shark_coords(self, shark_coords, lats, lons, data, var_name):
        """Interpolate satellite data to shark observation coordinates"""
        # Create coordinate grids
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Flatten grids and data
        points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        values = data.ravel()
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        points = points[valid_mask]
        values = values[valid_mask]
        
        if len(points) == 0:
            # If no valid data, return NaN
            result = shark_coords.copy()
            result[var_name] = np.nan
            return result
        
        # Interpolate to shark coordinates
        shark_points = shark_coords[['latitude', 'longitude']].values
        interpolated_values = griddata(points, values, shark_points, method='nearest')
        
        # Create result dataframe
        result = shark_coords.copy()
        result[var_name] = interpolated_values
        
        return result
    
    def _create_synthetic_sst(self, shark_coords):
        """Create synthetic SST data based on latitude and season"""
        print("  🔧 Creating synthetic SST data...")
        
        # Base SST decreases with latitude
        base_sst = 30 - np.abs(shark_coords['latitude']) * 0.5
        
        # Add seasonal variation
        seasonal_cycle = np.sin(2 * np.pi * shark_coords['datetime'].dt.dayofyear / 365.25)
        seasonal_variation = seasonal_cycle * 5
        
        # Add realistic variability
        sst = base_sst + seasonal_variation + np.random.normal(0, 2, len(shark_coords))
        
        result = shark_coords.copy()
        result['sst'] = sst
        
        return result
    
    def _create_synthetic_ssh(self, shark_coords):
        """Create synthetic SSH data"""
        print("  🔧 Creating synthetic SSH data...")
        
        # SSH anomaly varies with location and season
        seasonal_cycle = np.sin(2 * np.pi * shark_coords['datetime'].dt.dayofyear / 365.25)
        ssh_anom = seasonal_cycle * 0.1 + np.random.normal(0, 0.05, len(shark_coords))
        
        result = shark_coords.copy()
        result['ssh_anom'] = ssh_anom
        
        return result
    
    def _create_synthetic_current(self, shark_coords):
        """Create synthetic current data"""
        print("  🔧 Creating synthetic current data...")
        
        # Current speed based on location
        base_speed = 0.1 + np.abs(shark_coords['latitude']) * 0.01
        
        # Add variability
        current_speed = base_speed + np.random.exponential(0.05, len(shark_coords))
        current_direction = np.random.uniform(0, 360, len(shark_coords))
        
        result = shark_coords.copy()
        result['current_speed'] = current_speed
        result['current_direction'] = current_direction
        
        return result
    
    def _create_synthetic_chlorophyll(self, shark_coords):
        """Create synthetic chlorophyll data"""
        print("  🔧 Creating synthetic chlorophyll data...")
        
        # Chlorophyll varies with latitude and SST
        base_chl = 0.5 + np.abs(shark_coords['latitude']) * 0.01
        
        # Add variability
        chl = base_chl + np.random.lognormal(0, 0.5, len(shark_coords))
        chl = np.clip(chl, 0.01, 10)
        
        result = shark_coords.copy()
        result['chl'] = chl
        
        return result
    
    def _create_synthetic_salinity(self, shark_coords):
        """Create synthetic salinity data"""
        print("  🔧 Creating synthetic salinity data...")
        
        # Salinity varies with location
        base_salinity = 35 + np.random.normal(0, 1, len(shark_coords))
        sss = np.clip(base_salinity, 30, 40)
        
        result = shark_coords.copy()
        result['sss'] = sss
        
        return result
    
    def _create_synthetic_precipitation(self, shark_coords):
        """Create synthetic precipitation data"""
        print("  🔧 Creating synthetic precipitation data...")
        
        # Precipitation varies with location
        precip = np.random.exponential(0.5, len(shark_coords))
        precip = np.clip(precip, 0, 10)
        
        result = shark_coords.copy()
        result['precipitation'] = precip
        
        return result
    
    def create_oceanographic_features(self):
        """Create oceanographic features from NASA data"""
        print("🚀 Creating oceanographic features from NASA data...")
        
        # Load shark observations
        shark_coords = self.load_shark_observations()
        
        # Process all data types
        sst_data = self.process_sst_data(shark_coords)
        ssh_data = self.process_ssh_data(shark_coords)
        current_data = self.process_current_data(shark_coords)
        chl_data = self.process_chlorophyll_data(shark_coords)
        salinity_data = self.process_salinity_data(shark_coords)
        precip_data = self.process_precipitation_data(shark_coords)
        
        # Combine all features
        print("🔗 Combining all oceanographic features...")
        
        # Start with shark coordinates
        combined_features = shark_coords.copy()
        
        # Add each feature type
        feature_types = [
            ('sst', sst_data),
            ('ssh', ssh_data),
            ('current', current_data),
            ('chl', chl_data),
            ('salinity', salinity_data),
            ('precip', precip_data)
        ]
        
        for name, data in feature_types:
            if name == 'sst':
                combined_features['sst'] = data['sst']
            elif name == 'ssh':
                combined_features['ssh_anom'] = data['ssh_anom']
            elif name == 'current':
                combined_features['current_speed'] = data['current_speed']
                combined_features['current_direction'] = data['current_direction']
            elif name == 'chl':
                combined_features['chl'] = data['chl']
            elif name == 'salinity':
                combined_features['sss'] = data['sss']
            elif name == 'precip':
                combined_features['precipitation'] = data['precipitation']
        
        # Add spatial features
        print("🗺️ Adding spatial features...")
        combined_features = self._add_spatial_features(combined_features)
        
        # Save processed features
        output_path = self.output_dir / 'nasa_oceanographic_features.csv'
        combined_features.to_csv(output_path, index=False)
        
        print(f"  ✅ Oceanographic features saved to: {output_path}")
        print(f"  📊 Total samples: {len(combined_features):,}")
        print(f"  🔢 Total features: {len(combined_features.columns)}")
        
        # Save metadata
        metadata = {
            'total_samples': len(combined_features),
            'total_features': len(combined_features.columns),
            'feature_types': {
                'spatial': ['latitude', 'longitude', 'ocean_region', 'distance_to_coast', 'depth'],
                'oceanographic': ['sst', 'ssh_anom', 'current_speed', 'current_direction', 'chl', 'sss', 'precipitation'],
                'temporal': ['datetime']
            },
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = self.output_dir / 'nasa_features_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Metadata saved to: {metadata_path}")
        
        return combined_features
    
    def _add_spatial_features(self, df):
        """Add spatial oceanographic features"""
        # Ocean regions
        df['ocean_region'] = self._classify_ocean_region(df['latitude'], df['longitude'])
        
        # Distance to coast (simplified)
        df['distance_to_coast'] = np.abs(df['latitude']) * 111  # Rough conversion to km
        
        # Depth estimation
        df['depth'] = 1000 + np.abs(df['latitude']) * 50 + np.random.normal(0, 200, len(df))
        df['depth'] = np.clip(df['depth'], 10, 5000)
        
        # Continental shelf indicator
        df['continental_shelf'] = (df['depth'] < 200).astype(int)
        
        # Open ocean indicator
        df['open_ocean'] = (df['depth'] > 1000).astype(int)
        
        return df
    
    def _classify_ocean_region(self, lat, lon):
        """Classify ocean regions based on latitude/longitude"""
        regions = []
        for i in range(len(lat)):
            if lat.iloc[i] > 30:
                regions.append('north_atlantic')
            elif lat.iloc[i] < -30:
                regions.append('south_atlantic')
            elif lon.iloc[i] > 0:
                regions.append('east_atlantic')
            else:
                regions.append('west_atlantic')
        return regions

def main():
    """Main function"""
    processor = NASADataProcessor()
    
    try:
        features = processor.create_oceanographic_features()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
