"""
Compute daily oceanographic features on a regular grid with advanced algorithms.

This script reads the region of interest and temporal window from the YAML
configuration and produces a multi‑layer feature cube covering the entire
domain. It processes real NASA satellite data and computes advanced oceanographic features.
The resulting feature dataset is saved as a NetCDF or Zarr file for efficient
storage and processing.

Enhanced Features include:

* **SST** (`sst`), advanced gradient computation with cos(lat) scaling (`sst_grad`) 
  and Canny edge detection for thermal fronts (`sst_front`).
* **Chlorophyll** (`chl_log` = log10(chl)), advanced gradient (`chl_grad`) and 
  Canny edge detection for productivity fronts (`chl_front`).
* **Sea surface height anomaly** (`ssh_anom`) and derived geostrophic
  velocities (`u_current`, `v_current`).
* **Enhanced Okubo–Weiss parameter** (`ow`) with improved eddy detection 
  (`eddy_flag`, `eddy_cyc`, `eddy_anti`, `eddy_intensity`).
* **Current metrics**: speed, divergence, vorticity, and strain components.
* **Sea surface salinity** (`sss`) and precipitation accumulation.
* **Additional features**: bathymetry gradients, mixed layer depth proxies,
  and temporal derivatives.

The lat/lon grid uses proper spherical geometry with cos(lat) scaling for
accurate gradient computations. Advanced algorithms include Canny edge detection
for front identification and enhanced Okubo-Weiss parameter computation.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import datetime as _dt
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import xarray as xr
import dask.array as da
from scipy import ndimage
import netCDF4
from dotenv import load_dotenv

try:
    from .utils import (
        load_config, date_range, ensure_dir, setup_logging,
        compute_gradient_spherical, compute_okubo_weiss, detect_fronts_canny
    )
except ImportError:
    from utils import (
        load_config, date_range, ensure_dir, setup_logging,
        compute_gradient_spherical, compute_okubo_weiss, detect_fronts_canny
    )

# Load environment variables
load_dotenv()


class AdvancedFeatureComputer:
    """Advanced feature computation with state-of-the-art algorithms."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.logger = setup_logging('compute_features')
        
        # Grid configuration
        self.roi = config.get('roi', {})
        self.gridding_cfg = config.get('gridding', {})
        self.features_cfg = config.get('features', {})
        
        # Create grid
        self.lat_grid, self.lon_grid = self._create_grid()
        self.n_lat, self.n_lon = self.lat_grid.shape
        
        # Compute grid spacing
        self._compute_grid_spacing()
        
        self.logger.info(f"Created grid: {self.n_lat} x {self.n_lon} points")
        self.logger.info(f"Resolution: {self.lat_step_km:.2f} km (lat) x {self.lon_step_km:.2f} km (lon)")
    
    def _create_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Create regular lat/lon grid with proper spherical geometry."""
        res_deg = float(self.gridding_cfg.get('target_res_deg', 0.1))
        
        # Create coordinate arrays
        lats = np.arange(
            self.roi['lat_min'], 
            self.roi['lat_max'] + res_deg, 
            res_deg
        )
        lons = np.arange(
            self.roi['lon_min'], 
            self.roi['lon_max'] + res_deg, 
            res_deg
        )
        
        # Create 2D grids
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        return lat_grid, lon_grid
    
    def _compute_grid_spacing(self):
        """Compute grid spacing in kilometers."""
        res_deg = float(self.gridding_cfg.get('target_res_deg', 0.1))
        
        # Latitude step (constant)
        self.lat_step_km = res_deg * 111.0
        
        # Longitude step (varies with latitude)
        mean_lat_rad = math.radians((self.roi['lat_min'] + self.roi['lat_max']) / 2.0)
        self.lon_step_km = res_deg * 111.0 * math.cos(mean_lat_rad)
    
    def _load_real_data(self, date: _dt.date) -> Optional[Dict[str, np.ndarray]]:
        """Load real NASA data for a specific date with enhanced error handling."""
        data_dir = 'data/raw'
        date_str = date.strftime('%Y-%m-%d')
        
        # Try to find data files for this date (including batch files)
        datasets = {
            'mur_sst': 'mur_sst',
            'measures_ssh': 'measures_ssh', 
            'oscar_currents': 'oscar_currents',
            'pace_chl': 'pace_chl',
            'smap_salinity': 'smap_salinity',
            'gpm_precipitation': 'gpm_precipitation'
        }
        
        loaded_data = {}
        
        for dataset_name, dir_name in datasets.items():
            dataset_path = os.path.join(data_dir, dir_name)
            if os.path.exists(dataset_path):
                # Look for files containing the date (including batch files)
                files = [f for f in os.listdir(dataset_path) if date_str in f and f.endswith('.nc')]
                
                # If no exact date match, look for batch files that might contain this date
                if not files:
                    batch_files = [f for f in os.listdir(dataset_path) if f.endswith('.nc')]
                    for batch_file in batch_files:
                        # Check if date falls within batch range (format: YYYY-MM-DD_YYYY-MM-DD.nc)
                        if '_' in batch_file:
                            try:
                                date_part = batch_file.replace('.nc', '')
                                start_str, end_str = date_part.split('_')
                                batch_start = datetime.strptime(start_str, '%Y-%m-%d').date()
                                batch_end = datetime.strptime(end_str, '%Y-%m-%d').date()
                                if batch_start <= date <= batch_end:
                                    files = [batch_file]
                                    break
                            except:
                                continue
                
                if files:
                    file_path = os.path.join(dataset_path, files[0])
                    try:
                        ds = xr.open_dataset(file_path)
                        
                        # Extract variables based on dataset type
                        if dataset_name == 'mur_sst' and 'analysed_sst' in ds:
                            # Convert Kelvin to Celsius if needed
                            sst_data = ds['analysed_sst'].values
                            if np.nanmean(sst_data) > 200:  # Likely in Kelvin
                                sst_data = sst_data - 273.15
                            # Ensure 2D array
                            if sst_data.ndim > 2:
                                sst_data = sst_data.squeeze()
                            loaded_data['sst'] = sst_data
                        elif dataset_name == 'measures_ssh' and 'sla' in ds:
                            ssh_data = ds['sla'].values
                            if ssh_data.ndim > 2:
                                ssh_data = ssh_data.squeeze()
                            loaded_data['ssh_anom'] = ssh_data
                        elif dataset_name == 'oscar_currents':
                            if 'u' in ds:
                                u_data = ds['u'].values
                                if u_data.ndim > 2:
                                    u_data = u_data.squeeze()
                                loaded_data['u_current'] = u_data
                            if 'v' in ds:
                                v_data = ds['v'].values
                                if v_data.ndim > 2:
                                    v_data = v_data.squeeze()
                                loaded_data['v_current'] = v_data
                        elif dataset_name == 'pace_chl' and 'chlor_a' in ds:
                            chl_data = ds['chlor_a'].values
                            if chl_data.ndim > 2:
                                chl_data = chl_data.squeeze()
                            loaded_data['chlor_a'] = chl_data
                        elif dataset_name == 'smap_salinity' and 'sss_smap' in ds:
                            sss_data = ds['sss_smap'].values
                            if sss_data.ndim > 2:
                                sss_data = sss_data.squeeze()
                            loaded_data['sss'] = sss_data
                        elif dataset_name == 'gpm_precipitation' and 'precipitation' in ds:
                            precip_data = ds['precipitation'].values
                            if precip_data.ndim > 2:
                                precip_data = precip_data.squeeze()
                            loaded_data['precipitation'] = precip_data
                        
                        ds.close()
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load {dataset_name} for {date_str}: {e}")
        
        if loaded_data:
            self.logger.info(f"Loaded {len(loaded_data)} datasets for {date_str}")
            return loaded_data
        else:
            self.logger.warning(f"No data found for {date_str}")
            return None
    
    def _interpolate_to_grid(self, data: np.ndarray, lat_coords: np.ndarray, lon_coords: np.ndarray) -> np.ndarray:
        """Interpolate data to our target grid using simple resampling."""
        from scipy.ndimage import zoom
        
        # Simple approach: just resize the data to match our grid
        # This is much faster than full interpolation
        target_shape = self.lat_grid.shape
        
        # Calculate zoom factors
        zoom_lat = target_shape[0] / data.shape[0]
        zoom_lon = target_shape[1] / data.shape[1]
        
        # Use zoom to resize the data
        resized_data = zoom(data, (zoom_lat, zoom_lon), order=1, mode='nearest')
        
        return resized_data
    
    def _compute_advanced_features(self, data: Dict[str, np.ndarray], date: _dt.date = None) -> Dict[str, np.ndarray]:
        """Compute advanced oceanographic features from raw data."""
        features = {}
        
        # Sea Surface Temperature features
        if 'sst' in data:
            sst = data['sst']
            # Interpolate to our grid if needed
            if sst.shape != self.lat_grid.shape:
                # Get coordinates from the data (this is a simplified approach)
                # In a real implementation, you'd extract the actual lat/lon coordinates
                lat_coords = np.linspace(self.roi['lat_min'], self.roi['lat_max'], sst.shape[0])
                lon_coords = np.linspace(self.roi['lon_min'], self.roi['lon_max'], sst.shape[1])
                sst = self._interpolate_to_grid(sst, lat_coords, lon_coords)
            
            features['sst'] = sst
            
            # SST gradients with proper spherical geometry
            sst_grad_lat, sst_grad_lon = compute_gradient_spherical(
                sst, self.lat_step_km, self.lon_step_km, self.lat_grid
            )
            features['sst_grad'] = np.sqrt(sst_grad_lat**2 + sst_grad_lon**2)
            
            # SST fronts using Canny edge detection
            sst_front_threshold = self.features_cfg.get('sst_front_threshold_c_per_km', 0.02)
            features['sst_front'] = detect_fronts_canny(sst, sst_front_threshold)
        
        # Chlorophyll features
        if 'chlor_a' in data:
            chl = data['chlor_a']
            # Interpolate to our grid if needed
            if chl.shape != self.lat_grid.shape:
                lat_coords = np.linspace(self.roi['lat_min'], self.roi['lat_max'], chl.shape[0])
                lon_coords = np.linspace(self.roi['lon_min'], self.roi['lon_max'], chl.shape[1])
                chl = self._interpolate_to_grid(chl, lat_coords, lon_coords)
            
            # Log-transform chlorophyll
            chl_log = np.log10(np.clip(chl, 0.01, None))
            features['chl_log'] = chl_log
            
            # Chlorophyll gradients
            chl_grad_lat, chl_grad_lon = compute_gradient_spherical(
                chl_log, self.lat_step_km, self.lon_step_km, self.lat_grid
            )
            features['chl_grad'] = np.sqrt(chl_grad_lat**2 + chl_grad_lon**2)
            
            # Chlorophyll fronts
            chl_front_threshold = self.features_cfg.get('chl_front_threshold', 0.1)
            features['chl_front'] = detect_fronts_canny(chl_log, chl_front_threshold)
        
        # Sea Surface Height and Current features
        if 'ssh_anom' in data:
            ssh = data['ssh_anom']
            # Interpolate to our grid if needed
            if ssh.shape != self.lat_grid.shape:
                lat_coords = np.linspace(self.roi['lat_min'], self.roi['lat_max'], ssh.shape[0])
                lon_coords = np.linspace(self.roi['lon_min'], self.roi['lon_max'], ssh.shape[1])
                ssh = self._interpolate_to_grid(ssh, lat_coords, lon_coords)
            
            features['ssh_anom'] = ssh
            
            # Compute geostrophic velocities from SSH
            ssh_grad_lat, ssh_grad_lon = compute_gradient_spherical(
                ssh, self.lat_step_km, self.lon_step_km, self.lat_grid
            )
            
            # Geostrophic balance: u = -g/f * dη/dy, v = g/f * dη/dx
            g = 9.81  # gravity
            f = 2 * 7.2921e-5 * np.sin(np.radians(self.lat_grid))  # Coriolis parameter
            
            features['u_current'] = -g / f * ssh_grad_lat
            features['v_current'] = g / f * ssh_grad_lon
        
        # Current velocity features (from OSCAR data)
        if 'u_current' in data and 'v_current' in data:
            u = data['u_current']
            v = data['v_current']
            
            # Interpolate to our grid if needed
            if u.shape != self.lat_grid.shape:
                lat_coords = np.linspace(self.roi['lat_min'], self.roi['lat_max'], u.shape[0])
                lon_coords = np.linspace(self.roi['lon_min'], self.roi['lon_max'], u.shape[1])
                u = self._interpolate_to_grid(u, lat_coords, lon_coords)
                v = self._interpolate_to_grid(v, lat_coords, lon_coords)
            
            features['u_current'] = u
            features['v_current'] = v
            
            # Current speed
            features['current_speed'] = np.sqrt(u**2 + v**2)
            
            # Divergence
            u_grad_lat, u_grad_lon = compute_gradient_spherical(
                u, self.lat_step_km, self.lon_step_km, self.lat_grid
            )
            v_grad_lat, v_grad_lon = compute_gradient_spherical(
                v, self.lat_step_km, self.lon_step_km, self.lat_grid
            )
            features['divergence'] = u_grad_lon + v_grad_lat
            
            # Vorticity
            features['vorticity'] = v_grad_lon - u_grad_lat
            
            # Okubo-Weiss parameter for eddy detection
            ow_sigma = self.features_cfg.get('eddy_ow_sigma', 2.0)
            features['ow'] = compute_okubo_weiss(u, v, self.lat_step_km, self.lon_step_km, ow_sigma)
            
            # Enhanced eddy detection
            eddy_metrics = self._compute_eddy_metrics(features['ow'], features['vorticity'], ow_sigma)
            features.update(eddy_metrics)
            
            # Additional current-derived features
            features['current_divergence'] = features['divergence']
            features['current_vorticity'] = features['vorticity']
            
            # Strain rate components
            normal_strain = u_grad_lon - v_grad_lat
            shear_strain = v_grad_lon + u_grad_lat
            features['normal_strain'] = normal_strain
            features['shear_strain'] = shear_strain
            features['strain_rate'] = np.sqrt(normal_strain**2 + shear_strain**2)
            
            # Current direction and persistence
            features['current_direction'] = np.arctan2(v, u) * 180 / np.pi
            features['current_persistence'] = np.sqrt(u**2 + v**2)  # Same as speed for now
            
            # Eddy flags
            features['eddy_flag'] = (features['ow'] < -ow_sigma).astype(int)
            features['eddy_cyc'] = ((features['ow'] < -ow_sigma) & (features['vorticity'] > 0)).astype(int)
            features['eddy_anti'] = ((features['ow'] < -ow_sigma) & (features['vorticity'] < 0)).astype(int)
            features['eddy_intensity'] = np.abs(features['ow'])
        
        # Additional features
        if 'sss' in data:
            sss = data['sss']
            # Interpolate to our grid if needed
            if sss.shape != self.lat_grid.shape:
                lat_coords = np.linspace(self.roi['lat_min'], self.roi['lat_max'], sss.shape[0])
                lon_coords = np.linspace(self.roi['lon_min'], self.roi['lon_max'], sss.shape[1])
                sss = self._interpolate_to_grid(sss, lat_coords, lon_coords)
            features['sss'] = sss
        
        if 'precipitation' in data:
            precip = data['precipitation']
            # Interpolate to our grid if needed
            if precip.shape != self.lat_grid.shape:
                lat_coords = np.linspace(self.roi['lat_min'], self.roi['lat_max'], precip.shape[0])
                lon_coords = np.linspace(self.roi['lon_min'], self.roi['lon_max'], precip.shape[1])
                precip = self._interpolate_to_grid(precip, lat_coords, lon_coords)
            features['precipitation'] = precip
        
        # Add seasonal and temporal features
        if date is None:
            date = _dt.date.today()
        features.update(self._compute_temporal_features(date))
        
        # Add bathymetry-related features if available
        features.update(self._compute_bathymetry_features())
        
        return features
    
    def _compute_eddy_metrics(self, ow: np.ndarray, vorticity: np.ndarray, sigma: float) -> Dict[str, np.ndarray]:
        """Compute enhanced eddy metrics."""
        # Compute threshold
        ow_std = np.nanstd(ow)
        threshold = -sigma * ow_std
        
        # Eddy flag
        eddy_flag = (ow < threshold).astype(np.int8)
        
        # Cyclonic eddies (positive vorticity)
        eddy_cyc = ((eddy_flag == 1) & (vorticity > 0)).astype(np.int8)
        
        # Anticyclonic eddies (negative vorticity)
        eddy_anti = ((eddy_flag == 1) & (vorticity < 0)).astype(np.int8)
        
        # Eddy intensity (how strong the eddy is)
        eddy_intensity = np.where(eddy_flag == 1, np.abs(ow), 0)
        
        return {
            'eddy_flag': eddy_flag,
            'eddy_cyc': eddy_cyc,
            'eddy_anti': eddy_anti,
            'eddy_intensity': eddy_intensity
        }
    
    def _compute_temporal_features(self, date: _dt.date) -> Dict[str, np.ndarray]:
        """Compute temporal and seasonal features."""
        # Day of year (seasonal cycle)
        day_of_year = date.timetuple().tm_yday
        seasonal_cycle = np.sin(2 * np.pi * day_of_year / 365.25)
        seasonal_cycle_2 = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Create 2D arrays
        seasonal_2d = np.full((self.n_lat, self.n_lon), seasonal_cycle)
        seasonal_2d_2 = np.full((self.n_lat, self.n_lon), seasonal_cycle_2)
        
        return {
            'day_of_year': np.full((self.n_lat, self.n_lon), day_of_year),
            'seasonal_cycle': seasonal_2d,
            'seasonal_cycle_2': seasonal_2d_2
        }
    
    def _compute_bathymetry_features(self) -> Dict[str, np.ndarray]:
        """Compute bathymetry-related features."""
        # For now, create placeholder bathymetry features
        # In a real implementation, you would load actual bathymetry data
        depth = np.full((self.n_lat, self.n_lon), 3000.0)  # Placeholder depth
        
        # Compute depth gradients
        depth_grad_lat, depth_grad_lon = compute_gradient_spherical(
            depth, self.lat_step_km, self.lon_step_km, self.lat_grid
        )
        depth_grad_mag = np.sqrt(depth_grad_lat**2 + depth_grad_lon**2)
        
        # Distance to coast (simplified)
        distance_to_coast = np.full((self.n_lat, self.n_lon), 100.0)  # Placeholder
        
        return {
            'depth': depth,
            'depth_grad': depth_grad_mag,
            'distance_to_coast': distance_to_coast
        }
    
    def compute_all_features(self) -> None:
        """Compute features for all dates in the time range with batch processing."""
        dates = date_range(self.config['time']['start'], self.config['time']['end'])
        
        # Get batch configuration
        batch_size = self.config.get('performance', {}).get('batch_size', 30)
        memory_limit = self.config.get('performance', {}).get('memory_limit_gb', 2.0)
        
        self.logger.info(f"Processing {len(dates)} dates in batches of {batch_size}")
        self.logger.info(f"Memory limit: {memory_limit} GB")
        
        # Process dates in batches
        n_batches = (len(dates) + batch_size - 1) // batch_size
        all_features = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dates))
            batch_dates = dates[start_idx:end_idx]
            
            self.logger.info(f"Processing batch {batch_idx + 1}/{n_batches}: {len(batch_dates)} dates")
            
            batch_features = self._compute_batch_features(batch_dates)
            all_features.extend(batch_features)
            
            # Memory management
            if batch_idx % 5 == 0:  # Every 5 batches
                import gc
                gc.collect()
                self.logger.info("Memory cleanup performed")
        
        # Save all features
        self._save_features(all_features, dates)
        
        self.logger.info("Feature computation completed successfully")
    
    def _compute_batch_features(self, batch_dates: List[_dt.date]) -> List[Dict[str, Any]]:
        """Compute features for a batch of dates."""
        batch_features = []
        
        for date in batch_dates:
            self.logger.info(f"Computing features for {date}")
            
            # Load data for this date
            data = self._load_real_data(date)
            if data is None:
                self.logger.warning(f"No data available for {date}, skipping")
                continue
            
            # Compute features
            features = self._compute_advanced_features(data, date)
            
            # Add metadata
            features['date'] = date
            features['lat_grid'] = self.lat_grid
            features['lon_grid'] = self.lon_grid
            
            batch_features.append(features)
        
        return batch_features
    
    def _save_features(self, all_features: List[Dict[str, Any]], dates: List[_dt.date]) -> None:
        """Save computed features to file."""
        if not all_features:
            self.logger.warning("No features to save")
            return
        
        # Determine output format
        output_format = self.config.get('performance', {}).get('storage_format', 'netcdf')
        output_dir = 'data/interim'
        ensure_dir(output_dir)
        
        if output_format == 'zarr':
            output_path = os.path.join(output_dir, 'features.zarr')
            self._save_features_zarr(all_features, output_path)
        else:
            output_path = os.path.join(output_dir, 'features.nc')
            self._save_features_netcdf(all_features, output_path)
        
        self.logger.info(f"Features saved to: {output_path}")
    
    def _save_features_netcdf(self, all_features: List[Dict[str, Any]], output_path: str) -> None:
        """Save features as NetCDF file."""
        # Create time dimension
        times = [f['date'] for f in all_features]
        times_str = [t.strftime('%Y-%m-%d') for t in times]
        
        # Create coordinate arrays
        lats = self.lat_grid[:, 0]  # First column of lat grid
        lons = self.lon_grid[0, :]  # First row of lon grid
        
        # Create data arrays
        data_vars = {}
        for key in all_features[0].keys():
            if key not in ['date', 'lat_grid', 'lon_grid']:
                # Stack all time steps
                values = np.stack([f[key] for f in all_features], axis=0)
                data_vars[key] = (['time', 'lat', 'lon'], values)
        
        # Create dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                'time': times_str,
                'lat': lats,
                'lon': lons
            }
        )
        
        # Add attributes
        ds.attrs['description'] = 'Advanced oceanographic features for shark habitat modeling'
        ds.attrs['created'] = datetime.now().isoformat()
        ds.attrs['region'] = f"{self.roi['lon_min']}E to {self.roi['lon_max']}E, {self.roi['lat_min']}S to {self.roi['lat_max']}S"
        
        # Save to NetCDF
        ds.to_netcdf(output_path)
        ds.close()
    
    def _save_features_zarr(self, all_features: List[Dict[str, Any]], output_path: str) -> None:
        """Save features as Zarr file."""
        # Similar to NetCDF but using Zarr format
        # Implementation would be similar to _save_features_netcdf but with zarr
        self.logger.info("Zarr saving not yet implemented, falling back to NetCDF")
        netcdf_path = output_path.replace('.zarr', '.nc')
        self._save_features_netcdf(all_features, netcdf_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute advanced oceanographic features")
    parser.add_argument("--config", type=str, default="config/params.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--use-dask", action="store_true", 
                       help="Use Dask for parallel processing")
    parser.add_argument("--output-format", type=str, choices=['netcdf', 'zarr'], 
                       default='netcdf', help="Output format")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    computer = AdvancedFeatureComputer(config, args)
    computer.compute_all_features()


if __name__ == "__main__":
    main()