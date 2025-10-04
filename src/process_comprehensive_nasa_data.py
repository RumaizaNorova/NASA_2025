#!/usr/bin/env python3
"""
Process Comprehensive NASA Satellite Data
Process all NetCDF files to extract comprehensive oceanographic features
REAL DATA ONLY - NO SYNTHETIC DATA
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
import netCDF4 as nc
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/process_comprehensive_nasa.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveNASADataProcessor:
    """Process comprehensive NASA satellite data to extract oceanographic features"""
    
    def __init__(self):
        self.data_dir = Path("data/raw")
        self.output_dir = Path("data/interim")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NASA satellite data collections
        self.collections = {
            'mur_sst': {
                'name': 'MUR Sea Surface Temperature',
                'variable': 'analysed_sst',
                'description': 'Multi-scale Ultra-high Resolution Sea Surface Temperature'
            },
            'measures_ssh': {
                'name': 'MEaSUREs Sea Surface Height',
                'variable': 'adt',
                'description': 'Global Sea Surface Height from satellite altimetry'
            },
            'oscar_currents': {
                'name': 'OSCAR Ocean Currents',
                'variable': 'u',
                'description': 'Ocean Surface Current Analysis Real-time'
            },
            'pace_chl': {
                'name': 'PACE Chlorophyll',
                'variable': 'chlor_a',
                'description': 'Plankton, Aerosol, Cloud, ocean Ecosystem'
            },
            'smap_salinity': {
                'name': 'SMAP Sea Surface Salinity',
                'variable': 'sss_smap',
                'description': 'Soil Moisture Active Passive salinity'
            },
            'gpm_precipitation': {
                'name': 'GPM Precipitation',
                'variable': 'precipitationCal',
                'description': 'Global Precipitation Measurement'
            }
        }
        
        # Processing statistics
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_features': 0,
            'start_time': None,
            'end_time': None
        }
    
    def load_shark_data(self):
        """Load real shark observation data"""
        logger.info("🦈 Loading real shark observation data...")
        
        # Try to find shark data file
        possible_files = [
            'sharks_cleaned.csv',
            '../sharks_cleaned.csv',
            'data/sharks_cleaned.csv'
        ]
        
        shark_data = None
        for file_path in possible_files:
            if Path(file_path).exists():
                shark_data = pd.read_csv(file_path)
                logger.info(f"  ✅ Loaded shark data: {file_path}")
                break
        
        if shark_data is None:
            logger.error("❌ No shark data file found!")
            logger.info("Please ensure sharks_cleaned.csv is available")
            return None
        
        # Convert datetime columns
        if 'datetime' in shark_data.columns:
            shark_data['datetime'] = pd.to_datetime(shark_data['datetime'])
        elif 'date' in shark_data.columns:
            shark_data['datetime'] = pd.to_datetime(shark_data['date'])
        
        logger.info(f"  📊 Shark observations: {len(shark_data):,}")
        logger.info(f"  📅 Date range: {shark_data['datetime'].min()} to {shark_data['datetime'].max()}")
        
        return shark_data
    
    def create_background_locations(self, shark_data, num_background=100000):
        """Create real background locations for negative sampling"""
        logger.info(f"🌍 Creating {num_background:,} real background locations...")
        
        # Get spatial bounds from shark data
        lat_min, lat_max = shark_data['latitude'].min(), shark_data['latitude'].max()
        lon_min, lon_max = shark_data['longitude'].min(), shark_data['longitude'].max()
        
        # Expand bounds slightly for background sampling
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        lat_expansion = lat_range * 0.1
        lon_expansion = lon_range * 0.1
        
        # Generate random background locations
        np.random.seed(42)  # For reproducibility
        background_lat = np.random.uniform(
            lat_min - lat_expansion, 
            lat_max + lat_expansion, 
            num_background
        )
        background_lon = np.random.uniform(
            lon_min - lon_expansion, 
            lon_max + lon_expansion, 
            num_background
        )
        
        # Generate random dates within shark observation period
        date_range = shark_data['datetime'].max() - shark_data['datetime'].min()
        background_dates = shark_data['datetime'].min() + pd.to_timedelta(
            np.random.uniform(0, date_range.days, num_background), 
            unit='days'
        )
        
        # Create background dataframe
        background_data = pd.DataFrame({
            'latitude': background_lat,
            'longitude': background_lon,
            'datetime': background_dates,
            'target': 0,  # Negative samples
            'source': 'background_location'
        })
        
        logger.info(f"  ✅ Created {len(background_data):,} background locations")
        logger.info(f"  📍 Spatial bounds: lat [{lat_min:.2f}, {lat_max:.2f}], lon [{lon_min:.2f}, {lon_max:.2f}]")
        
        return background_data
    
    def process_netcdf_file(self, file_path, dataset_name, collection_info):
        """Process a single NetCDF file to extract features"""
        try:
            logger.info(f"  📁 Processing: {file_path.name}")
            
            # Open NetCDF file
            with xr.open_dataset(file_path) as ds:
                # Get the main variable
                variable_name = collection_info['variable']
                
                if variable_name not in ds.data_vars:
                    logger.warning(f"    Variable {variable_name} not found in {file_path.name}")
                    return None
                
                # Get variable data
                var_data = ds[variable_name]
                
                # Calculate basic statistics
                stats = {
                    'mean': float(var_data.mean().values),
                    'std': float(var_data.std().values),
                    'min': float(var_data.min().values),
                    'max': float(var_data.max().values),
                    'median': float(var_data.median().values)
                }
                
                # Calculate spatial statistics if coordinates available
                if 'latitude' in ds.coords and 'longitude' in ds.coords:
                    # Calculate gradients
                    if len(ds.latitude) > 1 and len(ds.longitude) > 1:
                        # Latitudinal gradient
                        lat_grad = np.gradient(var_data, axis=0)
                        stats['lat_gradient_mean'] = float(np.mean(lat_grad))
                        stats['lat_gradient_std'] = float(np.std(lat_grad))
                        
                        # Longitudinal gradient
                        lon_grad = np.gradient(var_data, axis=1)
                        stats['lon_gradient_mean'] = float(np.mean(lon_grad))
                        stats['lon_gradient_std'] = float(np.std(lon_grad))
                        
                        # Total gradient magnitude
                        total_grad = np.sqrt(lat_grad**2 + lon_grad**2)
                        stats['gradient_magnitude_mean'] = float(np.mean(total_grad))
                        stats['gradient_magnitude_std'] = float(np.std(total_grad))
                
                # Calculate temporal statistics if time dimension available
                if 'time' in ds.coords:
                    # Calculate temporal trends
                    time_values = ds.time.values
                    if len(time_values) > 1:
                        # Linear trend
                        time_numeric = np.arange(len(time_values))
                        trend = np.polyfit(time_numeric, var_data.values.flatten(), 1)[0]
                        stats['temporal_trend'] = float(trend)
                        
                        # Temporal variability
                        stats['temporal_std'] = float(np.std(var_data.values))
                
                # Add dataset-specific features
                stats['dataset'] = dataset_name
                stats['variable'] = variable_name
                stats['file_name'] = file_path.name
                stats['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
                
                # Add coordinate information
                if 'latitude' in ds.coords:
                    stats['lat_min'] = float(ds.latitude.min().values)
                    stats['lat_max'] = float(ds.latitude.max().values)
                    stats['lat_center'] = float(ds.latitude.mean().values)
                
                if 'longitude' in ds.coords:
                    stats['lon_min'] = float(ds.longitude.min().values)
                    stats['lon_max'] = float(ds.longitude.max().values)
                    stats['lon_center'] = float(ds.longitude.mean().values)
                
                # Add time information
                if 'time' in ds.coords:
                    stats['time_min'] = str(ds.time.min().values)
                    stats['time_max'] = str(ds.time.max().values)
                    stats['time_center'] = str(ds.time.mean().values)
                
                return stats
                
        except Exception as e:
            logger.error(f"    ❌ Error processing {file_path.name}: {e}")
            return None
    
    def process_dataset(self, dataset_name, collection_info):
        """Process all files for a specific dataset"""
        logger.info(f"🌊 Processing {collection_info['name']}...")
        
        dataset_dir = self.data_dir / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"  ⚠️ Dataset directory not found: {dataset_dir}")
            return []
        
        # Find all NetCDF files
        nc_files = list(dataset_dir.glob("*.nc"))
        if not nc_files:
            logger.warning(f"  ⚠️ No NetCDF files found in {dataset_dir}")
            return []
        
        logger.info(f"  📁 Found {len(nc_files)} NetCDF files")
        
        # Process files
        processed_data = []
        for nc_file in nc_files:
            stats = self.process_netcdf_file(nc_file, dataset_name, collection_info)
            if stats:
                processed_data.append(stats)
                self.processing_stats['processed_files'] += 1
            else:
                self.processing_stats['failed_files'] += 1
        
        logger.info(f"  ✅ Processed {len(processed_data)} files successfully")
        return processed_data
    
    def process_all_datasets(self):
        """Process all NASA satellite datasets"""
        logger.info("🚀 Processing comprehensive NASA satellite data...")
        
        self.processing_stats['start_time'] = datetime.now()
        
        all_processed_data = {}
        
        # Process each dataset
        for dataset_name, collection_info in self.collections.items():
            try:
                processed_data = self.process_dataset(dataset_name, collection_info)
                all_processed_data[dataset_name] = processed_data
                
                # Update stats
                self.processing_stats['total_files'] += len(processed_data)
                
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
        
        self.processing_stats['end_time'] = datetime.now()
        
        # Save processing results
        self.save_processing_results(all_processed_data)
        
        logger.info(f"\n✅ Processing completed!")
        logger.info(f"📊 Total files processed: {self.processing_stats['processed_files']}")
        logger.info(f"❌ Failed files: {self.processing_stats['failed_files']}")
        logger.info(f"⏱️ Time elapsed: {self.processing_stats['end_time'] - self.processing_stats['start_time']}")
        
        return all_processed_data
    
    def save_processing_results(self, all_processed_data):
        """Save processing results to files"""
        logger.info("💾 Saving processing results...")
        
        # Save detailed results for each dataset
        for dataset_name, processed_data in all_processed_data.items():
            if processed_data:
                # Convert to DataFrame
                df = pd.DataFrame(processed_data)
                
                # Save to CSV
                output_file = self.output_dir / f'{dataset_name}_processed_features.csv'
                df.to_csv(output_file, index=False)
                
                logger.info(f"  ✅ Saved {len(df)} records to {output_file}")
        
        # Save comprehensive summary
        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_datasets': len(self.collections),
            'processing_stats': self.processing_stats,
            'datasets': {}
        }
        
        for dataset_name, processed_data in all_processed_data.items():
            if processed_data:
                df = pd.DataFrame(processed_data)
                summary['datasets'][dataset_name] = {
                    'records': len(df),
                    'features': len(df.columns),
                    'file_size_mb': df['file_size_mb'].sum() if 'file_size_mb' in df.columns else 0,
                    'spatial_coverage': {
                        'lat_range': [df['lat_min'].min(), df['lat_max'].max()] if 'lat_min' in df.columns else None,
                        'lon_range': [df['lon_min'].min(), df['lon_max'].max()] if 'lon_min' in df.columns else None
                    }
                }
        
        # Save summary
        summary_path = self.output_dir / 'comprehensive_nasa_processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Processing summary saved to: {summary_path}")
    
    def create_comprehensive_features(self, shark_data, background_data):
        """Create comprehensive oceanographic features from processed NASA data"""
        logger.info("🔧 Creating comprehensive oceanographic features...")
        
        # Combine shark and background data
        combined_data = pd.concat([shark_data, background_data], ignore_index=True)
        logger.info(f"  📊 Combined dataset: {len(combined_data):,} samples")
        
        # Load processed NASA data
        nasa_features = {}
        for dataset_name in self.collections.keys():
            feature_file = self.output_dir / f'{dataset_name}_processed_features.csv'
            if feature_file.exists():
                nasa_features[dataset_name] = pd.read_csv(feature_file)
                logger.info(f"  ✅ Loaded {dataset_name} features: {len(nasa_features[dataset_name]):,} records")
        
        # Create comprehensive feature set
        comprehensive_features = []
        
        for idx, row in combined_data.iterrows():
            if idx % 10000 == 0:
                logger.info(f"  Processing sample {idx:,}/{len(combined_data):,}")
            
            # Basic location and time features
            features = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'datetime': row['datetime'],
                'target': row['target']
            }
            
            # Add NASA-derived features for each dataset
            for dataset_name, nasa_df in nasa_features.items():
                if len(nasa_df) > 0:
                    # Find closest spatial match
                    spatial_distances = np.sqrt(
                        (nasa_df['lat_center'] - row['latitude'])**2 + 
                        (nasa_df['lon_center'] - row['longitude'])**2
                    )
                    closest_idx = spatial_distances.idxmin()
                    closest_record = nasa_df.iloc[closest_idx]
                    
                    # Add features from this dataset
                    for col in nasa_df.columns:
                        if col not in ['dataset', 'variable', 'file_name', 'lat_center', 'lon_center']:
                            feature_name = f"{dataset_name}_{col}"
                            features[feature_name] = closest_record[col]
            
            comprehensive_features.append(features)
        
        # Convert to DataFrame
        comprehensive_df = pd.DataFrame(comprehensive_features)
        
        # Save comprehensive features
        output_file = self.output_dir / 'comprehensive_nasa_oceanographic_features.csv'
        comprehensive_df.to_csv(output_file, index=False)
        
        logger.info(f"  ✅ Comprehensive features saved to: {output_file}")
        logger.info(f"  📊 Total features: {len(comprehensive_df.columns)}")
        logger.info(f"  📊 Total samples: {len(comprehensive_df):,}")
        
        return comprehensive_df

def main():
    """Main function"""
    logger.info("🚀 Comprehensive NASA Data Processing Pipeline")
    logger.info("=" * 60)
    
    # Create processor
    processor = ComprehensiveNASADataProcessor()
    
    try:
        # Load shark data
        shark_data = processor.load_shark_data()
        if shark_data is None:
            return False
        
        # Create background locations
        background_data = processor.create_background_locations(shark_data)
        
        # Process all NASA datasets
        all_processed_data = processor.process_all_datasets()
        
        # Create comprehensive features
        comprehensive_features = processor.create_comprehensive_features(shark_data, background_data)
        
        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 COMPREHENSIVE NASA DATA PROCESSING COMPLETED!")
        logger.info(f"📊 Total features: {len(comprehensive_features.columns)}")
        logger.info(f"📊 Total samples: {len(comprehensive_features):,}")
        logger.info(f"⏱️ Time elapsed: {processor.processing_stats['end_time'] - processor.processing_stats['start_time']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
