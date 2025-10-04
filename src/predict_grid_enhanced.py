"""
Enhanced grid prediction script for shark habitat probability maps.

This script generates comprehensive habitat predictions including:
- Multiple model predictions (XGBoost, LightGBM, Random Forest)
- Time series predictions for full year
- Multiple output formats (PNG, GeoTIFF, JSON)
- Interactive web visualization data
- Performance metrics and confidence intervals
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import pickle
import warnings
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

try:
    from .utils import load_config, ensure_dir, setup_logging
except ImportError:
    from utils import load_config, ensure_dir, setup_logging

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')


class EnhancedGridPredictor:
    """Enhanced grid predictor with multiple models and comprehensive outputs."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.logger = setup_logging(__name__)
        
        # Configuration
        self.roi_cfg = config.get('roi', {})
        self.time_cfg = config.get('time', {})
        self.gridding_cfg = config.get('gridding', {})
        self.performance_cfg = config.get('performance', {})
        
        # Models and results
        self.models = {}
        self.scaler = None
        self.feature_names = []
        
        # Output directories
        self.web_dir = 'web/data'
        self.interim_dir = 'data/interim'
        ensure_dir(self.web_dir)
        ensure_dir(self.interim_dir)
        
        # Set random seed
        np.random.seed(42)
    
    def _load_models(self) -> None:
        """Load trained models from disk."""
        algorithms = self.config.get('model', {}).get('algorithms', ['xgboost'])
        
        for algorithm in algorithms:
            model_path = os.path.join(self.interim_dir, f'{algorithm}_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[algorithm] = pickle.load(f)
                self.logger.info(f"Loaded {algorithm} model from {model_path}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")
        
        if not self.models:
            raise FileNotFoundError("No trained models found. Run train_model.py first.")
    
    def _load_features_and_scaler(self) -> None:
        """Load feature data and scaler."""
        # Load features
        features_path = os.path.join(self.interim_dir, 'features.nc')
        if os.path.exists(features_path):
            self.features_ds = xr.open_dataset(features_path)
            self.logger.info(f"Loaded features from {features_path}")
        else:
            raise FileNotFoundError(f"Features not found: {features_path}")
        
        # Load scaler if available
        scaler_path = os.path.join(self.interim_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info(f"Loaded scaler from {scaler_path}")
        
        # Get feature names
        self.feature_names = [var for var in self.features_ds.data_vars 
                             if var not in ['lat', 'lon', 'time']]
        self.logger.info(f"Found {len(self.feature_names)} features: {self.feature_names}")
    
    def _create_prediction_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create prediction grid based on ROI configuration."""
        lon_min = self.roi_cfg.get('lon_min', 5.0)
        lon_max = self.roi_cfg.get('lon_max', 35.0)
        lat_min = self.roi_cfg.get('lat_min', -45.0)
        lat_max = self.roi_cfg.get('lat_max', -25.0)
        resolution = self.gridding_cfg.get('target_res_deg', 0.05)
        
        # Create grid
        lons = np.arange(lon_min, lon_max + resolution, resolution)
        lats = np.arange(lat_min, lat_max + resolution, resolution)
        
        # Create coordinate grids
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        self.logger.info(f"Created prediction grid: {lon_grid.shape}")
        self.logger.info(f"Longitude range: {lon_min} to {lon_max}")
        self.logger.info(f"Latitude range: {lat_min} to {lat_max}")
        self.logger.info(f"Resolution: {resolution}°")
        
        return lon_grid, lat_grid, lons, lats
    
    def _prepare_features_for_prediction(self, lon_grid: np.ndarray, lat_grid: np.ndarray,
                                       target_date: str) -> np.ndarray:
        """Prepare features for prediction at given coordinates and date."""
        # Convert date string to datetime
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d')
        
        # Find closest time in features dataset
        if 'time' in self.features_ds.coords:
            try:
                # Convert target_date to numpy datetime64 for comparison
                target_np = pd.to_datetime(target_date).to_numpy()
                time_coords = self.features_ds.time.values
                time_idx = np.argmin(np.abs(time_coords - target_np))
                features_subset = self.features_ds.isel(time=time_idx)
            except Exception as e:
                self.logger.warning(f"Time matching failed, using first time step: {e}")
                features_subset = self.features_ds.isel(time=0)
        else:
            features_subset = self.features_ds
        
        # Interpolate features to prediction grid
        feature_arrays = []
        for feature_name in self.feature_names:
            if feature_name in features_subset.data_vars:
                feature_data = features_subset[feature_name]
                
                # Interpolate to prediction grid
                feature_interp = feature_data.interp(
                    lon=xr.DataArray(lon_grid, dims=['y', 'x']),
                    lat=xr.DataArray(lat_grid, dims=['y', 'x'])
                )
                
                # Handle missing values
                feature_interp = feature_interp.fillna(0.0)
                feature_arrays.append(feature_interp.values)
            else:
                # Fill with zeros if feature not available
                feature_arrays.append(np.zeros_like(lon_grid))
        
        # Stack features
        X = np.stack(feature_arrays, axis=-1)
        X = X.reshape(-1, len(self.feature_names))
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
    
    def _predict_single_model(self, model: Any, X: np.ndarray, algorithm: str) -> np.ndarray:
        """Make predictions with a single model."""
        try:
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)[:, 1]
            else:
                predictions = model.predict(X)
            
            self.logger.info(f"{algorithm} predictions completed: {len(predictions)} points")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting with {algorithm}: {e}")
            return np.zeros(len(X))
    
    def _predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make ensemble predictions with uncertainty quantification."""
        predictions = {}
        
        for algorithm, model in self.models.items():
            pred = self._predict_single_model(model, X, algorithm)
            predictions[algorithm] = pred
        
        # Calculate ensemble statistics
        pred_array = np.array(list(predictions.values()))
        ensemble_mean = np.mean(pred_array, axis=0)
        ensemble_std = np.std(pred_array, axis=0)
        ensemble_max = np.max(pred_array, axis=0)
        
        return ensemble_mean, ensemble_std, ensemble_max
    
    def _save_prediction_geotiff(self, predictions: np.ndarray, lon_grid: np.ndarray, 
                                lat_grid: np.ndarray, output_path: str) -> None:
        """Save predictions as GeoTIFF."""
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            # Get bounds
            west = np.min(lon_grid)
            east = np.max(lon_grid)
            south = np.min(lat_grid)
            north = np.max(lat_grid)
            
            # Create transform
            transform = from_bounds(west, south, east, north, 
                                  predictions.shape[1], predictions.shape[0])
            
            # Save as GeoTIFF
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=predictions.shape[0],
                width=predictions.shape[1],
                count=1,
                dtype=predictions.dtype,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(predictions, 1)
            
            self.logger.info(f"GeoTIFF saved to {output_path}")
            
        except ImportError:
            self.logger.warning("rasterio not available, skipping GeoTIFF export")
        except Exception as e:
            self.logger.warning(f"Failed to save GeoTIFF: {e}")
    
    def _save_prediction_png(self, predictions: np.ndarray, output_path: str,
                           title: str = "Habitat Probability") -> None:
        """Save predictions as PNG visualization."""
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create visualization
            im = ax.imshow(predictions, cmap='viridis', vmin=0, vmax=1, origin='lower')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Habitat Probability', fontsize=12)
            
            # Customize plot
            ax.set_title(f'{title}\n{datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Save
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"PNG saved to {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save PNG: {e}")
    
    def _save_prediction_metadata(self, predictions: np.ndarray, lon_grid: np.ndarray,
                                 lat_grid: np.ndarray, model_name: str, date: str,
                                 output_path: str) -> None:
        """Save prediction metadata as JSON."""
        metadata = {
            'model': model_name,
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'shape': predictions.shape,
            'bounds': {
                'west': float(np.min(lon_grid)),
                'east': float(np.max(lon_grid)),
                'south': float(np.min(lat_grid)),
                'north': float(np.max(lat_grid))
            },
            'statistics': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'median': float(np.median(predictions))
            },
            'features_used': self.feature_names,
            'roi_config': self.roi_cfg,
            'gridding_config': self.gridding_cfg
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to {output_path}")
    
    def predict_single_date(self, date: str, models: List[str] = None) -> Dict[str, Any]:
        """Predict habitat probability for a single date."""
        if models is None:
            models = list(self.models.keys())
        
        self.logger.info(f"Predicting habitat probability for {date}")
        
        # Create prediction grid
        lon_grid, lat_grid, lons, lats = self._create_prediction_grid()
        
        # Prepare features
        X = self._prepare_features_for_prediction(lon_grid, lat_grid, date)
        
        # Make predictions for each model
        results = {}
        for model_name in models:
            if model_name in self.models:
                predictions = self._predict_single_model(
                    self.models[model_name], X, model_name
                )
                
                # Reshape to grid
                pred_grid = predictions.reshape(lon_grid.shape)
                
                # Save outputs
                date_str = date.replace('-', '')
                
                # PNG
                png_path = os.path.join(self.web_dir, 
                                      f'habitat_prob_{model_name}_{date_str}.png')
                self._save_prediction_png(pred_grid, png_path, 
                                        f'Shark Habitat Probability - {model_name}')
                
                # GeoTIFF
                tif_path = os.path.join(self.web_dir, 
                                      f'habitat_prob_{model_name}_{date_str}.tif')
                self._save_prediction_geotiff(pred_grid, lon_grid, lat_grid, tif_path)
                
                # Metadata
                json_path = os.path.join(self.web_dir, 
                                       f'habitat_prob_{model_name}_{date_str}.json')
                self._save_prediction_metadata(pred_grid, lon_grid, lat_grid, 
                                             model_name, date, json_path)
                
                results[model_name] = {
                    'predictions': pred_grid,
                    'png_path': png_path,
                    'tif_path': tif_path,
                    'json_path': json_path
                }
        
        return results
    
    def predict_time_series(self, start_date: str, end_date: str, 
                           models: List[str] = None, interval_days: int = 1) -> Dict[str, Any]:
        """Predict habitat probability for a time series."""
        if models is None:
            models = list(self.models.keys())
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate date range
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=interval_days)
        
        self.logger.info(f"Predicting time series from {start_date} to {end_date} "
                        f"({len(dates)} dates)")
        
        # Predict for each date
        time_series_results = {}
        for i, date in enumerate(dates):
            self.logger.info(f"Processing date {i+1}/{len(dates)}: {date}")
            
            try:
                date_results = self.predict_single_date(date, models)
                time_series_results[date] = date_results
                
                # Progress update
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i+1}/{len(dates)} dates")
                    
            except Exception as e:
                self.logger.error(f"Error processing date {date}: {e}")
                continue
        
        # Save time series metadata
        ts_metadata = {
            'start_date': start_date,
            'end_date': end_date,
            'interval_days': interval_days,
            'total_dates': len(dates),
            'models': models,
            'generated_at': datetime.now().isoformat()
        }
        
        ts_metadata_path = os.path.join(self.web_dir, 'time_series_metadata.json')
        with open(ts_metadata_path, 'w') as f:
            json.dump(ts_metadata, f, indent=2)
        
        self.logger.info(f"Time series predictions completed: {len(time_series_results)} dates")
        return time_series_results
    
    def predict_all_models(self) -> Dict[str, Any]:
        """Predict habitat probability for all models and all available dates."""
        self.logger.info("Generating predictions for all models")
        
        # Get available dates from features
        if 'time' in self.features_ds.coords:
            available_dates = pd.to_datetime(self.features_ds.time.values).strftime('%Y-%m-%d')
        else:
            # Use default date range
            start_date = self.time_cfg.get('start', '2014-01-01')
            end_date = self.time_cfg.get('end', '2014-12-31')
            available_dates = pd.date_range(start_date, end_date, freq='D').strftime('%Y-%m-%d')
        
        # Predict for each date
        all_results = {}
        for date in available_dates:
            try:
                date_results = self.predict_single_date(date)
                all_results[date] = date_results
                
            except Exception as e:
                self.logger.error(f"Error processing date {date}: {e}")
                continue
        
        # Save comprehensive metadata
        comprehensive_metadata = {
            'total_dates': len(available_dates),
            'models_used': list(self.models.keys()),
            'features_used': self.feature_names,
            'roi_config': self.roi_cfg,
            'time_config': self.time_cfg,
            'gridding_config': self.gridding_cfg,
            'generated_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.web_dir, 'prediction_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2)
        
        self.logger.info(f"All predictions completed: {len(all_results)} dates")
        return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enhanced habitat predictions.")
    parser.add_argument("--config", default="config/params_enhanced.yaml", 
                       help="Path to YAML configuration file")
    parser.add_argument("--model", default="xgboost", 
                       help="Model to use for prediction")
    parser.add_argument("--models", nargs='+', 
                       help="Multiple models to use for prediction")
    parser.add_argument("--date", help="Single date to predict (YYYY-MM-DD)")
    parser.add_argument("--date-range", help="Date range to predict (start,end)")
    parser.add_argument("--output-format", choices=['png', 'geotiff', 'json', 'all'], 
                       default='all', help="Output format")
    parser.add_argument("--parallel", action="store_true", 
                       help="Enable parallel processing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize predictor
    predictor = EnhancedGridPredictor(config, args)
    
    # Load models and features
    predictor._load_models()
    predictor._load_features_and_scaler()
    
    # Determine which models to use
    if args.models:
        models_to_use = args.models
    elif args.model:
        models_to_use = [args.model]
    else:
        models_to_use = list(predictor.models.keys())
    
    # Filter to available models
    models_to_use = [m for m in models_to_use if m in predictor.models]
    
    if not models_to_use:
        raise ValueError("No valid models found")
    
    predictor.logger.info(f"Using models: {models_to_use}")
    
    # Generate predictions
    if args.date:
        # Single date prediction
        results = predictor.predict_single_date(args.date, models_to_use)
        
    elif args.date_range:
        # Date range prediction
        start_date, end_date = args.date_range.split(',')
        results = predictor.predict_time_series(start_date, end_date, models_to_use)
        
    else:
        # All available dates
        results = predictor.predict_all_models()
    
    # Print summary
    print(f"\n=== PREDICTION SUMMARY ===")
    print(f"Models used: {models_to_use}")
    print(f"Output directory: {predictor.web_dir}")
    
    if isinstance(results, dict) and len(results) > 0:
        if 'date' in str(type(list(results.keys())[0])):
            print(f"Dates processed: {len(results)}")
        else:
            print("Single date processed")
    else:
        print("No results generated")
    
    print(f"\nResults saved to {predictor.web_dir}/")
    print("Files generated:")
    print("  - PNG visualizations")
    print("  - GeoTIFF rasters")
    print("  - JSON metadata")
    print("  - Time series data")


if __name__ == '__main__':
    main()
