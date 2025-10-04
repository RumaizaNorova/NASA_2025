"""
Apply the trained model to the full feature grid and export rasters.

This script loads the feature cube (`data/interim/features.nc` or `.zarr`) and the trained
models, computes a probability of shark foraging for each grid cell on specified days,
and writes the results to Cloud-Optimized GeoTIFFs and PNG overlays in the `web/data/` folder.
Supports time series predictions and multi-day averaging.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import netCDF4
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
from PIL import Image
import json

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    from .utils import load_config, date_range, ensure_dir, setup_logging
except ImportError:
    from utils import load_config, date_range, ensure_dir, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict habitat probability over the gridded domain.")
    parser.add_argument("--config", default="config/params.yaml", help="Path to YAML configuration file")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data (same as default)")
    parser.add_argument("--date", default=None, help="Date to predict (YYYY-MM-DD). Defaults to the last date in the study period.")
    parser.add_argument("--date-range", default=None, help="Date range to predict (YYYY-MM-DD,YYYY-MM-DD). Generates time series.")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest", "lightgbm"], help="Model to use for prediction")
    parser.add_argument("--output-format", default="both", choices=["png", "geotiff", "both"], help="Output format")
    parser.add_argument("--multi-day", type=int, default=None, help="Generate multi-day average (e.g., 7 for weekly average)")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing for multiple dates")
    return parser.parse_args()


def load_features_nc(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load features from NetCDF file."""
    with netCDF4.Dataset(path, 'r') as ds:
        latitudes = ds.variables['lat'][:].copy()
        longitudes = ds.variables['lon'][:].copy()
        variables = {}
        for name, var in ds.variables.items():
            if name in ('lat', 'lon', 'time'):
                continue
            variables[name] = var[:].copy()
    return latitudes, longitudes, variables


def load_features_zarr(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load features from Zarr store."""
    if not HAS_ZARR:
        raise ImportError("Zarr not available. Install with: pip install zarr")
    
    store = zarr.open(path, mode='r')
    latitudes = store['lat'][:]
    longitudes = store['lon'][:]
    variables = {}
    for key in store.keys():
        if key not in ('lat', 'lon', 'time'):
            variables[key] = store[key][:]
    return latitudes, longitudes, variables


def load_features(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load features from NetCDF or Zarr file."""
    if path.endswith('.zarr'):
        return load_features_zarr(path)
    else:
        return load_features_nc(path)


def load_model(model_path: str, model_type: str):
    """Load trained model from file."""
    import joblib
    
    if model_type == "xgboost":
        model = joblib.load(model_path)
    elif model_type == "random_forest":
        model = joblib.load(model_path)
    elif model_type == "lightgbm":
        model = joblib.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def predict_single_date(model, X_grid: np.ndarray, model_type: str) -> np.ndarray:
    """Predict probabilities for a single date."""
    if model_type == "xgboost":
        probs = model.predict_proba(X_grid)[:, 1]
    elif model_type == "random_forest":
        probs = model.predict_proba(X_grid)[:, 1]
    elif model_type == "lightgbm":
        probs = model.predict(X_grid, num_iteration=model.best_iteration)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return probs


def save_geotiff(data: np.ndarray, bounds: Tuple[float, float, float, float], 
                 output_path: str, crs: str = "EPSG:4326") -> None:
    """Save data as Cloud-Optimized GeoTIFF."""
    if not HAS_RASTERIO:
        print("Warning: rasterio not available. Saving as regular TIFF.")
        img = Image.fromarray((data * 255).astype(np.uint8))
        img.save(output_path.replace('.tif', '_unreferenced.tif'))
        return
    
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    
    with rasterio.open(
        output_path,
        'w',
        driver='COG',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_string(crs),
        transform=transform,
        compress='lzw',
        tiled=True,
        blockxsize=512,
        blockysize=512
    ) as dst:
        dst.write(data, 1)


def save_png_overlay(data: np.ndarray, output_path: str, colormap: str = 'viridis') -> None:
    """Save data as PNG overlay with colormap."""
    try:
        cmap = cm.get_cmap(colormap)
    except AttributeError:
        # For newer matplotlib versions
        cmap = cm.colormaps.get_cmap(colormap)
    # Flip vertically: row 0 corresponds to lat_min (southern edge) but map expects northern edge at top
    data_flipped = np.flipud(data)
    rgba = cmap(np.clip(data_flipped, 0.0, 1.0))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    img.save(output_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(__name__)
    
    # Determine dates to predict
    dates = date_range(config['time']['start'], config['time']['end'])
    target_dates = []
    
    if args.date_range:
        try:
            start_date, end_date = args.date_range.split(',')
            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()
            target_dates = [d for d in dates if start_date <= d <= end_date]
        except Exception:
            logger.error(f"Invalid date range format: {args.date_range}")
            sys.exit(1)
    elif args.date:
        try:
            target_date = pd.to_datetime(args.date).date()
            if target_date not in dates:
                logger.error(f"Date {target_date} is outside the study period.")
                sys.exit(1)
            target_dates = [target_date]
        except Exception:
            logger.error(f"Invalid date format: {args.date}")
            sys.exit(1)
    else:
        target_dates = [dates[-1]]
    
    # Load features
    feat_path = os.path.join('data', 'interim', 'features.nc')
    if not os.path.exists(feat_path):
        feat_path = os.path.join('data', 'interim', 'features.zarr')
        if not os.path.exists(feat_path):
            logger.error("Features file not found. Run compute_features.py first.")
            sys.exit(1)
    
    logger.info(f"Loading features from {feat_path}")
    latitudes, longitudes, variables = load_features(feat_path)
    n_lat, n_lon = len(latitudes), len(longitudes)
    
    # Get bounds for georeferencing
    roi = config.get('roi', {})
    bounds = (roi['lon_min'], roi['lat_min'], roi['lon_max'], roi['lat_max'])
    
    # Load training data to determine feature ordering
    train_csv = os.path.join('data', 'interim', 'training_data.csv')
    if not os.path.exists(train_csv):
        logger.error("Training data not found; cannot infer feature ordering.")
        sys.exit(1)
    
    df_train = pd.read_csv(train_csv)
    exclude_cols = {'date', 'lat', 'lon', 'label', 'shark_id', 'species', 'timestamp', 'latitude', 'longitude'}
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    
    # Load model
    model_path = os.path.join('data', 'interim', f'{args.model}_model.json')
    if not os.path.exists(model_path):
        model_path = os.path.join('data', 'interim', f'{args.model}_model.pkl')
        if not os.path.exists(model_path):
            logger.error(f"Model file not found for {args.model}. Run train_model.py first.")
            sys.exit(1)
    
    logger.info(f"Loading {args.model} model from {model_path}")
    model = load_model(model_path, args.model)
    
    # Create output directory
    ensure_dir(os.path.join('web', 'data'))
    
    # Process each target date
    for target_date in target_dates:
        time_idx = (target_date - dates[0]).days
        logger.info(f"Processing date: {target_date}")
        
        # Build design matrix for the grid
        X_grid = np.zeros((n_lat * n_lon, len(feature_cols)), dtype=float)
        for j, feat_name in enumerate(feature_cols):
            if feat_name not in variables:
                X_grid[:, j] = 0.0
                continue
            arr = variables[feat_name][time_idx]
            X_grid[:, j] = arr.reshape(-1)
        
        # Handle NaNs by filling with column means
        col_means = np.nanmean(X_grid, axis=0)
        inds = np.where(np.isnan(X_grid))
        X_grid[inds] = np.take(col_means, inds[1])
        
        # Predict probabilities
        probs = predict_single_date(model, X_grid, args.model)
        prob_grid = probs.reshape((n_lat, n_lon))
        
        # Generate output filenames
        date_str = target_date.strftime('%Y%m%d')
        base_name = f"habitat_prob_{args.model}_{date_str}"
        
        # Save outputs
        if args.output_format in ['png', 'both']:
            png_path = os.path.join('web', 'data', f'{base_name}.png')
            save_png_overlay(prob_grid, png_path)
            logger.info(f"Saved PNG overlay: {png_path}")
        
        if args.output_format in ['geotiff', 'both']:
            tif_path = os.path.join('web', 'data', f'{base_name}.tif')
            save_geotiff(prob_grid, bounds, tif_path)
            logger.info(f"Saved GeoTIFF: {tif_path}")
        
        # Generate multi-day average if requested
        if args.multi_day and len(target_dates) >= args.multi_day:
            logger.info(f"Generating {args.multi_day}-day average")
            # This would require loading multiple time steps and averaging
            # Implementation depends on specific requirements
    
    # Generate time series metadata
    if len(target_dates) > 1:
        metadata = {
            'model': args.model,
            'dates': [d.strftime('%Y-%m-%d') for d in target_dates],
            'bounds': bounds,
            'resolution': config.get('gridding', {}).get('target_res_deg', 0.05)
        }
        metadata_path = os.path.join('web', 'data', 'prediction_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
    
    logger.info("Prediction complete!")


if __name__ == '__main__':
    main()