"""Utility functions shared across the sharks‑from‑space pipeline.

This module contains helpers for loading configuration, parsing environment
variables, logging, and computing simple geographic operations.  Keeping these
functions separate avoids circular imports between scripts.
"""

from __future__ import annotations

import os
import yaml
import logging
import datetime as _dt
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

def load_config(path: str) -> Dict[str, Any]:
    """Read a YAML configuration file and return it as a Python dict.

    Parameters
    ----------
    path : str
        Path to the YAML file (relative or absolute).

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def date_range(start: str, end: str) -> List[_dt.date]:
    """Generate a list of daily dates (inclusive) between two ISO strings.

    Parameters
    ----------
    start : str
        Start date in ISO format (YYYY-MM-DD).
    end : str
        End date in ISO format (YYYY-MM-DD).

    Returns
    -------
    list[datetime.date]
        List of dates from start to end inclusive.
    """
    start_dt = _dt.datetime.fromisoformat(start).date()
    end_dt = _dt.datetime.fromisoformat(end).date()
    days = (end_dt - start_dt).days
    return [start_dt + _dt.timedelta(days=i) for i in range(days + 1)]

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great‑circle distance between two points on Earth.

    Uses the haversine formula for spherical distance.  All inputs and
    outputs are in degrees and kilometres.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Latitude and longitude of the two points (degrees).

    Returns
    -------
    float
        Distance in kilometres.
    """
    import math
    # Earth radius in kilometres
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return R * c

def ensure_dir(path: str) -> None:
    """Ensure that a directory exists.  If it doesn't, create it recursively."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def setup_logging(name: str, level: str = 'INFO') -> logging.Logger:
    """Set up logging for a module.
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__).
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    log_dir = 'logs'
    ensure_dir(log_dir)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name.replace(".", "_")}.log')
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def compute_gradient_with_cos_lat(data: np.ndarray, lat: np.ndarray, 
                                 lon: np.ndarray, method: str = 'advanced') -> Tuple[np.ndarray, np.ndarray]:
    """Compute gradients with proper latitude scaling.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of data values.
    lat : np.ndarray
        1D array of latitudes in degrees.
    lon : np.ndarray
        1D array of longitudes in degrees.
    method : str
        'simple' for constant km/degree, 'advanced' for cos(lat) scaling.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (grad_lat, grad_lon) gradients in physical units per km.
    """
    if method == 'simple':
        # Simple constant conversion
        lat_step_km = 111.0  # km per degree
        lon_step_km = 111.0  # km per degree (approximate)
        
        grad_lat = np.gradient(data, axis=0) / lat_step_km
        grad_lon = np.gradient(data, axis=1) / lon_step_km
        
    else:  # advanced
        # Proper latitude scaling
        lat_step_km = 111.0  # km per degree (constant)
        
        # Longitude step varies with latitude
        lat_2d = np.tile(lat, (len(lon), 1)).T
        lon_step_km = 111.0 * np.cos(np.radians(lat_2d))
        
        grad_lat = np.gradient(data, axis=0) / lat_step_km
        grad_lon = np.gradient(data, axis=1) / lon_step_km
    
    return grad_lat, grad_lon


def detect_fronts_canny(data: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Detect fronts using Canny edge detection.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of data values.
    threshold : float
        Threshold for edge detection.
        
    Returns
    -------
    np.ndarray
        Binary array indicating front locations.
    """
    # Ensure data is 2D
    if data.ndim != 2:
        if data.ndim > 2:
            data = data.squeeze()
        if data.ndim != 2:
            # Return zeros if still not 2D
            return np.zeros_like(data, dtype=np.int8)
    
    # Check if array is large enough for Canny edge detection
    if data.shape[0] < 3 or data.shape[1] < 3:
        # Fallback to simple threshold method for small arrays
        try:
            gradient_magnitude = np.sqrt(
                np.gradient(data, axis=0)**2 + np.gradient(data, axis=1)**2
            )
            threshold = np.nanpercentile(gradient_magnitude, 95)
            return (gradient_magnitude > threshold).astype(np.int8)
        except:
            return np.zeros_like(data, dtype=np.int8)
    
    try:
        from skimage import feature
        
        # Normalize data to 0-1 range
        data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        
        # Apply Canny edge detection
        edges = feature.canny(
            data_norm, 
            low_threshold=threshold * 0.5,
            high_threshold=threshold,
            sigma=1.0
        )
        
        return edges.astype(np.int8)
        
    except ImportError:
        # Fallback to simple threshold method
        try:
            gradient_magnitude = np.sqrt(
                np.gradient(data, axis=0)**2 + np.gradient(data, axis=1)**2
            )
            threshold = np.nanpercentile(gradient_magnitude, 95)
            return (gradient_magnitude > threshold).astype(np.int8)
        except:
            return np.zeros_like(data, dtype=np.int8)


def enhanced_okubo_weiss(u: np.ndarray, v: np.ndarray, lat: np.ndarray, 
                        lon: np.ndarray) -> np.ndarray:
    """Compute enhanced Okubo-Weiss parameter with proper scaling.
    
    Parameters
    ----------
    u : np.ndarray
        Eastward velocity component.
    v : np.ndarray
        Northward velocity component.
    lat : np.ndarray
        Latitude array.
    lon : np.ndarray
        Longitude array.
        
    Returns
    -------
    np.ndarray
        Enhanced Okubo-Weiss parameter.
    """
    # Compute gradients with proper latitude scaling
    dudx, dudy = compute_gradient_with_cos_lat(u, lat, lon, method='advanced')
    dvdx, dvdy = compute_gradient_with_cos_lat(v, lat, lon, method='advanced')
    
    # Strain components
    normal_strain = dudx - dvdy
    shear_strain = dvdx + dudy
    
    # Relative vorticity
    vorticity = dvdx - dudy
    
    # Okubo-Weiss parameter
    ow = normal_strain**2 + shear_strain**2 - vorticity**2
    
    # Enhanced version: apply smoothing to reduce noise
    from scipy import ndimage
    ow_smooth = ndimage.gaussian_filter(ow, sigma=1.0)
    
    return ow_smooth


def compute_eddy_metrics(ow: np.ndarray, vorticity: np.ndarray, 
                        sigma_threshold: float = 1.0) -> Dict[str, np.ndarray]:
    """Compute comprehensive eddy metrics.
    
    Parameters
    ----------
    ow : np.ndarray
        Okubo-Weiss parameter.
    vorticity : np.ndarray
        Relative vorticity.
    sigma_threshold : float
        Sigma multiplier for eddy detection threshold.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing eddy flags and metrics.
    """
    # Compute threshold
    ow_std = np.nanstd(ow)
    threshold = -sigma_threshold * ow_std
    
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


def compute_gradient_spherical(data: np.ndarray, lat_step_km: float, lon_step_km: float, 
                              lat_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient with spherical geometry.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of data values.
    lat_step_km : float
        Latitude step in kilometers.
    lon_step_km : float
        Longitude step in kilometers.
    lat_grid : np.ndarray
        2D latitude grid.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (grad_lat, grad_lon) gradients in physical units per km.
    """
    # Check if array is large enough for gradient computation
    if data.shape[0] < 2 or data.shape[1] < 2:
        # Return zero gradients for small arrays
        grad_lat = np.zeros_like(data)
        grad_lon = np.zeros_like(data)
        return grad_lat, grad_lon
    
    # Compute gradients
    grad_lat = np.gradient(data, axis=0) / lat_step_km
    
    # Longitude gradient with latitude correction
    lon_step_corrected = lon_step_km * np.cos(np.radians(lat_grid))
    grad_lon = np.gradient(data, axis=1) / lon_step_corrected
    
    return grad_lat, grad_lon


def compute_okubo_weiss(u: np.ndarray, v: np.ndarray, lat_step_km: float, 
                       lon_step_km: float, sigma: float = 2.0) -> np.ndarray:
    """Compute Okubo-Weiss parameter for eddy detection.
    
    Parameters
    ----------
    u : np.ndarray
        Eastward velocity component.
    v : np.ndarray
        Northward velocity component.
    lat_step_km : float
        Latitude step in kilometers.
    lon_step_km : float
        Longitude step in kilometers.
    sigma : float
        Sigma multiplier for eddy detection.
        
    Returns
    -------
    np.ndarray
        Okubo-Weiss parameter.
    """
    # Compute velocity gradients
    dudy, dudx = np.gradient(u)
    dvdy, dvdx = np.gradient(v)
    
    # Convert to physical units
    dudy /= lat_step_km
    dudx /= lon_step_km
    dvdy /= lat_step_km
    dvdx /= lon_step_km
    
    # Strain components
    normal_strain = dudx - dvdy
    shear_strain = dvdx + dudy
    
    # Relative vorticity
    vorticity = dvdx - dudy
    
    # Okubo-Weiss parameter
    ow = normal_strain**2 + shear_strain**2 - vorticity**2
    
    return ow


def validate_environment() -> Dict[str, bool]:
    """Validate that required environment variables are set.
    
    Returns
    -------
    Dict[str, bool]
        Dictionary indicating which credentials are available.
    """
    return {
        'earthdata_token': bool(os.getenv('EARTHDATA_TOKEN')),
        'earthdata_credentials': bool(os.getenv('EARTHDATA_USERNAME') and os.getenv('EARTHDATA_PASSWORD')),
        'mapbox_token': bool(os.getenv('MAPBOX_PUBLIC_TOKEN')),
        'api_key': bool(os.getenv('API_DATA_GOV_KEY')),
        'shark_csv_available': bool(os.getenv('SHARK_CSV') and os.path.exists(os.getenv('SHARK_CSV', '')))
    }