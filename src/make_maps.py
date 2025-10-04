"""
Generate an interactive web map visualising the predicted habitat probability.

This script writes an `index.html` file into the `web/` directory with enhanced
features including time sliders, multi-day overlays, and interactive controls.
The resulting page uses Mapbox GL JS when a `MAPBOX_PUBLIC_TOKEN` environment
variable is set, otherwise falls back to MapLibre GL JS with OpenStreetMap
basemaps. Supports time series visualization and multiple model comparisons.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import glob
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

try:
    from .utils import load_config, ensure_dir, setup_logging
except ImportError:
    from utils import load_config, ensure_dir, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the interactive HTML map for shark habitat.")
    parser.add_argument("--config", default="config/params.yaml", help="Path to YAML configuration file")
    parser.add_argument("--demo", action="store_true", help="Ignored for compatibility")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest", "lightgbm"], help="Model to visualize")
    parser.add_argument("--time-series", action="store_true", help="Enable time series visualization with slider")
    parser.add_argument("--multi-model", action="store_true", help="Enable multi-model comparison")
    parser.add_argument("--overlay-days", type=int, default=None, help="Number of days to overlay (e.g., 7 for weekly)")
    return parser.parse_args()


def discover_prediction_files(data_dir: str) -> Dict[str, List[str]]:
    """Discover available prediction files and organize by model and date."""
    files = {
        'xgboost': [],
        'random_forest': [],
        'lightgbm': []
    }
    
    # Look for PNG files
    png_pattern = os.path.join(data_dir, 'habitat_prob_*.png')
    for file_path in glob.glob(png_pattern):
        filename = os.path.basename(file_path)
        # Parse filename: habitat_prob_{model}_{date}.png
        parts = filename.replace('habitat_prob_', '').replace('.png', '').split('_')
        if len(parts) >= 2:
            model = parts[0]
            date_str = parts[1]
            if model in files:
                files[model].append((date_str, file_path))
    
    # Sort by date
    for model in files:
        files[model].sort(key=lambda x: x[0])
    
    return files


def load_metadata(data_dir: str) -> Optional[Dict]:
    """Load prediction metadata if available."""
    metadata_path = os.path.join(data_dir, 'prediction_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def generate_time_series_html(files: Dict[str, List[str]], metadata: Optional[Dict], 
                            config: Dict, token: Optional[str]) -> List[str]:
    """Generate HTML for time series visualization."""
    html = []
    
    # Get available dates from the first model
    available_models = [model for model, file_list in files.items() if file_list]
    if not available_models:
        return html
    
    first_model = available_models[0]
    dates = [item[0] for item in files[first_model]]
    
    html.append("    <div id='time-controls' style='position: absolute; top: 10px; left: 10px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>")
    html.append("      <h4 style='margin: 0 0 10px 0;'>Time Series Controls</h4>")
    html.append("      <label for='date-slider'>Date: <span id='current-date'>" + dates[0] + "</span></label><br>")
    html.append("      <input type='range' id='date-slider' min='0' max='" + str(len(dates)-1) + "' value='0' style='width: 200px;'><br>")
    html.append("      <button id='play-pause' style='margin-top: 5px;'>Play</button>")
    html.append("      <button id='reset' style='margin-top: 5px;'>Reset</button>")
    html.append("    </div>")
    
    # Model selection
    if len(available_models) > 1:
        html.append("    <div id='model-controls' style='position: absolute; top: 10px; right: 10px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>")
        html.append("      <h4 style='margin: 0 0 10px 0;'>Model Selection</h4>")
        for model in available_models:
            checked = "checked" if model == first_model else ""
            html.append(f"      <input type='radio' id='model-{model}' name='model' value='{model}' {checked}>")
            html.append(f"      <label for='model-{model}'>{model.replace('_', ' ').title()}</label><br>")
        html.append("    </div>")
    
    return html


def generate_enhanced_script(files: Dict[str, List[str]], metadata: Optional[Dict], 
                           config: Dict, token: Optional[str], args: argparse.Namespace) -> List[str]:
    """Generate enhanced JavaScript for the interactive map."""
    html = []
    
    # Get available dates and models
    available_models = [model for model, file_list in files.items() if file_list]
    if not available_models:
        return html
    
    first_model = available_models[0]
    dates = [item[0] for item in files[first_model]]
    
    html.append("  <script>")
    
    # Map initialization
    if token:
        html.append(f"    mapboxgl.accessToken = '{token}';")
        html.append("    const map = new mapboxgl.Map({")
        html.append("      container: 'map',")
        html.append("      style: 'mapbox://styles/mapbox/light-v10',")
    else:
        html.append("    const map = new maplibregl.Map({")
        html.append("      container: 'map',")
        html.append("      style: 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',")
    
    roi = config.get('roi', {})
    center_lon = (roi['lon_min'] + roi['lon_max']) / 2.0
    center_lat = (roi['lat_min'] + roi['lat_max']) / 2.0
    html.append(f"      center: [{center_lon:.4f}, {center_lat:.4f}],")
    html.append("      zoom: 5")
    html.append("    });")
    
    # Time series data
    html.append("    const timeSeriesData = {")
    for model in available_models:
        html.append(f"      '{model}': {json.dumps([item[1] for item in files[model]])},")
    html.append("    };")
    
    html.append(f"    const dates = {json.dumps(dates)};")
    html.append("    let currentDateIndex = 0;")
    html.append("    let currentModel = '" + first_model + "';")
    html.append("    let isPlaying = false;")
    html.append("    let playInterval;")
    
    # Load image function
    html.append("    function loadImageForDate(dateIndex, model) {")
    html.append("      const imagePath = timeSeriesData[model][dateIndex];")
    html.append("      fetch(imagePath)")
    html.append("        .then(response => response.blob())")
    html.append("        .then(blob => {")
    html.append("          const reader = new FileReader();")
    html.append("          reader.onload = function() {")
    html.append("            const dataUri = reader.result;")
    html.append("            if (map.getSource('habitat')) {")
    html.append("              map.getSource('habitat').setData({")
    html.append("                'type': 'image',")
    html.append("                'url': dataUri,")
    html.append("                'coordinates': [")
    html.append(f"                  [{roi['lon_min']:.4f}, {roi['lat_max']:.4f}],")
    html.append(f"                  [{roi['lon_max']:.4f}, {roi['lat_max']:.4f}],")
    html.append(f"                  [{roi['lon_max']:.4f}, {roi['lat_min']:.4f}],")
    html.append(f"                  [{roi['lon_min']:.4f}, {roi['lat_min']:.4f}]")
    html.append("                ]")
    html.append("              });")
    html.append("            } else {")
    html.append("              map.addSource('habitat', {")
    html.append("                'type': 'image',")
    html.append("                'url': dataUri,")
    html.append("                'coordinates': [")
    html.append(f"                  [{roi['lon_min']:.4f}, {roi['lat_max']:.4f}],")
    html.append(f"                  [{roi['lon_max']:.4f}, {roi['lat_max']:.4f}],")
    html.append(f"                  [{roi['lon_max']:.4f}, {roi['lat_min']:.4f}],")
    html.append(f"                  [{roi['lon_min']:.4f}, {roi['lat_min']:.4f}]")
    html.append("                ]")
    html.append("              });")
    html.append("              map.addLayer({")
    html.append("                'id': 'habitat-layer',")
    html.append("                'type': 'raster',")
    html.append("                'source': 'habitat',")
    html.append("                'paint': { 'raster-opacity': 0.85 }")
    html.append("              });")
    html.append("            }")
    html.append("          };")
    html.append("          reader.readAsDataURL(blob);")
    html.append("        });")
    html.append("    }")
    
    # Event handlers
    html.append("    map.on('load', function() {")
    html.append("      // Load initial image")
    html.append("      loadImageForDate(0, currentModel);")
    html.append("      map.fitBounds([[" + f"{roi['lon_min']:.4f}, {roi['lat_min']:.4f}" + "], [" + f"{roi['lon_max']:.4f}, {roi['lat_max']:.4f}" + "]]);")
    html.append("    });")
    
    # Time slider controls
    if args.time_series:
        html.append("    document.getElementById('date-slider').addEventListener('input', function(e) {")
        html.append("      currentDateIndex = parseInt(e.target.value);")
        html.append("      document.getElementById('current-date').textContent = dates[currentDateIndex];")
        html.append("      loadImageForDate(currentDateIndex, currentModel);")
        html.append("    });")
        
        html.append("    document.getElementById('play-pause').addEventListener('click', function() {")
        html.append("      if (isPlaying) {")
        html.append("        clearInterval(playInterval);")
        html.append("        this.textContent = 'Play';")
        html.append("        isPlaying = false;")
        html.append("      } else {")
        html.append("        this.textContent = 'Pause';")
        html.append("        isPlaying = true;")
        html.append("        playInterval = setInterval(function() {")
        html.append("          currentDateIndex = (currentDateIndex + 1) % dates.length;")
        html.append("          document.getElementById('date-slider').value = currentDateIndex;")
        html.append("          document.getElementById('current-date').textContent = dates[currentDateIndex];")
        html.append("          loadImageForDate(currentDateIndex, currentModel);")
        html.append("        }, 500);")
        html.append("      }")
        html.append("    });")
        
        html.append("    document.getElementById('reset').addEventListener('click', function() {")
        html.append("      if (isPlaying) {")
        html.append("        clearInterval(playInterval);")
        html.append("        document.getElementById('play-pause').textContent = 'Play';")
        html.append("        isPlaying = false;")
        html.append("      }")
        html.append("      currentDateIndex = 0;")
        html.append("      document.getElementById('date-slider').value = 0;")
        html.append("      document.getElementById('current-date').textContent = dates[0];")
        html.append("      loadImageForDate(0, currentModel);")
        html.append("    });")
    
    # Model selection
    if len(available_models) > 1:
        html.append("    document.querySelectorAll('input[name=\"model\"]').forEach(function(radio) {")
        html.append("      radio.addEventListener('change', function() {")
        html.append("        if (this.checked) {")
        html.append("          currentModel = this.value;")
        html.append("          loadImageForDate(currentDateIndex, currentModel);")
        html.append("        }")
        html.append("      });")
        html.append("    });")
    
    html.append("  </script>")
    
    return html


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(__name__)
    
    # Check for Mapbox token
    token = os.environ.get('MAPBOX_PUBLIC_TOKEN')
    
    # Discover available prediction files
    data_dir = os.path.join('web', 'data')
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Run predict_grid.py first to generate prediction files.")
        sys.exit(1)
    
    files = discover_prediction_files(data_dir)
    available_models = [model for model, file_list in files.items() if file_list]
    
    if not available_models:
        logger.warning("No prediction files found. Using fallback single image.")
        # Fallback to original behavior
        return generate_fallback_map(config, token, logger)
    
    logger.info(f"Found prediction files for models: {available_models}")
    
    # Load metadata if available
    metadata = load_metadata(data_dir)
    
    # Generate enhanced HTML
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append("  <meta charset='utf-8'>")
    html.append("  <title>Shark Habitat Probability Map - Interactive</title>")
    html.append("  <meta name='viewport' content='initial-scale=1,maximum-scale=1,user-scalable=no' />")
    
    if token:
        html.append("  <link href='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css' rel='stylesheet' />")
    else:
        html.append("  <link href='https://cdn.jsdelivr.net/npm/maplibre-gl@3.3.0/dist/maplibre-gl.css' rel='stylesheet' />")
    
    html.append("  <style>")
    html.append("    body { margin: 0; padding: 0; font-family: Arial, sans-serif; }")
    html.append("    #map { position: absolute; top: 0; bottom: 0; width: 100%; }")
    html.append("    .control-panel { background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }")
    html.append("    .control-panel h4 { margin: 0 0 10px 0; color: #333; }")
    html.append("    .control-panel label { display: block; margin: 5px 0; }")
    html.append("    .control-panel input[type='range'] { width: 100%; }")
    html.append("    .control-panel button { margin: 2px; padding: 5px 10px; border: 1px solid #ccc; border-radius: 3px; background: #f9f9f9; cursor: pointer; }")
    html.append("    .control-panel button:hover { background: #e9e9e9; }")
    html.append("    .info-panel { position: absolute; bottom: 10px; left: 10px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); max-width: 300px; }")
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")
    html.append("  <div id='map'></div>")
    
    # Add time series controls if enabled
    if args.time_series:
        html.extend(generate_time_series_html(files, metadata, config, token))
    
    # Add info panel
    html.append("    <div class='info-panel'>")
    html.append("      <h4>Shark Habitat Probability</h4>")
    html.append("      <p>Interactive map showing predicted shark habitat probability based on environmental conditions.</p>")
    if metadata:
        html.append(f"      <p><strong>Model:</strong> {metadata.get('model', 'Unknown')}</p>")
        html.append(f"      <p><strong>Resolution:</strong> {metadata.get('resolution', 'Unknown')}°</p>")
        html.append(f"      <p><strong>Dates:</strong> {len(metadata.get('dates', []))} available</p>")
    html.append("    </div>")
    
    # Add JavaScript libraries
    if token:
        html.append("  <script src='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js'></script>")
    else:
        html.append("  <script src='https://cdn.jsdelivr.net/npm/maplibre-gl@3.3.0/dist/maplibre-gl.js'></script>")
    
    # Generate enhanced JavaScript
    html.extend(generate_enhanced_script(files, metadata, config, token, args))
    
    html.append("</body>")
    html.append("</html>")
    
    # Write HTML file
    ensure_dir('web')
    out_path = os.path.join('web', 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html))
    
    logger.info(f"Generated enhanced interactive map: {out_path}")
    if args.time_series:
        logger.info("Time series controls enabled")
    if len(available_models) > 1:
        logger.info("Multi-model comparison enabled")


def generate_fallback_map(config: Dict, token: Optional[str], logger) -> None:
    """Generate fallback map with single image (original behavior)."""
    import base64
    
    roi = config.get('roi', {})
    lon_min = float(roi['lon_min'])
    lon_max = float(roi['lon_max'])
    lat_min = float(roi['lat_min'])
    lat_max = float(roi['lat_max'])
    
    # Compute map centre
    center_lon = (lon_min + lon_max) / 2.0
    center_lat = (lat_min + lat_max) / 2.0
    
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append("  <meta charset='utf-8'>")
    html.append("  <title>Shark Habitat Probability Map</title>")
    html.append("  <meta name='viewport' content='initial-scale=1,maximum-scale=1,user-scalable=no' />")
    
    if token:
        html.append("  <link href='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css' rel='stylesheet' />")
    else:
        html.append("  <link href='https://cdn.jsdelivr.net/npm/maplibre-gl@3.3.0/dist/maplibre-gl.css' rel='stylesheet' />")
    
    html.append("  <style>")
    html.append("    body { margin: 0; padding: 0; }")
    html.append("    #map { position: absolute; top: 0; bottom: 0; width: 100%; }")
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")
    html.append("  <div id='map'></div>")
    
    if token:
        html.append("  <script src='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js'></script>")
    else:
        html.append("  <script src='https://cdn.jsdelivr.net/npm/maplibre-gl@3.3.0/dist/maplibre-gl.js'></script>")
    
    # Embed the PNG as a base64 data URI
    png_path = os.path.join('web', 'data', 'habitat_prob.png')
    try:
        with open(png_path, 'rb') as img_f:
            b64_img = base64.b64encode(img_f.read()).decode('utf-8')
        data_uri = f"data:image/png;base64,{b64_img}"
    except Exception:
        data_uri = ''
        logger.warning("Could not load habitat probability image")
    
    html.append("  <script>")
    if token:
        html.append(f"    mapboxgl.accessToken = '{token}';")
        html.append("    const map = new mapboxgl.Map({")
        html.append("      container: 'map',")
        html.append("      style: 'mapbox://styles/mapbox/light-v10',")
    else:
        html.append("    const map = new maplibregl.Map({")
        html.append("      container: 'map',")
        html.append("      style: 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',")
    
    html.append(f"      center: [{center_lon:.4f}, {center_lat:.4f}],")
    html.append("      zoom: 5")
    html.append("    });")
    
    html.append("    map.on('load', function() {")
    html.append("      map.addSource('habitat', {")
    html.append("        'type': 'image',")
    html.append(f"        'url': '{data_uri}',")
    html.append("        'coordinates': [")
    html.append(f"          [{lon_min:.4f}, {lat_max:.4f}],")
    html.append(f"          [{lon_max:.4f}, {lat_max:.4f}],")
    html.append(f"          [{lon_max:.4f}, {lat_min:.4f}],")
    html.append(f"          [{lon_min:.4f}, {lat_min:.4f}]")
    html.append("        ]")
    html.append("      });")
    html.append("      map.addLayer({")
    html.append("        'id': 'habitat-layer',")
    html.append("        'type': 'raster',")
    html.append("        'source': 'habitat',")
    html.append("        'paint': { 'raster-opacity': 0.85 }")
    html.append("      });")
    html.append("      map.fitBounds([[" + f"{lon_min:.4f}, {lat_min:.4f}" + "], [" + f"{lon_max:.4f}, {lat_max:.4f}" + "]]);")
    html.append("    });")
    html.append("  </script>")
    html.append("</body>")
    html.append("</html>")
    
    # Write HTML file
    ensure_dir('web')
    out_path = os.path.join('web', 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html))
    
    logger.info(f"Generated fallback map: {out_path}")


if __name__ == '__main__':
    main()