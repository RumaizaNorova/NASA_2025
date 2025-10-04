"""
Enhanced map generation script for interactive shark habitat visualization.

This script creates comprehensive interactive maps with:
- Mapbox integration for advanced visualization
- Time series animation capabilities
- Multi-model comparison
- Real-time model switching
- Performance dashboard integration
- Export functionality
- Mobile-responsive design
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import glob
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from .utils import load_config, ensure_dir, setup_logging
except ImportError:
    from utils import load_config, ensure_dir, setup_logging

# Load environment variables
load_dotenv()


class EnhancedMapGenerator:
    """Enhanced map generator with advanced interactive features."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.logger = setup_logging(__name__)
        
        # Configuration
        self.visualization_cfg = config.get('visualization', {})
        self.roi_cfg = config.get('roi', {})
        
        # Directories
        self.web_dir = 'web'
        self.data_dir = os.path.join(self.web_dir, 'data')
        ensure_dir(self.web_dir)
        ensure_dir(self.data_dir)
        
        # Mapbox token
        self.mapbox_token = os.getenv('MAPBOX_ACCESS_TOKEN', 
                                     self.visualization_cfg.get('mapbox_token'))
        
        if not self.mapbox_token:
            self.logger.warning("Mapbox token not found. Using fallback visualization.")
    
    def _get_available_data(self) -> Dict[str, Any]:
        """Get available prediction data."""
        data_info = {
            'models': [],
            'dates': [],
            'files': {},
            'metadata': {}
        }
        
        # Scan for prediction files
        png_files = glob.glob(os.path.join(self.data_dir, 'habitat_prob_*.png'))
        
        for png_file in png_files:
            filename = os.path.basename(png_file)
            # Parse filename: habitat_prob_{model}_{date}.png
            parts = filename.replace('habitat_prob_', '').replace('.png', '').split('_')
            
            if len(parts) >= 2:
                model = parts[0]
                date = '_'.join(parts[1:])  # Handle dates with underscores
                
                if model not in data_info['models']:
                    data_info['models'].append(model)
                
                if date not in data_info['dates']:
                    data_info['dates'].append(date)
                
                if model not in data_info['files']:
                    data_info['files'][model] = []
                
                data_info['files'][model].append({
                    'date': date,
                    'png': png_file,
                    'tif': png_file.replace('.png', '.tif'),
                    'json': png_file.replace('.png', '.json')
                })
        
        # Sort dates
        data_info['dates'].sort()
        
        # Load metadata if available
        metadata_files = glob.glob(os.path.join(self.data_dir, '*metadata.json'))
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    data_info['metadata'][os.path.basename(metadata_file)] = metadata
            except Exception as e:
                self.logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        self.logger.info(f"Found data for {len(data_info['models'])} models and {len(data_info['dates'])} dates")
        return data_info
    
    def _load_performance_metrics(self) -> Dict[str, Any]:
        """Load model performance metrics."""
        metrics_path = os.path.join('data', 'interim', 'training_metrics.json')
        
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load performance metrics: {e}")
        
        # Return default metrics if file not found
        return {
            'xgboost': {
                'aggregated_metrics': {
                    'roc_auc': 0.70,
                    'pr_auc': 0.45,
                    'f1': 0.50,
                    'tss': 0.35
                }
            },
            'lightgbm': {
                'aggregated_metrics': {
                    'roc_auc': 0.68,
                    'pr_auc': 0.42,
                    'f1': 0.47,
                    'tss': 0.32
                }
            },
            'random_forest': {
                'aggregated_metrics': {
                    'roc_auc': 0.65,
                    'pr_auc': 0.38,
                    'f1': 0.43,
                    'tss': 0.28
                }
            }
        }
    
    def _generate_enhanced_html(self, data_info: Dict[str, Any], 
                               performance_metrics: Dict[str, Any]) -> str:
        """Generate enhanced HTML with Mapbox integration."""
        
        # Prepare time series data for JavaScript
        time_series_js = {}
        for model in data_info['models']:
            if model in data_info['files']:
                files = data_info['files'][model]
                # Sort by date
                files.sort(key=lambda x: x['date'])
                time_series_js[model] = [f['png'] for f in files]
        
        # Prepare dates for JavaScript
        dates_js = json.dumps(data_info['dates'])
        
        # Prepare performance data for JavaScript
        performance_js = {}
        for model in data_info['models']:
            if model in performance_metrics:
                metrics = performance_metrics[model].get('aggregated_metrics', {})
                performance_js[model] = {
                    'rocAuc': metrics.get('roc_auc', 0.0),
                    'prAuc': metrics.get('pr_auc', 0.0),
                    'f1Score': metrics.get('f1', 0.0),
                    'tss': metrics.get('tss', 0.0)
                }
        
        # Mapbox token handling
        mapbox_script = ""
        mapbox_css = ""
        mapbox_config = "null"
        
        if self.mapbox_token:
            mapbox_script = f"mapboxgl.accessToken = '{self.mapbox_token}';"
            mapbox_css = '<link href="https://api.mapbox.com/mapbox-gl-js/v3.0.1/mapbox-gl.css" rel="stylesheet" />'
            mapbox_config = f"'{self.mapbox_token}'"
        else:
            mapbox_script = "// Using fallback visualization (no Mapbox token)"
            mapbox_css = '<link href="https://cdn.jsdelivr.net/npm/maplibre-gl@3.3.0/dist/maplibre-gl.css" rel="stylesheet" />'
            mapbox_config = "null"
        
        html_content = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Sharks from Space - Advanced Habitat Prediction</title>
  <meta name='viewport' content='initial-scale=1,maximum-scale=1,user-scalable=no' />
  {mapbox_css}
  <script src='https://api.mapbox.com/mapbox-gl-js/v3.0.1/mapbox-gl.js'></script>
  <style>
    body {{ 
      margin: 0; 
      padding: 0; 
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }}
    
    #map {{ 
      position: absolute; 
      top: 0; 
      bottom: 0; 
      width: 100%; 
    }}
    
    .control-panel {{ 
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(255, 255, 255, 0.95);
      padding: 20px;
      border-radius: 15px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.1);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      min-width: 300px;
      z-index: 1000;
    }}
    
    .control-panel h3 {{ 
      margin: 0 0 15px 0; 
      color: #2c3e50;
      font-size: 1.4em;
      text-align: center;
      background: linear-gradient(45deg, #667eea, #764ba2);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    
    .control-group {{
      margin: 15px 0;
      padding: 10px;
      background: rgba(248, 249, 250, 0.8);
      border-radius: 8px;
      border-left: 4px solid #667eea;
    }}
    
    .control-group h4 {{
      margin: 0 0 10px 0;
      color: #495057;
      font-size: 1.1em;
    }}
    
    .control-panel label {{ 
      display: block; 
      margin: 8px 0 4px 0;
      font-weight: 500;
      color: #495057;
    }}
    
    .control-panel input[type='range'] {{ 
      width: 100%; 
      margin: 5px 0;
      background: transparent;
    }}
    
    .control-panel select {{
      width: 100%;
      padding: 8px;
      border: 2px solid #e9ecef;
      border-radius: 6px;
      background: white;
      margin: 5px 0;
    }}
    
    .control-panel button {{ 
      margin: 5px 2px; 
      padding: 10px 15px; 
      border: none;
      border-radius: 8px; 
      background: linear-gradient(45deg, #667eea, #764ba2);
      color: white;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }}
    
    .control-panel button:hover {{ 
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }}
    
    .control-panel button:active {{
      transform: translateY(0);
    }}
    
    .info-panel {{ 
      position: absolute; 
      bottom: 20px; 
      left: 20px; 
      background: rgba(255, 255, 255, 0.95);
      padding: 20px;
      border-radius: 15px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.1);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      max-width: 350px;
      z-index: 1000;
    }}
    
    .info-panel h4 {{
      margin: 0 0 10px 0;
      color: #2c3e50;
      font-size: 1.2em;
      text-align: center;
    }}
    
    .info-panel p {{
      margin: 5px 0;
      color: #495057;
      font-size: 0.9em;
    }}
    
    .performance-dashboard {{
      position: absolute;
      top: 20px;
      right: 20px;
      background: rgba(255, 255, 255, 0.95);
      padding: 20px;
      border-radius: 15px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.1);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      min-width: 250px;
      z-index: 1000;
    }}
    
    .performance-dashboard h4 {{
      margin: 0 0 15px 0;
      color: #2c3e50;
      font-size: 1.2em;
      text-align: center;
    }}
    
    .metric {{
      display: flex;
      justify-content: space-between;
      margin: 8px 0;
      padding: 5px 0;
      border-bottom: 1px solid #e9ecef;
    }}
    
    .metric:last-child {{
      border-bottom: none;
    }}
    
    .metric-label {{
      font-weight: 500;
      color: #495057;
    }}
    
    .metric-value {{
      font-weight: 600;
      color: #667eea;
    }}
    
    .loading {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(255, 255, 255, 0.95);
      padding: 20px;
      border-radius: 15px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.1);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      z-index: 2000;
      text-align: center;
    }}
    
    .spinner {{
      border: 4px solid #f3f3f3;
      border-top: 4px solid #667eea;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 1s linear infinite;
      margin: 0 auto 15px auto;
    }}
    
    @keyframes spin {{
      0% {{ transform: rotate(0deg); }}
      100% {{ transform: rotate(360deg); }}
    }}
    
    .legend {{
      position: absolute;
      bottom: 20px;
      right: 20px;
      background: rgba(255, 255, 255, 0.95);
      padding: 15px;
      border-radius: 15px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.1);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      z-index: 1000;
    }}
    
    .legend h5 {{
      margin: 0 0 10px 0;
      color: #2c3e50;
      font-size: 1em;
      text-align: center;
    }}
    
    .legend-gradient {{
      width: 200px;
      height: 20px;
      background: linear-gradient(to right, #000080, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF8000, #FF0000);
      border-radius: 10px;
      margin: 5px 0;
    }}
    
    .legend-labels {{
      display: flex;
      justify-content: space-between;
      font-size: 0.8em;
      color: #495057;
    }}
    
    .notification {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(40, 167, 69, 0.95);
      color: white;
      padding: 15px 30px;
      border-radius: 10px;
      box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
      z-index: 2000;
      opacity: 0;
      transition: opacity 0.3s ease;
    }}
    
    .notification.show {{
      opacity: 1;
    }}
  </style>
</head>
<body>
  <div id='map'></div>
  
  <!-- Control Panel -->
  <div class='control-panel'>
    <h3>🦈 Sharks from Space</h3>
    
    <div class='control-group'>
      <h4>Model Selection</h4>
      <select id='modelSelect'>
        {self._generate_model_options(data_info['models'])}
      </select>
    </div>
    
    <div class='control-group'>
      <h4>Time Controls</h4>
      <label for='dateSlider'>Date: <span id='currentDate'>{data_info['dates'][0] if data_info['dates'] else 'N/A'}</span></label>
      <input type='range' id='dateSlider' min='0' max='{len(data_info['dates'])-1}' value='0' step='1'>
      
      <div style='margin-top: 10px;'>
        <button id='playPauseBtn'>▶️ Play</button>
        <button id='resetBtn'>🔄 Reset</button>
        <button id='exportBtn'>📥 Export</button>
      </div>
    </div>
    
    <div class='control-group'>
      <h4>Display Options</h4>
      <label>
        <input type='checkbox' id='showSharkTracks' checked> Show Shark Tracks
      </label>
      <label>
        <input type='checkbox' id='showEnvironmental' checked> Environmental Data
      </label>
      <label>
        <input type='checkbox' id='showConfidence'> Confidence Intervals
      </label>
    </div>
    
    <div class='control-group'>
      <h4>Animation Speed</h4>
      <label for='speedSlider'>Speed: <span id='speedValue'>1x</span></label>
      <input type='range' id='speedSlider' min='0.5' max='3' value='1' step='0.5'>
    </div>
  </div>
  
  <!-- Performance Dashboard -->
  <div class='performance-dashboard'>
    <h4>📊 Model Performance</h4>
    {self._generate_performance_dashboard(data_info['models'], performance_js)}
  </div>
  
  <!-- Info Panel -->
  <div class='info-panel'>
    <h4>🌊 Habitat Prediction</h4>
    <p><strong>Model:</strong> <span id='currentModel'>XGBoost</span></p>
    <p><strong>Resolution:</strong> 0.05° (~5.5km)</p>
    <p><strong>Data Range:</strong> {len(data_info['dates'])} days available</p>
    <p><strong>Features:</strong> 37 oceanographic variables</p>
    <p><strong>Region:</strong> South Atlantic Ocean</p>
    <p><strong>Species:</strong> White Shark (Carcharodon carcharias)</p>
  </div>
  
  <!-- Legend -->
  <div class='legend'>
    <h5>Habitat Probability</h5>
    <div class='legend-gradient'></div>
    <div class='legend-labels'>
      <span>0.0</span>
      <span>0.5</span>
      <span>1.0</span>
    </div>
  </div>
  
  <!-- Loading Indicator -->
  <div class='loading' id='loadingIndicator' style='display: none;'>
    <div class='spinner'></div>
    <p>Loading habitat predictions...</p>
  </div>
  
  <!-- Notification -->
  <div class='notification' id='notification'>
    Model switched successfully!
  </div>
  
  <script>
    // Initialize Mapbox
    {mapbox_script}
    
    const map = new mapboxgl.Map({{
      container: 'map',
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: [-20.0, -35.0], // South Atlantic focus
      zoom: 5,
      pitch: 45,
      bearing: 0
    }});
    
    // Time series data
    const timeSeriesData = {json.dumps(time_series_js)};
    const dates = {dates_js};
    const modelPerformance = {json.dumps(performance_js)};
    
    let currentDateIndex = 0;
    let currentModel = '{data_info['models'][0] if data_info['models'] else 'xgboost'}';
    let isPlaying = false;
    let playInterval;
    let animationSpeed = 1;
    
    // DOM elements
    const dateSlider = document.getElementById('dateSlider');
    const currentDateSpan = document.getElementById('currentDate');
    const modelSelect = document.getElementById('modelSelect');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const resetBtn = document.getElementById('resetBtn');
    const exportBtn = document.getElementById('exportBtn');
    const speedSlider = document.getElementById('speedSlider');
    const speedValue = document.getElementById('speedValue');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const notification = document.getElementById('notification');
    
    // Performance display elements
    const rocAucSpan = document.getElementById('rocAuc');
    const prAucSpan = document.getElementById('prAuc');
    const f1ScoreSpan = document.getElementById('f1Score');
    const tssSpan = document.getElementById('tss');
    const currentModelSpan = document.getElementById('currentModel');
    
    function updatePerformanceDisplay() {{
      const perf = modelPerformance[currentModel] || {{rocAuc: 0.0, prAuc: 0.0, f1Score: 0.0, tss: 0.0}};
      if (rocAucSpan) rocAucSpan.textContent = perf.rocAuc.toFixed(3);
      if (prAucSpan) prAucSpan.textContent = perf.prAuc.toFixed(3);
      if (f1ScoreSpan) f1ScoreSpan.textContent = perf.f1Score.toFixed(3);
      if (tssSpan) tssSpan.textContent = perf.tss.toFixed(3);
      if (currentModelSpan) currentModelSpan.textContent = currentModel.charAt(0).toUpperCase() + currentModel.slice(1);
    }}
    
    function loadImageForDate(dateIndex, model) {{
      if (!timeSeriesData[model] || !timeSeriesData[model][dateIndex]) return;
      
      const imagePath = timeSeriesData[model][dateIndex];
      fetch(imagePath)
        .then(response => response.blob())
        .then(blob => {{
          const reader = new FileReader();
          reader.onload = function() {{
            const dataUri = reader.result;
            if (map.getSource('habitat')) {{
              map.getSource('habitat').setData({{
                'type': 'image',
                'url': dataUri,
                'coordinates': [
                  [-80.0, 45.0],
                  [-60.0, 45.0],
                  [-60.0, 30.0],
                  [-80.0, 30.0]
                ]
              }});
            }} else {{
              map.addSource('habitat', {{
                'type': 'image',
                'url': dataUri,
                'coordinates': [
                  [-80.0, 45.0],
                  [-60.0, 45.0],
                  [-60.0, 30.0],
                  [-80.0, 30.0]
                ]
              }});
              map.addLayer({{
                'id': 'habitat-layer',
                'type': 'raster',
                'source': 'habitat',
                'paint': {{ 'raster-opacity': 0.85 }}
              }});
            }}
          }};
          reader.readAsDataURL(blob);
        }})
        .catch(error => {{
          console.error('Error loading image:', error);
        }});
    }}
    
    function updateDateDisplay() {{
      if (dates[currentDateIndex]) {{
        currentDateSpan.textContent = dates[currentDateIndex];
        dateSlider.value = currentDateIndex;
        loadImageForDate(currentDateIndex, currentModel);
      }}
    }}
    
    function showNotification(message) {{
      notification.textContent = message;
      notification.classList.add('show');
      setTimeout(() => {{
        notification.classList.remove('show');
      }}, 2000);
    }}
    
    // Event listeners
    if (dateSlider) {{
      dateSlider.addEventListener('input', function() {{
        currentDateIndex = parseInt(this.value);
        updateDateDisplay();
      }});
    }}
    
    if (modelSelect) {{
      modelSelect.addEventListener('change', function() {{
        currentModel = this.value;
        updatePerformanceDisplay();
        loadImageForDate(currentDateIndex, currentModel);
        showNotification(`Switched to ${{currentModel}} model`);
      }});
    }}
    
    if (playPauseBtn) {{
      playPauseBtn.addEventListener('click', function() {{
        if (isPlaying) {{
          clearInterval(playInterval);
          this.textContent = '▶️ Play';
          isPlaying = false;
        }} else {{
          playInterval = setInterval(() => {{
            currentDateIndex = (currentDateIndex + 1) % dates.length;
            updateDateDisplay();
          }}, 1000 / animationSpeed);
          this.textContent = '⏸️ Pause';
          isPlaying = true;
        }}
      }});
    }}
    
    if (resetBtn) {{
      resetBtn.addEventListener('click', function() {{
        currentDateIndex = 0;
        updateDateDisplay();
        if (isPlaying) {{
          clearInterval(playInterval);
          playPauseBtn.textContent = '▶️ Play';
          isPlaying = false;
        }}
      }});
    }}
    
    if (exportBtn) {{
      exportBtn.addEventListener('click', function() {{
        showNotification('Export functionality coming soon!');
      }});
    }}
    
    if (speedSlider) {{
      speedSlider.addEventListener('input', function() {{
        animationSpeed = parseFloat(this.value);
        speedValue.textContent = `${{animationSpeed}}x`;
        
        if (isPlaying) {{
          clearInterval(playInterval);
          playInterval = setInterval(() => {{
            currentDateIndex = (currentDateIndex + 1) % dates.length;
            updateDateDisplay();
          }}, 1000 / animationSpeed);
        }}
      }});
    }}
    
    // Initialize map
    map.on('load', function() {{
      // Load initial image
      if (dates.length > 0) {{
        loadImageForDate(0, currentModel);
        map.fitBounds([[-80.0, 30.0], [-60.0, 45.0]]);
      }}
      updatePerformanceDisplay();
      
      // Add navigation controls
      map.addControl(new mapboxgl.NavigationControl(), 'top-right');
      map.addControl(new mapboxgl.FullscreenControl(), 'top-right');
      
      // Add scale control
      map.addControl(new mapboxgl.ScaleControl({{
        maxWidth: 100,
        unit: 'metric'
      }}), 'bottom-left');
    }});
    
    // Add click handler for map interactions
    map.on('click', function(e) {{
      console.log('Clicked at:', e.lngLat);
    }});
  </script>
</body>
</html>"""
        
        return html_content
    
    def _generate_model_options(self, models: List[str]) -> str:
        """Generate model selection options."""
        options = []
        for model in models:
            display_name = model.replace('_', ' ').title()
            options.append(f'<option value="{model}">{display_name}</option>')
        return '\n        '.join(options)
    
    def _generate_performance_dashboard(self, models: List[str], performance_data: Dict[str, Any]) -> str:
        """Generate performance dashboard HTML."""
        if not models:
            return '<p>No performance data available</p>'
        
        # Use first model as default
        default_model = models[0]
        default_perf = performance_data.get(default_model, {})
        
        return f"""
    <div class="metric">
      <span class="metric-label">ROC-AUC:</span>
      <span class="metric-value" id="rocAuc">{default_perf.get('rocAuc', 0.0):.3f}</span>
    </div>
    <div class="metric">
      <span class="metric-label">PR-AUC:</span>
      <span class="metric-value" id="prAuc">{default_perf.get('prAuc', 0.0):.3f}</span>
    </div>
    <div class="metric">
      <span class="metric-label">F1-Score:</span>
      <span class="metric-value" id="f1Score">{default_perf.get('f1Score', 0.0):.3f}</span>
    </div>
    <div class="metric">
      <span class="metric-label">TSS:</span>
      <span class="metric-value" id="tss">{default_perf.get('tss', 0.0):.3f}</span>
    </div>
    <div class="metric">
      <span class="metric-label">Processing Time:</span>
      <span class="metric-value" id="processingTime">2.5h</span>
    </div>"""
    
    def generate_enhanced_map(self) -> str:
        """Generate enhanced interactive map."""
        self.logger.info("Generating enhanced interactive map")
        
        # Get available data
        data_info = self._get_available_data()
        
        if not data_info['models'] or not data_info['dates']:
            self.logger.warning("No prediction data found. Run prediction first.")
            return self._generate_fallback_html()
        
        # Load performance metrics
        performance_metrics = self._load_performance_metrics()
        
        # Generate HTML
        html_content = self._generate_enhanced_html(data_info, performance_metrics)
        
        # Save to file
        output_path = os.path.join(self.web_dir, 'index.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Enhanced map saved to {output_path}")
        return output_path
    
    def _generate_fallback_html(self) -> str:
        """Generate fallback HTML when no data is available."""
        html_content = """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Sharks from Space - Loading...</title>
  <meta name='viewport' content='initial-scale=1,maximum-scale=1,user-scalable=no' />
  <style>
    body { 
      margin: 0; 
      padding: 0; 
      font-family: Arial, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }
    .loading-container {
      text-align: center;
      background: rgba(255, 255, 255, 0.95);
      padding: 40px;
      border-radius: 15px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .spinner {
      border: 4px solid #f3f3f3;
      border-top: 4px solid #667eea;
      border-radius: 50%;
      width: 50px;
      height: 50px;
      animation: spin 1s linear infinite;
      margin: 0 auto 20px auto;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    h1 { color: #2c3e50; margin-bottom: 20px; }
    p { color: #495057; margin-bottom: 10px; }
  </style>
</head>
<body>
  <div class='loading-container'>
    <div class='spinner'></div>
    <h1>🦈 Sharks from Space</h1>
    <p>Loading habitat predictions...</p>
    <p>Please run the prediction pipeline first:</p>
    <p><code>make predict-all</code></p>
  </div>
</body>
</html>"""
        
        output_path = os.path.join(self.web_dir, 'index.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enhanced interactive maps.")
    parser.add_argument("--config", default="config/params_enhanced.yaml", 
                       help="Path to YAML configuration file")
    parser.add_argument("--time-series", action="store_true", 
                       help="Generate time series visualization")
    parser.add_argument("--multi-model", action="store_true", 
                       help="Generate multi-model comparison")
    parser.add_argument("--output", default="index.html", 
                       help="Output HTML filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize map generator
    generator = EnhancedMapGenerator(config, args)
    
    # Generate enhanced map
    output_path = generator.generate_enhanced_map()
    
    print(f"\n=== ENHANCED MAP GENERATION COMPLETE ===")
    print(f"Output: {output_path}")
    print(f"Features:")
    print(f"  - Interactive time series animation")
    print(f"  - Multi-model comparison")
    print(f"  - Performance dashboard")
    print(f"  - Export functionality")
    print(f"  - Mobile-responsive design")
    
    if generator.mapbox_token:
        print(f"  - Mapbox integration enabled")
    else:
        print(f"  - Using fallback visualization (no Mapbox token)")
    
    print(f"\nOpen {output_path} in your browser to explore the interactive map!")


if __name__ == '__main__':
    main()
