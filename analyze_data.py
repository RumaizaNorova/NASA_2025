"""
Data Analysis Script to Determine Optimal Parameters
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('../sharks_cleaned.csv')

print("=== SHARK DATA ANALYSIS ===")
print(f"Total observations: {len(df):,}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
print(f"Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
print(f"Unique sharks: {df['id'].nunique()}")
print(f"Species: {df['species'].unique()}")

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date

# Filter for 2014 data
df_2014 = df[df['datetime'].dt.year == 2014]
print(f"\n2014 observations: {len(df_2014):,}")

# Check spatial distribution in expanded ROI
roi_expanded = {'lon_min': 5.0, 'lon_max': 35.0, 'lat_min': -45.0, 'lat_max': -25.0}
df_roi = df_2014[
    (df_2014['latitude'] >= roi_expanded['lat_min']) & 
    (df_2014['latitude'] <= roi_expanded['lat_max']) &
    (df_2014['longitude'] >= roi_expanded['lon_min']) & 
    (df_2014['longitude'] <= roi_expanded['lon_max'])
]
print(f"Observations in expanded ROI: {len(df_roi):,}")

# Check spatial distribution in original ROI
roi_original = {'lon_min': 15.0, 'lon_max': 25.0, 'lat_min': -40.0, 'lat_max': -30.0}
df_roi_orig = df_2014[
    (df_2014['latitude'] >= roi_original['lat_min']) & 
    (df_2014['latitude'] <= roi_original['lat_max']) &
    (df_2014['longitude'] >= roi_original['lon_min']) & 
    (df_2014['longitude'] <= roi_original['lon_max'])
]
print(f"Observations in original ROI: {len(df_roi_orig):,}")

# Monthly distribution for 2014
monthly_counts = df_2014.groupby(df_2014['datetime'].dt.month).size()
print(f"\nMonthly distribution (2014):")
for month, count in monthly_counts.items():
    print(f"  Month {month}: {count:,} observations")

# Spatial density analysis
print(f"\n=== SPATIAL ANALYSIS ===")
print(f"Latitude range in ROI: {df_roi['latitude'].min():.2f} to {df_roi['latitude'].max():.2f}")
print(f"Longitude range in ROI: {df_roi['longitude'].min():.2f} to {df_roi['longitude'].max():.2f}")

# Calculate spatial extent
lat_extent = df_roi['latitude'].max() - df_roi['latitude'].min()
lon_extent = df_roi['longitude'].max() - df_roi['longitude'].min()
print(f"Spatial extent: {lat_extent:.2f}° lat x {lon_extent:.2f}° lon")

# Estimate grid points for different resolutions
resolutions = [0.1, 0.05, 0.2]
roi_area = (roi_expanded['lon_max'] - roi_expanded['lon_min']) * (roi_expanded['lat_max'] - roi_expanded['lat_min'])

print(f"\n=== GRID ANALYSIS ===")
print(f"ROI area: {roi_area:.1f} square degrees")
for res in resolutions:
    grid_points = roi_area / (res * res)
    print(f"Resolution {res}°: ~{grid_points:,.0f} grid points")

print(f"\n=== PARAMETER RECOMMENDATIONS ===")

# Calculate optimal pseudo-absence ratio
positive_samples = len(df_roi)
print(f"Positive samples (shark observations): {positive_samples:,}")

# Recommended ratios based on ML best practices
ratios = [1, 2, 5, 10, 20, 50]
print(f"\nPseudo-absence recommendations:")
for ratio in ratios:
    pseudo_absences = positive_samples * ratio
    total_samples = positive_samples + pseudo_absences
    print(f"  Ratio 1:{ratio} → {pseudo_absences:,} pseudo-absences, {total_samples:,} total samples")

# Memory and processing time estimates
print(f"\n=== PERFORMANCE ESTIMATES ===")
print(f"Feature extraction time per 1000 points: ~30 seconds (optimized)")
print(f"Model training time per 10K samples: ~2-5 minutes")

for ratio in [5, 10, 20]:
    pseudo_absences = positive_samples * ratio
    total_samples = positive_samples + pseudo_absences
    extraction_time = (total_samples / 1000) * 30 / 60  # minutes
    training_time = (total_samples / 10000) * 5  # minutes
    total_time = extraction_time + training_time
    print(f"  Ratio 1:{ratio}: ~{total_time:.1f} minutes total processing")

print(f"\n=== FINAL RECOMMENDATIONS ===")
print(f"Based on analysis:")
print(f"- Use ratio 1:10 (balanced performance vs time)")
print(f"- Resolution 0.1° (good balance of detail vs processing)")
print(f"- Full year data (365 days)")
print(f"- Expanded ROI (20° x 20°)")
print(f"- Estimated processing time: ~15-20 minutes")

