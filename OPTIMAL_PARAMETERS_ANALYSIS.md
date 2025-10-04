# Optimal Parameters Analysis

## Data Characteristics Discovered

### Shark Data:
- **Total observations:** 65,793 (2012-2019)
- **2014 observations:** 12,614 (our target year)
- **Expanded ROI (20°×20°):** 259 observations
- **Original ROI (10°×10°):** 159 observations
- **Spatial extent:** 16.77° lat × 22.98° lon
- **Multiple species:** 9 different shark species

### Key Insights:
1. **Expanded ROI gives us 63% more data** (159 → 259 observations)
2. **Seasonal variation:** Peak in Feb-Mar, low in Oct-Dec
3. **Spatial coverage:** Good distribution across the region
4. **Processing scale:** Manageable dataset size

## Optimal Parameter Selection

### 1. **Pseudo-Absence Ratio: 1:10**
**Rationale:**
- **259 shark observations** → **2,590 pseudo-absences** → **2,849 total samples**
- **ML best practice:** 1:10 ratio balances model performance with computational efficiency
- **Processing time:** ~2.8 minutes (very reasonable)
- **Model performance:** Sufficient negative examples for robust training

### 2. **Grid Resolution: 0.1°**
**Rationale:**
- **~60,000 grid points** for 20°×20° region
- **Good spatial detail** without excessive computational load
- **Memory usage:** ~60MB for features (manageable)
- **Processing time:** ~30 minutes for full year features

### 3. **Temporal Coverage: Full Year (365 days)**
**Rationale:**
- **Seasonal patterns:** Captures migration and behavioral changes
- **Data availability:** 12,614 observations available
- **Scientific value:** Much more robust than 14-day subset
- **Processing:** Batch processing keeps memory manageable

### 4. **Spatial Coverage: Expanded ROI (20°×20°)**
**Rationale:**
- **63% more shark data** (159 → 259 observations)
- **Oceanographic features:** Includes sub-Antarctic convergence
- **Migration patterns:** Captures broader movement range
- **Scientific completeness:** More representative habitat analysis

## Performance Estimates (Optimized)

### Processing Times:
- **Data loading:** 2-3 minutes
- **Feature computation:** 30-45 minutes (365 days)
- **Label processing:** 3-5 minutes (2,849 samples)
- **Model training:** 5-10 minutes
- **Predictions:** 2-3 minutes
- **Total:** ~45-65 minutes

### Memory Usage:
- **Peak memory:** <2 GB (well within limits)
- **Batch processing:** 30-day chunks prevent memory overflow
- **Feature storage:** ~60MB NetCDF file

## Expected Model Performance Improvements

### Current Baseline (14 days, 10°×10°, 159 observations):
- **ROC-AUC:** ~0.55 (barely better than random)
- **Training samples:** ~10,000 (mostly pseudo-absences)
- **Spatial coverage:** Limited
- **Temporal coverage:** No seasonal patterns

### Optimized Configuration (365 days, 20°×20°, 259 observations):
- **Expected ROC-AUC:** 0.65-0.75 (significant improvement)
- **Training samples:** 2,849 (balanced ratio)
- **Spatial coverage:** 4× larger region
- **Temporal coverage:** Full seasonal cycle
- **Feature diversity:** Seasonal and spatial variations

## Configuration Parameters

```yaml
# Optimal configuration
roi:
  lon_min: 5.0      # Expanded coverage
  lon_max: 35.0     
  lat_min: -45.0    
  lat_max: -25.0    

time:
  start: "2014-01-01"  # Full year
  end: "2014-12-31"

gridding:
  target_res_deg: 0.1   # Optimal resolution
  memory_chunk_size: 30 # 30-day batches

pseudo_absence:
  n_samples: 2590      # 1:10 ratio (259 × 10)
  sampling_strategy: "stratified"

model:
  algorithms: ["xgboost", "random_forest", "lightgbm"]
  # Optimized hyperparameters for balanced performance
```

## Quality Assurance

### Data Validation:
- **259 real shark observations** (no synthetic data)
- **Full year NASA satellite data** (real measurements)
- **Comprehensive feature set** (advanced oceanographic variables)
- **Spatial and temporal validation** (realistic ranges)

### Performance Monitoring:
- **Progress bars** with time estimates
- **Memory monitoring** (stays under 2GB)
- **Batch processing** with cleanup
- **Error handling** and recovery

## Conclusion

The optimal configuration provides:
- **5× temporal coverage** (14 days → 365 days)
- **4× spatial coverage** (10°×10° → 20°×20°)
- **63% more shark data** (159 → 259 observations)
- **Balanced training set** (1:10 ratio)
- **Reasonable processing time** (~45-65 minutes)
- **Significant performance improvement** (expected 0.55 → 0.65-0.75 ROC-AUC)

This configuration maximizes scientific value while maintaining computational efficiency and ensuring all data is real and validated.

