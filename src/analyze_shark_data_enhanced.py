"""
Enhanced Shark Data Analysis and Expansion

This script analyzes the full shark dataset to:
1. Identify all available shark observations
2. Expand temporal coverage beyond the current 14-day period
3. Analyze data quality and coordinate accuracy
4. Engineer new features based on insights
5. Generate comprehensive data analysis reports
6. Create expanded training datasets

Features:
- Comprehensive shark data analysis
- Temporal expansion strategies
- Data quality validation
- Feature engineering recommendations
- AI-powered data insights
- Automated report generation
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from .utils import load_config, ensure_dir, setup_logging
    from .ai_analysis import AIAnalysisEngine
except ImportError:
    from utils import load_config, ensure_dir, setup_logging
    from ai_analysis import AIAnalysisEngine


class SharkDataAnalyzer:
    """Enhanced shark data analyzer for comprehensive dataset exploration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = setup_logging(__name__)
        
        # Initialize AI analysis engine
        self.ai_engine = AIAnalysisEngine(config)
        
        # Data storage
        self.shark_data = None
        self.analysis_results = {}
        self.data_quality_report = {}
        self.expansion_recommendations = {}
        
        # Configuration
        self.data_cfg = self.config.get('data', {})
        self.analysis_cfg = self.config.get('analysis', {})
        
    def load_shark_data(self, csv_path: str = None) -> pd.DataFrame:
        """Load shark tracking data from CSV file."""
        if csv_path is None:
            csv_path = os.getenv('SHARK_CSV', 'sharks_cleaned.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Shark data file not found: {csv_path}")
        
        self.logger.info(f"Loading shark data from {csv_path}")
        
        # Load data with proper parsing
        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded {len(df)} shark observations")
            
            # Parse datetime
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['date'] = df['datetime'].dt.date
                df['year'] = df['datetime'].dt.year
                df['month'] = df['datetime'].dt.month
                df['day_of_year'] = df['datetime'].dt.dayofyear
                df['hour'] = df['datetime'].dt.hour
                df['day_of_week'] = df['datetime'].dt.dayofweek
            
            self.shark_data = df
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load shark data: {e}")
            raise
    
    def analyze_data_overview(self) -> Dict[str, Any]:
        """Analyze basic dataset overview and statistics."""
        if self.shark_data is None:
            raise ValueError("No shark data loaded. Call load_shark_data() first.")
        
        df = self.shark_data
        
        overview = {
            'total_observations': len(df),
            'unique_sharks': df['id'].nunique() if 'id' in df.columns else 0,
            'species_count': df['species'].nunique() if 'species' in df.columns else 0,
            'date_range': {
                'start': str(df['datetime'].min()) if 'datetime' in df.columns else 'Unknown',
                'end': str(df['datetime'].max()) if 'datetime' in df.columns else 'Unknown',
                'duration_days': (df['datetime'].max() - df['datetime'].min()).days if 'datetime' in df.columns else 0
            },
            'geographic_range': {
                'latitude': {
                    'min': df['latitude'].min(),
                    'max': df['latitude'].max(),
                    'mean': df['latitude'].mean()
                },
                'longitude': {
                    'min': df['longitude'].min(),
                    'max': df['longitude'].max(),
                    'mean': df['longitude'].mean()
                }
            }
        }
        
        # Species breakdown
        if 'species' in df.columns:
            species_counts = df['species'].value_counts()
            overview['species_breakdown'] = species_counts.to_dict()
        
        # Individual shark analysis
        if 'id' in df.columns:
            shark_counts = df['id'].value_counts()
            overview['individual_shark_stats'] = {
                'most_observations': shark_counts.max(),
                'least_observations': shark_counts.min(),
                'mean_observations': shark_counts.mean(),
                'sharks_with_100_plus_obs': (shark_counts >= 100).sum(),
                'sharks_with_500_plus_obs': (shark_counts >= 500).sum()
            }
        
        self.analysis_results['overview'] = overview
        return overview
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in shark observations."""
        if self.shark_data is None:
            raise ValueError("No shark data loaded.")
        
        df = self.shark_data
        
        temporal_analysis = {}
        
        if 'datetime' in df.columns:
            # Yearly patterns
            yearly_counts = df.groupby(df['datetime'].dt.year).size()
            temporal_analysis['yearly_patterns'] = {
                'observations_per_year': yearly_counts.to_dict(),
                'most_active_year': yearly_counts.idxmax(),
                'least_active_year': yearly_counts.idxmin(),
                'years_covered': len(yearly_counts)
            }
            
            # Monthly patterns
            monthly_counts = df.groupby(df['datetime'].dt.month).size()
            temporal_analysis['monthly_patterns'] = {
                'observations_per_month': monthly_counts.to_dict(),
                'most_active_month': monthly_counts.idxmax(),
                'least_active_month': monthly_counts.idxmin(),
                'seasonal_variation': monthly_counts.std() / monthly_counts.mean()
            }
            
            # Daily patterns
            daily_counts = df.groupby(df['datetime'].dt.dayofyear).size()
            temporal_analysis['daily_patterns'] = {
                'most_active_day': daily_counts.idxmax(),
                'least_active_day': daily_counts.idxmin(),
                'temporal_variation': daily_counts.std() / daily_counts.mean()
            }
            
            # Hourly patterns
            hourly_counts = df.groupby(df['datetime'].dt.hour).size()
            temporal_analysis['hourly_patterns'] = {
                'observations_per_hour': hourly_counts.to_dict(),
                'most_active_hour': hourly_counts.idxmax(),
                'least_active_hour': hourly_counts.idxmin(),
                'diel_variation': hourly_counts.std() / hourly_counts.mean()
            }
            
            # Day of week patterns
            dow_counts = df.groupby(df['datetime'].dt.dayofweek).size()
            temporal_analysis['day_of_week_patterns'] = {
                'observations_per_day': dow_counts.to_dict(),
                'most_active_day': dow_counts.idxmax(),
                'least_active_day': dow_counts.idxmin()
            }
        
        self.analysis_results['temporal_patterns'] = temporal_analysis
        return temporal_analysis
    
    def analyze_spatial_patterns(self) -> Dict[str, Any]:
        """Analyze spatial patterns in shark observations."""
        if self.shark_data is None:
            raise ValueError("No shark data loaded.")
        
        df = self.shark_data
        
        spatial_analysis = {}
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Geographic distribution
            spatial_analysis['geographic_distribution'] = {
                'latitude_range': df['latitude'].max() - df['latitude'].min(),
                'longitude_range': df['longitude'].max() - df['longitude'].min(),
                'centroid': {
                    'latitude': df['latitude'].mean(),
                    'longitude': df['longitude'].mean()
                },
                'spatial_extent_km': self._calculate_spatial_extent(df)
            }
            
            # Density analysis
            lat_bins = 20
            lon_bins = 20
            
            lat_binned = pd.cut(df['latitude'], bins=lat_bins, labels=False)
            lon_binned = pd.cut(df['longitude'], bins=lon_bins, labels=False)
            
            density_grid = pd.crosstab(lat_binned, lon_binned)
            spatial_analysis['density_analysis'] = {
                'max_density': density_grid.max().max(),
                'min_density': density_grid.min().min(),
                'mean_density': density_grid.mean().mean(),
                'density_variation': density_grid.std().std()
            }
            
            # Movement analysis
            if 'id' in df.columns:
                movement_stats = self._analyze_individual_movements(df)
                spatial_analysis['movement_analysis'] = movement_stats
        
        self.analysis_results['spatial_patterns'] = spatial_analysis
        return spatial_analysis
    
    def _calculate_spatial_extent(self, df: pd.DataFrame) -> float:
        """Calculate approximate spatial extent in kilometers."""
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lon1, lat1, lon2, lat2):
            """Calculate the great circle distance between two points on earth."""
            # Convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Radius of earth in kilometers
            return c * r
        
        # Calculate extent from min/max coordinates
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        # Approximate extent
        lat_extent = haversine(lon_min, lat_min, lon_min, lat_max)
        lon_extent = haversine(lon_min, lat_min, lon_max, lat_min)
        
        return max(lat_extent, lon_extent)
    
    def _analyze_individual_movements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual shark movement patterns."""
        movement_stats = {}
        
        for shark_id in df['id'].unique():
            shark_data = df[df['id'] == shark_id].sort_values('datetime')
            
            if len(shark_data) > 1:
                # Calculate distances between consecutive points
                distances = []
                for i in range(1, len(shark_data)):
                    lat1, lon1 = shark_data.iloc[i-1]['latitude'], shark_data.iloc[i-1]['longitude']
                    lat2, lon2 = shark_data.iloc[i]['latitude'], shark_data.iloc[i]['longitude']
                    
                    # Simple distance calculation (not great circle)
                    dist = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # Approximate km
                    distances.append(dist)
                
                if distances:
                    movement_stats[shark_id] = {
                        'total_distance_km': sum(distances),
                        'mean_distance_km': np.mean(distances),
                        'max_distance_km': np.max(distances),
                        'movement_observations': len(distances)
                    }
        
        return movement_stats
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and identify issues."""
        if self.shark_data is None:
            raise ValueError("No shark data loaded.")
        
        df = self.shark_data
        
        quality_report = {
            'missing_values': {},
            'data_types': {},
            'outliers': {},
            'duplicates': {},
            'coordinate_validation': {},
            'temporal_validation': {}
        }
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        quality_report['missing_values'] = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Data types analysis
        quality_report['data_types'] = df.dtypes.to_dict()
        
        # Outlier detection for coordinates
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_outliers = self._detect_outliers(df['latitude'])
            lon_outliers = self._detect_outliers(df['longitude'])
            
            quality_report['outliers'] = {
                'latitude_outliers': len(lat_outliers),
                'longitude_outliers': len(lon_outliers),
                'total_outliers': len(lat_outliers) + len(lon_outliers)
            }
        
        # Coordinate validation
        if 'latitude' in df.columns and 'longitude' in df.columns:
            invalid_coords = df[(df['latitude'] < -90) | (df['latitude'] > 90) | 
                               (df['longitude'] < -180) | (df['longitude'] > 180)]
            
            quality_report['coordinate_validation'] = {
                'invalid_coordinates': len(invalid_coords),
                'valid_coordinates': len(df) - len(invalid_coords),
                'coordinate_accuracy': ((len(df) - len(invalid_coords)) / len(df)) * 100
            }
        
        # Temporal validation
        if 'datetime' in df.columns:
            future_dates = df[df['datetime'] > datetime.now()]
            very_old_dates = df[df['datetime'] < datetime(1900, 1, 1)]
            
            quality_report['temporal_validation'] = {
                'future_dates': len(future_dates),
                'very_old_dates': len(very_old_dates),
                'valid_dates': len(df) - len(future_dates) - len(very_old_dates),
                'temporal_accuracy': ((len(df) - len(future_dates) - len(very_old_dates)) / len(df)) * 100
            }
        
        # Duplicate detection
        duplicates = df.duplicated().sum()
        quality_report['duplicates'] = {
            'duplicate_rows': duplicates,
            'duplicate_percentage': (duplicates / len(df)) * 100
        }
        
        self.data_quality_report = quality_report
        return quality_report
    
    def _detect_outliers(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detect outliers using z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores[z_scores > threshold].index.tolist()
    
    def identify_expansion_opportunities(self) -> Dict[str, Any]:
        """Identify opportunities for expanding the dataset."""
        if self.shark_data is None:
            raise ValueError("No shark data loaded.")
        
        df = self.shark_data
        
        expansion_opportunities = {
            'temporal_expansion': {},
            'spatial_expansion': {},
            'individual_expansion': {},
            'feature_expansion': {}
        }
        
        # Temporal expansion opportunities
        if 'datetime' in df.columns:
            date_range = df['datetime'].max() - df['datetime'].min()
            total_days = date_range.days
            
            expansion_opportunities['temporal_expansion'] = {
                'current_coverage_days': total_days,
                'current_coverage_years': total_days / 365.25,
                'recommended_expansion_days': min(365, total_days * 2),  # Expand to at least 1 year
                'potential_observations': self._estimate_potential_observations(df, 'temporal')
            }
        
        # Spatial expansion opportunities
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_range = df['latitude'].max() - df['latitude'].min()
            lon_range = df['longitude'].max() - df['longitude'].min()
            
            expansion_opportunities['spatial_expansion'] = {
                'current_lat_range': lat_range,
                'current_lon_range': lon_range,
                'recommended_expansion_factor': 1.5,  # Expand by 50%
                'potential_observations': self._estimate_potential_observations(df, 'spatial')
            }
        
        # Individual expansion opportunities
        if 'id' in df.columns:
            individual_counts = df['id'].value_counts()
            expansion_opportunities['individual_expansion'] = {
                'current_individuals': len(individual_counts),
                'individuals_with_100_plus_obs': (individual_counts >= 100).sum(),
                'individuals_with_500_plus_obs': (individual_counts >= 500).sum(),
                'recommended_min_observations_per_individual': 50,
                'potential_observations': self._estimate_potential_observations(df, 'individual')
            }
        
        # Feature expansion opportunities
        expansion_opportunities['feature_expansion'] = {
            'current_features': len(df.columns),
            'recommended_new_features': [
                'distance_to_coast',
                'bathymetry',
                'tidal_phase',
                'moon_phase',
                'seasonal_indicators',
                'movement_speed',
                'movement_direction',
                'habitat_type',
                'prey_abundance_proxy',
                'water_clarity'
            ],
            'feature_engineering_priority': 'high'
        }
        
        self.expansion_recommendations = expansion_opportunities
        return expansion_opportunities
    
    def _estimate_potential_observations(self, df: pd.DataFrame, expansion_type: str) -> int:
        """Estimate potential observations for different expansion types."""
        current_obs = len(df)
        
        if expansion_type == 'temporal':
            # Estimate based on temporal density
            if 'datetime' in df.columns:
                date_range = df['datetime'].max() - df['datetime'].min()
                current_days = date_range.days
                target_days = min(365, current_days * 2)
                return int(current_obs * (target_days / current_days))
        
        elif expansion_type == 'spatial':
            # Estimate based on spatial expansion
            return int(current_obs * 1.5)
        
        elif expansion_type == 'individual':
            # Estimate based on individual coverage
            if 'id' in df.columns:
                current_individuals = df['id'].nunique()
                target_individuals = current_individuals * 1.2  # 20% more individuals
                return int(current_obs * (target_individuals / current_individuals))
        
        return current_obs
    
    def create_expanded_training_data(self, target_date_range: Tuple[str, str] = None) -> pd.DataFrame:
        """Create expanded training dataset with more observations."""
        if self.shark_data is None:
            raise ValueError("No shark data loaded.")
        
        df = self.shark_data.copy()
        
        # Filter by target date range if provided
        if target_date_range and 'datetime' in df.columns:
            start_date, end_date = target_date_range
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        
        # Create expanded dataset
        expanded_data = []
        
        # Add original observations as positive samples
        for _, row in df.iterrows():
            expanded_data.append({
                'date': row['datetime'].date() if 'datetime' in row else row.get('date'),
                'lat': row['latitude'],
                'lon': row['longitude'],
                'label': 1,  # Positive sample
                'shark_id': row['id'],
                'species': row.get('species', 'Unknown'),
                'timestamp': row['datetime'] if 'datetime' in row else None
            })
        
        # Add pseudo-absence samples (negative samples)
        # Use stratified sampling across the spatial and temporal domain
        pseudo_absences = self._generate_pseudo_absences(df, n_samples=len(expanded_data) * 10)
        
        for pseudo_absence in pseudo_absences:
            expanded_data.append({
                'date': pseudo_absence['date'],
                'lat': pseudo_absence['lat'],
                'lon': pseudo_absence['lon'],
                'label': 0,  # Negative sample
                'shark_id': None,
                'species': None,
                'timestamp': pseudo_absence['timestamp']
            })
        
        expanded_df = pd.DataFrame(expanded_data)
        
        self.logger.info(f"Created expanded training dataset:")
        self.logger.info(f"  Positive samples: {len(expanded_df[expanded_df['label'] == 1])}")
        self.logger.info(f"  Negative samples: {len(expanded_df[expanded_df['label'] == 0])}")
        self.logger.info(f"  Total samples: {len(expanded_df)}")
        
        return expanded_df
    
    def _generate_pseudo_absences(self, df: pd.DataFrame, n_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate pseudo-absence samples for training."""
        pseudo_absences = []
        
        if 'datetime' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
            # Get spatial and temporal bounds
            lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
            lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
            date_min, date_max = df['datetime'].min(), df['datetime'].max()
            
            # Generate random samples within bounds
            for _ in range(n_samples):
                # Random location within bounds
                lat = np.random.uniform(lat_min, lat_max)
                lon = np.random.uniform(lon_min, lon_max)
                
                # Random date within bounds
                date_range = (date_max - date_min).days
                random_days = np.random.randint(0, date_range)
                random_date = date_min + timedelta(days=random_days)
                
                pseudo_absences.append({
                    'lat': lat,
                    'lon': lon,
                    'date': random_date.date(),
                    'timestamp': random_date
                })
        
        return pseudo_absences
    
    def generate_ai_insights(self) -> Dict[str, Any]:
        """Generate AI-powered insights about the shark data."""
        if self.shark_data is None:
            raise ValueError("No shark data loaded.")
        
        self.logger.info("Generating AI-powered insights...")
        
        # Analyze shark behavior patterns
        behavior_analysis = self.ai_engine.analyze_shark_behavior(self.shark_data)
        
        # Generate recommendations
        recommendations = self.ai_engine.generate_improvement_recommendations(
            self.analysis_results
        )
        
        # Combine all insights
        ai_insights = {
            'behavior_analysis': behavior_analysis,
            'recommendations': recommendations,
            'data_quality_insights': self._generate_data_quality_insights(),
            'expansion_insights': self._generate_expansion_insights()
        }
        
        return ai_insights
    
    def _generate_data_quality_insights(self) -> Dict[str, Any]:
        """Generate insights about data quality."""
        if not self.data_quality_report:
            return {}
        
        insights = {
            'overall_quality': 'good' if self.data_quality_report['missing_values']['missing_percentage'] < 5 else 'needs_improvement',
            'key_issues': [],
            'recommendations': []
        }
        
        # Check for key issues
        if self.data_quality_report['missing_values']['missing_percentage'] > 5:
            insights['key_issues'].append('High percentage of missing values')
            insights['recommendations'].append('Implement data imputation strategies')
        
        if self.data_quality_report['outliers']['total_outliers'] > 0:
            insights['key_issues'].append('Coordinate outliers detected')
            insights['recommendations'].append('Review and clean coordinate data')
        
        if self.data_quality_report['duplicates']['duplicate_percentage'] > 1:
            insights['key_issues'].append('Duplicate observations present')
            insights['recommendations'].append('Remove or merge duplicate records')
        
        return insights
    
    def _generate_expansion_insights(self) -> Dict[str, Any]:
        """Generate insights about data expansion opportunities."""
        if not self.expansion_recommendations:
            return {}
        
        insights = {
            'priority_expansions': [],
            'feasibility_scores': {},
            'expected_improvements': {}
        }
        
        # Prioritize expansions
        if self.expansion_recommendations['temporal_expansion']['current_coverage_days'] < 365:
            insights['priority_expansions'].append('temporal_expansion')
            insights['feasibility_scores']['temporal_expansion'] = 0.8
            insights['expected_improvements']['temporal_expansion'] = 'High - will improve seasonal pattern learning'
        
        if self.expansion_recommendations['individual_expansion']['current_individuals'] < 10:
            insights['priority_expansions'].append('individual_expansion')
            insights['feasibility_scores']['individual_expansion'] = 0.6
            insights['expected_improvements']['individual_expansion'] = 'Medium - will improve individual variation modeling'
        
        insights['priority_expansions'].append('feature_expansion')
        insights['feasibility_scores']['feature_expansion'] = 0.9
        insights['expected_improvements']['feature_expansion'] = 'High - will improve environmental modeling'
        
        return insights
    
    def generate_comprehensive_report(self, output_dir: str = "data/interim") -> str:
        """Generate comprehensive analysis report."""
        ensure_dir(output_dir)
        
        # Generate AI insights
        ai_insights = self.generate_ai_insights()
        
        # Create comprehensive report
        report_path = os.path.join(output_dir, "shark_data_analysis_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Shark Data Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            overview = self.analysis_results.get('overview', {})
            f.write(f"- **Total Observations**: {overview.get('total_observations', 'Unknown')}\n")
            f.write(f"- **Unique Sharks**: {overview.get('unique_sharks', 'Unknown')}\n")
            f.write(f"- **Species Count**: {overview.get('species_count', 'Unknown')}\n")
            f.write(f"- **Date Range**: {overview.get('date_range', {}).get('start', 'Unknown')} to {overview.get('date_range', {}).get('end', 'Unknown')}\n")
            f.write(f"- **Duration**: {overview.get('date_range', {}).get('duration_days', 'Unknown')} days\n\n")
            
            # Data Quality
            f.write("## Data Quality Assessment\n\n")
            quality = self.data_quality_report
            f.write(f"- **Missing Values**: {quality.get('missing_values', {}).get('missing_percentage', 0):.2f}%\n")
            f.write(f"- **Coordinate Accuracy**: {quality.get('coordinate_validation', {}).get('coordinate_accuracy', 0):.2f}%\n")
            f.write(f"- **Temporal Accuracy**: {quality.get('temporal_validation', {}).get('temporal_accuracy', 0):.2f}%\n")
            f.write(f"- **Duplicates**: {quality.get('duplicates', {}).get('duplicate_percentage', 0):.2f}%\n\n")
            
            # Temporal Patterns
            f.write("## Temporal Patterns\n\n")
            temporal = self.analysis_results.get('temporal_patterns', {})
            if temporal:
                monthly = temporal.get('monthly_patterns', {})
                f.write(f"- **Most Active Month**: {monthly.get('most_active_month', 'Unknown')}\n")
                f.write(f"- **Seasonal Variation**: {monthly.get('seasonal_variation', 0):.2f}\n")
                
                hourly = temporal.get('hourly_patterns', {})
                f.write(f"- **Most Active Hour**: {hourly.get('most_active_hour', 'Unknown')}\n")
                f.write(f"- **Diel Variation**: {hourly.get('diel_variation', 0):.2f}\n\n")
            
            # Spatial Patterns
            f.write("## Spatial Patterns\n\n")
            spatial = self.analysis_results.get('spatial_patterns', {})
            if spatial:
                geo_dist = spatial.get('geographic_distribution', {})
                f.write(f"- **Spatial Extent**: {geo_dist.get('spatial_extent_km', 0):.2f} km\n")
                f.write(f"- **Centroid**: ({geo_dist.get('centroid', {}).get('latitude', 0):.2f}, {geo_dist.get('centroid', {}).get('longitude', 0):.2f})\n\n")
            
            # Expansion Opportunities
            f.write("## Expansion Opportunities\n\n")
            expansion = self.expansion_recommendations
            f.write("### Temporal Expansion\n")
            temporal_exp = expansion.get('temporal_expansion', {})
            f.write(f"- **Current Coverage**: {temporal_exp.get('current_coverage_days', 0)} days\n")
            f.write(f"- **Recommended Expansion**: {temporal_exp.get('recommended_expansion_days', 0)} days\n")
            f.write(f"- **Potential Observations**: {temporal_exp.get('potential_observations', 0)}\n\n")
            
            f.write("### Feature Expansion\n")
            feature_exp = expansion.get('feature_expansion', {})
            f.write(f"- **Current Features**: {feature_exp.get('current_features', 0)}\n")
            f.write(f"- **Recommended New Features**: {len(feature_exp.get('recommended_new_features', []))}\n\n")
            
            # AI Insights
            f.write("## AI-Powered Insights\n\n")
            if ai_insights:
                behavior = ai_insights.get('behavior_analysis', {})
                if behavior:
                    f.write("### Behavior Analysis\n")
                    f.write(f"{behavior.get('raw_response', 'Analysis not available')}\n\n")
                
                recommendations = ai_insights.get('recommendations', {})
                if recommendations:
                    f.write("### Recommendations\n")
                    f.write(f"{recommendations.get('raw_response', 'Recommendations not available')}\n\n")
            
            # Next Steps
            f.write("## Recommended Next Steps\n\n")
            f.write("1. **Expand Temporal Coverage**: Collect data for at least 1 year\n")
            f.write("2. **Improve Data Quality**: Address missing values and outliers\n")
            f.write("3. **Feature Engineering**: Implement recommended environmental features\n")
            f.write("4. **Model Enhancement**: Use expanded dataset for training\n")
            f.write("5. **Validation**: Implement robust cross-validation strategies\n\n")
            
            f.write("---\n")
            f.write("*Report generated by AI-enhanced shark data analyzer*\n")
        
        self.logger.info(f"Comprehensive analysis report saved to {report_path}")
        return report_path
    
    def save_expanded_training_data(self, output_dir: str = "data/interim") -> str:
        """Save expanded training data to file."""
        ensure_dir(output_dir)
        
        # Create expanded dataset
        expanded_data = self.create_expanded_training_data()
        
        # Save to CSV
        output_path = os.path.join(output_dir, "expanded_training_data.csv")
        expanded_data.to_csv(output_path, index=False)
        
        self.logger.info(f"Expanded training data saved to {output_path}")
        return output_path


def main():
    """Main function for shark data analysis."""
    # Load configuration
    config = load_config("config/params_ai_enhanced.yaml")
    
    # Initialize analyzer
    analyzer = SharkDataAnalyzer(config)
    
    # Load shark data
    shark_data = analyzer.load_shark_data()
    
    # Perform comprehensive analysis
    print("Analyzing shark data...")
    analyzer.analyze_data_overview()
    analyzer.analyze_temporal_patterns()
    analyzer.analyze_spatial_patterns()
    analyzer.validate_data_quality()
    analyzer.identify_expansion_opportunities()
    
    # Generate reports
    print("Generating comprehensive report...")
    report_path = analyzer.generate_comprehensive_report()
    
    print("Creating expanded training data...")
    expanded_data_path = analyzer.save_expanded_training_data()
    
    print(f"\nAnalysis complete!")
    print(f"Report: {report_path}")
    print(f"Expanded data: {expanded_data_path}")
    
    # Print summary
    overview = analyzer.analysis_results.get('overview', {})
    print(f"\nDataset Summary:")
    print(f"  Total observations: {overview.get('total_observations', 'Unknown')}")
    print(f"  Unique sharks: {overview.get('unique_sharks', 'Unknown')}")
    print(f"  Date range: {overview.get('date_range', {}).get('duration_days', 'Unknown')} days")
    
    expansion = analyzer.expansion_recommendations.get('temporal_expansion', {})
    print(f"  Recommended expansion: {expansion.get('recommended_expansion_days', 0)} days")


if __name__ == "__main__":
    main()
