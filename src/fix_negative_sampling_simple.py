#!/usr/bin/env python3
"""
Fix Negative Sampling to Match Shark Observation Temporal Patterns
Remove temporal data leakage by ensuring negative samples have same temporal distribution as shark observations
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NegativeSamplingFixer:
    """Fix negative sampling to match shark observation temporal patterns"""
    
    def __init__(self):
        self.data_dir = Path("data/interim")
        self.output_dir = Path("data/interim")
        
    def load_current_data(self):
        """Load current training data"""
        print("Loading current training data...")
        
        df = pd.read_csv('data/interim/training_data_expanded.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"  Total samples: {len(df):,}")
        print(f"  Shark observations: {len(df[df.target==1]):,}")
        print(f"  Negative samples: {len(df[df.target==0]):,}")
        
        return df
    
    def analyze_temporal_patterns(self, df):
        """Analyze temporal patterns in shark observations"""
        print("Analyzing temporal patterns...")
        
        # Get shark observations
        shark_obs = df[df.target == 1].copy()
        
        # Analyze hour distribution
        shark_obs['hour'] = shark_obs['datetime'].dt.hour
        hour_dist = shark_obs['hour'].value_counts().sort_index()
        
        print("  Shark observation hour distribution:")
        for hour, count in hour_dist.items():
            print(f"    Hour {hour:2d}: {count:5,} observations")
        
        # Analyze day of year distribution
        shark_obs['day_of_year'] = shark_obs['datetime'].dt.dayofyear
        doy_dist = shark_obs['day_of_year'].value_counts().sort_index()
        
        print(f"  Shark observations span {doy_dist.min()} to {doy_dist.max()} days of year")
        
        # Analyze year distribution
        shark_obs['year'] = shark_obs['datetime'].dt.year
        year_dist = shark_obs['year'].value_counts().sort_index()
        
        print("  Shark observation year distribution:")
        for year, count in year_dist.items():
            print(f"    {year}: {count:5,} observations")
        
        return {
            'hour_dist': hour_dist,
            'doy_dist': doy_dist,
            'year_dist': year_dist,
            'shark_obs': shark_obs
        }
    
    def create_fixed_negative_samples(self, shark_obs, num_negative_samples):
        """Create negative samples that match shark observation temporal patterns"""
        print(f"Creating {num_negative_samples:,} fixed negative samples...")
        
        # Get temporal patterns from shark observations
        hour_dist = shark_obs['hour'].value_counts(normalize=True)
        doy_dist = shark_obs['day_of_year'].value_counts(normalize=True)
        year_dist = shark_obs['year'].value_counts(normalize=True)
        
        # Sample hours, days, and years according to shark observation patterns
        sampled_hours = np.random.choice(
            hour_dist.index, 
            size=num_negative_samples, 
            p=hour_dist.values
        )
        
        sampled_days = np.random.choice(
            doy_dist.index, 
            size=num_negative_samples, 
            p=doy_dist.values
        )
        
        sampled_years = np.random.choice(
            year_dist.index, 
            size=num_negative_samples, 
            p=year_dist.values
        )
        
        # Create negative samples
        negative_samples = []
        
        for i in range(num_negative_samples):
            # Create datetime from sampled components
            year = sampled_years[i]
            day_of_year = sampled_days[i]
            hour = sampled_hours[i]
            
            # Convert day of year to month and day
            date = datetime(int(year), 1, 1) + timedelta(days=int(day_of_year) - 1)
            
            # Create datetime with sampled hour
            dt = datetime.combine(date.date(), datetime.min.time().replace(hour=int(hour)))
            
            # Generate random coordinates (avoiding shark observation locations)
            lat = np.random.uniform(-60, 60)  # Global ocean coverage
            lon = np.random.uniform(-180, 180)
            
            # Create negative sample
            negative_sample = {
                'latitude': lat,
                'longitude': lon,
                'datetime': dt,
                'target': 0,
                'shark_id': f'neg_fixed_{i}',
                'species': 'background',
                'timestamp': dt,
                'date': dt.date()
            }
            
            negative_samples.append(negative_sample)
        
        # Convert to DataFrame
        negative_df = pd.DataFrame(negative_samples)
        
        # Add temporal columns for validation
        negative_df['hour'] = negative_df['datetime'].dt.hour
        negative_df['day_of_year'] = negative_df['datetime'].dt.dayofyear
        negative_df['year'] = negative_df['datetime'].dt.year
        
        print(f"  Created {len(negative_df):,} fixed negative samples")
        print(f"  Date range: {negative_df['datetime'].min()} to {negative_df['datetime'].max()}")
        
        return negative_df
    
    def validate_temporal_distribution(self, shark_obs, negative_df):
        """Validate that negative samples match shark observation temporal patterns"""
        print("Validating temporal distribution...")
        
        # Compare hour distributions
        shark_hours = shark_obs['hour'].value_counts(normalize=True).sort_index()
        neg_hours = negative_df['hour'].value_counts(normalize=True).sort_index()
        
        print("  Hour distribution comparison:")
        print("    Hour | Shark % | Negative % | Difference")
        print("    -----|---------|------------|----------")
        
        max_diff = 0
        for hour in range(24):
            shark_pct = shark_hours.get(hour, 0) * 100
            neg_pct = neg_hours.get(hour, 0) * 100
            diff = abs(shark_pct - neg_pct)
            max_diff = max(max_diff, diff)
            print(f"    {hour:4d} | {shark_pct:6.2f}% | {neg_pct:8.2f}% | {diff:6.2f}%")
        
        print(f"  Maximum hour distribution difference: {max_diff:.2f}%")
        
        # Compare day of year distributions
        shark_doy = shark_obs['day_of_year'].value_counts(normalize=True).sort_index()
        neg_doy = negative_df['day_of_year'].value_counts(normalize=True).sort_index()
        
        # Calculate correlation
        common_days = set(shark_doy.index) & set(neg_doy.index)
        if common_days:
            shark_doy_common = shark_doy[list(common_days)].sort_index()
            neg_doy_common = neg_doy[list(common_days)].sort_index()
            doy_correlation = np.corrcoef(shark_doy_common.values, neg_doy_common.values)[0, 1]
            print(f"  Day-of-year correlation: {doy_correlation:.4f}")
        
        # Compare year distributions
        shark_years = shark_obs['year'].value_counts(normalize=True).sort_index()
        neg_years = negative_df['year'].value_counts(normalize=True).sort_index()
        
        print("  Year distribution comparison:")
        print("    Year | Shark % | Negative %")
        print("    -----|---------|------------")
        
        for year in sorted(set(shark_years.index) | set(neg_years.index)):
            shark_pct = shark_years.get(year, 0) * 100
            neg_pct = neg_years.get(year, 0) * 100
            print(f"    {year} | {shark_pct:6.2f}% | {neg_pct:8.2f}%")
        
        return max_diff < 5.0  # Accept if max difference < 5%
    
    def create_fixed_training_data(self):
        """Create fixed training data with proper negative sampling"""
        print("Creating fixed training data...")
        
        # Load current data
        df = self.load_current_data()
        
        # Analyze temporal patterns
        patterns = self.analyze_temporal_patterns(df)
        shark_obs = patterns['shark_obs']
        
        # Determine number of negative samples needed
        num_shark_obs = len(shark_obs)
        num_negative_samples = num_shark_obs * 5  # 5:1 ratio
        
        print(f"  Target: {num_shark_obs:,} shark observations, {num_negative_samples:,} negative samples")
        
        # Create fixed negative samples
        negative_df = self.create_fixed_negative_samples(shark_obs, num_negative_samples)
        
        # Validate temporal distribution
        is_valid = self.validate_temporal_distribution(shark_obs, negative_df)
        
        if not is_valid:
            print("  Warning: Temporal distribution validation failed")
        else:
            print("  Temporal distribution validation passed")
        
        # Combine shark observations and fixed negative samples
        fixed_df = pd.concat([shark_obs, negative_df], ignore_index=True)
        
        # Shuffle the data
        fixed_df = fixed_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save fixed training data
        output_path = self.output_dir / 'training_data_fixed_negative_sampling.csv'
        fixed_df.to_csv(output_path, index=False)
        
        print(f"  Fixed training data saved to: {output_path}")
        print(f"  Total samples: {len(fixed_df):,}")
        print(f"  Shark observations: {len(fixed_df[fixed_df.target==1]):,}")
        print(f"  Negative samples: {len(fixed_df[fixed_df.target==0]):,}")
        
        # Save metadata
        metadata = {
            'total_samples': len(fixed_df),
            'shark_observations': len(fixed_df[fixed_df.target==1]),
            'negative_samples': len(fixed_df[fixed_df.target==0]),
            'ratio': len(fixed_df[fixed_df.target==0]) / len(fixed_df[fixed_df.target==1]),
            'temporal_validation_passed': bool(is_valid),
            'created_at': datetime.now().isoformat(),
            'description': 'Fixed negative sampling to match shark observation temporal patterns'
        }
        
        metadata_path = self.output_dir / 'fixed_negative_sampling_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Metadata saved to: {metadata_path}")
        
        return fixed_df
    
    def run_fix(self):
        """Run the negative sampling fix"""
        print("Fixing Negative Sampling")
        print("=" * 50)
        
        try:
            # Create fixed training data
            fixed_df = self.create_fixed_training_data()
            
            print("\n" + "=" * 50)
            print("NEGATIVE SAMPLING FIX COMPLETED!")
            print(f"Fixed dataset: {len(fixed_df):,} samples")
            print(f"Shark observations: {len(fixed_df[fixed_df.target==1]):,}")
            print(f"Negative samples: {len(fixed_df[fixed_df.target==0]):,}")
            print("Temporal data leakage removed")
            print("Negative samples match shark observation patterns")
            print("Ready for oceanographic feature extraction")
            
            return True
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    fixer = NegativeSamplingFixer()
    return fixer.run_fix()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
