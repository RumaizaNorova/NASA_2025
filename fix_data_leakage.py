#!/usr/bin/env python3
"""
Fix Critical Data Leakage in AI-Enhanced Shark Habitat Prediction System
Main script to orchestrate the complete fix of temporal data leakage
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataLeakageFixer:
    """Main orchestrator for fixing data leakage"""
    
    def __init__(self):
        self.scripts_dir = Path("src")
        self.data_dir = Path("data/interim")
        
        # Script execution order
        self.scripts = [
            ("fix_negative_sampling.py", "Fix negative sampling to match shark observation temporal patterns"),
            ("create_oceanographic_features.py", "Create oceanographic-only features from NASA data"),
            ("spatial_cross_validation.py", "Implement spatial cross-validation to prevent data leakage"),
            ("validate_realistic_performance.py", "Validate realistic performance expectations (ROC-AUC 0.65-0.75)")
        ]
        
    def run_script(self, script_name, description):
        """Run a single script"""
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"📄 Script: {script_name}")
        print(f"{'='*60}")
        
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        try:
            # Run the script
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                print("✅ Script completed successfully")
                if result.stdout:
                    print("📄 Output:")
                    print(result.stdout)
                return True
            else:
                print("❌ Script failed")
                if result.stderr:
                    print("📄 Error output:")
                    print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error running script: {e}")
            return False
    
    def check_prerequisites(self):
        """Check if prerequisites are met"""
        print("🔍 Checking prerequisites...")
        
        # Check if training data exists
        training_data_path = self.data_dir / 'training_data_expanded.csv'
        if not training_data_path.exists():
            print("❌ Training data not found. Please ensure training_data_expanded.csv exists")
            return False
        
        print("✅ Prerequisites met")
        return True
    
    def run_complete_fix(self):
        """Run the complete data leakage fix"""
        print("🚀 FIXING CRITICAL DATA LEAKAGE")
        print("="*60)
        print("This will fix the temporal data leakage that causes invalid 0.997 ROC-AUC")
        print("Expected result: Realistic performance (ROC-AUC 0.65-0.75)")
        print("="*60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Run each script in order
        success_count = 0
        total_scripts = len(self.scripts)
        
        for i, (script_name, description) in enumerate(self.scripts, 1):
            print(f"\n📋 Step {i}/{total_scripts}")
            
            success = self.run_script(script_name, description)
            
            if success:
                success_count += 1
                print(f"✅ Step {i} completed successfully")
            else:
                print(f"❌ Step {i} failed")
                print("🛑 Stopping execution due to failure")
                break
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful steps: {success_count}/{total_scripts}")
        
        if success_count == total_scripts:
            print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
            print("✅ Data leakage has been fixed")
            print("✅ Model now uses only oceanographic features")
            print("✅ Performance should be realistic (ROC-AUC 0.65-0.75)")
            print("✅ Ready for NASA Space Apps Challenge submission")
            
            # Final validation message
            print(f"\n🔍 FINAL VALIDATION:")
            print("   - Check spatial_cv_results.json for cross-validation results")
            print("   - Check realistic_performance_validation.json for performance validation")
            print("   - Verify that ROC-AUC is between 0.65-0.75")
            print("   - Ensure no temporal features remain in the model")
            
            return True
        else:
            print("❌ SOME STEPS FAILED")
            print("🔧 Please check the error messages above and fix issues")
            return False
    
    def show_help(self):
        """Show help information"""
        print("🚀 Data Leakage Fixer for Shark Habitat Prediction")
        print("="*60)
        print("This script fixes critical temporal data leakage in the AI-enhanced")
        print("shark habitat prediction system.")
        print()
        print("PROBLEM:")
        print("  - Current model achieves 0.997 ROC-AUC due to temporal leakage")
        print("  - Shark observations clustered at specific times (mostly midnight)")
        print("  - Negative samples created with random times")
        print("  - Model learns TIME patterns, not HABITAT patterns")
        print()
        print("SOLUTION:")
        print("  1. Fix negative sampling to match shark observation temporal patterns")
        print("  2. Remove all temporal features (diurnal_cycle, annual_cycle, etc.)")
        print("  3. Create oceanographic-only features from NASA data")
        print("  4. Implement spatial cross-validation")
        print("  5. Validate realistic performance expectations")
        print()
        print("EXPECTED RESULT:")
        print("  - Realistic performance (ROC-AUC 0.65-0.75)")
        print("  - Model predicts habitat based on oceanographic conditions")
        print("  - No temporal leakage or artificial performance inflation")
        print("  - Scientifically valid results for NASA challenge")
        print()
        print("USAGE:")
        print("  python fix_data_leakage.py")
        print()
        print("REQUIREMENTS:")
        print("  - training_data_expanded.csv must exist in data/interim/")
        print("  - All Python dependencies must be installed")
        print("  - Sufficient disk space for processed data")

def main():
    """Main function"""
    fixer = DataLeakageFixer()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        fixer.show_help()
        return True
    
    try:
        success = fixer.run_complete_fix()
        return success
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
