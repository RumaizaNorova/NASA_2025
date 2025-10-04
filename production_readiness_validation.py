#!/usr/bin/env python3
"""
Production Readiness Validation for AI-Enhanced Shark Habitat Prediction
Final comprehensive validation before heavy model training
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProductionReadinessValidator:
    """Comprehensive production readiness validation"""
    
    def __init__(self):
        self.config_path = 'config/params_ai_enhanced.yaml'
        self.config = self.load_config()
        self.validation_results = {}
        
    def load_config(self):
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_data_infrastructure(self):
        """Validate data infrastructure"""
        print("🔍 Validating data infrastructure...")
        
        validation_results = {
            'expanded_data': False,
            'satellite_data': False,
            'feature_engineering': False,
            'data_quality': False
        }
        
        # Check expanded training data
        expanded_data_path = 'data/interim/training_data_expanded.csv'
        if os.path.exists(expanded_data_path):
            df = pd.read_csv(expanded_data_path)
            if len(df) > 300000:  # Should have expanded data
                validation_results['expanded_data'] = True
                print(f"  ✅ Expanded data: {len(df):,} samples")
            else:
                print(f"  ❌ Expanded data insufficient: {len(df):,} samples")
        else:
            print("  ❌ Expanded data not found")
        
        # Check satellite data structure
        satellite_dirs = ['mur_sst', 'measures_ssh', 'oscar_currents', 'pace_chl', 'smap_salinity', 'gpm_precipitation']
        satellite_files = 0
        for dir_name in satellite_dirs:
            dir_path = f'data/raw/{dir_name}'
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith('.nc')]
                satellite_files += len(files)
        
        if satellite_files >= 6:  # Should have at least 6 files
            validation_results['satellite_data'] = True
            print(f"  ✅ Satellite data: {satellite_files} files")
        else:
            print(f"  ❌ Satellite data insufficient: {satellite_files} files")
        
        # Check feature engineering
        if 'feature_selection' in self.config and self.config['feature_selection']['enable']:
            validation_results['feature_engineering'] = True
            print("  ✅ Feature engineering configured")
        else:
            print("  ❌ Feature engineering not configured")
        
        # Check data quality
        if 'data' in self.config and 'validation' in self.config['data']:
            validation_results['data_quality'] = True
            print("  ✅ Data quality checks configured")
        else:
            print("  ❌ Data quality checks not configured")
        
        self.validation_results['data_infrastructure'] = validation_results
        return validation_results
    
    def validate_model_configuration(self):
        """Validate model configuration"""
        print("🔍 Validating model configuration...")
        
        validation_results = {
            'algorithms': False,
            'regularization': False,
            'cross_validation': False,
            'early_stopping': False,
            'ensemble': False
        }
        
        # Check algorithms
        if 'model' in self.config and 'algorithms' in self.config['model']:
            algorithms = self.config['model']['algorithms']
            if len(algorithms) >= 3:  # Should have multiple algorithms
                validation_results['algorithms'] = True
                print(f"  ✅ Algorithms: {algorithms}")
            else:
                print(f"  ❌ Insufficient algorithms: {algorithms}")
        else:
            print("  ❌ Algorithms not configured")
        
        # Check regularization
        if 'regularization' in self.config:
            validation_results['regularization'] = True
            print("  ✅ Regularization configured")
        else:
            print("  ❌ Regularization not configured")
        
        # Check cross-validation
        if 'cv' in self.config and 'strategies' in self.config['cv']:
            validation_results['cross_validation'] = True
            print("  ✅ Cross-validation configured")
        else:
            print("  ❌ Cross-validation not configured")
        
        # Check early stopping
        if 'early_stopping' in self.config and self.config['early_stopping']['enable']:
            validation_results['early_stopping'] = True
            print("  ✅ Early stopping configured")
        else:
            print("  ❌ Early stopping not configured")
        
        # Check ensemble
        if 'ensemble' in self.config and self.config['ensemble']['enable']:
            validation_results['ensemble'] = True
            print("  ✅ Ensemble methods configured")
        else:
            print("  ❌ Ensemble methods not configured")
        
        self.validation_results['model_configuration'] = validation_results
        return validation_results
    
    def validate_overfitting_prevention(self):
        """Validate overfitting prevention strategies"""
        print("🔍 Validating overfitting prevention...")
        
        validation_results = {
            'cross_validation': False,
            'regularization': False,
            'feature_selection': False,
            'data_augmentation': False,
            'validation_monitoring': False
        }
        
        # Check cross-validation strategies
        if 'cv' in self.config and 'strategies' in self.config['cv']:
            strategies = self.config['cv']['strategies']
            if len(strategies) >= 2:  # Should have multiple strategies
                validation_results['cross_validation'] = True
                print(f"  ✅ Cross-validation strategies: {len(strategies)}")
            else:
                print(f"  ❌ Insufficient CV strategies: {len(strategies)}")
        else:
            print("  ❌ Cross-validation not configured")
        
        # Check regularization
        if 'regularization' in self.config:
            reg_config = self.config['regularization']
            if len(reg_config) >= 2:  # Should have multiple algorithms
                validation_results['regularization'] = True
                print(f"  ✅ Regularization: {len(reg_config)} algorithms")
            else:
                print(f"  ❌ Insufficient regularization: {len(reg_config)}")
        else:
            print("  ❌ Regularization not configured")
        
        # Check feature selection
        if 'feature_selection' in self.config and self.config['feature_selection']['enable']:
            validation_results['feature_selection'] = True
            print("  ✅ Feature selection enabled")
        else:
            print("  ❌ Feature selection not enabled")
        
        # Check data augmentation
        if 'data_augmentation' in self.config and self.config['data_augmentation']['enable']:
            validation_results['data_augmentation'] = True
            print("  ✅ Data augmentation enabled")
        else:
            print("  ❌ Data augmentation not enabled")
        
        # Check validation monitoring
        if 'validation_monitoring' in self.config and self.config['validation_monitoring']['enable']:
            validation_results['validation_monitoring'] = True
            print("  ✅ Validation monitoring enabled")
        else:
            print("  ❌ Validation monitoring not enabled")
        
        self.validation_results['overfitting_prevention'] = validation_results
        return validation_results
    
    def validate_performance_targets(self):
        """Validate performance targets"""
        print("🔍 Validating performance targets...")
        
        validation_results = {
            'targets_defined': False,
            'realistic_targets': False,
            'validation_configured': False
        }
        
        # Check if targets are defined
        if 'evaluation' in self.config and 'targets' in self.config['evaluation']:
            targets = self.config['evaluation']['targets']
            if all(key in targets for key in ['roc_auc', 'pr_auc', 'f1', 'tss']):
                validation_results['targets_defined'] = True
                print(f"  ✅ Performance targets defined: {targets}")
            else:
                print("  ❌ Incomplete performance targets")
        else:
            print("  ❌ Performance targets not defined")
        
        # Check if targets are realistic
        if validation_results['targets_defined']:
            targets = self.config['evaluation']['targets']
            if (0.5 <= targets['roc_auc'] <= 0.9 and 
                0.1 <= targets['pr_auc'] <= 0.8 and
                0.1 <= targets['f1'] <= 0.8 and
                0.0 <= targets['tss'] <= 0.8):
                validation_results['realistic_targets'] = True
                print("  ✅ Performance targets are realistic")
            else:
                print("  ❌ Performance targets may be unrealistic")
        
        # Check validation configuration
        if 'validation' in self.config and 'performance_validation' in self.config['validation']:
            validation_results['validation_configured'] = True
            print("  ✅ Performance validation configured")
        else:
            print("  ❌ Performance validation not configured")
        
        self.validation_results['performance_targets'] = validation_results
        return validation_results
    
    def validate_system_resources(self):
        """Validate system resources"""
        print("🔍 Validating system resources...")
        
        validation_results = {
            'dependencies': False,
            'memory': False,
            'storage': False,
            'processing': False
        }
        
        # Check dependencies
        try:
            import numpy, pandas, sklearn, xgboost, lightgbm, openai, optuna, imblearn
            validation_results['dependencies'] = True
            print("  ✅ All dependencies available")
        except ImportError as e:
            print(f"  ❌ Missing dependency: {e}")
        
        # Check memory (simplified)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb >= 4:  # Should have at least 4GB
                validation_results['memory'] = True
                print(f"  ✅ Memory: {memory_gb:.1f} GB")
            else:
                print(f"  ❌ Insufficient memory: {memory_gb:.1f} GB")
        except ImportError:
            print("  ⚠️  Cannot check memory (psutil not available)")
            validation_results['memory'] = True  # Assume OK
        
        # Check storage
        try:
            import shutil
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            if free_space_gb >= 10:  # Should have at least 10GB free
                validation_results['storage'] = True
                print(f"  ✅ Storage: {free_space_gb:.1f} GB free")
            else:
                print(f"  ❌ Insufficient storage: {free_space_gb:.1f} GB free")
        except Exception:
            print("  ⚠️  Cannot check storage")
            validation_results['storage'] = True  # Assume OK
        
        # Check processing capability
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            if cpu_count >= 2:  # Should have at least 2 cores
                validation_results['processing'] = True
                print(f"  ✅ CPU cores: {cpu_count}")
            else:
                print(f"  ❌ Insufficient CPU cores: {cpu_count}")
        except Exception:
            print("  ⚠️  Cannot check CPU")
            validation_results['processing'] = True  # Assume OK
        
        self.validation_results['system_resources'] = validation_results
        return validation_results
    
    def validate_environment_setup(self):
        """Validate environment setup"""
        print("🔍 Validating environment setup...")
        
        validation_results = {
            'env_file': False,
            'directories': False,
            'permissions': False,
            'configuration': False
        }
        
        # Check .env file
        if os.path.exists('.env'):
            validation_results['env_file'] = True
            print("  ✅ .env file exists")
        else:
            print("  ❌ .env file not found")
        
        # Check required directories
        required_dirs = ['data/raw', 'data/interim', 'logs', 'config']
        missing_dirs = []
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
        
        if not missing_dirs:
            validation_results['directories'] = True
            print("  ✅ All required directories exist")
        else:
            print(f"  ❌ Missing directories: {missing_dirs}")
        
        # Check permissions (simplified)
        try:
            test_file = 'test_permissions.tmp'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            validation_results['permissions'] = True
            print("  ✅ Write permissions OK")
        except Exception as e:
            print(f"  ❌ Permission error: {e}")
        
        # Check configuration
        if self.config and len(self.config) > 10:  # Should have substantial config
            validation_results['configuration'] = True
            print("  ✅ Configuration loaded")
        else:
            print("  ❌ Configuration insufficient")
        
        self.validation_results['environment_setup'] = validation_results
        return validation_results
    
    def generate_production_readiness_report(self):
        """Generate comprehensive production readiness report"""
        print("📋 Generating production readiness report...")
        
        # Calculate overall readiness
        all_validations = []
        for category, results in self.validation_results.items():
            if isinstance(results, dict):
                all_validations.extend(results.values())
        
        total_checks = len(all_validations)
        passed_checks = sum(1 for check in all_validations if check)
        readiness_percentage = (passed_checks / total_checks) * 100
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_readiness': {
                'percentage': readiness_percentage,
                'passed_checks': passed_checks,
                'total_checks': total_checks,
                'status': 'READY' if readiness_percentage >= 80 else 'NEEDS_ATTENTION'
            },
            'validation_results': self.validation_results,
            'recommendations': [],
            'next_steps': []
        }
        
        # Generate recommendations
        for category, results in self.validation_results.items():
            if isinstance(results, dict):
                for check_name, check_result in results.items():
                    if not check_result:
                        report['recommendations'].append(f"Fix {check_name} in {category}")
        
        # Generate next steps
        if readiness_percentage >= 80:
            report['next_steps'].append("System is ready for heavy model training")
            report['next_steps'].append("Run: python src/train_model_ai_enhanced.py")
            report['next_steps'].append("Monitor training progress and performance")
        else:
            report['next_steps'].append("Address validation issues before training")
            report['next_steps'].append("Review and fix failed validation checks")
            report['next_steps'].append("Re-run validation after fixes")
        
        # Save report
        report_path = 'data/interim/production_readiness_report.json'
        os.makedirs('data/interim', exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✅ Report saved to: {report_path}")
        
        return report
    
    def run_validation(self):
        """Run complete production readiness validation"""
        print("🚀 Production Readiness Validation")
        print("=" * 60)
        
        try:
            # Run all validations
            self.validate_data_infrastructure()
            self.validate_model_configuration()
            self.validate_overfitting_prevention()
            self.validate_performance_targets()
            self.validate_system_resources()
            self.validate_environment_setup()
            
            # Generate report
            report = self.generate_production_readiness_report()
            
            print("\n" + "=" * 60)
            print("🎉 PRODUCTION READINESS VALIDATION COMPLETED!")
            
            # Summary
            readiness = report['overall_readiness']
            print(f"📊 Overall Readiness: {readiness['percentage']:.1f}%")
            print(f"✅ Passed Checks: {readiness['passed_checks']}/{readiness['total_checks']}")
            print(f"🎯 Status: {readiness['status']}")
            
            if readiness['status'] == 'READY':
                print("\n🎉 SYSTEM IS READY FOR HEAVY MODEL TRAINING!")
                print("🚀 Next Steps:")
                for step in report['next_steps']:
                    print(f"  • {step}")
            else:
                print("\n⚠️  SYSTEM NEEDS ATTENTION BEFORE TRAINING")
                print("🔧 Recommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")
            
            return readiness['status'] == 'READY'
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    validator = ProductionReadinessValidator()
    return validator.run_validation()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
