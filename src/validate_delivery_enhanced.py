"""
Enhanced delivery validation script for Sharks from Space project.

This script performs comprehensive validation of the entire pipeline including:
- Data integrity checks
- Model performance validation
- Output file verification
- Web interface functionality
- Performance benchmarks
- Documentation completeness
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import glob
import hashlib
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


class EnhancedDeliveryValidator:
    """Enhanced delivery validator with comprehensive checks."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.logger = setup_logging(__name__)
        
        # Validation results
        self.validation_results = {
            'overall_status': 'PENDING',
            'checks': {},
            'summary': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Directories to check
        self.directories = {
            'data_raw': 'data/raw',
            'data_interim': 'data/interim',
            'web_data': 'web/data',
            'web': 'web',
            'src': 'src',
            'config': 'config',
            'logs': 'logs'
        }
    
    def _check_directory_structure(self) -> Dict[str, Any]:
        """Check if all required directories exist."""
        check_name = "Directory Structure"
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        required_dirs = [
            'data/raw', 'data/interim', 'web/data', 'web', 
            'src', 'config', 'logs', 'tests'
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                results['details'].append(f"✓ {dir_path} exists")
            else:
                results['errors'].append(f"✗ {dir_path} missing")
                results['status'] = 'FAIL'
        
        # Check for optional directories
        optional_dirs = ['notebooks', 'docs', 'exports']
        for dir_path in optional_dirs:
            if os.path.exists(dir_path):
                results['details'].append(f"✓ {dir_path} exists (optional)")
            else:
                results['details'].append(f"- {dir_path} not found (optional)")
        
        return results
    
    def _check_data_files(self) -> Dict[str, Any]:
        """Check data files integrity."""
        check_name = "Data Files"
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        # Check raw data files
        raw_data_patterns = [
            'data/raw/*/*.nc',
            'sharks_cleaned.csv'
        ]
        
        raw_files_found = 0
        for pattern in raw_data_patterns:
            files = glob.glob(pattern)
            if files:
                raw_files_found += len(files)
                results['details'].append(f"✓ Found {len(files)} files matching {pattern}")
            else:
                results['errors'].append(f"✗ No files found for {pattern}")
        
        if raw_files_found == 0:
            results['status'] = 'FAIL'
            results['errors'].append("No raw data files found")
        
        # Check interim data files
        interim_files = [
            'data/interim/features.nc',
            'data/interim/training_data.csv',
            'data/interim/training_metrics.json'
        ]
        
        for file_path in interim_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                results['details'].append(f"✓ {file_path} exists ({file_size:.1f} MB)")
            else:
                results['errors'].append(f"✗ {file_path} missing")
                results['status'] = 'FAIL'
        
        # Check web data files
        web_data_files = glob.glob('web/data/*.png')
        if web_data_files:
            results['details'].append(f"✓ Found {len(web_data_files)} visualization files")
        else:
            results['errors'].append("✗ No visualization files found")
            results['status'] = 'FAIL'
        
        return results
    
    def _check_model_performance(self) -> Dict[str, Any]:
        """Check model performance meets requirements."""
        check_name = "Model Performance"
        results = {'status': 'PASS', 'details': [], 'errors': [], 'warnings': []}
        
        metrics_file = 'data/interim/training_metrics.json'
        if not os.path.exists(metrics_file):
            results['errors'].append("✗ Training metrics file not found")
            results['status'] = 'FAIL'
            return results
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Performance targets
            targets = {
                'roc_auc': 0.65,  # Minimum acceptable
                'pr_auc': 0.30,
                'f1': 0.30,
                'tss': 0.20
            }
            
            best_model = None
            best_roc_auc = 0
            
            for model_name, model_metrics in metrics.items():
                aggregated = model_metrics.get('aggregated_metrics', {})
                
                # Check if model meets targets
                model_status = 'PASS'
                for metric, target in targets.items():
                    value = aggregated.get(metric, 0)
                    if value >= target:
                        results['details'].append(f"✓ {model_name} {metric}: {value:.3f} (target: {target})")
                    else:
                        results['warnings'].append(f"⚠ {model_name} {metric}: {value:.3f} (target: {target})")
                        model_status = 'WARN'
                
                # Track best model
                roc_auc = aggregated.get('roc_auc', 0)
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model = model_name
            
            # Overall performance assessment
            if best_roc_auc >= 0.70:
                results['details'].append(f"✓ Excellent performance: {best_model} ROC-AUC {best_roc_auc:.3f}")
            elif best_roc_auc >= 0.65:
                results['details'].append(f"✓ Good performance: {best_model} ROC-AUC {best_roc_auc:.3f}")
            else:
                results['errors'].append(f"✗ Performance below target: best ROC-AUC {best_roc_auc:.3f}")
                results['status'] = 'FAIL'
            
            # Add performance summary
            results['best_model'] = best_model
            results['best_roc_auc'] = best_roc_auc
            
        except Exception as e:
            results['errors'].append(f"✗ Error reading metrics: {e}")
            results['status'] = 'FAIL'
        
        return results
    
    def _check_web_interface(self) -> Dict[str, Any]:
        """Check web interface functionality."""
        check_name = "Web Interface"
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        # Check main HTML file
        html_files = ['web/index.html', 'web/index_enhanced.html']
        html_found = False
        
        for html_file in html_files:
            if os.path.exists(html_file):
                html_found = True
                file_size = os.path.getsize(html_file) / 1024  # KB
                results['details'].append(f"✓ {html_file} exists ({file_size:.1f} KB)")
                
                # Check for key features in HTML
                try:
                    with open(html_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    features = [
                        ('Interactive map', 'mapbox' in content.lower() or 'maplibre' in content.lower()),
                        ('Time controls', 'dateSlider' in content),
                        ('Model selection', 'modelSelect' in content),
                        ('Performance dashboard', 'performance-dashboard' in content),
                        ('Animation controls', 'playPauseBtn' in content)
                    ]
                    
                    for feature_name, has_feature in features:
                        if has_feature:
                            results['details'].append(f"  ✓ {feature_name} implemented")
                        else:
                            results['warnings'].append(f"  ⚠ {feature_name} not found")
                
                except Exception as e:
                    results['warnings'].append(f"  ⚠ Error reading HTML content: {e}")
                
                break
        
        if not html_found:
            results['errors'].append("✗ No HTML interface found")
            results['status'] = 'FAIL'
        
        # Check for visualization data
        data_files = glob.glob('web/data/*.png')
        if data_files:
            results['details'].append(f"✓ Found {len(data_files)} visualization files")
        else:
            results['errors'].append("✗ No visualization data found")
            results['status'] = 'FAIL'
        
        return results
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration files."""
        check_name = "Configuration"
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        # Check main config file
        config_files = ['config/params.yaml', 'config/params_enhanced.yaml']
        config_found = False
        
        for config_file in config_files:
            if os.path.exists(config_file):
                config_found = True
                results['details'].append(f"✓ {config_file} exists")
                
                # Validate config content
                try:
                    config = load_config(config_file)
                    required_sections = ['roi', 'time', 'model', 'features']
                    
                    for section in required_sections:
                        if section in config:
                            results['details'].append(f"  ✓ {section} section present")
                        else:
                            results['errors'].append(f"  ✗ {section} section missing")
                            results['status'] = 'FAIL'
                    
                    # Check model algorithms
                    algorithms = config.get('model', {}).get('algorithms', [])
                    if algorithms:
                        results['details'].append(f"  ✓ Configured algorithms: {', '.join(algorithms)}")
                    else:
                        results['errors'].append("  ✗ No algorithms configured")
                        results['status'] = 'FAIL'
                
                except Exception as e:
                    results['errors'].append(f"  ✗ Error reading config: {e}")
                    results['status'] = 'FAIL'
                
                break
        
        if not config_found:
            results['errors'].append("✗ No configuration file found")
            results['status'] = 'FAIL'
        
        # Check environment file
        if os.path.exists('.env.example'):
            results['details'].append("✓ .env.example exists")
        else:
            results['warnings'].append("⚠ .env.example not found")
        
        return results
    
    def _check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        check_name = "Documentation"
        results = {'status': 'PASS', 'details': [], 'errors': [], 'warnings': []}
        
        # Check main documentation files
        doc_files = [
            'README.md',
            'QUICK_START.md',
            'DELIVERY_REPORT.md',
            'FINAL_DELIVERY_SUMMARY.md'
        ]
        
        docs_found = 0
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                docs_found += 1
                file_size = os.path.getsize(doc_file) / 1024  # KB
                results['details'].append(f"✓ {doc_file} exists ({file_size:.1f} KB)")
            else:
                results['warnings'].append(f"⚠ {doc_file} not found")
        
        if docs_found == 0:
            results['errors'].append("✗ No documentation files found")
            results['status'] = 'FAIL'
        elif docs_found < len(doc_files) / 2:
            results['warnings'].append("⚠ Documentation appears incomplete")
        
        # Check for code documentation
        src_files = glob.glob('src/*.py')
        documented_files = 0
        
        for src_file in src_files:
            try:
                with open(src_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        documented_files += 1
            except:
                pass
        
        if src_files:
            doc_ratio = documented_files / len(src_files)
            if doc_ratio >= 0.8:
                results['details'].append(f"✓ Good code documentation ({doc_ratio:.1%})")
            else:
                results['warnings'].append(f"⚠ Code documentation could be improved ({doc_ratio:.1%})")
        
        return results
    
    def _check_environment_setup(self) -> Dict[str, Any]:
        """Check environment setup files."""
        check_name = "Environment Setup"
        results = {'status': 'PASS', 'details': [], 'errors': [], 'warnings': []}
        
        # Check conda environment file
        env_files = ['environment.yml', 'requirements.txt']
        env_found = False
        
        for env_file in env_files:
            if os.path.exists(env_file):
                env_found = True
                results['details'].append(f"✓ {env_file} exists")
                
                # Check for key dependencies
                try:
                    with open(env_file, 'r') as f:
                        content = f.read()
                    
                    key_deps = ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'matplotlib']
                    for dep in key_deps:
                        if dep in content.lower():
                            results['details'].append(f"  ✓ {dep} included")
                        else:
                            results['warnings'].append(f"  ⚠ {dep} not found")
                
                except Exception as e:
                    results['warnings'].append(f"  ⚠ Error reading {env_file}: {e}")
                
                break
        
        if not env_found:
            results['errors'].append("✗ No environment setup file found")
            results['status'] = 'FAIL'
        
        # Check Makefile
        if os.path.exists('Makefile'):
            results['details'].append("✓ Makefile exists")
            
            # Check for key targets
            try:
                with open('Makefile', 'r') as f:
                    content = f.read()
                
                key_targets = ['all', 'train', 'predict', 'map', 'help']
                for target in key_targets:
                    if target in content:
                        results['details'].append(f"  ✓ {target} target present")
                    else:
                        results['warnings'].append(f"  ⚠ {target} target not found")
            
            except Exception as e:
                results['warnings'].append(f"  ⚠ Error reading Makefile: {e}")
        else:
            results['errors'].append("✗ Makefile not found")
            results['status'] = 'FAIL'
        
        return results
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check overall performance
        performance_check = self.validation_results['checks'].get('Model Performance', {})
        if performance_check.get('status') == 'FAIL':
            recommendations.append("Improve model performance by adjusting hyperparameters or feature engineering")
        
        # Check documentation
        doc_check = self.validation_results['checks'].get('Documentation', {})
        if len(doc_check.get('warnings', [])) > 0:
            recommendations.append("Complete documentation files for better project understanding")
        
        # Check web interface
        web_check = self.validation_results['checks'].get('Web Interface', {})
        if len(web_check.get('warnings', [])) > 0:
            recommendations.append("Enhance web interface with additional interactive features")
        
        # Check data files
        data_check = self.validation_results['checks'].get('Data Files', {})
        if data_check.get('status') == 'FAIL':
            recommendations.append("Ensure all required data files are present and properly formatted")
        
        # General recommendations
        recommendations.extend([
            "Consider adding unit tests for better code reliability",
            "Implement continuous integration for automated testing",
            "Add performance monitoring and logging",
            "Consider containerization with Docker for easier deployment"
        ])
        
        return recommendations
    
    def run_validation(self) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info("Starting comprehensive delivery validation")
        
        # Run all checks
        checks = [
            ('Directory Structure', self._check_directory_structure),
            ('Data Files', self._check_data_files),
            ('Model Performance', self._check_model_performance),
            ('Web Interface', self._check_web_interface),
            ('Configuration', self._check_configuration),
            ('Documentation', self._check_documentation),
            ('Environment Setup', self._check_environment_setup)
        ]
        
        for check_name, check_func in checks:
            self.logger.info(f"Running check: {check_name}")
            try:
                results = check_func()
                self.validation_results['checks'][check_name] = results
            except Exception as e:
                self.logger.error(f"Error in {check_name}: {e}")
                self.validation_results['checks'][check_name] = {
                    'status': 'ERROR',
                    'details': [],
                    'errors': [f"Check failed with error: {e}"]
                }
        
        # Determine overall status
        statuses = [check['status'] for check in self.validation_results['checks'].values()]
        
        if 'FAIL' in statuses:
            self.validation_results['overall_status'] = 'FAIL'
        elif 'ERROR' in statuses:
            self.validation_results['overall_status'] = 'ERROR'
        elif 'WARN' in statuses:
            self.validation_results['overall_status'] = 'WARN'
        else:
            self.validation_results['overall_status'] = 'PASS'
        
        # Generate summary
        total_checks = len(self.validation_results['checks'])
        passed_checks = sum(1 for check in self.validation_results['checks'].values() 
                           if check['status'] == 'PASS')
        
        self.validation_results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'overall_status': self.validation_results['overall_status']
        }
        
        # Generate recommendations
        self.validation_results['recommendations'] = self._generate_recommendations()
        
        self.logger.info(f"Validation complete. Overall status: {self.validation_results['overall_status']}")
        return self.validation_results
    
    def save_results(self, output_path: str = None) -> str:
        """Save validation results to file."""
        if output_path is None:
            output_path = 'validation_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        self.logger.info(f"Validation results saved to {output_path}")
        return output_path
    
    def print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "="*60)
        print("SHARKS FROM SPACE - DELIVERY VALIDATION SUMMARY")
        print("="*60)
        
        # Overall status
        status = self.validation_results['overall_status']
        status_emoji = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'ERROR': '💥'}.get(status, '❓')
        print(f"\nOverall Status: {status_emoji} {status}")
        
        # Summary statistics
        summary = self.validation_results['summary']
        print(f"Checks Passed: {summary['passed_checks']}/{summary['total_checks']} ({summary['pass_rate']:.1%})")
        
        # Individual check results
        print(f"\nDetailed Results:")
        for check_name, check_results in self.validation_results['checks'].items():
            status = check_results['status']
            emoji = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'ERROR': '💥'}.get(status, '❓')
            print(f"  {emoji} {check_name}: {status}")
            
            # Show errors if any
            if check_results.get('errors'):
                for error in check_results['errors'][:3]:  # Show first 3 errors
                    print(f"    - {error}")
                if len(check_results['errors']) > 3:
                    print(f"    ... and {len(check_results['errors']) - 3} more errors")
            
            # Show warnings if any
            if check_results.get('warnings'):
                for warning in check_results['warnings'][:2]:  # Show first 2 warnings
                    print(f"    ⚠ {warning}")
                if len(check_results['warnings']) > 2:
                    print(f"    ... and {len(check_results['warnings']) - 2} more warnings")
        
        # Recommendations
        if self.validation_results['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(self.validation_results['recommendations'][:5], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n" + "="*60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Sharks from Space delivery.")
    parser.add_argument("--config", default="config/params_enhanced.yaml", 
                       help="Path to YAML configuration file")
    parser.add_argument("--output", default="validation_results.json", 
                       help="Output file for validation results")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize validator
    validator = EnhancedDeliveryValidator(config, args)
    
    # Run validation
    results = validator.run_validation()
    
    # Save results
    validator.save_results(args.output)
    
    # Print summary
    validator.print_summary()
    
    # Exit with appropriate code
    if results['overall_status'] == 'PASS':
        sys.exit(0)
    elif results['overall_status'] == 'WARN':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()
