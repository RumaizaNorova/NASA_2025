"""
AI-Powered Analysis Module for Shark Habitat Prediction

This module integrates OpenAI's GPT models to provide intelligent analysis of:
- Model performance and insights
- Feature importance interpretation
- Shark behavior pattern analysis
- Automated report generation
- Natural language explanations of predictions
- Recommendations for model improvement

Features:
- Automated model result analysis
- Natural language insights generation
- Feature importance interpretation
- Performance improvement recommendations
- Comprehensive reporting with AI insights
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import openai
except ImportError:
    openai = None

try:
    from .utils import load_config, ensure_dir, setup_logging
except ImportError:
    from utils import load_config, ensure_dir, setup_logging


class AIAnalysisEngine:
    """AI-powered analysis engine for shark habitat model results."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = setup_logging(__name__)
        
        # Initialize OpenAI client
        self.openai_client = None
        self._initialize_openai()
        
        # Analysis results storage
        self.analysis_results = {}
        self.insights = {}
        self.recommendations = {}
        
    def _initialize_openai(self):
        """Initialize OpenAI client with API key."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.warning("OPENAI_API_KEY not found in environment variables")
            return
        
        if openai is None:
            self.logger.error("OpenAI library not installed. Install with: pip install openai")
            return
        
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def analyze_model_performance(self, metrics: Dict[str, Any], 
                                feature_importance: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze model performance using AI and generate insights."""
        if not self.openai_client:
            return self._fallback_analysis(metrics, feature_importance)
        
        try:
            # Prepare context for AI analysis
            context = self._prepare_performance_context(metrics, feature_importance)
            
            # Generate AI analysis
            prompt = self._create_performance_analysis_prompt(context)
            response = self._call_openai(prompt)
            
            # Parse and structure the analysis
            analysis = self._parse_ai_response(response, "performance")
            
            self.analysis_results['performance'] = analysis
            return analysis
            
        except Exception as e:
            self.logger.error(f"AI performance analysis failed: {e}")
            return self._fallback_analysis(metrics, feature_importance)
    
    def analyze_feature_importance(self, feature_importance: Dict[str, Any], 
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance using AI insights."""
        if not self.openai_client:
            return self._fallback_feature_analysis(feature_importance, feature_names)
        
        try:
            # Prepare feature context
            context = self._prepare_feature_context(feature_importance, feature_names)
            
            # Generate AI analysis
            prompt = self._create_feature_analysis_prompt(context)
            response = self._call_openai(prompt)
            
            # Parse and structure the analysis
            analysis = self._parse_ai_response(response, "features")
            
            self.analysis_results['features'] = analysis
            return analysis
            
        except Exception as e:
            self.logger.error(f"AI feature analysis failed: {e}")
            return self._fallback_feature_analysis(feature_importance, feature_names)
    
    def analyze_shark_behavior(self, shark_data: pd.DataFrame, 
                             predictions: np.ndarray = None) -> Dict[str, Any]:
        """Analyze shark behavior patterns using AI."""
        if not self.openai_client:
            return self._fallback_behavior_analysis(shark_data)
        
        try:
            # Prepare shark behavior context
            context = self._prepare_behavior_context(shark_data, predictions)
            
            # Generate AI analysis
            prompt = self._create_behavior_analysis_prompt(context)
            response = self._call_openai(prompt)
            
            # Parse and structure the analysis
            analysis = self._parse_ai_response(response, "behavior")
            
            self.analysis_results['behavior'] = analysis
            return analysis
            
        except Exception as e:
            self.logger.error(f"AI behavior analysis failed: {e}")
            return self._fallback_behavior_analysis(shark_data)
    
    def generate_improvement_recommendations(self, current_metrics: Dict[str, Any],
                                           target_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate AI-powered recommendations for model improvement."""
        if not self.openai_client:
            return self._fallback_recommendations(current_metrics)
        
        try:
            # Prepare recommendations context
            context = self._prepare_recommendations_context(current_metrics, target_metrics)
            
            # Generate AI recommendations
            prompt = self._create_recommendations_prompt(context)
            response = self._call_openai(prompt)
            
            # Parse and structure the recommendations
            recommendations = self._parse_ai_response(response, "recommendations")
            
            self.recommendations = recommendations
            return recommendations
            
        except Exception as e:
            self.logger.error(f"AI recommendations generation failed: {e}")
            return self._fallback_recommendations(current_metrics)
    
    def generate_comprehensive_report(self, output_dir: str = "data/interim") -> str:
        """Generate a comprehensive AI-powered analysis report."""
        try:
            # Prepare comprehensive context
            context = self._prepare_comprehensive_context()
            
            # Generate comprehensive report
            prompt = self._create_comprehensive_report_prompt(context)
            response = self._call_openai(prompt)
            
            # Save report
            ensure_dir(output_dir)
            report_path = os.path.join(output_dir, "ai_analysis_report.md")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(response)
            
            self.logger.info(f"AI analysis report saved to {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Comprehensive report generation failed: {e}")
            return self._generate_fallback_report(output_dir)
    
    def _call_openai(self, prompt: str, model: str = "gpt-4") -> str:
        """Call OpenAI API with the given prompt."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert marine biologist and machine learning scientist specializing in shark habitat modeling and oceanographic data analysis. Provide detailed, scientific, and actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _prepare_performance_context(self, metrics: Dict[str, Any], 
                                   feature_importance: Dict[str, Any] = None) -> str:
        """Prepare context for performance analysis."""
        context = f"""
        SHARK HABITAT MODEL PERFORMANCE ANALYSIS
        
        Current Performance Metrics:
        """
        
        # Add model metrics
        if isinstance(metrics, dict):
            for model_name, model_metrics in metrics.items():
                context += f"\n{model_name.upper()} Model:\n"
                if isinstance(model_metrics, dict):
                    for metric, value in model_metrics.items():
                        if isinstance(value, (int, float)):
                            context += f"  - {metric}: {value:.4f}\n"
                        elif isinstance(value, dict):
                            context += f"  - {metric}:\n"
                            for sub_metric, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    context += f"    - {sub_metric}: {sub_value:.4f}\n"
        
        # Add feature importance if available
        if feature_importance:
            context += f"\nFeature Importance Summary:\n"
            for model_name, importance in feature_importance.items():
                if isinstance(importance, (list, np.ndarray)):
                    context += f"  - {model_name}: {len(importance)} features\n"
        
        context += f"""
        
        Analysis Requirements:
        1. Evaluate current performance levels
        2. Identify strengths and weaknesses
        3. Explain what these metrics mean for shark habitat prediction
        4. Suggest specific improvements
        5. Provide scientific context for the results
        """
        
        return context
    
    def _prepare_feature_context(self, feature_importance: Dict[str, Any], 
                               feature_names: List[str]) -> str:
        """Prepare context for feature importance analysis."""
        context = f"""
        SHARK HABITAT MODEL FEATURE IMPORTANCE ANALYSIS
        
        Available Features ({len(feature_names)} total):
        """
        
        # Group features by category
        oceanographic_features = [f for f in feature_names if any(keyword in f.lower() for keyword in 
                           ['sst', 'salinity', 'chl', 'precip', 'ssh', 'current'])]
        
        context += f"\nOceanographic Features ({len(oceanographic_features)}):\n"
        for feature in oceanographic_features[:10]:  # Show first 10
            context += f"  - {feature}\n"
        
        if len(oceanographic_features) > 10:
            context += f"  ... and {len(oceanographic_features) - 10} more\n"
        
        # Add feature importance data
        context += f"\nFeature Importance Results:\n"
        for model_name, importance in feature_importance.items():
            context += f"\n{model_name.upper()} Model:\n"
            if isinstance(importance, (list, np.ndarray)):
                # Get top 10 most important features
                importance_array = np.array(importance)
                top_indices = np.argsort(importance_array)[-10:][::-1]
                
                for i, idx in enumerate(top_indices):
                    if idx < len(feature_names):
                        context += f"  {i+1}. {feature_names[idx]}: {importance_array[idx]:.4f}\n"
        
        context += f"""
        
        Analysis Requirements:
        1. Interpret which oceanographic variables are most important
        2. Explain biological relevance for shark habitat preferences
        3. Identify potentially missing important features
        4. Suggest feature engineering improvements
        5. Provide scientific justification for feature importance patterns
        """
        
        return context
    
    def _prepare_behavior_context(self, shark_data: pd.DataFrame, 
                                predictions: np.ndarray = None) -> str:
        """Prepare context for shark behavior analysis."""
        context = f"""
        SHARK BEHAVIOR PATTERN ANALYSIS
        
        Dataset Overview:
        - Total observations: {len(shark_data)}
        - Species: {shark_data.get('species', 'Unknown').nunique() if 'species' in shark_data.columns else 'Unknown'}
        - Date range: {shark_data['datetime'].min()} to {shark_data['datetime'].max() if 'datetime' in shark_data.columns else 'Unknown'}
        - Geographic range: Lat {shark_data['latitude'].min():.2f} to {shark_data['latitude'].max():.2f}, 
          Lon {shark_data['longitude'].min():.2f} to {shark_data['longitude'].max():.2f}
        """
        
        # Add spatial analysis
        if 'latitude' in shark_data.columns and 'longitude' in shark_data.columns:
            context += f"\nSpatial Distribution:\n"
            context += f"- Mean latitude: {shark_data['latitude'].mean():.2f}\n"
            context += f"- Mean longitude: {shark_data['longitude'].mean():.2f}\n"
            context += f"- Latitude std: {shark_data['latitude'].std():.2f}\n"
            context += f"- Longitude std: {shark_data['longitude'].std():.2f}\n"
        
        # Add temporal analysis
        if 'datetime' in shark_data.columns:
            shark_data['datetime'] = pd.to_datetime(shark_data['datetime'])
            context += f"\nTemporal Patterns:\n"
            context += f"- Most active month: {shark_data['datetime'].dt.month.mode().iloc[0] if len(shark_data) > 0 else 'Unknown'}\n"
            context += f"- Most active hour: {shark_data['datetime'].dt.hour.mode().iloc[0] if len(shark_data) > 0 else 'Unknown'}\n"
        
        # Add movement analysis if distance data available
        if 'dist_total' in shark_data.columns:
            context += f"\nMovement Analysis:\n"
            context += f"- Mean total distance: {shark_data['dist_total'].mean():.2f} km\n"
            context += f"- Max total distance: {shark_data['dist_total'].max():.2f} km\n"
        
        context += f"""
        
        Analysis Requirements:
        1. Identify key behavioral patterns in shark movements
        2. Explain environmental factors driving habitat selection
        3. Analyze seasonal and diel patterns
        4. Suggest implications for habitat modeling
        5. Provide biological insights about shark behavior
        """
        
        return context
    
    def _prepare_recommendations_context(self, current_metrics: Dict[str, Any],
                                       target_metrics: Dict[str, Any] = None) -> str:
        """Prepare context for improvement recommendations."""
        context = f"""
        SHARK HABITAT MODEL IMPROVEMENT RECOMMENDATIONS
        
        Current Performance:
        """
        
        # Add current metrics
        for model_name, metrics in current_metrics.items():
            context += f"\n{model_name.upper()}:\n"
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        context += f"  - {metric}: {value:.4f}\n"
        
        # Add target metrics if provided
        if target_metrics:
            context += f"\nTarget Performance:\n"
            for metric, value in target_metrics.items():
                context += f"  - {metric}: {value:.4f}\n"
        
        context += f"""
        
        Current Challenges:
        - Extreme class imbalance (2 positive vs 2500 negative samples)
        - Limited temporal coverage (14 days)
        - Small dataset size for robust training
        - Basic feature engineering
        
        Recommendation Requirements:
        1. Specific hyperparameter optimization strategies
        2. Advanced sampling techniques for class imbalance
        3. Feature engineering improvements
        4. Data collection and expansion strategies
        5. Model architecture enhancements
        6. Validation and evaluation improvements
        """
        
        return context
    
    def _prepare_comprehensive_context(self) -> str:
        """Prepare context for comprehensive report generation."""
        context = f"""
        COMPREHENSIVE SHARK HABITAT MODEL ANALYSIS REPORT
        
        Generate a comprehensive scientific report covering:
        
        1. Executive Summary
        2. Model Performance Analysis
        3. Feature Importance Interpretation
        4. Shark Behavior Insights
        5. Scientific Implications
        6. Improvement Recommendations
        7. Future Research Directions
        
        Report should be:
        - Scientifically rigorous and well-referenced
        - Accessible to both technical and non-technical audiences
        - Actionable with specific recommendations
        - Comprehensive yet concise
        - Professional in tone and format
        """
        
        return context
    
    def _create_performance_analysis_prompt(self, context: str) -> str:
        """Create prompt for performance analysis."""
        return f"""
        {context}
        
        Please provide a detailed analysis of the shark habitat model performance, including:
        1. Performance evaluation and interpretation
        2. Strengths and weaknesses identification
        3. Scientific context and implications
        4. Specific improvement suggestions
        5. Comparison to expected performance levels
        
        Format your response as structured insights with clear headings and actionable recommendations.
        """
    
    def _create_feature_analysis_prompt(self, context: str) -> str:
        """Create prompt for feature importance analysis."""
        return f"""
        {context}
        
        Please provide a detailed analysis of feature importance for shark habitat prediction, including:
        1. Interpretation of most important oceanographic variables
        2. Biological relevance and scientific explanation
        3. Identification of missing potentially important features
        4. Feature engineering recommendations
        5. Domain knowledge integration
        
        Format your response with scientific explanations and practical recommendations.
        """
    
    def _create_behavior_analysis_prompt(self, context: str) -> str:
        """Create prompt for shark behavior analysis."""
        return f"""
        {context}
        
        Please provide a detailed analysis of shark behavior patterns, including:
        1. Key behavioral patterns and movement strategies
        2. Environmental drivers of habitat selection
        3. Temporal and spatial patterns
        4. Biological implications for habitat modeling
        5. Scientific insights about shark ecology
        
        Format your response with biological insights and modeling implications.
        """
    
    def _create_recommendations_prompt(self, context: str) -> str:
        """Create prompt for improvement recommendations."""
        return f"""
        {context}
        
        Please provide specific, actionable recommendations for improving the shark habitat model, including:
        1. Hyperparameter optimization strategies
        2. Advanced sampling techniques for class imbalance
        3. Feature engineering improvements
        4. Data collection and expansion strategies
        5. Model architecture enhancements
        6. Validation and evaluation improvements
        
        Format your response with prioritized, specific, and implementable recommendations.
        """
    
    def _create_comprehensive_report_prompt(self, context: str) -> str:
        """Create prompt for comprehensive report generation."""
        return f"""
        {context}
        
        Please generate a comprehensive scientific report in Markdown format with proper headings, 
        subheadings, and formatting. Include all the sections mentioned and ensure the report is:
        - Professional and well-structured
        - Scientifically accurate and referenced
        - Comprehensive yet readable
        - Actionable with clear recommendations
        
        Use proper Markdown formatting including headers, bullet points, and emphasis where appropriate.
        """
    
    def _parse_ai_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse AI response and structure it appropriately."""
        return {
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'raw_response': response,
            'insights': self._extract_insights(response),
            'recommendations': self._extract_recommendations(response)
        }
    
    def _extract_insights(self, response: str) -> List[str]:
        """Extract key insights from AI response."""
        # Simple extraction - could be enhanced with more sophisticated parsing
        insights = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ('insight' in line.lower() or 'finding' in line.lower() or 
                        line.startswith('-') or line.startswith('•')):
                insights.append(line)
        
        return insights[:10]  # Limit to top 10 insights
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from AI response."""
        # Simple extraction - could be enhanced with more sophisticated parsing
        recommendations = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ('recommend' in line.lower() or 'suggest' in line.lower() or 
                        'improve' in line.lower() or line.startswith('-') or line.startswith('•')):
                recommendations.append(line)
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _fallback_analysis(self, metrics: Dict[str, Any], 
                          feature_importance: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback analysis when AI is not available."""
        return {
            'analysis_type': 'fallback_performance',
            'timestamp': datetime.now().isoformat(),
            'insights': [
                "Model performance analysis requires AI integration",
                "Current metrics show baseline performance levels",
                "Further optimization needed for production deployment"
            ],
            'recommendations': [
                "Implement hyperparameter optimization",
                "Add advanced sampling techniques",
                "Expand feature engineering",
                "Increase training data volume"
            ],
            'raw_response': 'AI analysis not available - using fallback insights'
        }
    
    def _fallback_feature_analysis(self, feature_importance: Dict[str, Any], 
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Fallback feature analysis when AI is not available."""
        return {
            'analysis_type': 'fallback_features',
            'timestamp': datetime.now().isoformat(),
            'insights': [
                f"Analyzing {len(feature_names)} oceanographic features",
                "Feature importance shows environmental preferences",
                "Oceanographic variables drive habitat selection"
            ],
            'recommendations': [
                "Focus on top-performing features",
                "Engineer additional environmental features",
                "Consider temporal feature interactions"
            ],
            'raw_response': 'AI feature analysis not available - using fallback insights'
        }
    
    def _fallback_behavior_analysis(self, shark_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback behavior analysis when AI is not available."""
        return {
            'analysis_type': 'fallback_behavior',
            'timestamp': datetime.now().isoformat(),
            'insights': [
                f"Analyzing {len(shark_data)} shark observations",
                "Spatial patterns show habitat preferences",
                "Temporal patterns reveal behavioral cycles"
            ],
            'recommendations': [
                "Expand temporal coverage",
                "Analyze seasonal patterns",
                "Include more shark individuals"
            ],
            'raw_response': 'AI behavior analysis not available - using fallback insights'
        }
    
    def _fallback_recommendations(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback recommendations when AI is not available."""
        return {
            'analysis_type': 'fallback_recommendations',
            'timestamp': datetime.now().isoformat(),
            'insights': [
                "Current performance shows room for improvement",
                "Class imbalance is the primary challenge",
                "Data volume limitations affect model robustness"
            ],
            'recommendations': [
                "Implement SMOTE for oversampling",
                "Use cost-sensitive learning",
                "Add ensemble methods",
                "Expand training data period",
                "Optimize hyperparameters with Optuna"
            ],
            'raw_response': 'AI recommendations not available - using fallback recommendations'
        }
    
    def _generate_fallback_report(self, output_dir: str) -> str:
        """Generate fallback report when AI is not available."""
        ensure_dir(output_dir)
        report_path = os.path.join(output_dir, "fallback_analysis_report.md")
        
        report_content = f"""# Shark Habitat Model Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report provides analysis of the shark habitat prediction model. AI-powered analysis is not currently available, so this report contains basic insights and recommendations.

## Current Status
- Model training completed with multiple algorithms
- Performance metrics calculated and stored
- Feature importance analysis available
- Basic validation completed

## Recommendations
1. **Implement AI Analysis**: Set up OpenAI integration for advanced insights
2. **Optimize Hyperparameters**: Use Optuna for systematic optimization
3. **Address Class Imbalance**: Implement SMOTE and cost-sensitive learning
4. **Expand Features**: Add temporal and interaction features
5. **Increase Data**: Expand temporal coverage and add more observations

## Next Steps
1. Complete AI integration setup
2. Implement advanced sampling techniques
3. Optimize model hyperparameters
4. Expand feature engineering
5. Validate improvements with cross-validation

---
*Report generated by fallback analysis system*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Fallback analysis report saved to {report_path}")
        return report_path


def main():
    """Main function for testing AI analysis engine."""
    # Load configuration
    config = load_config("config/params.yaml")
    
    # Initialize AI analysis engine
    ai_engine = AIAnalysisEngine(config)
    
    # Test with sample data
    sample_metrics = {
        'lightgbm': {'roc_auc': 0.536, 'pr_auc': 0.438, 'tss': -0.142},
        'xgboost': {'roc_auc': 0.349, 'pr_auc': 0.405, 'tss': -0.125},
        'random_forest': {'roc_auc': 0.413, 'pr_auc': 0.420, 'tss': -0.083}
    }
    
    # Analyze performance
    performance_analysis = ai_engine.analyze_model_performance(sample_metrics)
    print("Performance Analysis:", performance_analysis)
    
    # Generate recommendations
    recommendations = ai_engine.generate_improvement_recommendations(sample_metrics)
    print("Recommendations:", recommendations)
    
    # Generate comprehensive report
    report_path = ai_engine.generate_comprehensive_report()
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
