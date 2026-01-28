#!/usr/bin/env python3
"""
Multicollinearity Diagnosis Tool

This script performs comprehensive multicollinearity analysis for regression models,
including continuous, categorical, and mixed variable types.

Output directory: results/multicollinearity_analysis/
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, Dict

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class TeeOutput:
    """Redirect print output to both console and file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()
        sys.stdout = self.terminal


class MulticollinearityDiagnoser:
    """Comprehensive multicollinearity diagnosis tool"""
    
    def __init__(self, data_path: str, classification_path: str, include_ras: bool = False):
        """
        Initialize the diagnoser
        
        Args:
            data_path: Path to unified_dataset.csv or unified_dataset_with_ras.csv
            classification_path: Path to refactoring_classification.xlsx
            include_ras: Whether to include RAS variables in the analysis
        """
        self.data_path = Path(data_path)
        self.classification_path = Path(classification_path)
        self.include_ras = include_ras
        self.df = None
        self.df_encoded = None
        self.classification_df = None
        self.action_mapping = {}
        self.scope_mapping = {}
        self.test_interface_mapping = {}
        
    def load_data(self):
        """Load and preprocess data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        self.classification_df = pd.read_excel(self.classification_path)
        print(f"Loaded {len(self.df)} records")
        
        # Check if RAS columns exist when include_ras is True
        if self.include_ras:
            ras_cols = ['max_ras', 'mean_ras', 'ras_count']
            missing_cols = [col for col in ras_cols if col not in self.df.columns]
            if missing_cols:
                print(f"Warning: RAS columns not found: {missing_cols}")
                print("Setting include_ras to False")
                self.include_ras = False
            else:
                print(f"RAS columns found: {ras_cols}")
        
    def create_mappings(self):
        """Create mapping dictionaries from refactoring types to three dimensions"""
        # Action dimension mapping
        for action in ['Add', 'Remove', 'Adjust']:
            refactoring_types = set(
                self.classification_df[
                    self.classification_df['Action'] == action
                ]['Refactoring'].tolist()
            )
            self.action_mapping[action] = refactoring_types
        
        # Scope dimension mapping
        for scope in ['Method', 'Class', 'Local Variable']:
            refactoring_types = set(
                self.classification_df[
                    self.classification_df['Scope'] == scope
                ]['Refactoring'].tolist()
            )
            self.scope_mapping[scope] = refactoring_types
        
        # Test Interface dimension mapping
        for affect in ['Yes', 'No']:
            refactoring_types = set(
                self.classification_df[
                    self.classification_df['Can Affect Test Interface？'] == affect
                ]['Refactoring'].tolist()
            )
            self.test_interface_mapping[affect] = refactoring_types
    
    def parse_refactoring_types(self, refactoring_str: str) -> set:
        """Parse refactoring types JSON string"""
        try:
            if pd.isna(refactoring_str) or refactoring_str == '{}':
                return set()
            refactoring_dict = json.loads(refactoring_str)
            return set(refactoring_dict.keys())
        except:
            return set()
    
    def generate_has_variables(self):
        """Generate has_* variables"""
        print("Generating treatment variables...")
        # Parse refactoring types for each record
        self.df['refactoring_types'] = self.df['agent_refactoring_type_count'].apply(
            self.parse_refactoring_types
        )
        
        # Generate has_* variables for Action dimension
        for action in ['Add', 'Remove', 'Adjust']:
            col_name = f'has_{action.lower()}'
            self.df[col_name] = self.df['refactoring_types'].apply(
                lambda types: int(len(types & self.action_mapping[action]) > 0)
            )
        
        # Generate has_* variables for Scope dimension
        for scope in ['Method', 'Class', 'Local Variable']:
            col_name = f'has_{scope.lower().replace(" ", "_")}'
            self.df[col_name] = self.df['refactoring_types'].apply(
                lambda types: int(len(types & self.scope_mapping[scope]) > 0)
            )
        
        # Generate has_* variables for Test Interface dimension
        self.df['has_affects_test_interface'] = self.df['refactoring_types'].apply(
            lambda types: int(len(types & self.test_interface_mapping['Yes']) > 0)
        )
    
    def preprocess_features(self):
        """Handle confounding variables and feature engineering"""
        print("Preprocessing features...")
        # Process dependent variables
        self.df['is_issue_solved_binary'] = (self.df['is_issue_solved'] == 'resolved').astype(int)
        
        # Process compilation outcome variable (unified column only)
        if 'is_compile_ok' in self.df.columns:
            self.df['is_compile_ok_binary'] = self.df['is_compile_ok'].astype(int)
            print(f"  is_compile_ok: {self.df['is_compile_ok_binary'].sum()} success / {len(self.df)} total")

        # Apply log transformation to skewed variables (matching modeling script)
        if 'patch_size' in self.df.columns:
            self.df['patch_size_log'] = np.log1p(self.df['patch_size'])
        
        if 'issue_length' in self.df.columns:
            self.df['issue_length_log'] = np.log1p(self.df['issue_length'])
        
        if 'golden_patch_length' in self.df.columns:
            self.df['golden_patch_length_log'] = np.log1p(self.df['golden_patch_length'])
        
        if 'golden_patch_modified_files' in self.df.columns:
            self.df['golden_patch_modified_files_log'] = np.log1p(self.df['golden_patch_modified_files'])
        
        # Process RAS variables if included
        if self.include_ras:
            print("Processing RAS variables...")
            # Binary variable: has high RAS (max_ras >= 0.8)
            self.df['has_high_ras'] = (self.df['max_ras'] >= 0.8).astype(int)
            
            print(f"  - has_high_ras: {self.df['has_high_ras'].sum()} instances with high RAS")

        # One-hot encode categorical variables
        categorical_vars = ['task_difficulty', 'llm_model', 'agent_framework', 'issue_type']
        self.df_encoded = pd.get_dummies(
            self.df,
            columns=categorical_vars,
            prefix=categorical_vars,
            drop_first=True
        )

        # Convert boolean columns to integers
        bool_columns = self.df_encoded.select_dtypes(include=[bool]).columns
        self.df_encoded[bool_columns] = self.df_encoded[bool_columns].astype(int)

        # Standardize numerical variables (use log-transformed versions)
        numerical_vars = ['patch_size_log', 'file_coverage', 'line_coverage',
                         'issue_length_log', 'golden_patch_length_log',
                         'golden_patch_modified_files_log']
        
        # Note: has_high_ras is binary, not continuous, so no scaling needed
        
        scaler = StandardScaler()
        for var in numerical_vars:
            if var in self.df_encoded.columns:
                self.df_encoded[f'{var}_scaled'] = scaler.fit_transform(
                    self.df_encoded[[var]]
                ).flatten()
    
    def get_control_variables(self):
        """Get list of control variables"""
        control_vars = []
        
        # Add standardized numerical variables
        available_scaled_vars = [col for col in self.df_encoded.columns if col.endswith('_scaled')]
        control_vars.extend(available_scaled_vars)
        
        # Add One-hot encoded categorical variables
        categorical_prefixes = ['task_difficulty_', 'llm_model_', 'agent_framework_', 'issue_type_']
        for prefix in categorical_prefixes:
            control_vars.extend([col for col in self.df_encoded.columns if col.startswith(prefix)])
        
        # Add has_high_ras binary variable if RAS analysis is included
        if self.include_ras and 'has_high_ras' in self.df_encoded.columns:
            control_vars.append('has_high_ras')
        
        return control_vars
    
    def correlation_ratio(self, categories: pd.Series, values: pd.Series) -> float:
        """
        Calculate correlation ratio (eta) for categorical vs continuous variables
        
        Args:
            categories: Categorical variable
            values: Continuous variable
            
        Returns:
            Eta (correlation ratio)
        """
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        
        for i in range(cat_num):
            cat_measures = values[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures) if len(cat_measures) > 0 else 0
        
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
        denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
        
        if numerator == 0 or denominator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator / denominator)
        
        return eta
    
    def cramers_v(self, var1: pd.Series, var2: pd.Series) -> float:
        """
        Calculate Cramér's V for categorical vs categorical variables
        
        Args:
            var1: First categorical variable
            var2: Second categorical variable
            
        Returns:
            Cramér's V coefficient
        """
        confusion_matrix = pd.crosstab(var1, var2)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1
        
        if min_dim == 0:
            return 0.0
        
        return np.sqrt(chi2 / (n * min_dim))
    
    def analyze_variable_distributions(self, output_dir: str) -> Dict:
        """
        Analyze distributions of numerical variables and recommend transformations
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with distribution analysis and transformation recommendations
        """
        print("\n" + "="*70)
        print("DISTRIBUTION ANALYSIS: Skewness & Transformation Recommendations")
        print("="*70)
        
        numerical_vars = ['patch_size', 'file_coverage', 'line_coverage', 
                         'issue_length', 'golden_patch_length', 
                         'golden_patch_modified_files']
        
        # Note: has_high_ras is binary, not continuous, so it doesn't need distribution analysis
        
        distribution_results = []
        transformation_recommendations = {
            'log_transform': [],
            'keep_original': [],
            'already_transformed': []
        }
        
        for var in numerical_vars:
            if var not in self.df.columns:
                continue
                
            data = self.df[var].dropna()
            
            # Calculate statistics
            skewness = data.skew()
            min_val = data.min()
            max_val = data.max()
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            # Determine if log transformation is needed
            # Criteria: skewness > 2 indicates severe right-skewness
            needs_log = abs(skewness) > 2.0
            
            # Special handling for proportion/ratio variables [0, 1]
            is_proportion = (min_val >= 0) and (max_val <= 1)
            
            if is_proportion:
                recommendation = "Keep original (proportion/ratio variable)"
                transformation_recommendations['keep_original'].append(var)
            elif needs_log:
                recommendation = "Apply log transformation (log1p)"
                transformation_recommendations['log_transform'].append(var)
            else:
                recommendation = "Keep original (low skewness)"
                transformation_recommendations['keep_original'].append(var)
            
            distribution_results.append({
                'Variable': var,
                'Min': min_val,
                'Max': max_val,
                'Mean': mean_val,
                'Median': median_val,
                'Std': std_val,
                'Skewness': skewness,
                'Recommendation': recommendation
            })
        
        # Save distribution analysis
        dist_df = pd.DataFrame(distribution_results)
        dist_df.to_csv(os.path.join(output_dir, 'distribution_analysis.csv'), index=False)
        
        print("\nDistribution Analysis Results:")
        print(dist_df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print("TRANSFORMATION RECOMMENDATIONS:")
        print(f"{'='*70}")
        
        if transformation_recommendations['log_transform']:
            print("\n✓ Variables requiring LOG TRANSFORMATION (skewness > 2):")
            for var in transformation_recommendations['log_transform']:
                skew = dist_df[dist_df['Variable'] == var]['Skewness'].values[0]
                print(f"  - {var:30s} (skewness={skew:6.2f})")
        
        if transformation_recommendations['keep_original']:
            print("\n✓ Variables to KEEP ORIGINAL:")
            for var in transformation_recommendations['keep_original']:
                skew = dist_df[dist_df['Variable'] == var]['Skewness'].values[0]
                reason = "proportion" if dist_df[dist_df['Variable'] == var]['Max'].values[0] <= 1 else "low skewness"
                print(f"  - {var:30s} (skewness={skew:6.2f}, {reason})")
        
        # Generate code snippet
        code_snippet = self._generate_transformation_code(transformation_recommendations)
        
        with open(os.path.join(output_dir, 'transformation_code.py'), 'w') as f:
            f.write(code_snippet)
        
        print(f"\nTransformation code saved to: {output_dir}/transformation_code.py")
        print(f"Distribution analysis saved to: {output_dir}/distribution_analysis.csv")
        
        return {
            'distribution_df': dist_df,
            'recommendations': transformation_recommendations,
            'code_snippet': code_snippet
        }
    
    def _generate_transformation_code(self, recommendations: Dict) -> str:
        """Generate Python code snippet for variable transformations"""
        lines = []
        lines.append("# Variable Transformation Code")
        lines.append("# Copy this to your modeling script (refactoring_logistic_regression_analysis.py)")
        lines.append("")
        lines.append("import numpy as np")
        lines.append("")
        lines.append("# Apply log transformation to skewed variables")
        
        if recommendations['log_transform']:
            for var in sorted(recommendations['log_transform']):
                lines.append(f"if '{var}' in self.df.columns:")
                lines.append(f"    self.df['{var}_log'] = np.log1p(self.df['{var}'])")
                lines.append(f"    print(f\"Applied log transformation to {var}\")")
                lines.append("")
        
        lines.append("# Define numerical variables list for modeling")
        lines.append("numerical_vars = [")
        
        # Add transformed variables
        if recommendations['log_transform']:
            for var in sorted(recommendations['log_transform']):
                lines.append(f"    '{var}_log',  # Log-transformed")
        
        # Add original variables
        if recommendations['keep_original']:
            for var in sorted(recommendations['keep_original']):
                lines.append(f"    '{var}',  # Keep original")
        
        lines.append("]")
        lines.append("")
        lines.append("# Note: Variables already ending with '_log' are considered transformed")
        
        return "\n".join(lines)
    
    def analyze_continuous_correlation(self, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Phase 1: Analyze correlation among continuous variables
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Tuple of (correlation_matrix, high_correlation_pairs)
        """
        print("\n" + "="*70)
        print("PHASE 1: Continuous Variables Correlation Analysis")
        print("="*70)
        
        numerical_vars = ['patch_size_log', 'file_coverage', 'line_coverage',
                         'issue_length_log', 'golden_patch_length_log',
                         'golden_patch_modified_files_log']
        
        # Note: has_high_ras is binary, not continuous, so it's not included in correlation analysis

        # Calculate correlation matrix
        corr_matrix = self.df[numerical_vars].corr()
        
        # Save correlation matrix
        corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
        print(f"\nCorrelation Matrix saved to: {output_dir}/correlation_matrix.csv")
        
        # Generate heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix - Continuous Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
        plt.close()
        print(f"Correlation Heatmap saved to: {output_dir}/correlation_heatmap.png")
        
        # Identify high correlation pairs
        high_corr_pairs = []
        for i in range(len(numerical_vars)):
            for j in range(i+1, len(numerical_vars)):
                corr_val = corr_matrix.loc[numerical_vars[i], numerical_vars[j]]
                abs_corr = abs(corr_val)
                if abs_corr > 0.7:
                    high_corr_pairs.append({
                        'Variable_1': numerical_vars[i],
                        'Variable_2': numerical_vars[j],
                        'Correlation': corr_val,
                        'Abs_Correlation': abs_corr,
                        'Level': 'Very High (>0.8)' if abs_corr > 0.8 else 'High (0.7-0.8)'
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Abs_Correlation', ascending=False)
        else:
            high_corr_df = pd.DataFrame(columns=['Variable_1', 'Variable_2', 'Correlation', 'Abs_Correlation', 'Level'])
        
        high_corr_df.to_csv(os.path.join(output_dir, 'high_correlation_pairs.csv'), index=False)
        
        print(f"\n{len(high_corr_pairs)} high correlation pairs (|r| > 0.7) found:")
        if not high_corr_df.empty:
            print(high_corr_df.to_string(index=False))
        else:
            print("None")
        
        return corr_matrix, high_corr_df
    
    def analyze_continuous_categorical_association(self, output_dir: str) -> pd.DataFrame:
        """
        Phase 2: Analyze association between continuous and categorical variables
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            DataFrame with association results
        """
        print("\n" + "="*70)
        print("PHASE 2: Continuous vs Categorical Association Analysis")
        print("="*70)
        
        numerical_vars = ['patch_size_log', 'file_coverage', 'line_coverage',
                         'issue_length_log', 'golden_patch_length_log',
                         'golden_patch_modified_files_log']
        
        categorical_vars = ['task_difficulty', 'llm_model', 'agent_framework', 'issue_type']
        
        # Add has_high_ras if RAS analysis is included
        if self.include_ras:
            categorical_vars.append('has_high_ras')
        
        results = []
        
        for cat_var in categorical_vars:
            for num_var in numerical_vars:
                # Calculate correlation ratio
                eta = self.correlation_ratio(self.df[cat_var], self.df[num_var])
                eta_squared = eta ** 2
                
                # ANOVA F-test
                groups = [group[num_var].values for name, group in self.df.groupby(cat_var)]
                f_stat, p_value = stats.f_oneway(*groups)
                
                results.append({
                    'Categorical_Variable': cat_var,
                    'Continuous_Variable': num_var,
                    'Eta': eta,
                    'Eta_Squared': eta_squared,
                    'F_Statistic': f_stat,
                    'P_Value': p_value,
                    'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                })
        
        results_df = pd.DataFrame(results).sort_values('Eta_Squared', ascending=False)
        results_df.to_csv(os.path.join(output_dir, 'continuous_categorical_association.csv'), index=False)
        
        print(f"\nTop 10 associations (by Eta-squared):")
        print(results_df.head(10).to_string(index=False))
        print(f"\nNote: Eta-squared > 0.25 indicates strong association")
        print(f"Full results saved to: {output_dir}/continuous_categorical_association.csv")
        
        return results_df
    
    def analyze_categorical_association(self, output_dir: str) -> pd.DataFrame:
        """
        Phase 3: Analyze association between categorical variables
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            DataFrame with association results
        """
        print("\n" + "="*70)
        print("PHASE 3: Categorical vs Categorical Association Analysis")
        print("="*70)
        
        categorical_vars = ['task_difficulty', 'llm_model', 'agent_framework', 'issue_type']
        
        # Add has_high_ras if RAS analysis is included
        if self.include_ras:
            categorical_vars.append('has_high_ras')
        
        results = []
        
        for i in range(len(categorical_vars)):
            for j in range(i+1, len(categorical_vars)):
                var1, var2 = categorical_vars[i], categorical_vars[j]
                
                # Calculate Cramér's V
                cramers = self.cramers_v(self.df[var1], self.df[var2])
                
                # Chi-square test
                confusion_matrix = pd.crosstab(self.df[var1], self.df[var2])
                chi2, p_value, dof, expected = chi2_contingency(confusion_matrix)
                
                results.append({
                    'Variable_1': var1,
                    'Variable_2': var2,
                    'Cramers_V': cramers,
                    'Chi_Square': chi2,
                    'P_Value': p_value,
                    'Degrees_of_Freedom': dof,
                    'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                })
        
        results_df = pd.DataFrame(results).sort_values('Cramers_V', ascending=False)
        results_df.to_csv(os.path.join(output_dir, 'categorical_association.csv'), index=False)
        
        print(f"\nCategorical associations:")
        print(results_df.to_string(index=False))
        print(f"\nNote: Cramér's V > 0.3 indicates strong association")
        print(f"Full results saved to: {output_dir}/categorical_association.csv")
        
        return results_df
    
    def calculate_vif_analysis(self, output_dir: str) -> pd.DataFrame:
        """
        Calculate VIF for all control variables after encoding
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            DataFrame with VIF results
        """
        print("\n" + "="*70)
        print("VIF ANALYSIS: Comprehensive Multicollinearity Check")
        print("="*70)
        
        control_vars = self.get_control_variables()
        X = self.df_encoded[control_vars].copy()
        X = X.fillna(0)
        
        print(f"\nCalculating VIF for {len(control_vars)} variables...")
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        # Categorize VIF levels
        vif_data['Multicollinearity_Level'] = vif_data['VIF'].apply(
            lambda x: 'Severe (≥10)' if x >= 10 else 'Moderate (5-10)' if x >= 5 else 'Low (<5)'
        )
        
        # Save results
        vif_data.to_csv(os.path.join(output_dir, 'vif_results.csv'), index=False)
        
        # Generate bar plot for top 20 variables
        plt.figure(figsize=(12, 8))
        top_20 = vif_data.head(20)
        colors = ['red' if x >= 10 else 'orange' if x >= 5 else 'green' for x in top_20['VIF']]
        plt.barh(range(len(top_20)), top_20['VIF'], color=colors)
        plt.yticks(range(len(top_20)), top_20['Variable'])
        plt.xlabel('VIF Value', fontsize=12)
        plt.title('Top 20 Variables by VIF', fontsize=14, fontweight='bold')
        plt.axvline(x=5, color='orange', linestyle='--', label='VIF = 5 (Moderate)')
        plt.axvline(x=10, color='red', linestyle='--', label='VIF = 10 (Severe)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vif_barplot.png'), dpi=300)
        plt.close()
        
        print(f"\nTop 20 variables by VIF:")
        print(top_20.to_string(index=False))
        
        severe = len(vif_data[vif_data['VIF'] >= 10])
        moderate = len(vif_data[(vif_data['VIF'] >= 5) & (vif_data['VIF'] < 10)])
        low = len(vif_data[vif_data['VIF'] < 5])
        
        print(f"\nVIF Summary:")
        print(f"  - Severe multicollinearity (VIF ≥ 10): {severe} variables")
        print(f"  - Moderate multicollinearity (5 ≤ VIF < 10): {moderate} variables")
        print(f"  - Low multicollinearity (VIF < 5): {low} variables")
        print(f"\nFull results saved to: {output_dir}/vif_results.csv")
        print(f"VIF bar plot saved to: {output_dir}/vif_barplot.png")
        
        return vif_data
    
    def generate_multicollinearity_recommendations(self, 
                                                  corr_data: pd.DataFrame,
                                                  vif_data: pd.DataFrame,
                                                  eta_data: pd.DataFrame,
                                                  cramers_data: pd.DataFrame,
                                                  output_dir: str) -> Dict:
        """
        Generate recommendations for variable removal based on all analyses
        
        Args:
            corr_data: Correlation matrix
            vif_data: VIF analysis results
            eta_data: Continuous-categorical association results
            cramers_data: Categorical-categorical association results
            output_dir: Directory to save results
            
        Returns:
            Dictionary with recommendations
        """
        print("\n" + "="*70)
        print("GENERATING RECOMMENDATIONS")
        print("="*70)
        
        recommendations = {
            'high_vif_variables': [],
            'highly_correlated_pairs': [],
            'strong_associations': [],
            'removal_suggestions': []
        }
        
        # Identify high VIF variables
        high_vif = vif_data[vif_data['VIF'] >= 10].copy()
        recommendations['high_vif_variables'] = high_vif.to_dict('records')
        
        # Check correlation for high VIF continuous variables
        numerical_vars = ['patch_size_log', 'file_coverage', 'line_coverage',
                         'issue_length_log', 'golden_patch_length_log',
                         'golden_patch_modified_files_log']
        
        # Note: has_high_ras is binary, not continuous, so it's not included in correlation checks
        
        for _, row in high_vif.iterrows():
            var_name = row['Variable']
            
            # Check if it's a scaled numerical variable
            if var_name.endswith('_scaled'):
                original_var = var_name.replace('_scaled', '')
                
                if original_var in numerical_vars:
                    # Find correlated variables
                    corr_values = corr_data[original_var].abs().sort_values(ascending=False)
                    correlated_vars = corr_values[corr_values > 0.7].index.tolist()
                    if original_var in correlated_vars:
                        correlated_vars.remove(original_var)  # Remove self
                    
                    if correlated_vars:
                        recommendations['removal_suggestions'].append({
                            'variable': original_var,
                            'vif': row['VIF'],
                            'reason': f'High VIF and correlated with: {", ".join(correlated_vars)}',
                            'correlated_variables': correlated_vars
                        })
        
        # Strong categorical associations
        strong_cramers = cramers_data[cramers_data['Cramers_V'] > 0.3]
        if not strong_cramers.empty:
            recommendations['strong_associations'] = strong_cramers.to_dict('records')
        
        # Save recommendations
        with open(os.path.join(output_dir, 'diagnosis_summary.json'), 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Generate text report
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("MULTICOLLINEARITY DIAGNOSIS REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        
        report_lines.append(f"Total variables analyzed: {len(vif_data)}")
        report_lines.append(f"Variables with severe multicollinearity (VIF ≥ 10): {len(high_vif)}")
        report_lines.append("")
        
        if recommendations['removal_suggestions']:
            report_lines.append("VARIABLES TO CONSIDER REMOVING:")
            report_lines.append("-" * 70)
            for suggestion in recommendations['removal_suggestions']:
                report_lines.append(f"\n  Variable: {suggestion['variable']}")
                report_lines.append(f"  VIF: {suggestion['vif']:.2f}")
                report_lines.append(f"  Reason: {suggestion['reason']}")
        else:
            report_lines.append("No strong recommendations for variable removal.")
        
        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("DECISION CRITERIA:")
        report_lines.append("When choosing which variable to remove from a correlated pair:")
        report_lines.append("  1. Keep the variable with stronger theoretical justification")
        report_lines.append("  2. Keep the variable with fewer missing values")
        report_lines.append("  3. Keep the variable with stronger correlation to outcome (is_issue_solved)")
        report_lines.append("  4. Consider business interpretability")
        report_lines.append("="*70)
        
        report_text = "\n".join(report_lines)
        
        with open(os.path.join(output_dir, 'removal_recommendations.txt'), 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nDiagnosis summary saved to: {output_dir}/diagnosis_summary.json")
        print(f"Recommendations saved to: {output_dir}/removal_recommendations.txt")
        
        return recommendations
    
    def run_diagnosis(self, output_dir='results/multicollinearity_analysis'):
        """
        Run complete multicollinearity diagnosis workflow
        
        Args:
            output_dir: Directory to save all results and visualizations
        """
        # Create output directory first
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup output redirection to both console and file
        log_file = os.path.join(output_dir, 'analysis_output.txt')
        tee = TeeOutput(log_file)
        sys.stdout = tee
        
        print("\n" + "="*80)
        print(" "*20 + "MULTICOLLINEARITY DIAGNOSIS TOOL")
        print("="*80)
        print(f"\nOutput directory: {output_dir}")
        print(f"Log file: {log_file}")
        print(f"Created/verified output directory: {output_dir}")
        
        try:
            # Load and prepare data
            self.load_data()
            self.create_mappings()
            self.generate_has_variables()
            self.preprocess_features()
            
            # Distribution Analysis (must run before correlation analysis)
            distribution_analysis = self.analyze_variable_distributions(output_dir)
            
            # Phase 1: Continuous correlation
            corr_matrix, high_corr_pairs = self.analyze_continuous_correlation(output_dir)
            
            # Phase 2: Continuous-Categorical association
            eta_results = self.analyze_continuous_categorical_association(output_dir)
            
            # Phase 3: Categorical-Categorical association
            cramers_results = self.analyze_categorical_association(output_dir)
            
            # VIF Analysis
            vif_results = self.calculate_vif_analysis(output_dir)
            
            # Generate recommendations
            recommendations = self.generate_multicollinearity_recommendations(
                corr_matrix, vif_results, eta_results, cramers_results, output_dir
            )
            
            print("\n" + "="*80)
            print(" "*25 + "DIAGNOSIS COMPLETE")
            print("="*80)
            print(f"\nAll results saved to: {output_dir}/")
            print("\nGenerated files:")
            print(f"  - analysis_output.txt                # Terminal output log")
            print(f"  - distribution_analysis.csv          # Variable distributions & skewness")
            print(f"  - transformation_code.py             # Code to copy to modeling script")
            print(f"  - correlation_matrix.csv             # Continuous variable correlations")
            print(f"  - correlation_heatmap.png            # Correlation heatmap visualization")
            print(f"  - high_correlation_pairs.csv         # High correlation pairs (|r| > 0.7)")
            print(f"  - continuous_categorical_association.csv  # Continuous-Categorical associations")
            print(f"  - categorical_association.csv        # Categorical-Categorical associations")
            print(f"  - vif_results.csv                    # VIF analysis results")
            print(f"  - vif_barplot.png                    # VIF bar plot (top 20)")
            print(f"  - diagnosis_summary.json             # Comprehensive diagnosis summary")
            print(f"  - removal_recommendations.txt        # Variable removal recommendations")
            print("\n" + "="*80)
            print("NEXT STEPS:")
            print("="*80)
            print("1. Review distribution_analysis.csv for transformation recommendations")
            print("2. Copy code from transformation_code.py to your modeling script")
            print("3. Check VIF results for multicollinearity issues")
            print("4. Run your regression analysis with properly transformed variables")
            
        except Exception as e:
            print(f"\nError during diagnosis: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore stdout and close log file
            tee.close()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multicollinearity Diagnosis Tool')
    parser.add_argument('--with-ras', action='store_true', 
                       help='Include RAS variables in the analysis')
    args = parser.parse_args()
    
    # Set file paths
    if args.with_ras:
        data_path = "data/unified_data_with_ras.csv"
        output_dir = "output/RQ2/multicollinearity_analysis_with_ras"
        print("Running diagnosis WITH RAS variables")
    else:
        data_path = "data/unified_data.csv"
        output_dir = "output/RQ2/multicollinearity_analysis"
        print("Running diagnosis WITHOUT RAS variables")
    
    classification_path = "data/refactoring_classification.xlsx"
    
    # Create diagnoser and run
    diagnoser = MulticollinearityDiagnoser(
        data_path, 
        classification_path, 
        include_ras=args.with_ras
    )
    diagnoser.run_diagnosis(output_dir)


if __name__ == "__main__":
    main()

