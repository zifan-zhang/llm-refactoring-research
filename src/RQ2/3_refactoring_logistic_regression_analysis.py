#!/usr/bin/env python3
"""
Logistic Regression Analysis: Impact of Refactoring on Task Resolution Rate
Study the impact of two-dimensional refactoring classification (Action + Scope) on is_issue_solved and is_compile_ok
"""

import argparse
import pandas as pd
import numpy as np
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
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

class RefactoringLogisticAnalysis:
    """Main class for logistic regression analysis"""
    
    def __init__(self, data_path: str, classification_path: str):
        """
        Initialize the analyzer
        
        Args:
            data_path: Path to unified_dataset.csv
            classification_path: Path to refactoring_classification.xlsx
        """
        self.data_path = Path(data_path)
        self.classification_path = Path(classification_path)
        self.df = None
        self.classification_df = None
        self.action_mapping = {}
        self.scope_mapping = {}
        self.test_interface_mapping = {}
        
    def load_data(self):
        """Load and preprocess data"""
        self.df = pd.read_csv(self.data_path)
        self.classification_df = pd.read_excel(self.classification_path)
        print(f"  Loaded dataset with {len(self.df)} records")
        print(f"  Loaded refactoring classification with {len(self.classification_df)} types")
        
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
        
        # Note: Test Interface dimension is not used in this analysis
    
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
            print(f"  {col_name}: {self.df[col_name].sum()} instances")
        
        # Generate has_* variables for Scope dimension
        for scope in ['Method', 'Class', 'Local Variable']:
            col_name = f'has_{scope.lower().replace(" ", "_")}'
            self.df[col_name] = self.df['refactoring_types'].apply(
                lambda types: int(len(types & self.scope_mapping[scope]) > 0)
            )
            print(f"  {col_name}: {self.df[col_name].sum()} instances")
        
        # Note: Test Interface dimension is not used in this analysis
        # Focus on Action and Scope dimensions only
    
    def preprocess_features(self):
        """Handle confounding variables and feature engineering"""
        # Process dependent variables (already 0/1 in CSV)
        self.df['is_issue_solved_binary'] = self.df['is_issue_solved'].astype(int)
        
        # Process compilation outcome variable if it exists (unified compile result)
        self.has_compile_data = False
        if 'is_compile_ok' in self.df.columns:
            self.df['is_compile_ok_binary'] = self.df['is_compile_ok'].astype(int)
            self.has_compile_data = True
            
            print(f"  Outcome variable distributions:")
            print(f"    - is_issue_solved: {self.df['is_issue_solved_binary'].sum()} resolved / {len(self.df)} total")
            print(f"    - is_compile_ok: {self.df['is_compile_ok_binary'].sum()} success / {len(self.df)} total")
        else:
            print(f"  Note: is_compile_ok column not found in data.")
            print(f"  Only analyzing is_issue_solved outcome variable.")
            print(f"  Outcome variable distribution:")
            print(f"    - is_issue_solved: {self.df['is_issue_solved_binary'].sum()} resolved / {len(self.df)} total")

        # Apply log transformation to skewed variables
        if 'patch_size' in self.df.columns:
            self.df['patch_size_log'] = np.log1p(self.df['patch_size'])
            print(f"  Applied log transformation to patch_size")

        if 'issue_length' in self.df.columns:
            self.df['issue_length_log'] = np.log1p(self.df['issue_length'])
            print(f"  Applied log transformation to issue_length")

        if 'golden_patch_length' in self.df.columns:
            self.df['golden_patch_length_log'] = np.log1p(self.df['golden_patch_length'])
            print(f"  Applied log transformation to golden_patch_length")

        # Define numerical variables
        numerical_vars = [
            'patch_size_log',
            'file_coverage',
            'line_coverage',
            'issue_length_log',
            'golden_patch_length_log'
        ]

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

        # Standardize numerical variables
        scaler = StandardScaler()
        for var in numerical_vars:
            if var in self.df_encoded.columns:
                self.df_encoded[f'{var}_scaled'] = scaler.fit_transform(
                    self.df_encoded[[var]]
                ).flatten()
        
        print(f"Preprocessing complete. Total features: {len(self.df_encoded.columns)}")
    
    
    
    def get_control_variables(self) -> List[str]:
        """Get list of control variables"""
        control_vars = []
        
        # Add standardized numerical variables
        available_scaled_vars = [col for col in self.df_encoded.columns if col.endswith('_scaled')]
        control_vars.extend(available_scaled_vars)
        
        # Add One-hot encoded categorical variables
        categorical_prefixes = ['task_difficulty_', 'llm_model_', 'agent_framework_', 'issue_type_']
        for prefix in categorical_prefixes:
            control_vars.extend([col for col in self.df_encoded.columns if col.startswith(prefix)])
        
        return control_vars
    
    def fit_logistic_model(self, treatment_vars: List[str], model_name: str, outcome_var: str = 'is_issue_solved_binary') -> Dict[str, Any]:
        """Fit a single logistic regression model
        
        Args:
            treatment_vars: List of treatment variable names
            model_name: Name of the model
            outcome_var: Name of the outcome variable (default: 'is_issue_solved_binary')
        """
        
        # Prepare features
        control_vars = self.get_control_variables()
        all_vars = treatment_vars + control_vars
        
        # Check if variables exist
        missing_vars = [var for var in all_vars if var not in self.df_encoded.columns]
        if missing_vars:
            print(f"Warning: The following variables do not exist: {missing_vars}")
            all_vars = [var for var in all_vars if var in self.df_encoded.columns]
        
        # Prepare data
        X = self.df_encoded[all_vars].copy()
        y = self.df_encoded[outcome_var]
        
        # Store X before filling NA for variable-specific N calculation
        X_before_fillna = X.copy()
        
        # Check and handle missing values
        X = X.fillna(0)
        
        # Add constant term
        X = sm.add_constant(X)
        
        # Fit model
        try:
            model = sm.Logit(y, X).fit(disp=0)
            
            # Calculate McFadden Pseudo R²
            mcfadden_r2 = 1 - (model.llf / model.llnull)
            adj_mcfadden_r2 = 1 - ((model.llf - model.df_model) / (model.llnull - 1))
            
            
            # Extract results
            results = {
                'model': model,
                'model_name': model_name,
                'treatment_vars': treatment_vars,
                'control_vars': control_vars,
                'n_obs': int(model.nobs),
                'X_data': X_before_fillna,  # Store X data for variable-specific N calculation
                'mcfadden_r2': mcfadden_r2,
                'adj_mcfadden_r2': adj_mcfadden_r2,
                'aic': model.aic,
                'bic': model.bic,
                'llf': model.llf
            }
            
            return results
            
        except Exception as e:
            print(f"Model {model_name} fitting failed: {e}")
            return None
    
    def build_models(self):
        """Build two-dimensional (Action + Scope) logistic regression models"""
        self.models = {}

        print("  Building two-dimensional refactoring models (Action + Scope)...")
        
        # Combined model with Action + Scope dimensions
        action_vars = ['has_add', 'has_remove', 'has_adjust']
        scope_vars = ['has_method', 'has_class', 'has_local_variable']
        combined_vars = action_vars + scope_vars

        # Build model for is_issue_solved (always present)
        print("    - Building model for is_issue_solved...")
        self.models['combined_is_issue_solved'] = self.fit_logistic_model(
            combined_vars, 
            'Two-Dimensional Refactoring Model (Action + Scope, Outcome: is_issue_solved)',
            outcome_var='is_issue_solved_binary'
        )
        
        # Build model for compilation outcome if available
        if self.has_compile_data:
            print("    - Building model for is_compile_ok...")
            self.models['combined_is_compile_ok'] = self.fit_logistic_model(
                combined_vars,
                'Two-Dimensional Refactoring Model (Action + Scope, Outcome: is_compile_ok)',
                outcome_var='is_compile_ok_binary'
            )
            
            print(f"  2 models built successfully (is_issue_solved and is_compile_ok)")
        else:
            print(f"  1 model built successfully (is_issue_solved only)")
    
    def extract_results_table(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        """Extract results table for a single model
        
        Args:
            model_results: Dictionary containing model results
        """
        if model_results is None:
            return pd.DataFrame()

        model = model_results['model']
        treatment_vars = model_results['treatment_vars']
        X_data = model_results['X_data']

        # Extract coefficients, standard errors and p values
        results_data = []

        for var in model.params.index:
            is_treatment = any(var == tvar for tvar in treatment_vars)
            is_const = (var == 'const')

            if is_const:
                var_type = 'Intercept'
            elif is_treatment:
                var_type = 'Treatment'
            else:
                var_type = 'Control'

            # Clean variable name for display: remove _scaled suffix
            display_var = self._clean_variable_name(var)

            # Calculate N_Observations for this specific variable
            if var == 'const':
                n_obs_var = int(model.nobs)
            elif var in X_data.columns:
                # For dummy variables (0/1), count how many times the category appears (value = 1)
                # For continuous variables, count non-missing values
                if X_data[var].dtype in ['int64', 'float64'] and set(X_data[var].dropna().unique()).issubset({0, 1}):
                    # This is a dummy variable (only contains 0 and 1)
                    n_obs_var = int((X_data[var] == 1).sum())
                else:
                    # This is a continuous variable
                    n_obs_var = int(X_data[var].notna().sum())
            else:
                n_obs_var = int(model.nobs)

            # Format p values and check significance
            p_val = model.pvalues[var]
            if p_val < 0.001:
                p_formatted = f"{p_val:.2e}"
                significant = '***'
            elif p_val < 0.01:
                p_formatted = f"{p_val:.3f}"
                significant = '**'
            elif p_val < 0.05:
                p_formatted = f"{p_val:.3f}"
                significant = '*'
            else:
                p_formatted = f"{p_val:.3f}"
                significant = ''

            # Calculate Odds Ratios (skip for intercept)
            if var == 'const':
                odds_ratio = np.nan
                or_ci_lower = np.nan
                or_ci_upper = np.nan
            else:
                odds_ratio = np.exp(model.params[var])
                or_ci_lower = np.exp(model.conf_int()[0][var])
                or_ci_upper = np.exp(model.conf_int()[1][var])

            results_data.append({
                'Variable': display_var,
                'Original_Variable': var,  # Keep original for reference
                'Type': var_type,
                'N_Observations': n_obs_var,
                'Estimate': model.params[var],
                'Std_Error': model.bse[var],
                'P_value': p_val,
                'P_value_formatted': p_formatted,
                'CI_Lower': model.conf_int()[0][var],
                'CI_Upper': model.conf_int()[1][var],
                'Odds_Ratio': odds_ratio,
                'OR_CI_Lower': or_ci_lower,
                'OR_CI_Upper': or_ci_upper,
                'Significant': significant
            })

        return pd.DataFrame(results_data)
    
    def _clean_variable_name(self, var: str) -> str:
        """
        Clean variable name for display by removing _scaled suffix
        Keep _log for log-transformed variables
        """
        if var == 'const':
            return var
        
        # Remove _scaled suffix
        if var.endswith('_scaled'):
            return var.replace('_scaled', '')
        
        return var
    
    
    def generate_results(self):
        """Generate and display results for the combined model"""
        print("\n" + "="*80)
        print("REFACTORING LOGISTIC REGRESSION ANALYSIS RESULTS")
        print("="*80)

        for model_name, model_results in self.models.items():
            if model_results is None:
                continue

            print(f"\n{'='*80}")
            print(f"{model_results['model_name']}")
            print(f"{'='*80}")
            print(f"Observations: {model_results['n_obs']}")
            print(f"McFadden R²: {model_results['mcfadden_r2']:.4f}")
            print(f"Adjusted McFadden R²: {model_results['adj_mcfadden_r2']:.4f}")
            print(f"AIC: {model_results['aic']:.2f}")
            print(f"BIC: {model_results['bic']:.2f}")

            # Get all variables results
            detailed_results = self.extract_results_table(model_results)

            # Display results organized by dimension
            self.display_model_results(detailed_results)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)


    def display_model_results(self, detailed_results: pd.DataFrame):
        """Display results for combined model organized by dimension"""
        # Define dimension groupings
        action_vars = ['has_add', 'has_remove', 'has_adjust']
        scope_vars = ['has_method', 'has_class', 'has_local_variable']

        # Filter out intercept for cleaner display
        results_no_intercept = detailed_results[detailed_results['Variable'] != 'const'].copy()

        # Treatment variables organized by dimension
        action_results = results_no_intercept[results_no_intercept['Variable'].isin(action_vars)]
        scope_results = results_no_intercept[results_no_intercept['Variable'].isin(scope_vars)]
        
        # All treatment variables
        all_treatment_results = pd.concat([action_results, scope_results])

        # Control variables
        control_results = results_no_intercept[results_no_intercept['Type'] == 'Control']

        # Display Action dimension
        if not action_results.empty:
            print(f"\n{'Action Dimension':-^80}")
            print(f"{'Variable':<30} {'N_Obs':<8} {'Estimate':<12} {'Odds Ratio':<12} {'95% CI':<25} {'P-Value':<12} {'Sig.':<5}")
            print("-" * 80)
            for _, row in action_results.iterrows():
                p_val = row['P_value']
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                estimate_str = f"{row['Estimate']:.4f}"
                or_str = f"{row['Odds_Ratio']:.4f}"
                ci_str = f"[{row['OR_CI_Lower']:.4f}, {row['OR_CI_Upper']:.4f}]"
                p_formatted = f"{p_val:.4f}" if p_val >= 0.001 else f"{p_val:.2e}"
                n_obs = f"{row['N_Observations']:,}"
                print(f"{row['Variable']:<30} {n_obs:<8} {estimate_str:<12} {or_str:<12} {ci_str:<25} {p_formatted:<12} {sig:<5}")

        # Display Scope dimension
        if not scope_results.empty:
            print(f"\n{'Scope Dimension':-^80}")
            print(f"{'Variable':<30} {'N_Obs':<8} {'Estimate':<12} {'Odds Ratio':<12} {'95% CI':<25} {'P-Value':<12} {'Sig.':<5}")
            print("-" * 80)
            for _, row in scope_results.iterrows():
                p_val = row['P_value']
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                estimate_str = f"{row['Estimate']:.4f}"
                or_str = f"{row['Odds_Ratio']:.4f}"
                ci_str = f"[{row['OR_CI_Lower']:.4f}, {row['OR_CI_Upper']:.4f}]"
                p_formatted = f"{p_val:.4f}" if p_val >= 0.001 else f"{p_val:.2e}"
                n_obs = f"{row['N_Observations']:,}"
                print(f"{row['Variable']:<30} {n_obs:<8} {estimate_str:<12} {or_str:<12} {ci_str:<25} {p_formatted:<12} {sig:<5}")

        # Display all control variables
        if not control_results.empty:
            print(f"\n{'Control Variables (All Variables)':-^80}")
            print(f"{'Variable':<30} {'N_Obs':<8} {'Estimate':<12} {'Odds Ratio':<12} {'P-Value':<12} {'Sig.':<5}")
            print("-" * 80)
            for _, row in control_results.iterrows():
                p_val = row['P_value']
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                estimate_str = f"{row['Estimate']:.4f}"
                or_str = f"{row['Odds_Ratio']:.4f}"
                p_formatted = f"{p_val:.4f}" if p_val >= 0.001 else f"{p_val:.2e}"
                n_obs = f"{row['N_Observations']:,}"
                print(f"{row['Variable']:<30} {n_obs:<8} {estimate_str:<12} {or_str:<12} {p_formatted:<12} {sig:<5}")
            
            # Show summary statistics
            significant_count = len(control_results[control_results['P_value'] < 0.05])
            print(f"\nTotal control variables: {len(control_results)}, Significant (p < 0.05): {significant_count}")
    
    

    def save_results(self):
        """Save analysis results to files"""
        output_dir = Path("output/RQ2/refactoring_logistic_regression")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  - analysis_output.txt")
        
        # Save detailed results for each model
        for model_name, model_results in self.models.items():
            if model_results is None:
                continue
            
            # Extract outcome name from model_name
            if 'is_issue_solved' in model_name:
                outcome = 'is_issue_solved'
            elif 'is_compile_ok' in model_name:
                outcome = 'is_compile_ok'
            else:
                outcome = 'unknown'
            
            # Get detailed results table
            detailed_results = self.extract_results_table(model_results)
            
            # Save as CSV
            csv_path = output_dir / f"detailed_results_{outcome}.csv"
            detailed_results.to_csv(csv_path, index=False)
            print(f"  - {csv_path}")
            
            # Save model statistics
            stats = {
                'model_name': model_results['model_name'],
                'outcome': outcome,
                'n_observations': model_results['n_obs'],
                'mcfadden_r2': model_results['mcfadden_r2'],
                'adj_mcfadden_r2': model_results['adj_mcfadden_r2'],
                'aic': model_results['aic'],
                'bic': model_results['bic']
            }
            
            json_path = output_dir / f"model_statistics_{outcome}.json"
            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"  - {json_path}")
        
        print(f"\nResults saved to {output_dir}/")
    
    def run_complete_analysis(self):
        """Run two-dimensional refactoring analysis"""
        # Setup output directory and redirection
        output_dir = Path("output/RQ2/refactoring_logistic_regression")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = output_dir / 'analysis_output.txt'
        tee = TeeOutput(str(log_file))
        sys.stdout = tee
        
        print(f"Output log file: {log_file}\n")
        
        try:
            print("\nLoading data...")
            self.load_data()
            
            print("\nCreating refactoring dimension mappings...")
            self.create_mappings()
            
            print("\nGenerating refactoring dimension variables...")
            self.generate_has_variables()
            
            print("\nPreprocessing features...")
            self.preprocess_features()
            
            print("\nBuilding models...")
            self.build_models()
            
            # Display core results
            self.generate_results()
            
            # Save results
            print("\nSaving results...")
            self.save_results()
            
            print("\nAnalysis completed successfully.")
                        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore stdout and close log file
            tee.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run refactoring logistic regression analysis.")
    parser.add_argument("--data", type=str, default="data/unified_data.csv",
                        help="Path to unified data CSV file")
    parser.add_argument("--classification", type=str, default="data/refactoring_classification.xlsx",
                        help="Path to refactoring classification Excel file")
    args = parser.parse_args()

    # Set file paths
    data_path = args.data
    classification_path = args.classification
    
    # Check if classification file exists
    if not Path(classification_path).exists():
        print(f"Error: Classification file not found at {classification_path}")
        print("Please provide the refactoring_classification.xlsx file.")
        return
    
    # Create analyzer and run
    analyzer = RefactoringLogisticAnalysis(data_path, classification_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
