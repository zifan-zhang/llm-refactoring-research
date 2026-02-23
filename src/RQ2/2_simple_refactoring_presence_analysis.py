#!/usr/bin/env python3
"""
Simple Refactoring Presence Analysis: Impact of Having Refactoring on Task Resolution
Study the basic effect of agent_has_refactoring (binary: Yes/No) on task outcomes
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings

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

class SimpleRefactoringAnalysis:
    """Analyze the basic effect of having refactoring"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.df_encoded = None
        self.has_compile_data = False
        
    def load_and_preprocess(self):
        """Load and preprocess data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records")
        
        # Create binary outcome variable (already 0/1 in CSV)
        self.df['is_issue_solved_binary'] = self.df['is_issue_solved'].astype(int)
        
        # Check compilation data availability (only use unified is_compile_ok field)
        if 'is_compile_ok' in self.df.columns:
            self.df['is_compile_ok_binary'] = self.df['is_compile_ok'].astype(int)
            self.has_compile_data = True
            print("Compilation data found: will analyze is_compile_ok")
        else:
            print("Compilation data (is_compile_ok) not found: will only analyze is_issue_solved")
        
        # Ensure agent_has_refactoring is binary integer
        self.df['agent_has_refactoring'] = self.df['agent_has_refactoring'].astype(int)
        
        print(f"\n{'='*80}")
        print("DESCRIPTIVE STATISTICS")
        print(f"{'='*80}")
        print(f"\nDistribution of agent_has_refactoring:")
        refactoring_dist = self.df['agent_has_refactoring'].value_counts().sort_index()
        for val, count in refactoring_dist.items():
            label = "Has Refactoring" if val == 1 else "No Refactoring"
            print(f"  {label} ({val}): {count} instances ({count/len(self.df)*100:.1f}%)")
        
        print(f"\nSolve rate by refactoring presence:")
        solve_by_ref = self.df.groupby('agent_has_refactoring')['is_issue_solved_binary'].agg(['mean', 'count'])
        for idx, row in solve_by_ref.iterrows():
            label = "Has Refactoring" if idx == 1 else "No Refactoring"
            print(f"  {label}: {row['mean']*100:.2f}% solved (n={int(row['count'])})")
        
        if self.has_compile_data:
            print(f"\nCompilation success rate (is_compile_ok) by refactoring presence:")
            compile_by_ref = self.df.groupby('agent_has_refactoring')['is_compile_ok_binary'].agg(['mean', 'count'])
            for idx, row in compile_by_ref.iterrows():
                label = "Has Refactoring" if idx == 1 else "No Refactoring"
                print(f"  {label}: {row['mean']*100:.2f}% success (n={int(row['count'])})")
        
        # Apply transformations and encoding (same as other scripts)
        print(f"\nApplying transformations...")
        self.df['modified_lines_log'] = np.log1p(self.df['modified_lines'])
        self.df['modified_files_log'] = np.log1p(self.df['modified_files'])
        self.df['issue_length_log'] = np.log1p(self.df['issue_length'])
        
        # One-hot encode categorical variables
        categorical_vars = ['task_difficulty', 'llm_model', 'agent_framework', 'issue_type']
        self.df_encoded = pd.get_dummies(
            self.df,
            columns=categorical_vars,
            prefix=categorical_vars,
            drop_first=True
        )
        
        # Convert boolean columns
        bool_columns = self.df_encoded.select_dtypes(include=[bool]).columns
        self.df_encoded[bool_columns] = self.df_encoded[bool_columns].astype(int)
        
        # Standardize numerical variables
        numerical_vars = ['modified_lines_log', 'modified_files_log', 'file_coverage', 'line_coverage',
                         'issue_length_log']
        scaler = StandardScaler()
        for var in numerical_vars:
            if var in self.df_encoded.columns:
                self.df_encoded[f'{var}_scaled'] = scaler.fit_transform(
                    self.df_encoded[[var]]
                ).flatten()
        
        print(f"Preprocessing complete. Total features: {len(self.df_encoded.columns)}")
    
    def fit_model(self, outcome_var='is_issue_solved_binary'):
        """Fit logistic regression with agent_has_refactoring as treatment"""
        # Get control variables
        control_vars = []
        control_vars.extend([col for col in self.df_encoded.columns if col.endswith('_scaled')])
        categorical_prefixes = ['task_difficulty_', 'llm_model_', 'agent_framework_', 'issue_type_']
        for prefix in categorical_prefixes:
            control_vars.extend([col for col in self.df_encoded.columns if col.startswith(prefix)])
        
        # Prepare data
        all_vars = ['agent_has_refactoring'] + control_vars
        X = self.df_encoded[all_vars].copy()
        y = self.df_encoded[outcome_var]

        # Keep a copy before fillna for variable-specific N calculation
        X_before_fillna = X.copy()

        X = X.fillna(0)
        X = sm.add_constant(X)

        # Fit model
        model = sm.Logit(y, X).fit(disp=0)

        # Calculate metrics
        mcfadden_r2 = 1 - (model.llf / model.llnull)
        adj_mcfadden_r2 = 1 - ((model.llf - model.df_model) / (model.llnull - 1))

        return {
            'model': model,
            'outcome_var': outcome_var,
            'treatment_vars': ['agent_has_refactoring'],
            'control_vars': control_vars,
            'n_obs': int(model.nobs),
            'X_data': X_before_fillna,
            'mcfadden_r2': mcfadden_r2,
            'adj_mcfadden_r2': adj_mcfadden_r2,
            'aic': model.aic,
            'bic': model.bic
        }

    def _clean_variable_name(self, var: str) -> str:
        """Remove _scaled suffix for display consistency."""
        if var == 'const':
            return var
        if var.endswith('_scaled'):
            return var.replace('_scaled', '')
        return var

    def extract_results_table(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        """Extract detailed variable-level table with N per variable."""
        model = model_results['model']
        treatment_vars = model_results['treatment_vars']
        X_data = model_results['X_data']

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

            if var == 'const':
                n_obs_var = int(model.nobs)
            elif var in X_data.columns:
                non_na_series = X_data[var].dropna()
                if non_na_series.nunique() > 0 and set(non_na_series.unique()).issubset({0, 1}):
                    n_obs_var = int((X_data[var] == 1).sum())
                else:
                    n_obs_var = int(X_data[var].notna().sum())
            else:
                n_obs_var = int(model.nobs)

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

            if var == 'const':
                odds_ratio = np.nan
                or_ci_lower = np.nan
                or_ci_upper = np.nan
            else:
                odds_ratio = np.exp(model.params[var])
                or_ci_lower = np.exp(model.conf_int()[0][var])
                or_ci_upper = np.exp(model.conf_int()[1][var])

            results_data.append({
                'Variable': self._clean_variable_name(var),
                'Original_Variable': var,
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

    def display_model_results(self, detailed_results: pd.DataFrame):
        """Display treatment and all control variables, aligned with script 3."""
        results_no_intercept = detailed_results[detailed_results['Variable'] != 'const'].copy()
        treatment_results = results_no_intercept[results_no_intercept['Type'] == 'Treatment']
        control_results = results_no_intercept[results_no_intercept['Type'] == 'Control']

        if not treatment_results.empty:
            print(f"\n{'Treatment Variables':-^80}")
            print(f"{'Variable':<30} {'N_Obs':<8} {'Estimate':<12} {'Odds Ratio':<12} {'95% CI':<25} {'P-Value':<12} {'Sig.':<5}")
            print("-" * 80)
            for _, row in treatment_results.iterrows():
                p_val = row['P_value']
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                estimate_str = f"{row['Estimate']:.4f}"
                or_str = f"{row['Odds_Ratio']:.4f}"
                ci_str = f"[{row['OR_CI_Lower']:.4f}, {row['OR_CI_Upper']:.4f}]"
                p_formatted = f"{p_val:.4f}" if p_val >= 0.001 else f"{p_val:.2e}"
                n_obs = f"{row['N_Observations']:,}"
                print(f"{row['Variable']:<30} {n_obs:<8} {estimate_str:<12} {or_str:<12} {ci_str:<25} {p_formatted:<12} {sig:<5}")

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

            significant_count = len(control_results[control_results['P_value'] < 0.05])
            print(f"\nTotal control variables: {len(control_results)}, Significant (p < 0.05): {significant_count}")
    
    def run_analysis(self):
        """Run complete analysis"""
        # Setup output directory and redirection
        output_dir = Path("output/RQ2/simple_refactoring_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = output_dir / 'analysis_output.txt'
        tee = TeeOutput(str(log_file))
        sys.stdout = tee
        
        print(f"Output log file: {log_file}")
        
        self.load_and_preprocess()
        
        print("\n" + "="*80)
        print("LOGISTIC REGRESSION ANALYSIS RESULTS")
        print("="*80)
        
        # Analyze for each outcome variable
        outcomes = [('is_issue_solved_binary', 'is_issue_solved')]
        if self.has_compile_data and 'is_compile_ok_binary' in self.df_encoded.columns:
            outcomes.append(('is_compile_ok_binary', 'is_compile_ok'))
        
        results_summary = []
        detailed_results_by_outcome = {}
        
        for outcome_var, outcome_name in outcomes:
            print(f"\n{'='*80}")
            print(f"OUTCOME: {outcome_name.upper()}")
            print(f"{'='*80}")
            
            model_results = self.fit_model(outcome_var)
            model = model_results['model']
            r2 = model_results['mcfadden_r2']
            adj_r2 = model_results['adj_mcfadden_r2']

            print(f"\nModel Statistics:")
            print(f"  Observations: {int(model.nobs)}")
            print(f"  McFadden R²: {r2:.4f}")
            print(f"  Adjusted McFadden R²: {adj_r2:.4f}")
            print(f"  AIC: {model.aic:.2f}")
            print(f"  BIC: {model.bic:.2f}")

            detailed_results = self.extract_results_table(model_results)
            detailed_results_by_outcome[outcome_name] = {
                'model_results': model_results,
                'detailed_results': detailed_results
            }

            self.display_model_results(detailed_results)

            treatment_row = detailed_results[detailed_results['Original_Variable'] == 'agent_has_refactoring'].iloc[0]
            coef = treatment_row['Estimate']
            pval = treatment_row['P_value']
            odds_ratio = treatment_row['Odds_Ratio']
            or_ci_lower = treatment_row['OR_CI_Lower']
            or_ci_upper = treatment_row['OR_CI_Upper']
            
            # Store results
            results_summary.append({
                'outcome': outcome_name,
                'coefficient': coef,
                'odds_ratio': odds_ratio,
                'or_ci_lower': or_ci_lower,
                'or_ci_upper': or_ci_upper,
                'p_value': pval,
                'mcfadden_r2': r2,
                'adj_mcfadden_r2': adj_r2,
                'n_observations': int(model.nobs),
                'significant': bool(pval < 0.05)
            })
            
            # Interpretation
            print(f"\n{'Interpretation':-^80}")
            if pval < 0.05:
                direction = "increases" if coef > 0 else "decreases"
                percentage_change = (odds_ratio - 1) * 100
                if percentage_change > 0:
                    print(f"  ✓ Having refactoring significantly INCREASES the odds of {outcome_name}")
                    print(f"    by {percentage_change:.1f}% (p={pval:.4f})")
                else:
                    print(f"  ✓ Having refactoring significantly DECREASES the odds of {outcome_name}")
                    print(f"    by {abs(percentage_change):.1f}% (p={pval:.4f})")
                
                # Effect size interpretation
                if abs(percentage_change) < 10:
                    effect_size = "small"
                elif abs(percentage_change) < 30:
                    effect_size = "moderate"
                else:
                    effect_size = "large"
                print(f"    Effect size: {effect_size}")
            else:
                print(f"  ✗ No significant effect of having refactoring on {outcome_name}")
                print(f"    (p={pval:.4f} > 0.05)")
            
            print("-" * 80)
        
        # Final summary
        print(f"\n{'='*80}")
        print("SUMMARY OF RESULTS")
        print(f"{'='*80}")
        print(f"\n{'Outcome':<25} {'Odds Ratio':<15} {'95% CI':<30} {'P-value':<12} {'Sig.':<5}")
        print("-" * 80)
        for result in results_summary:
            ci_str = f"[{result['or_ci_lower']:.4f}, {result['or_ci_upper']:.4f}]"
            p_str = f"{result['p_value']:.6f}" if result['p_value'] >= 0.0001 else f"{result['p_value']:.2e}"
            sig_mark = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
            print(f"{result['outcome']:<25} {result['odds_ratio']:<15.4f} {ci_str:<30} {p_str:<12} {sig_mark:<5}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        # Save results
        self.save_results(results_summary, detailed_results_by_outcome)
        
        # Restore stdout and close log file
        tee.close()
    
    def save_results(self, results_summary, detailed_results_by_outcome):
        """Save results to file"""
        output_dir = Path("output/RQ2/simple_refactoring_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save summary as CSV
        df_results = pd.DataFrame(results_summary)
        df_results.to_csv(output_dir / 'summary.csv', index=False)

        # Save detailed results and model statistics for each outcome
        for outcome_name, result_bundle in detailed_results_by_outcome.items():
            if outcome_name == 'is_issue_solved':
                outcome_suffix = 'is_issue_solved'
            elif outcome_name == 'is_compile_ok':
                outcome_suffix = 'is_compile_ok'
            else:
                outcome_suffix = outcome_name

            detailed_csv_path = output_dir / f"detailed_results_{outcome_suffix}.csv"
            result_bundle['detailed_results'].to_csv(detailed_csv_path, index=False)

            model_results = result_bundle['model_results']
            stats = {
                'outcome': outcome_name,
                'n_observations': model_results['n_obs'],
                'mcfadden_r2': model_results['mcfadden_r2'],
                'adj_mcfadden_r2': model_results['adj_mcfadden_r2'],
                'aic': model_results['aic'],
                'bic': model_results['bic']
            }
            stats_json_path = output_dir / f"model_statistics_{outcome_suffix}.json"
            with open(stats_json_path, 'w') as f:
                json.dump(stats, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")
        print(f"  - analysis_output.txt")
        print(f"  - summary.json")
        print(f"  - summary.csv")
        print(f"  - detailed_results_is_issue_solved.csv")
        if self.has_compile_data:
            print(f"  - detailed_results_is_compile_ok.csv")
        print(f"  - model_statistics_is_issue_solved.json")
        if self.has_compile_data:
            print(f"  - model_statistics_is_compile_ok.json")

def main():
    """Main function"""
    data_path = "data/unified_data.csv"
    
    # Check if we have RAS data available
    if Path("data/unified_data_with_ras.csv").exists():
        print("Note: RAS data available at data/unified_data_with_ras.csv")
        print("This analysis uses the base unified_data.csv without RAS columns.\n")
    
    analyzer = SimpleRefactoringAnalysis(data_path)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()



