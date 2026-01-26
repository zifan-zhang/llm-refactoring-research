#!/usr/bin/env python3
"""
Simple Refactoring Presence Analysis: Impact of Having Refactoring on Task Resolution
Study the basic effect of agent_has_refactoring (binary: Yes/No) on task outcomes
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

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
        self.df['patch_size_log'] = np.log1p(self.df['patch_size'])
        self.df['issue_length_log'] = np.log1p(self.df['issue_length'])
        self.df['golden_patch_length_log'] = np.log1p(self.df['golden_patch_length'])
        
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
        numerical_vars = ['patch_size_log', 'file_coverage', 'line_coverage', 
                         'issue_length_log', 'golden_patch_length_log']
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
        
        X = X.fillna(0)
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.Logit(y, X).fit(disp=0)
        
        # Calculate metrics
        mcfadden_r2 = 1 - (model.llf / model.llnull)
        adj_mcfadden_r2 = 1 - ((model.llf - model.df_model) / (model.llnull - 1))
        
        return model, mcfadden_r2, adj_mcfadden_r2
    
    def run_analysis(self):
        """Run complete analysis"""
        self.load_and_preprocess()
        
        print("\n" + "="*80)
        print("LOGISTIC REGRESSION ANALYSIS RESULTS")
        print("="*80)
        
        # Analyze for each outcome variable
        outcomes = [('is_issue_solved_binary', 'is_issue_solved')]
        if self.has_compile_data and 'is_compile_ok_binary' in self.df_encoded.columns:
            outcomes.append(('is_compile_ok_binary', 'is_compile_ok'))
        
        results_summary = []
        
        for outcome_var, outcome_name in outcomes:
            print(f"\n{'='*80}")
            print(f"OUTCOME: {outcome_name.upper()}")
            print(f"{'='*80}")
            
            model, r2, adj_r2 = self.fit_model(outcome_var)
            
            print(f"\nModel Statistics:")
            print(f"  Observations: {int(model.nobs)}")
            print(f"  McFadden R²: {r2:.4f}")
            print(f"  Adjusted McFadden R²: {adj_r2:.4f}")
            print(f"  AIC: {model.aic:.2f}")
            print(f"  BIC: {model.bic:.2f}")
            
            # Extract agent_has_refactoring coefficient
            coef = model.params['agent_has_refactoring']
            se = model.bse['agent_has_refactoring']
            pval = model.pvalues['agent_has_refactoring']
            ci_lower, ci_upper = model.conf_int().loc['agent_has_refactoring']
            
            # Calculate odds ratio
            odds_ratio = np.exp(coef)
            or_ci_lower = np.exp(ci_lower)
            or_ci_upper = np.exp(ci_upper)
            
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            
            print(f"\n{'Treatment Variable: agent_has_refactoring':-^80}")
            print(f"  Coefficient (log-odds): {coef:.4f} (SE: {se:.4f})")
            print(f"  Odds Ratio: {odds_ratio:.4f}")
            print(f"  95% CI for OR: [{or_ci_lower:.4f}, {or_ci_upper:.4f}]")
            print(f"  P-value: {pval:.6f} {sig}")
            
            # Store results
            results_summary.append({
                'outcome': outcome_name,
                'coefficient': coef,
                'odds_ratio': odds_ratio,
                'or_ci_lower': or_ci_lower,
                'or_ci_upper': or_ci_upper,
                'p_value': pval,
                'significant': sig != ''
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
        self.save_results(results_summary)
    
    def save_results(self, results_summary):
        """Save results to file"""
        output_dir = Path("output/RQ2/simple_refactoring_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save summary as CSV
        df_results = pd.DataFrame(results_summary)
        df_results.to_csv(output_dir / 'summary.csv', index=False)
        
        print(f"\nResults saved to {output_dir}/")
        print(f"  - summary.json")
        print(f"  - summary.csv")

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



