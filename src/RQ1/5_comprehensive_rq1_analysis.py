"""
RQ1 Comprehensive Analysis: Agent vs Human Refactoring Behavior
Generates all statistics and LaTeX tables needed for RQ1 Results section

This script produces:
1. Overall comparison statistics (presence rate, operation counts)
2. Refactoring type distribution comparison
3. Action/Scope/TestImpact category analysis
4. Per-issue paired comparison
5. Framework and LLM comparisons
6. LaTeX tables for paper
"""

import csv
import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_refactoring_classification(excel_path):
    """Load refactoring classification from Excel file."""
    df = pd.read_excel(excel_path)
    
    classification = {}
    for _, row in df.iterrows():
        refactoring_type = row['Refactoring']
        classification[refactoring_type] = {
            'action': row['Action'],
            'scope': row['Scope'],
            'test_impact': row['Can Affect Test Interface？']
        }
    
    return classification


def parse_refactoring_dict(refactoring_str):
    """Parse refactoring type count from JSON string."""
    if not refactoring_str or refactoring_str == '{}':
        return {}
    try:
        return json.loads(refactoring_str)
    except json.JSONDecodeError:
        return {}


def load_unified_dataset(csv_path):
    """Load and parse the unified dataset."""
    data = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both numeric (0/1) and string ('True'/'False', 'resolved'/'unresolved') formats
            agent_has_ref_raw = row.get('agent_has_refactoring', '')
            golden_has_ref_raw = row.get('golden_has_refactoring', '')
            is_solved_raw = row.get('is_issue_solved', '')
            is_compile_raw = row.get('is_compile_ok', '')
            
            data.append({
                'instance_id': row.get('instance_id', ''),
                'agent_name': row.get('agent_name', ''),
                'llm_model': row.get('llm_model', ''),
                'agent_framework': row.get('agent_framework', ''),
                'agent_has_refactoring': agent_has_ref_raw in ('1', 'True'),
                'agent_refactoring_types': parse_refactoring_dict(row.get('agent_refactoring_type_count', '{}')),
                'golden_has_refactoring': golden_has_ref_raw in ('1', 'True'),
                'golden_refactoring_types': parse_refactoring_dict(row.get('golden_refactoring_type_count', '{}')),
                'is_solved': is_solved_raw in ('1', 'resolved'),
                'issue_type': row.get('issue_type', ''),
                'task_difficulty': row.get('task_difficulty', ''),
                'is_compile_ok': is_compile_raw in ('1', 'True')
            })
    
    return data


def calculate_overall_statistics(data, classification):
    """Calculate overall refactoring statistics for Agent vs Golden."""
    
    # Deduplicate golden patches by instance_id
    golden_by_instance = {}
    for row in data:
        instance_id = row['instance_id']
        if instance_id not in golden_by_instance:
            golden_by_instance[instance_id] = {
                'has_refactoring': row['golden_has_refactoring'],
                'refactoring_types': row['golden_refactoring_types'],
                'total_count': sum(row['golden_refactoring_types'].values())
            }
    
    # Agent statistics (all patches)
    agent_stats = {
        'total_patches': len(data),
        'patches_with_refactoring': sum(1 for row in data if row['agent_has_refactoring']),
        'refactoring_operations': [],
        'refactoring_types': defaultdict(int)
    }
    
    for row in data:
        count = sum(row['agent_refactoring_types'].values())
        agent_stats['refactoring_operations'].append(count)
        for ref_type, cnt in row['agent_refactoring_types'].items():
            agent_stats['refactoring_types'][ref_type] += cnt
    
    # Golden statistics (deduplicated)
    golden_stats = {
        'total_patches': len(golden_by_instance),
        'patches_with_refactoring': sum(1 for v in golden_by_instance.values() if v['has_refactoring']),
        'refactoring_operations': [v['total_count'] for v in golden_by_instance.values()],
        'refactoring_types': defaultdict(int)
    }
    
    for v in golden_by_instance.values():
        for ref_type, cnt in v['refactoring_types'].items():
            golden_stats['refactoring_types'][ref_type] += cnt
    
    # Statistical tests
    # Chi-square test for presence of refactoring
    agent_presence = agent_stats['patches_with_refactoring']
    agent_absence = agent_stats['total_patches'] - agent_presence
    golden_presence = golden_stats['patches_with_refactoring']
    golden_absence = golden_stats['total_patches'] - golden_presence
    
    chi2, p_value_presence = stats.chi2_contingency([
        [agent_presence, agent_absence],
        [golden_presence, golden_absence]
    ])[:2]
    
    # Mann-Whitney U test for refactoring counts
    u_stat, p_value_count = stats.mannwhitneyu(
        agent_stats['refactoring_operations'],
        golden_stats['refactoring_operations'],
        alternative='two-sided'
    )
    
    # Calculate summary statistics
    results = {
        'agent': {
            'total_patches': agent_stats['total_patches'],
            'with_refactoring': agent_presence,
            'presence_rate': agent_presence / agent_stats['total_patches'] * 100,
            'total_operations': sum(agent_stats['refactoring_operations']),
            'mean_operations': np.mean(agent_stats['refactoring_operations']),
            'median_operations': np.median(agent_stats['refactoring_operations']),
            'std_operations': np.std(agent_stats['refactoring_operations']),
            'unique_types': len(agent_stats['refactoring_types'])
        },
        'golden': {
            'total_patches': golden_stats['total_patches'],
            'with_refactoring': golden_presence,
            'presence_rate': golden_presence / golden_stats['total_patches'] * 100,
            'total_operations': sum(golden_stats['refactoring_operations']),
            'mean_operations': np.mean(golden_stats['refactoring_operations']),
            'median_operations': np.median(golden_stats['refactoring_operations']),
            'std_operations': np.std(golden_stats['refactoring_operations']),
            'unique_types': len(golden_stats['refactoring_types'])
        },
        'statistical_tests': {
            'chi2_presence': {
                'chi2': chi2,
                'p_value': p_value_presence,
                'significant': p_value_presence < 0.05
            },
            'mann_whitney_count': {
                'u_stat': u_stat,
                'p_value': p_value_count,
                'significant': p_value_count < 0.05
            }
        }
    }
    
    return results, agent_stats, golden_stats


def calculate_category_statistics(agent_types, golden_types, classification):
    """Calculate refactoring statistics by Action, Scope, and Test Impact."""
    
    def categorize_refactorings(ref_types_dict, classification):
        categories = {
            'action': defaultdict(int),
            'scope': defaultdict(int),
            'test_impact': {'Yes': 0, 'No': 0}
        }
        
        for ref_type, count in ref_types_dict.items():
            if ref_type in classification:
                cat = classification[ref_type]
                categories['action'][cat['action']] += count
                categories['scope'][cat['scope']] += count
                categories['test_impact'][cat['test_impact']] += count
        
        return categories
    
    agent_categories = categorize_refactorings(agent_types, classification)
    golden_categories = categorize_refactorings(golden_types, classification)
    
    return {
        'agent': agent_categories,
        'golden': golden_categories
    }


def calculate_paired_comparison(data):
    """Calculate per-issue paired comparison between Agent and Golden."""
    
    # Group by instance_id
    instance_groups = defaultdict(list)
    golden_by_instance = {}
    
    for row in data:
        instance_id = row['instance_id']
        instance_groups[instance_id].append(row)
        
        # Store golden patch info (deduplicated)
        if instance_id not in golden_by_instance:
            golden_by_instance[instance_id] = {
                'has_refactoring': row['golden_has_refactoring'],
                'refactoring_types': set(row['golden_refactoring_types'].keys())
            }
    
    # Calculate different scenarios
    scenarios = {
        'both_have': 0,           # Both Agent and Golden have refactoring
        'only_agent': 0,          # Only Agent has refactoring
        'only_golden': 0,         # Only Golden has refactoring
        'neither': 0,             # Neither has refactoring
        'agent_patches_per_scenario': defaultdict(int),
        # Performance metrics per scenario
        'both_have_compile_ok': 0,
        'both_have_solved': 0,
        'only_agent_compile_ok': 0,
        'only_agent_solved': 0,
        'only_golden_compile_ok': 0,
        'only_golden_solved': 0,
        'neither_compile_ok': 0,
        'neither_solved': 0,
    }
    
    overlap_stats = {
        'overlapping_types': [],   # Percentage of overlapping refactoring types
        'excess_types': []         # Number of excess types in Agent patches
    }
    
    for instance_id, agent_patches in instance_groups.items():
        golden_info = golden_by_instance[instance_id]
        golden_has = golden_info['has_refactoring']
        golden_types = golden_info['refactoring_types']
        
        for patch in agent_patches:
            agent_has = patch['agent_has_refactoring']
            agent_types = set(patch['agent_refactoring_types'].keys())
            is_compile_ok = patch['is_compile_ok']
            is_solved = patch['is_solved']
            
            # Categorize scenario
            if agent_has and golden_has:
                scenarios['both_have'] += 1
                scenarios['agent_patches_per_scenario']['both_have'] += 1
                if is_compile_ok:
                    scenarios['both_have_compile_ok'] += 1
                if is_solved:
                    scenarios['both_have_solved'] += 1
                
                # Calculate overlap
                if len(golden_types) > 0:
                    overlap = len(agent_types & golden_types) / len(golden_types) * 100
                    overlap_stats['overlapping_types'].append(overlap)
                
                # Calculate excess
                excess = len(agent_types - golden_types)
                overlap_stats['excess_types'].append(excess)
                
            elif agent_has and not golden_has:
                scenarios['only_agent'] += 1
                scenarios['agent_patches_per_scenario']['only_agent'] += 1
                if is_compile_ok:
                    scenarios['only_agent_compile_ok'] += 1
                if is_solved:
                    scenarios['only_agent_solved'] += 1
                overlap_stats['excess_types'].append(len(agent_types))
                
            elif not agent_has and golden_has:
                scenarios['only_golden'] += 1
                scenarios['agent_patches_per_scenario']['only_golden'] += 1
                if is_compile_ok:
                    scenarios['only_golden_compile_ok'] += 1
                if is_solved:
                    scenarios['only_golden_solved'] += 1
                
            else:
                scenarios['neither'] += 1
                scenarios['agent_patches_per_scenario']['neither'] += 1
                if is_compile_ok:
                    scenarios['neither_compile_ok'] += 1
                if is_solved:
                    scenarios['neither_solved'] += 1
    
    # Calculate summary statistics for overlap
    if overlap_stats['overlapping_types']:
        scenarios['mean_overlap_pct'] = np.mean(overlap_stats['overlapping_types'])
        scenarios['median_overlap_pct'] = np.median(overlap_stats['overlapping_types'])
    else:
        scenarios['mean_overlap_pct'] = 0
        scenarios['median_overlap_pct'] = 0
    
    scenarios['mean_excess'] = np.mean(overlap_stats['excess_types']) if overlap_stats['excess_types'] else 0
    scenarios['median_excess'] = np.median(overlap_stats['excess_types']) if overlap_stats['excess_types'] else 0
    
    # Calculate success rates for each scenario
    if scenarios['both_have'] > 0:
        scenarios['both_have_compile_rate'] = scenarios['both_have_compile_ok'] / scenarios['both_have'] * 100
        scenarios['both_have_solve_rate'] = scenarios['both_have_solved'] / scenarios['both_have'] * 100
    else:
        scenarios['both_have_compile_rate'] = 0
        scenarios['both_have_solve_rate'] = 0
    
    if scenarios['only_agent'] > 0:
        scenarios['only_agent_compile_rate'] = scenarios['only_agent_compile_ok'] / scenarios['only_agent'] * 100
        scenarios['only_agent_solve_rate'] = scenarios['only_agent_solved'] / scenarios['only_agent'] * 100
    else:
        scenarios['only_agent_compile_rate'] = 0
        scenarios['only_agent_solve_rate'] = 0
    
    if scenarios['only_golden'] > 0:
        scenarios['only_golden_compile_rate'] = scenarios['only_golden_compile_ok'] / scenarios['only_golden'] * 100
        scenarios['only_golden_solve_rate'] = scenarios['only_golden_solved'] / scenarios['only_golden'] * 100
    else:
        scenarios['only_golden_compile_rate'] = 0
        scenarios['only_golden_solve_rate'] = 0
    
    if scenarios['neither'] > 0:
        scenarios['neither_compile_rate'] = scenarios['neither_compile_ok'] / scenarios['neither'] * 100
        scenarios['neither_solve_rate'] = scenarios['neither_solved'] / scenarios['neither'] * 100
    else:
        scenarios['neither_compile_rate'] = 0
        scenarios['neither_solve_rate'] = 0
    
    return scenarios


def generate_latex_table_overall(stats, output_dir):
    """Generate LaTeX table for overall comparison."""
    
    agent = stats['agent']
    golden = stats['golden']
    tests = stats['statistical_tests']
    
    # Format p-values
    p_val_presence = '< 0.001' if tests['chi2_presence']['p_value'] < 0.001 else f"{tests['chi2_presence']['p_value']:.3f}"
    p_val_count = '< 0.001' if tests['mann_whitney_count']['p_value'] < 0.001 else f"{tests['mann_whitney_count']['p_value']:.3f}"
    
    latex = r"""\begin{table}[htbp]
  \centering
  \caption{Overall Refactoring Statistics: Agent vs Golden Patches}
  \label{tab:rq1_overall}
  \footnotesize
  \begin{tabular}{lrrr}
    \toprule
    \textbf{Metric} & \textbf{Agent Patches} & \textbf{Golden Patches} & \textbf{P-value} \\
    \midrule
    Total Patches & """ + f"{agent['total_patches']:,}" + r""" & """ + f"{golden['total_patches']:,}" + r""" & --- \\
    Patches with Refactoring & """ + f"{agent['with_refactoring']:,} ({agent['presence_rate']:.1f}\%)" + r""" & """ + f"{golden['with_refactoring']:,} ({golden['presence_rate']:.1f}\%)" + r""" & """ + p_val_presence + r"""$^a$ \\
    \midrule
    Total Refactoring Operations & """ + f"{agent['total_operations']:,}" + r""" & """ + f"{golden['total_operations']:,}" + r""" & --- \\
    Mean Operations per Patch & """ + f"{agent['mean_operations']:.3f}" + r""" & """ + f"{golden['mean_operations']:.3f}" + r""" & \multirow{2}{*}{""" + p_val_count + r"""$^b$} \\
    Median Operations per Patch & """ + f"{agent['median_operations']:.1f}" + r""" & """ + f"{golden['median_operations']:.1f}" + r""" & \\
    Std Dev & """ + f"{agent['std_operations']:.3f}" + r""" & """ + f"{golden['std_operations']:.3f}" + r""" & --- \\
    \midrule
    Unique Refactoring Types & """ + f"{agent['unique_types']}" + r""" & """ + f"{golden['unique_types']}" + r""" & --- \\
    \bottomrule
  \end{tabular}
  \begin{tablenotes}
    \footnotesize
    \item[$^a$] Chi-square test for refactoring presence
    \item[$^b$] Mann-Whitney U test for refactoring operation counts
  \end{tablenotes}
\end{table}
"""
    
    output_path = os.path.join(output_dir, 'latex_table_overall.tex')
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"Generated LaTeX table: {output_path}")
    return output_path


def generate_latex_table_categories(category_stats, output_dir):
    """Generate LaTeX table for category comparison."""
    
    agent_cat = category_stats['agent']
    golden_cat = category_stats['golden']
    
    # Calculate totals
    agent_total = sum(agent_cat['action'].values())
    golden_total = sum(golden_cat['action'].values())
    
    latex = r"""\begin{table}[htbp]
  \centering
  \caption{Refactoring Distribution by Action, Scope, and Test Impact}
  \label{tab:rq1_categories}
  \footnotesize
  \begin{tabular}{llrrrr}
    \toprule
    \textbf{Category} & \textbf{Type} & \multicolumn{2}{c}{\textbf{Agent}} & \multicolumn{2}{c}{\textbf{Golden}} \\
    \cmidrule(lr){3-4} \cmidrule(lr){5-6}
    & & Count & \% & Count & \% \\
    \midrule
"""
    
    # Action category
    latex += r"    \multirow{3}{*}{\textbf{Action}}"
    for i, action in enumerate(['Add', 'Remove', 'Adjust']):
        agent_count = agent_cat['action'].get(action, 0)
        golden_count = golden_cat['action'].get(action, 0)
        agent_pct = agent_count / agent_total * 100 if agent_total > 0 else 0
        golden_pct = golden_count / golden_total * 100 if golden_total > 0 else 0
        
        if i == 0:
            latex += f" & {action} & {agent_count:,} & {agent_pct:.1f} & {golden_count:,} & {golden_pct:.1f} \\\\\n"
        else:
            latex += f"    & {action} & {agent_count:,} & {agent_pct:.1f} & {golden_count:,} & {golden_pct:.1f} \\\\\n"
    
    latex += r"    \midrule" + "\n"
    
    # Scope category
    scopes = ['Class', 'Method', 'Local Variable']
    latex += r"    \multirow{3}{*}{\textbf{Scope}}"
    for i, scope in enumerate(scopes):
        agent_count = agent_cat['scope'].get(scope, 0)
        golden_count = golden_cat['scope'].get(scope, 0)
        agent_pct = agent_count / agent_total * 100 if agent_total > 0 else 0
        golden_pct = golden_count / golden_total * 100 if golden_total > 0 else 0
        
        if i == 0:
            latex += f" & {scope} & {agent_count:,} & {agent_pct:.1f} & {golden_count:,} & {golden_pct:.1f} \\\\\n"
        else:
            latex += f"    & {scope} & {agent_count:,} & {agent_pct:.1f} & {golden_count:,} & {golden_pct:.1f} \\\\\n"
    
    latex += r"    \midrule" + "\n"
    
    # Test Impact category
    latex += r"    \multirow{2}{*}{\textbf{Test Impact}}"
    for i, impact in enumerate(['Yes', 'No']):
        agent_count = agent_cat['test_impact'].get(impact, 0)
        golden_count = golden_cat['test_impact'].get(impact, 0)
        agent_pct = agent_count / agent_total * 100 if agent_total > 0 else 0
        golden_pct = golden_count / golden_total * 100 if golden_total > 0 else 0
        
        if i == 0:
            latex += f" & {impact} & {agent_count:,} & {agent_pct:.1f} & {golden_count:,} & {golden_pct:.1f} \\\\\n"
        else:
            latex += f"    & {impact} & {agent_count:,} & {agent_pct:.1f} & {golden_count:,} & {golden_pct:.1f} \\\\\n"
    
    latex += r"""    \bottomrule
  \end{tabular}
\end{table}
"""
    
    output_path = os.path.join(output_dir, 'latex_table_categories.tex')
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"Generated LaTeX table: {output_path}")
    return output_path


def generate_latex_table_paired(paired_stats, output_dir):
    """Generate LaTeX table for paired comparison."""
    
    total = sum([paired_stats['both_have'], paired_stats['only_agent'], 
                 paired_stats['only_golden'], paired_stats['neither']])
    
    latex = r"""\begin{table}[htbp]
  \centering
  \caption{Per-Issue Paired Comparison: Agent vs Golden Refactoring Behavior}
  \label{tab:rq1_paired}
  \footnotesize
  \begin{tabular}{lrr}
    \toprule
    \textbf{Scenario} & \textbf{Count} & \textbf{Percentage} \\
    \midrule
    Both Agent and Golden have refactoring & """ + f"{paired_stats['both_have']:,}" + r""" & """ + f"{paired_stats['both_have']/total*100:.1f}\%" + r""" \\
    Only Agent has refactoring & """ + f"{paired_stats['only_agent']:,}" + r""" & """ + f"{paired_stats['only_agent']/total*100:.1f}\%" + r""" \\
    Only Golden has refactoring & """ + f"{paired_stats['only_golden']:,}" + r""" & """ + f"{paired_stats['only_golden']/total*100:.1f}\%" + r""" \\
    Neither has refactoring & """ + f"{paired_stats['neither']:,}" + r""" & """ + f"{paired_stats['neither']/total*100:.1f}\%" + r""" \\
    \midrule
    \textbf{Total Agent Patches} & """ + f"{total:,}" + r""" & """ + r"""100.0\% \\
    \midrule
    \multicolumn{3}{l}{\textit{When both have refactoring:}} \\
    \quad Mean type overlap with Golden (\%) & """ + f"{paired_stats['mean_overlap_pct']:.1f}" + r""" & --- \\
    \quad Median type overlap with Golden (\%) & """ + f"{paired_stats['median_overlap_pct']:.1f}" + r""" & --- \\
    \midrule
    \multicolumn{3}{l}{\textit{Excess refactoring types:}} \\
    \quad Mean excess types per patch & """ + f"{paired_stats['mean_excess']:.2f}" + r""" & --- \\
    \quad Median excess types per patch & """ + f"{paired_stats['median_excess']:.1f}" + r""" & --- \\
    \bottomrule
  \end{tabular}
  \begin{tablenotes}
    \footnotesize
    \item Excess types: Refactoring types present in Agent patch but not in Golden patch
  \end{tablenotes}
\end{table}
"""
    
    output_path = os.path.join(output_dir, 'latex_table_paired.tex')
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"Generated LaTeX table: {output_path}")
    return output_path


def generate_top_types_comparison(agent_types, golden_types, output_dir, top_n=15):
    """Generate comparison of top refactoring types."""
    
    # Get all types and their counts
    all_types = set(agent_types.keys()) | set(golden_types.keys())
    type_comparison = []
    
    for ref_type in all_types:
        agent_count = agent_types.get(ref_type, 0)
        golden_count = golden_types.get(ref_type, 0)
        total = agent_count + golden_count
        type_comparison.append({
            'type': ref_type,
            'agent_count': agent_count,
            'golden_count': golden_count,
            'total': total,
            'agent_pct': agent_count / sum(agent_types.values()) * 100 if sum(agent_types.values()) > 0 else 0,
            'golden_pct': golden_count / sum(golden_types.values()) * 100 if sum(golden_types.values()) > 0 else 0
        })
    
    # Sort by total count
    type_comparison.sort(key=lambda x: -x['total'])
    
    # Save top N to CSV
    output_path = os.path.join(output_dir, 'top_refactoring_types_comparison.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'agent_count', 'agent_pct', 'golden_count', 'golden_pct', 'total'])
        writer.writeheader()
        for item in type_comparison[:top_n]:
            writer.writerow({
                'type': item['type'],
                'agent_count': item['agent_count'],
                'agent_pct': f"{item['agent_pct']:.2f}%",
                'golden_count': item['golden_count'],
                'golden_pct': f"{item['golden_pct']:.2f}%",
                'total': item['total']
            })
    
    print(f"Generated top types comparison: {output_path}")
    return type_comparison[:top_n]


def generate_summary_report(overall_stats, category_stats, paired_stats, output_dir):
    """Generate a comprehensive summary report."""
    
    report = f"""
================================================================================
RQ1 COMPREHENSIVE ANALYSIS SUMMARY
================================================================================

1. OVERALL COMPARISON: Agent vs Golden Patches
--------------------------------------------------------------------------------
Agent Patches:
  - Total patches: {overall_stats['agent']['total_patches']:,}
  - Patches with refactoring: {overall_stats['agent']['with_refactoring']:,} ({overall_stats['agent']['presence_rate']:.1f}%)
  - Total refactoring operations: {overall_stats['agent']['total_operations']:,}
  - Mean operations per patch: {overall_stats['agent']['mean_operations']:.3f}
  - Median operations per patch: {overall_stats['agent']['median_operations']:.1f}
  - Std dev: {overall_stats['agent']['std_operations']:.3f}
  - Unique refactoring types: {overall_stats['agent']['unique_types']}

Golden Patches:
  - Total patches: {overall_stats['golden']['total_patches']:,}
  - Patches with refactoring: {overall_stats['golden']['with_refactoring']:,} ({overall_stats['golden']['presence_rate']:.1f}%)
  - Total refactoring operations: {overall_stats['golden']['total_operations']:,}
  - Mean operations per patch: {overall_stats['golden']['mean_operations']:.3f}
  - Median operations per patch: {overall_stats['golden']['median_operations']:.1f}
  - Std dev: {overall_stats['golden']['std_operations']:.3f}
  - Unique refactoring types: {overall_stats['golden']['unique_types']}

Statistical Tests:
  - Chi-square test (presence): χ² = {overall_stats['statistical_tests']['chi2_presence']['chi2']:.3f}, 
    p-value = {overall_stats['statistical_tests']['chi2_presence']['p_value']:.6f}
    {'SIGNIFICANT' if overall_stats['statistical_tests']['chi2_presence']['significant'] else 'NOT SIGNIFICANT'} (α = 0.05)
  
  - Mann-Whitney U test (count): U = {overall_stats['statistical_tests']['mann_whitney_count']['u_stat']:.3f}, 
    p-value = {overall_stats['statistical_tests']['mann_whitney_count']['p_value']:.6f}
    {'SIGNIFICANT' if overall_stats['statistical_tests']['mann_whitney_count']['significant'] else 'NOT SIGNIFICANT'} (α = 0.05)


2. CATEGORY DISTRIBUTION COMPARISON
--------------------------------------------------------------------------------
Action Distribution:
"""
    
    agent_action_total = sum(category_stats['agent']['action'].values())
    golden_action_total = sum(category_stats['golden']['action'].values())
    
    for action in ['Add', 'Remove', 'Adjust']:
        agent_count = category_stats['agent']['action'].get(action, 0)
        golden_count = category_stats['golden']['action'].get(action, 0)
        agent_pct = agent_count / agent_action_total * 100 if agent_action_total > 0 else 0
        golden_pct = golden_count / golden_action_total * 100 if golden_action_total > 0 else 0
        report += f"  {action:10s}: Agent = {agent_count:4d} ({agent_pct:5.1f}%), Golden = {golden_count:4d} ({golden_pct:5.1f}%)\n"
    
    report += "\nScope Distribution:\n"
    for scope in ['Class', 'Method', 'Local Variable']:
        agent_count = category_stats['agent']['scope'].get(scope, 0)
        golden_count = category_stats['golden']['scope'].get(scope, 0)
        agent_pct = agent_count / agent_action_total * 100 if agent_action_total > 0 else 0
        golden_pct = golden_count / golden_action_total * 100 if golden_action_total > 0 else 0
        report += f"  {scope:15s}: Agent = {agent_count:4d} ({agent_pct:5.1f}%), Golden = {golden_count:4d} ({golden_pct:5.1f}%)\n"
    
    report += "\nTest Impact Distribution:\n"
    for impact in ['Yes', 'No']:
        agent_count = category_stats['agent']['test_impact'].get(impact, 0)
        golden_count = category_stats['golden']['test_impact'].get(impact, 0)
        agent_pct = agent_count / agent_action_total * 100 if agent_action_total > 0 else 0
        golden_pct = golden_count / golden_action_total * 100 if golden_action_total > 0 else 0
        report += f"  {impact:3s}:            Agent = {agent_count:4d} ({agent_pct:5.1f}%), Golden = {golden_count:4d} ({golden_pct:5.1f}%)\n"
    
    total_patches = sum([paired_stats['both_have'], paired_stats['only_agent'], 
                        paired_stats['only_golden'], paired_stats['neither']])
    
    report += f"""

3. PER-ISSUE PAIRED COMPARISON
--------------------------------------------------------------------------------
Total agent patches analyzed: {total_patches:,}

Refactoring Presence Scenarios:
  - Both Agent and Golden have refactoring: {paired_stats['both_have']:,} ({paired_stats['both_have']/total_patches*100:.1f}%)
  - Only Agent has refactoring:            {paired_stats['only_agent']:,} ({paired_stats['only_agent']/total_patches*100:.1f}%)
  - Only Golden has refactoring:           {paired_stats['only_golden']:,} ({paired_stats['only_golden']/total_patches*100:.1f}%)
  - Neither has refactoring:               {paired_stats['neither']:,} ({paired_stats['neither']/total_patches*100:.1f}%)

Performance by Refactoring Scenario:
  
  Both Agent and Golden have refactoring:
    - Compilation success: {paired_stats['both_have_compile_ok']:,} / {paired_stats['both_have']:,} ({paired_stats['both_have_compile_rate']:.1f}%)
    - Issue resolved:      {paired_stats['both_have_solved']:,} / {paired_stats['both_have']:,} ({paired_stats['both_have_solve_rate']:.1f}%)
  
  Only Agent has refactoring:
    - Compilation success: {paired_stats['only_agent_compile_ok']:,} / {paired_stats['only_agent']:,} ({paired_stats['only_agent_compile_rate']:.1f}%)
    - Issue resolved:      {paired_stats['only_agent_solved']:,} / {paired_stats['only_agent']:,} ({paired_stats['only_agent_solve_rate']:.1f}%)
  
  Only Golden has refactoring:
    - Compilation success: {paired_stats['only_golden_compile_ok']:,} / {paired_stats['only_golden']:,} ({paired_stats['only_golden_compile_rate']:.1f}%)
    - Issue resolved:      {paired_stats['only_golden_solved']:,} / {paired_stats['only_golden']:,} ({paired_stats['only_golden_solve_rate']:.1f}%)
  
  Neither has refactoring:
    - Compilation success: {paired_stats['neither_compile_ok']:,} / {paired_stats['neither']:,} ({paired_stats['neither_compile_rate']:.1f}%)
    - Issue resolved:      {paired_stats['neither_solved']:,} / {paired_stats['neither']:,} ({paired_stats['neither_solve_rate']:.1f}%)

Type Overlap Analysis (when both have refactoring):
  - Mean overlap with Golden types: {paired_stats['mean_overlap_pct']:.1f}%
  - Median overlap with Golden types: {paired_stats['median_overlap_pct']:.1f}%

Excess Refactoring Analysis:
  - Mean excess types per patch: {paired_stats['mean_excess']:.2f}
  - Median excess types per patch: {paired_stats['median_excess']:.1f}

KEY FINDINGS:
  - {paired_stats['only_agent']/total_patches*100:.1f}% of agent patches contain refactoring NOT present in golden patches
  - {paired_stats['only_golden']/total_patches*100:.1f}% of agent patches miss refactoring that IS present in golden patches


================================================================================
"""
    
    output_path = os.path.join(output_dir, 'RQ1_comprehensive_summary.txt')
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Generated summary report: {output_path}")
    print("\n" + report)
    
    return report


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    csv_path = os.path.join(project_root, 'data', 'unified_data.csv')
    excel_path = os.path.join(project_root, 'data', 'refactoring_classification.xlsx')
    output_dir = os.path.join(project_root, 'output', 'RQ1')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("RQ1 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n[1/7] Loading refactoring classification...")
    classification = load_refactoring_classification(excel_path)
    print(f"  Loaded {len(classification)} refactoring types with classifications")
    
    print("\n[2/7] Loading unified dataset...")
    data = load_unified_dataset(csv_path)
    print(f"  Loaded {len(data)} agent patches")
    
    # Calculate overall statistics
    print("\n[3/7] Calculating overall statistics...")
    overall_stats, agent_types_dict, golden_types_dict = calculate_overall_statistics(data, classification)
    
    # Calculate category statistics
    print("\n[4/7] Calculating category statistics...")
    category_stats = calculate_category_statistics(
        agent_types_dict['refactoring_types'],
        golden_types_dict['refactoring_types'],
        classification
    )
    
    # Calculate paired comparison
    print("\n[5/7] Calculating per-issue paired comparison...")
    paired_stats = calculate_paired_comparison(data)
    
    # Generate top types comparison
    print("\n[6/7] Generating top refactoring types comparison...")
    top_types = generate_top_types_comparison(
        agent_types_dict['refactoring_types'],
        golden_types_dict['refactoring_types'],
        output_dir,
        top_n=15
    )
    
    # Generate LaTeX tables
    print("\n[7/7] Generating LaTeX tables and summary report...")
    generate_latex_table_overall(overall_stats, output_dir)
    generate_latex_table_categories(category_stats, output_dir)
    generate_latex_table_paired(paired_stats, output_dir)
    
    # Generate summary report
    generate_summary_report(overall_stats, category_stats, paired_stats, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - RQ1_comprehensive_summary.txt")
    print("  - latex_table_overall.tex")
    print("  - latex_table_categories.tex")
    print("  - latex_table_paired.tex")
    print("  - top_refactoring_types_comparison.csv")


if __name__ == "__main__":
    main()
