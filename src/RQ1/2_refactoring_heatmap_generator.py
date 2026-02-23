"""
Refactoring Type Heatmap Generator - Paper Version
Statistics: Count how many patches contain each refactoring type (at least once)
Color theme: Red
"""

import csv
import json
import os
from collections import defaultdict
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def parse_refactoring_dict(refactoring_str):
    """Parse refactoring type count from JSON string."""
    if not refactoring_str or refactoring_str == '{}':
        return {}
    try:
        return json.loads(refactoring_str)
    except json.JSONDecodeError:
        return {}


def load_dataset_presence_based(csv_path):
    """
    Load dataset and count how many patches contain each refactoring type for each LLM model
    Also collect Human (golden patch) statistics
    (Count presence, not total occurrences)
    Note: Human stats are deduplicated by instance_id since golden patches are the same for all LLMs
    """
    llm_stats = defaultdict(lambda: {
        'total_patches': 0,
        'patches_with_refactoring': 0,
        'refactoring_type_presence': defaultdict(int),  # 记录每个类型出现在多少个 patch 中
    })
    
    # Add Human statistics - use set to track unique instances
    human_instances = set()  # Track unique instance_ids
    human_refactoring_instances = set()  # Track instances with refactoring
    human_refactoring_types = defaultdict(set)  # Track which instances have each refactoring type
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            llm_model = row.get('llm_model', '')
            if not llm_model:
                continue
            
            instance_id = row.get('instance_id', '')
            
            stats = llm_stats[llm_model]
            stats['total_patches'] += 1
            
            # Parse agent refactoring (handle both '1'/'0' and 'True'/'False')
            has_refactoring_raw = row.get('agent_has_refactoring', '')
            has_refactoring = has_refactoring_raw in ('1', 'True')
            if has_refactoring:
                stats['patches_with_refactoring'] += 1
            
            # 统计每个 refactoring 类型是否出现在这个 patch 中（不管出现几次）
            agent_ref = parse_refactoring_dict(row.get('agent_refactoring_type_count', '{}'))
            for ref_type in agent_ref.keys():  # 只要出现就记录，不管次数
                stats['refactoring_type_presence'][ref_type] += 1
            
            # Collect Human (golden) stats - deduplicate by instance_id
            human_instances.add(instance_id)
            
            golden_has_refactoring_raw = row.get('golden_has_refactoring', '')
            golden_has_refactoring = golden_has_refactoring_raw in ('1', 'True')
            if golden_has_refactoring:
                human_refactoring_instances.add(instance_id)
            
            golden_ref = parse_refactoring_dict(row.get('golden_refactoring_type_count', '{}'))
            for ref_type in golden_ref.keys():
                human_refactoring_types[ref_type].add(instance_id)
    
    # Build Human stats from deduplicated data
    human_stats = {
        'total_patches': len(human_instances),
        'patches_with_refactoring': len(human_refactoring_instances),
        'refactoring_type_presence': {ref_type: len(instances) 
                                     for ref_type, instances in human_refactoring_types.items()},
    }
    
    # Add Human to the results
    result = dict(llm_stats)
    result['Human'] = human_stats
    
    return result


def load_dataset_by_framework(csv_path):
    """
    Load dataset and count how many patches contain each refactoring type for each framework
    Also collect Human (golden patch) statistics
    (Count presence, not total occurrences)
    Note: Human stats are deduplicated by instance_id since golden patches are the same for all agents
    """
    framework_stats = defaultdict(lambda: {
        'total_patches': 0,
        'patches_with_refactoring': 0,
        'refactoring_type_presence': defaultdict(int),  # 记录每个类型出现在多少个 patch 中
    })
    
    # Add Human statistics - use set to track unique instances
    human_instances = set()  # Track unique instance_ids
    human_refactoring_instances = set()  # Track instances with refactoring
    human_refactoring_types = defaultdict(set)  # Track which instances have each refactoring type
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            framework = row.get('agent_framework', '')
            if not framework:
                continue
            
            instance_id = row.get('instance_id', '')
            
            # Collect agent framework stats
            stats = framework_stats[framework]
            stats['total_patches'] += 1
            
            # Parse agent refactoring (handle both '1'/'0' and 'True'/'False')
            has_refactoring_raw = row.get('agent_has_refactoring', '')
            has_refactoring = has_refactoring_raw in ('1', 'True')
            if has_refactoring:
                stats['patches_with_refactoring'] += 1
            
            # 统计每个 refactoring 类型是否出现在这个 patch 中（不管出现几次）
            agent_ref = parse_refactoring_dict(row.get('agent_refactoring_type_count', '{}'))
            for ref_type in agent_ref.keys():  # 只要出现就记录，不管次数
                stats['refactoring_type_presence'][ref_type] += 1
            
            # Collect Human (golden) stats - deduplicate by instance_id
            human_instances.add(instance_id)
            
            golden_has_refactoring_raw = row.get('golden_has_refactoring', '')
            golden_has_refactoring = golden_has_refactoring_raw in ('1', 'True')
            if golden_has_refactoring:
                human_refactoring_instances.add(instance_id)
            
            golden_ref = parse_refactoring_dict(row.get('golden_refactoring_type_count', '{}'))
            for ref_type in golden_ref.keys():
                human_refactoring_types[ref_type].add(instance_id)
    
    # Build Human stats from deduplicated data
    human_stats = {
        'total_patches': len(human_instances),
        'patches_with_refactoring': len(human_refactoring_instances),
        'refactoring_type_presence': {ref_type: len(instances) 
                                     for ref_type, instances in human_refactoring_types.items()},
    }
    
    # Add Human to the results
    result = dict(framework_stats)
    result['Human'] = human_stats
    
    return result


def plot_refactoring_types_heatmap(data, output_dir, top_n=15):
    """
    Create heatmap showing top N refactoring types in each LLM model's patches
    Using red color theme
    Human (golden patch) data is shown at the bottom for comparison
    """
    # Get all refactoring types and their total presence counts across all LLMs
    all_types = defaultdict(int)
    for llm, stats in data.items():
        for ref_type, presence_count in stats['refactoring_type_presence'].items():
            all_types[ref_type] += presence_count
    
    # Get top N types
    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    top_type_names = [t[0] for t in top_types]
    
    # Separate Human from LLM models
    llm_models = {k: v for k, v in data.items() if k != 'Human'}
    human_data = data.get('Human', None)
    
    # Sort LLMs by total refactoring presence
    llm_totals = [(llm, sum(stats['refactoring_type_presence'].values())) 
                  for llm, stats in llm_models.items()]
    llm_totals.sort(key=lambda x: -x[1])
    llms = [x[0] for x in llm_totals]
    
    # Add Human at the end
    if human_data:
        llms.append('Human')
    
    # Build matrix: rows are LLMs + Human, columns are refactoring types
    matrix = np.zeros((len(llms), len(top_type_names)))
    for i, llm in enumerate(llms):
        if llm == 'Human':
            source_data = human_data
        else:
            source_data = llm_models[llm]
        
        for j, ref_type in enumerate(top_type_names):
            matrix[i, j] = source_data['refactoring_type_presence'].get(ref_type, 0)
    
    # Create heatmap with red theme - adjust height for additional row
    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    # Use Reds colormap (red theme)
    im = ax.imshow(matrix, cmap='Reds', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=8)
    
    # Set axis labels
    ax.set_xticks(np.arange(len(top_type_names)))
    ax.set_yticks(np.arange(len(llms)))
    ax.set_xticklabels(top_type_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(llms, fontsize=8)
    
    # Add value annotations in each cell
    for i in range(len(llms)):
        for j in range(len(top_type_names)):
            value = int(matrix[i, j])
            if value > 0:
                # Adjust text color based on background intensity
                text_color = 'white' if value > matrix.max() * 0.5 else '#8B0000'
                ax.text(j, i, str(value), ha='center', va='center', 
                       color=text_color, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'refactoring_heatmap_top10_presence_red.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Heatmap saved: {output_path}")
    return output_path


def plot_refactoring_types_heatmap_by_framework(data, output_dir, top_n=15):
    """
    Create heatmap showing top N refactoring types in each framework's patches
    Using red color theme
    Human (golden patch) data is shown at the bottom for comparison
    """
    # Separate Human from agent frameworks
    agent_frameworks = {k: v for k, v in data.items() if k != 'Human'}
    human_data = data.get('Human', None)
    
    # Get all refactoring types and their total presence counts across agent frameworks only
    all_types = defaultdict(int)
    for framework, stats in agent_frameworks.items():
        for ref_type, presence_count in stats['refactoring_type_presence'].items():
            all_types[ref_type] += presence_count
    
    # Get top N types
    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    top_type_names = [t[0] for t in top_types]
    
    # Sort agent frameworks by total refactoring presence
    framework_totals = [(framework, sum(stats['refactoring_type_presence'].values())) 
                       for framework, stats in agent_frameworks.items()]
    framework_totals.sort(key=lambda x: -x[1])
    frameworks = [x[0] for x in framework_totals]
    
    # Add Human at the end
    if human_data:
        frameworks = frameworks + ['Human']
    
    # Build matrix: rows are frameworks (+ Human), columns are refactoring types
    matrix = np.zeros((len(frameworks), len(top_type_names)))
    for i, framework in enumerate(frameworks):
        if framework == 'Human':
            source_data = human_data
        else:
            source_data = agent_frameworks[framework]
        for j, ref_type in enumerate(top_type_names):
            matrix[i, j] = source_data['refactoring_type_presence'].get(ref_type, 0)
    
    # Create heatmap with red theme - height accounts for frameworks + Human row
    n_rows = len(frameworks)
    fig_height = max(4.0, 1.0 + n_rows * 1.0)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Use Reds colormap (red theme)
    im = ax.imshow(matrix, cmap='Reds', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=8)
    
    # Set axis labels
    ax.set_xticks(np.arange(len(top_type_names)))
    ax.set_yticks(np.arange(len(frameworks)))
    ax.set_xticklabels(top_type_names, rotation=45, ha='right', fontsize=8)
    # Remove leading 'M' from framework names and capitalize first letter; Human stays as "Human"
    framework_labels = []
    for f in frameworks:
        if f == 'Human':
            framework_labels.append('Human')
        elif f.startswith('M'):
            new_name = f[1:]  # Remove 'M'
            new_name = new_name[0].upper() + new_name[1:] if new_name else ''  # Capitalize first letter
            framework_labels.append(new_name)
        else:
            framework_labels.append(f)
    ax.set_yticklabels(framework_labels, fontsize=9)
    
    # Add value annotations in each cell
    for i in range(len(frameworks)):
        for j in range(len(top_type_names)):
            value = int(matrix[i, j])
            if value > 0:
                # Adjust text color based on background intensity
                text_color = 'white' if value > matrix.max() * 0.5 else '#8B0000'
                ax.text(j, i, str(value), ha='center', va='center', 
                       color=text_color, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'refactoring_heatmap_framework_top10_presence_red.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Framework heatmap saved: {output_path}")
    return output_path


def generate_complete_statistics_csv(data, output_dir, top_n=15):
    """
    Generate complete statistics CSV file
    Contains detailed data for all LLM models and top N refactoring types
    Human (golden) data is shown at the end
    """
    # Get all refactoring types and their total presence counts
    all_types = defaultdict(int)
    for llm, stats in data.items():
        for ref_type, presence_count in stats['refactoring_type_presence'].items():
            all_types[ref_type] += presence_count
    
    # Get top N types
    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    top_type_names = [t[0] for t in top_types]
    
    # Separate Human from LLM models
    llm_models = {k: v for k, v in data.items() if k != 'Human'}
    human_data = data.get('Human', None)
    
    # Sort LLMs by total presence count
    llm_totals = [(llm, sum(stats['refactoring_type_presence'].values()), 
                   stats['total_patches'], stats['patches_with_refactoring']) 
                  for llm, stats in llm_models.items()]
    llm_totals.sort(key=lambda x: -x[1])
    
    # Add Human at the end if available
    if human_data:
        human_total_presence = sum(human_data['refactoring_type_presence'].values())
        llm_totals.append(('Human', human_total_presence, 
                          human_data['total_patches'], 
                          human_data['patches_with_refactoring']))
    
    # Generate detailed statistics table
    output_path = os.path.join(output_dir, 'refactoring_statistics_top10_presence.csv')
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Build header
        fieldnames = ['LLM_Model', 'Total_Patches', 'Patches_With_Refactoring', 'Refactoring_Rate(%)'] + top_type_names + ['Total_Presence']
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        
        # Write data for each LLM
        for llm, total_presence, total_patches, patches_with_ref in llm_totals:
            stats = data[llm]
            ref_rate = patches_with_ref / total_patches * 100 if total_patches > 0 else 0
            
            row = [
                llm,
                total_patches,
                patches_with_ref,
                f'{ref_rate:.2f}%'
            ]
            
            # Add presence count for each refactoring type
            for ref_type in top_type_names:
                presence_count = stats['refactoring_type_presence'].get(ref_type, 0)
                row.append(presence_count)
            
            # Add total presence count
            row.append(total_presence)
            
            writer.writerow(row)
        
        # Add summary rows (excluding Human from summary)
        writer.writerow([])
        writer.writerow(['Summary Statistics (LLM Models Only)'])
        
        # Calculate total presence count for each refactoring type (excluding Human)
        summary_row = ['All LLMs Total', '', '', '']
        for ref_type in top_type_names:
            total = sum(llm_models[llm]['refactoring_type_presence'].get(ref_type, 0) 
                       for llm in llm_models.keys())
            summary_row.append(total)
        summary_row.append('')
        writer.writerow(summary_row)
    
    print(f"✓ Statistics CSV saved: {output_path}")
    return output_path


def generate_framework_statistics_csv(data, output_dir, top_n=15):
    """
    Generate complete statistics CSV file for frameworks
    Contains detailed data for all frameworks and top N refactoring types
    Human (golden) data is shown at the end
    """
    # Get all refactoring types and their total presence counts
    all_types = defaultdict(int)
    for framework, stats in data.items():
        for ref_type, presence_count in stats['refactoring_type_presence'].items():
            all_types[ref_type] += presence_count
    
    # Get top N types
    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    top_type_names = [t[0] for t in top_types]
    
    # Separate Human from agent frameworks
    agent_frameworks = {k: v for k, v in data.items() if k != 'Human'}
    human_data = data.get('Human', None)
    
    # Sort agent frameworks by total presence count
    framework_totals = [(framework, sum(stats['refactoring_type_presence'].values()), 
                        stats['total_patches'], stats['patches_with_refactoring']) 
                       for framework, stats in agent_frameworks.items()]
    framework_totals.sort(key=lambda x: -x[1])
    
    # Add Human at the end if available
    if human_data:
        human_total_presence = sum(human_data['refactoring_type_presence'].values())
        framework_totals.append(('Human', human_total_presence, 
                                human_data['total_patches'], 
                                human_data['patches_with_refactoring']))
    
    # Generate detailed statistics table
    output_path = os.path.join(output_dir, 'refactoring_statistics_framework_top10_presence.csv')
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Build header
        fieldnames = ['Framework', 'Total_Patches', 'Patches_With_Refactoring', 'Refactoring_Rate(%)'] + top_type_names + ['Total_Presence']
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        
        # Write data for each framework
        for framework, total_presence, total_patches, patches_with_ref in framework_totals:
            stats = data[framework]
            ref_rate = patches_with_ref / total_patches * 100 if total_patches > 0 else 0
            
            row = [
                framework,
                total_patches,
                patches_with_ref,
                f'{ref_rate:.2f}%'
            ]
            
            # Add presence count for each refactoring type
            for ref_type in top_type_names:
                presence_count = stats['refactoring_type_presence'].get(ref_type, 0)
                row.append(presence_count)
            
            # Add total presence count
            row.append(total_presence)
            
            writer.writerow(row)
        
        # Add summary rows (excluding Human from summary)
        writer.writerow([])
        writer.writerow(['Summary Statistics (Agent Frameworks Only)'])
        
        # Calculate total presence count for each refactoring type (excluding Human)
        summary_row = ['All Agent Frameworks Total', '', '', '']
        for ref_type in top_type_names:
            total = sum(agent_frameworks[framework]['refactoring_type_presence'].get(ref_type, 0) 
                       for framework in agent_frameworks.keys())
            summary_row.append(total)
        summary_row.append('')
        writer.writerow(summary_row)
    
    print(f"✓ Framework statistics CSV saved: {output_path}")
    return output_path


def generate_summary_report(data, output_dir, top_n=15):
    """Generate summary report (including Human)"""
    # Get all refactoring types and their total presence counts
    all_types = defaultdict(int)
    for llm, stats in data.items():
        for ref_type, presence_count in stats['refactoring_type_presence'].items():
            all_types[ref_type] += presence_count
    
    # Get top N types
    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    
    # Separate Human from LLM models
    llm_models = {k: v for k, v in data.items() if k != 'Human'}
    human_data = data.get('Human', None)
    
    num_llms = len(llm_models)
    if human_data:
        num_llms_text = f"{num_llms} LLM models (+ Human for comparison)"
    else:
        num_llms_text = f"{num_llms}"
    
    report = f"""
================================================================================
Refactoring Type Heatmap Statistics Report - Top {top_n}
================================================================================

Methodology:
  - Count how many patches contain each refactoring type (at least once)
  - Does not count total occurrences, only presence
  - Example: If a refactoring appears 5 times in one patch, it counts as 1

Data Overview:
  - Number of LLM models analyzed: {num_llms_text}
  - Number of top refactoring types: {top_n}
  - Total number of refactoring types: {len(all_types)}

Statistics by LLM Model:
"""
    
    # Sort LLM models by total presence count
    llm_totals = [(llm, sum(stats['refactoring_type_presence'].values()), 
                   stats['total_patches'], stats['patches_with_refactoring']) 
                  for llm, stats in llm_models.items()]
    llm_totals.sort(key=lambda x: -x[1])
    
    # Add Human at the end
    if human_data:
        human_total = sum(human_data['refactoring_type_presence'].values())
        llm_totals.append(('Human', human_total, 
                          human_data['total_patches'], 
                          human_data['patches_with_refactoring']))
    
    for i, (llm, total_presence, total_patches, patches_with_ref) in enumerate(llm_totals, 1):
        stats = data[llm]
        ref_rate = patches_with_ref / total_patches * 100 if total_patches > 0 else 0
        unique_types = len(stats['refactoring_type_presence'])
        
        marker = " (Golden Patch - for comparison)" if llm == 'Human' else ""
        
        report += f"""
{i}. {llm}{marker}
   - Total patches: {total_patches:,}
   - Patches with refactoring: {patches_with_ref:,} ({ref_rate:.1f}%)
   - Number of refactoring types involved: {unique_types}
   - Total presence count: {total_presence:,}
"""
    
    report += f"""

Top {top_n} Refactoring Types (sorted by total presence):
"""
    
    for i, (ref_type, total_count) in enumerate(top_types, 1):
        report += f"\n{i:2d}. {ref_type:40s} - Present in {total_count:,} patches"
    
    report += "\n\n" + "=" * 80 + "\n"
    
    # Save report
    output_path = os.path.join(output_dir, 'refactoring_statistics_summary.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Summary report saved: {output_path}")
    print(report)
    
    return output_path


def generate_framework_summary_report(data, output_dir, top_n=15):
    """Generate summary report for frameworks (including Human)"""
    # Get all refactoring types and their total presence counts
    all_types = defaultdict(int)
    for framework, stats in data.items():
        for ref_type, presence_count in stats['refactoring_type_presence'].items():
            all_types[ref_type] += presence_count
    
    # Get top N types
    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    
    # Separate Human from agent frameworks
    agent_frameworks = {k: v for k, v in data.items() if k != 'Human'}
    human_data = data.get('Human', None)
    
    num_frameworks = len(agent_frameworks)
    if human_data:
        num_frameworks_text = f"{num_frameworks} agent frameworks (+ Human for comparison)"
    else:
        num_frameworks_text = f"{num_frameworks}"
    
    report = f"""
================================================================================
Refactoring Type Heatmap Statistics Report (by Framework) - Top {top_n}
================================================================================

Methodology:
  - Count how many patches contain each refactoring type (at least once)
  - Does not count total occurrences, only presence
  - Example: If a refactoring appears 5 times in one patch, it counts as 1

Data Overview:
  - Number of frameworks analyzed: {num_frameworks_text}
  - Number of top refactoring types: {top_n}
  - Total number of refactoring types: {len(all_types)}

Statistics by Framework:
"""
    
    # Sort agent frameworks by total presence count
    framework_totals = [(framework, sum(stats['refactoring_type_presence'].values()), 
                        stats['total_patches'], stats['patches_with_refactoring']) 
                       for framework, stats in agent_frameworks.items()]
    framework_totals.sort(key=lambda x: -x[1])
    
    # Add Human at the end
    if human_data:
        human_total = sum(human_data['refactoring_type_presence'].values())
        framework_totals.append(('Human', human_total, 
                                human_data['total_patches'], 
                                human_data['patches_with_refactoring']))
    
    for i, (framework, total_presence, total_patches, patches_with_ref) in enumerate(framework_totals, 1):
        stats = data[framework]
        ref_rate = patches_with_ref / total_patches * 100 if total_patches > 0 else 0
        unique_types = len(stats['refactoring_type_presence'])
        
        marker = " (Golden Patch - for comparison)" if framework == 'Human' else ""
        
        report += f"""
{i}. {framework}{marker}
   - Total patches: {total_patches:,}
   - Patches with refactoring: {patches_with_ref:,} ({ref_rate:.1f}%)
   - Number of refactoring types involved: {unique_types}
   - Total presence count: {total_presence:,}
"""
    
    report += f"""

Top {top_n} Refactoring Types (sorted by total presence):
"""
    
    for i, (ref_type, total_count) in enumerate(top_types, 1):
        report += f"\n{i:2d}. {ref_type:40s} - Present in {total_count:,} patches"
    
    report += "\n\n" + "=" * 80 + "\n"
    
    # Save report
    output_path = os.path.join(output_dir, 'refactoring_statistics_framework_summary.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Framework summary report saved: {output_path}")
    print(report)
    
    return output_path


def main():
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    csv_path = os.path.join(project_root, 'data', 'unified_data.csv')
    output_dir = os.path.join(project_root, 'output', 'RQ1', 'paper')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Refactoring Type Heatmap Generator - Paper Version")
    print("=" * 80)
    print(f"\nData file: {csv_path}")
    print(f"Output directory: {output_dir}\n")
    
    # === Part 1: LLM Model Statistics ===
    print("\n" + "=" * 80)
    print("PART 1: LLM Model Statistics")
    print("=" * 80)
    
    # Load data (presence-based statistics)
    print("\n[1/4] Loading dataset by LLM model (counting refactoring type presence)...")
    llm_data = load_dataset_presence_based(csv_path)
    print(f"      Loaded: {len(llm_data)} LLM models")
    
    # Generate heatmap
    print("\n[2/4] Generating Top 10 refactoring types heatmap (red theme)...")
    plot_refactoring_types_heatmap(llm_data, output_dir, top_n=10)
    
    # Generate complete statistics CSV
    print("\n[3/4] Generating complete statistics CSV...")
    generate_complete_statistics_csv(llm_data, output_dir, top_n=10)
    
    # Generate summary report
    print("\n[4/4] Generating summary report...")
    generate_summary_report(llm_data, output_dir, top_n=10)
    
    # === Part 2: Framework Statistics ===
    print("\n" + "=" * 80)
    print("PART 2: Framework Statistics")
    print("=" * 80)
    
    # Load data by framework
    print("\n[1/4] Loading dataset by framework (counting refactoring type presence)...")
    framework_data = load_dataset_by_framework(csv_path)
    print(f"      Loaded: {len(framework_data)} frameworks")
    
    # Generate framework heatmap
    print("\n[2/4] Generating framework heatmap (red theme)...")
    plot_refactoring_types_heatmap_by_framework(framework_data, output_dir, top_n=10)
    
    # Generate framework statistics CSV
    print("\n[3/4] Generating framework statistics CSV...")
    generate_framework_statistics_csv(framework_data, output_dir, top_n=10)
    
    # Generate framework summary report
    print("\n[4/4] Generating framework summary report...")
    generate_framework_summary_report(framework_data, output_dir, top_n=10)
    
    # === Summary ===
    print("\n" + "=" * 80)
    print("✓ All files generated successfully!")
    print("=" * 80)
    print(f"\nGenerated files (LLM Models):")
    print(f"  1. Heatmap: {output_dir}/refactoring_heatmap_top10_presence_red.png")
    print(f"  2. Statistics CSV: {output_dir}/refactoring_statistics_top10_presence.csv")
    print(f"  3. Summary report: {output_dir}/refactoring_statistics_summary.txt")
    print(f"\nGenerated files (Frameworks):")
    print(f"  4. Heatmap: {output_dir}/refactoring_heatmap_framework_top10_presence_red.png")
    print(f"  5. Statistics CSV: {output_dir}/refactoring_statistics_framework_top10_presence.csv")
    print(f"  6. Summary report: {output_dir}/refactoring_statistics_framework_summary.txt")
    print()


if __name__ == "__main__":
    main()
