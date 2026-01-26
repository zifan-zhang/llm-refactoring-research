"""
RQ1 Visualization: Generate figures for comprehensive analysis
Creates publication-ready figures for RQ1 results section
"""

import csv
import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Define consistent color palette
BLUE_COLORS = {
    'agent': '#3B6FA0',      # Medium blue for Agent
    'golden': '#1E3A5F',     # Deep navy for Golden
    'both': '#6B9FD4',       # Light blue
    'only_agent': '#A8CCE8', # Pale blue
    'only_golden': '#2B5278',# Navy blue
    'neither': '#E8E8E8'     # Light gray
}


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
            # Handle both numeric (0/1) and string ('True'/'False') formats
            agent_has_ref_raw = row.get('agent_has_refactoring', '')
            golden_has_ref_raw = row.get('golden_has_refactoring', '')
            
            data.append({
                'instance_id': row.get('instance_id', ''),
                'agent_has_refactoring': agent_has_ref_raw in ('1', 'True'),
                'agent_refactoring_types': parse_refactoring_dict(row.get('agent_refactoring_type_count', '{}')),
                'golden_has_refactoring': golden_has_ref_raw in ('1', 'True'),
                'golden_refactoring_types': parse_refactoring_dict(row.get('golden_refactoring_type_count', '{}')),
            })
    return data


def plot_presence_comparison(data, output_dir):
    """Create pie charts comparing refactoring presence."""
    
    # Calculate statistics
    agent_with = sum(1 for row in data if row['agent_has_refactoring'])
    agent_without = len(data) - agent_with
    
    golden_by_instance = {}
    for row in data:
        instance_id = row['instance_id']
        if instance_id not in golden_by_instance:
            golden_by_instance[instance_id] = row['golden_has_refactoring']
    
    golden_with = sum(1 for v in golden_by_instance.values() if v)
    golden_without = len(golden_by_instance) - golden_with
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Agent patches
    ax1 = axes[0]
    sizes1 = [agent_with, agent_without]
    labels1 = ['With Refactoring', 'Without Refactoring']
    colors1 = [BLUE_COLORS['agent'], BLUE_COLORS['neither']]
    explode1 = (0.05, 0)
    
    wedges1, texts1, autotexts1 = ax1.pie(
        sizes1, labels=labels1, autopct='%1.1f%%',
        colors=colors1, explode=explode1, startangle=90,
        textprops={'fontsize': 13, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    for autotext in autotexts1:
        autotext.set_color('white')
    
    ax1.set_title(f'Agent Patches\n(n={len(data):,})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Golden patches
    ax2 = axes[1]
    sizes2 = [golden_with, golden_without]
    labels2 = ['With Refactoring', 'Without Refactoring']
    colors2 = [BLUE_COLORS['golden'], BLUE_COLORS['neither']]
    explode2 = (0.05, 0)
    
    wedges2, texts2, autotexts2 = ax2.pie(
        sizes2, labels=labels2, autopct='%1.1f%%',
        colors=colors2, explode=explode2, startangle=90,
        textprops={'fontsize': 13, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    for autotext in autotexts2:
        autotext.set_color('white')
    
    ax2.set_title(f'Golden Patches (Deduplicated)\n(n={len(golden_by_instance):,})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Refactoring Presence: Agent vs Golden Patches', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_presence_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Generated: {output_path}")
    return output_path


def plot_category_comparison(data, classification, output_dir):
    """Create grouped bar charts for category comparison."""
    
    # Get refactoring types
    agent_types = defaultdict(int)
    golden_types = defaultdict(int)
    
    seen_instances = set()
    for row in data:
        for ref_type, count in row['agent_refactoring_types'].items():
            agent_types[ref_type] += count
        
        instance_id = row['instance_id']
        if instance_id not in seen_instances:
            seen_instances.add(instance_id)
            for ref_type, count in row['golden_refactoring_types'].items():
                golden_types[ref_type] += count
    
    # Categorize
    def categorize_refactorings(ref_types_dict):
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
    
    agent_cat = categorize_refactorings(agent_types)
    golden_cat = categorize_refactorings(golden_types)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Action
    ax1 = axes[0]
    actions = ['Add', 'Remove', 'Adjust']
    agent_action_counts = [agent_cat['action'][a] for a in actions]
    golden_action_counts = [golden_cat['action'][a] for a in actions]
    
    x = np.arange(len(actions))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, agent_action_counts, width, 
                    label='Agent', color=BLUE_COLORS['agent'], edgecolor='white')
    bars2 = ax1.bar(x + width/2, golden_action_counts, width,
                    label='Golden', color=BLUE_COLORS['golden'], edgecolor='white')
    
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Action Type Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(actions, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    # Scope
    ax2 = axes[1]
    scopes = ['Class', 'Method', 'Local Variable']
    agent_scope_counts = [agent_cat['scope'][s] for s in scopes]
    golden_scope_counts = [golden_cat['scope'][s] for s in scopes]
    
    x = np.arange(len(scopes))
    
    bars1 = ax2.bar(x - width/2, agent_scope_counts, width,
                    label='Agent', color=BLUE_COLORS['agent'], edgecolor='white')
    bars2 = ax2.bar(x + width/2, golden_scope_counts, width,
                    label='Golden', color=BLUE_COLORS['golden'], edgecolor='white')
    
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Scope Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scopes, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    # Test Impact
    ax3 = axes[2]
    impacts = ['Yes', 'No']
    agent_impact_counts = [agent_cat['test_impact'][i] for i in impacts]
    golden_impact_counts = [golden_cat['test_impact'][i] for i in impacts]
    
    x = np.arange(len(impacts))
    
    bars1 = ax3.bar(x - width/2, agent_impact_counts, width,
                    label='Agent', color=BLUE_COLORS['agent'], edgecolor='white')
    bars2 = ax3.bar(x + width/2, golden_impact_counts, width,
                    label='Golden', color=BLUE_COLORS['golden'], edgecolor='white')
    
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Test Interface Impact', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Affects Test', 'No Impact'], fontsize=11)
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Refactoring Category Distribution: Agent vs Golden', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_category_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Generated: {output_path}")
    return output_path


def plot_paired_scenarios(data, output_dir):
    """Create stacked bar chart for paired comparison scenarios."""
    
    # Calculate scenarios
    instance_groups = defaultdict(list)
    golden_by_instance = {}
    
    for row in data:
        instance_id = row['instance_id']
        instance_groups[instance_id].append(row)
        
        if instance_id not in golden_by_instance:
            golden_by_instance[instance_id] = row['golden_has_refactoring']
    
    scenarios = {
        'both_have': 0,
        'only_agent': 0,
        'only_golden': 0,
        'neither': 0
    }
    
    for instance_id, agent_patches in instance_groups.items():
        golden_has = golden_by_instance[instance_id]
        
        for patch in agent_patches:
            agent_has = patch['agent_has_refactoring']
            
            if agent_has and golden_has:
                scenarios['both_have'] += 1
            elif agent_has and not golden_has:
                scenarios['only_agent'] += 1
            elif not agent_has and golden_has:
                scenarios['only_golden'] += 1
            else:
                scenarios['neither'] += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    total = sum(scenarios.values())
    categories = ['Agent Patches']
    
    both = scenarios['both_have']
    only_agent = scenarios['only_agent']
    only_golden = scenarios['only_golden']
    neither = scenarios['neither']
    
    # Create stacked bar
    bar1 = ax.barh(categories, [both], label='Both have refactoring',
                  color=BLUE_COLORS['both'], edgecolor='white', linewidth=2)
    bar2 = ax.barh(categories, [only_agent], left=[both],
                  label='Only Agent has refactoring',
                  color=BLUE_COLORS['only_agent'], edgecolor='white', linewidth=2)
    bar3 = ax.barh(categories, [only_golden], left=[both + only_agent],
                  label='Only Golden has refactoring',
                  color=BLUE_COLORS['only_golden'], edgecolor='white', linewidth=2)
    bar4 = ax.barh(categories, [neither], left=[both + only_agent + only_golden],
                  label='Neither has refactoring',
                  color=BLUE_COLORS['neither'], edgecolor='white', linewidth=2)
    
    # Add percentage labels
    def add_label(bar, value, left=0):
        if value > total * 0.03:  # Only show label if segment is large enough
            pct = value / total * 100
            ax.text(left + value/2, bar[0].get_y() + bar[0].get_height()/2,
                   f'{value:,}\n({pct:.1f}%)',
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   color='white' if pct > 15 else 'black')
    
    add_label(bar1, both, 0)
    add_label(bar2, only_agent, both)
    add_label(bar3, only_golden, both + only_agent)
    add_label(bar4, neither, both + only_agent + only_golden)
    
    ax.set_xlabel('Number of Agent Patches', fontsize=13, fontweight='bold')
    ax.set_title('Per-Issue Refactoring Behavior: Agent vs Golden Patches',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=11)
    ax.set_xlim(0, total)
    
    # Remove y-axis
    ax.set_yticks([])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_paired_scenarios.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Generated: {output_path}")
    return output_path


def plot_top_types_comparison(data, classification, output_dir, top_n=12):
    """Create horizontal bar chart for top refactoring types."""
    
    # Get refactoring types
    agent_types = defaultdict(int)
    golden_types = defaultdict(int)
    
    seen_instances = set()
    for row in data:
        for ref_type, count in row['agent_refactoring_types'].items():
            agent_types[ref_type] += count
        
        instance_id = row['instance_id']
        if instance_id not in seen_instances:
            seen_instances.add(instance_id)
            for ref_type, count in row['golden_refactoring_types'].items():
                golden_types[ref_type] += count
    
    # Get top types by total count
    all_types = set(agent_types.keys()) | set(golden_types.keys())
    type_totals = [(t, agent_types.get(t, 0) + golden_types.get(t, 0)) for t in all_types]
    type_totals.sort(key=lambda x: -x[1])
    top_types = [t[0] for t in type_totals[:top_n]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = np.arange(len(top_types))
    height = 0.35
    
    agent_counts = [agent_types.get(t, 0) for t in top_types]
    golden_counts = [golden_types.get(t, 0) for t in top_types]
    
    bars1 = ax.barh(y + height/2, agent_counts, height,
                    label='Agent', color=BLUE_COLORS['agent'], edgecolor='white')
    bars2 = ax.barh(y - height/2, golden_counts, height,
                    label='Golden', color=BLUE_COLORS['golden'], edgecolor='white')
    
    # Add value labels
    for bar in bars1:
        width = bar.get_width()
        if width > 0:
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {int(width)}', ha='left', va='center', fontsize=9)
    
    for bar in bars2:
        width = bar.get_width()
        if width > 0:
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {int(width)}', ha='left', va='center', fontsize=9)
    
    ax.set_yticks(y)
    ax.set_yticklabels(top_types, fontsize=10)
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Refactoring Types: Agent vs Golden Patches',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_top_types_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Generated: {output_path}")
    return output_path


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    csv_path = os.path.join(project_root, 'data', 'unified_data.csv')
    excel_path = os.path.join(project_root, 'data', 'refactoring_classification.xlsx')
    output_dir = os.path.join(project_root, 'output', 'RQ1')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING RQ1 COMPREHENSIVE FIGURES")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    classification = load_refactoring_classification(excel_path)
    data = load_unified_dataset(csv_path)
    print(f"  Loaded {len(data)} agent patches")
    
    # Generate figures
    print("\n[2/5] Generating presence comparison figure...")
    plot_presence_comparison(data, output_dir)
    
    print("\n[3/5] Generating category comparison figure...")
    plot_category_comparison(data, classification, output_dir)
    
    print("\n[4/5] Generating paired scenarios figure...")
    plot_paired_scenarios(data, output_dir)
    
    print("\n[5/5] Generating top types comparison figure...")
    plot_top_types_comparison(data, classification, output_dir)
    
    print("\n" + "=" * 80)
    print("ALL FIGURES GENERATED!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
