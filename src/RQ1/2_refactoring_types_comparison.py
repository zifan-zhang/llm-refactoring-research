"""
RQ1 Analysis: Compare refactoring types between Golden and Agent patches
Generates visualizations for:
1. Top refactoring types comparison (bar chart)
2. Total refactoring counts comparison (pie charts)
"""

import csv
import json
import os
from collections import defaultdict

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Define blue color palette
BLUE_COLORS = [
    '#1E3A5F',  # Deep navy blue
    '#3B6FA0',  # Medium blue
    '#6B9FD4',  # Light blue
    '#A8CCE8',  # Pale blue
    '#C5DCF0',  # Very pale blue
]
BLUE_ACCENT = '#1E3A5F'


def parse_refactoring_dict(refactoring_str):
    """Parse refactoring type count from JSON string."""
    if not refactoring_str or refactoring_str == '{}':
        return {}
    try:
        return json.loads(refactoring_str)
    except json.JSONDecodeError:
        return {}


def load_dataset(csv_path):
    """Load dataset and extract refactoring type statistics."""
    # Golden patch stats (deduplicated by instance_id)
    golden_types = defaultdict(int)
    golden_total_count = 0
    seen_instances = set()
    
    # Agent patch stats (all submissions)
    agent_types = defaultdict(int)
    agent_total_count = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_id = row.get('instance_id', '')
            
            # Parse agent refactoring types (no dedup)
            agent_ref = parse_refactoring_dict(row.get('agent_refactoring_type_count', '{}'))
            for ref_type, count in agent_ref.items():
                agent_types[ref_type] += count
                agent_total_count += count
            
            # Parse golden refactoring types (deduplicated by instance_id)
            if instance_id not in seen_instances:
                seen_instances.add(instance_id)
                golden_ref = parse_refactoring_dict(row.get('golden_refactoring_type_count', '{}'))
                for ref_type, count in golden_ref.items():
                    golden_types[ref_type] += count
                    golden_total_count += count
    
    return {
        'golden_types': dict(golden_types),
        'agent_types': dict(agent_types),
        'golden_total': golden_total_count,
        'agent_total': agent_total_count,
        'unique_instances': len(seen_instances)
    }


def plot_refactoring_types_comparison(data, output_dir, top_n=15):
    """Create horizontal bar chart comparing top refactoring types."""
    golden_types = data['golden_types']
    agent_types = data['agent_types']
    
    # Get all unique refactoring types
    all_types = set(golden_types.keys()) | set(agent_types.keys())
    
    # Sort by total count (golden + agent)
    type_totals = [(t, golden_types.get(t, 0) + agent_types.get(t, 0)) for t in all_types]
    type_totals.sort(key=lambda x: -x[1])
    
    # Take top N
    top_types = [t[0] for t in type_totals[:top_n]]
    
    golden_counts = [golden_types.get(t, 0) for t in top_types]
    agent_counts = [agent_types.get(t, 0) for t in top_types]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y = np.arange(len(top_types))
    height = 0.35
    
    # Create bars
    bars1 = ax.barh(y - height/2, golden_counts, height, label='Golden Patch', 
                    color=BLUE_COLORS[0], edgecolor='white')
    bars2 = ax.barh(y + height/2, agent_counts, height, label='Agent Patch',
                    color=BLUE_COLORS[2], edgecolor='white')
    
    # Add value labels
    for bar in bars1:
        width = bar.get_width()
        if width > 0:
            ax.annotate(f'{int(width)}',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(3, 0), textcoords="offset points",
                       ha='left', va='center', fontsize=9, color=BLUE_ACCENT)
    
    for bar in bars2:
        width = bar.get_width()
        if width > 0:
            ax.annotate(f'{int(width)}',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(3, 0), textcoords="offset points",
                       ha='left', va='center', fontsize=9, color=BLUE_COLORS[2])
    
    ax.set_xlabel('Count', fontsize=12, color=BLUE_ACCENT)
    ax.set_ylabel('Refactoring Type', fontsize=12, color=BLUE_ACCENT)
    ax.set_title(f'Top {top_n} Refactoring Types: Golden vs Agent Patches\n(Golden deduplicated by instance)',
                fontsize=14, fontweight='bold', color=BLUE_ACCENT, pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(top_types, fontsize=10)
    ax.legend(loc='lower right', fontsize=11)
    ax.invert_yaxis()  # Top types at top
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BLUE_COLORS[3])
    ax.spines['bottom'].set_color(BLUE_COLORS[3])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'refactoring_types_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_refactoring_summary(data, output_dir):
    """Create summary statistics visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Total refactoring counts
    ax1 = axes[0]
    categories = ['Golden Patch\n(Deduplicated)', 'Agent Patch\n(All Submissions)']
    counts = [data['golden_total'], data['agent_total']]
    colors = [BLUE_COLORS[0], BLUE_COLORS[2]]
    
    bars = ax1.bar(categories, counts, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax1.annotate(f'{count:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color=BLUE_ACCENT)
    
    ax1.set_ylabel('Total Refactoring Count', fontsize=12, color=BLUE_ACCENT)
    ax1.set_title('Total Refactoring Operations', fontsize=14, fontweight='bold', color=BLUE_ACCENT, pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: Unique refactoring types
    ax2 = axes[1]
    golden_unique = len(data['golden_types'])
    agent_unique = len(data['agent_types'])
    
    bars2 = ax2.bar(categories, [golden_unique, agent_unique], color=colors, edgecolor='white', linewidth=2)
    
    for bar, count in zip(bars2, [golden_unique, agent_unique]):
        ax2.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color=BLUE_ACCENT)
    
    ax2.set_ylabel('Number of Unique Types', fontsize=12, color=BLUE_ACCENT)
    ax2.set_title('Unique Refactoring Types', fontsize=14, fontweight='bold', color=BLUE_ACCENT, pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'refactoring_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def generate_summary_table(data, output_dir):
    """Generate a CSV summary table of all refactoring types."""
    golden_types = data['golden_types']
    agent_types = data['agent_types']
    
    all_types = set(golden_types.keys()) | set(agent_types.keys())
    
    rows = []
    for ref_type in sorted(all_types):
        golden_count = golden_types.get(ref_type, 0)
        agent_count = agent_types.get(ref_type, 0)
        rows.append({
            'refactoring_type': ref_type,
            'golden_count': golden_count,
            'agent_count': agent_count,
            'total': golden_count + agent_count
        })
    
    # Sort by total descending
    rows.sort(key=lambda x: -x['total'])
    
    output_path = os.path.join(output_dir, 'refactoring_types_table.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['refactoring_type', 'golden_count', 'agent_count', 'total'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved: {output_path}")
    return output_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(project_root, 'data', 'unified_data.csv')
    output_dir = os.path.join(project_root, 'output', 'RQ1')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from: {csv_path}")
    data = load_dataset(csv_path)
    
    # Print summary
    print("\n=== Refactoring Types Summary ===")
    print(f"Unique instances: {data['unique_instances']}")
    print(f"\nGolden Patch (deduplicated):")
    print(f"  Total refactoring operations: {data['golden_total']}")
    print(f"  Unique refactoring types: {len(data['golden_types'])}")
    
    print(f"\nAgent Patch (all submissions):")
    print(f"  Total refactoring operations: {data['agent_total']}")
    print(f"  Unique refactoring types: {len(data['agent_types'])}")
    
    print("\nTop 10 Golden refactoring types:")
    for ref_type, count in sorted(data['golden_types'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {ref_type}: {count}")
    
    print("\nTop 10 Agent refactoring types:")
    for ref_type, count in sorted(data['agent_types'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {ref_type}: {count}")
    
    # Generate visualizations
    print("\n=== Generating Figures ===")
    plot_refactoring_types_comparison(data, output_dir)
    plot_refactoring_summary(data, output_dir)
    generate_summary_table(data, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

