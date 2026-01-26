"""
RQ1 Basic Analysis: Issue Type Distribution and Refactoring Presence
Generates pie charts for:
1. Issue type distribution
2. Refactoring presence in golden and agent patches
"""

import csv
import os

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define blue color palette
BLUE_COLORS = [
    '#1E3A5F',  # Deep navy blue
    '#3B6FA0',  # Medium blue
    '#6B9FD4',  # Light blue
    '#A8CCE8',  # Pale blue
]

BLUE_ACCENT = '#1E3A5F'
BLUE_LIGHT = '#6B9FD4'


def load_dataset(csv_path):
    """Load and parse the unified dataset."""
    data = {
        'issue_type': {},
        'agent_has_refactoring': {'True': 0, 'False': 0},
        'golden_has_refactoring': {'True': 0, 'False': 0}
    }
    
    # Track seen instance_ids for deduplication
    seen_instances = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_id = row.get('instance_id', '')
            
            # Count agent refactoring presence (no dedup - each agent patch is unique)
            # Handle both '1'/'0' and 'True'/'False' formats
            agent_ref_raw = row.get('agent_has_refactoring', '')
            agent_ref = 'True' if agent_ref_raw in ('1', 'True') else 'False'
            data['agent_has_refactoring'][agent_ref] += 1
            
            # Deduplicate by instance_id for issue_type and golden_has_refactoring
            if instance_id not in seen_instances:
                seen_instances.add(instance_id)
                
                # Count issue types (deduplicated by instance_id)
                issue_type = row.get('issue_type', '')
                if issue_type:
                    data['issue_type'][issue_type] = data['issue_type'].get(issue_type, 0) + 1
                
                # Count golden refactoring presence (deduplicated by instance_id)
                golden_ref_raw = row.get('golden_has_refactoring', '')
                golden_ref = 'True' if golden_ref_raw in ('1', 'True') else 'False'
                data['golden_has_refactoring'][golden_ref] += 1
    
    data['unique_issues'] = len(seen_instances)
    return data


def plot_issue_type_distribution(data, output_dir):
    """Create a pie chart for issue type distribution."""
    issue_types = data['issue_type']
    
    # Sort by count descending
    sorted_items = sorted(issue_types.items(), key=lambda x: -x[1])
    labels = [item[0].replace('_', ' ').title() for item in sorted_items]
    sizes = [item[1] for item in sorted_items]
    
    # Calculate percentages
    total = sum(sizes)
    percentages = [s / total * 100 for s in sizes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart with blue colors
    colors = BLUE_COLORS[:len(labels)]
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%',
        colors=colors,
        startangle=90,
        explode=[0.02] * len(labels),
        shadow=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Style autopct texts
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    # Create legend
    legend_labels = [f'{label} (n={size:,})' for label, size in zip(labels, sizes)]
    ax.legend(
        wedges, 
        legend_labels,
        title="Issue Types",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=11
    )
    
    ax.set_title('Issue Type Distribution\n(Deduplicated by Instance)', fontsize=16, fontweight='bold', color=BLUE_ACCENT, pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'issue_type_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_refactoring_presence(data, output_dir):
    """Create pie charts for refactoring presence in golden and agent patches."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Define data for both charts with subtitles
    chart_data = [
        ('Golden Patch', '(Deduplicated by Instance)', data['golden_has_refactoring']),
        ('Agent Patch', '(All Agent Submissions)', data['agent_has_refactoring'])
    ]
    
    # Blue colors for True/False
    colors = [BLUE_COLORS[1], BLUE_COLORS[3]]  # Medium blue for True, pale blue for False
    
    for ax, (title, subtitle, ref_data) in zip(axes, chart_data):
        true_count = ref_data['True']
        false_count = ref_data['False']
        total = true_count + false_count
        
        sizes = [true_count, false_count]
        labels_display = ['Has Refactoring', 'No Refactoring']
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct=lambda pct: f'{pct:.1f}%',
            colors=colors,
            startangle=90,
            explode=[0.03, 0],
            shadow=False,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        # Style autopct texts
        for i, autotext in enumerate(autotexts):
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
            if i == 0:  # Has Refactoring (darker background)
                autotext.set_color('white')
            else:  # No Refactoring (lighter background)
                autotext.set_color(BLUE_ACCENT)
        
        # Create legend for this chart
        legend_labels = [
            f'{labels_display[0]} (n={true_count:,})',
            f'{labels_display[1]} (n={false_count:,})'
        ]
        ax.legend(
            wedges,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fontsize=11
        )
        
        ax.set_title(f'{title}\n{subtitle}\n(Total: {total:,})', fontsize=14, fontweight='bold', color=BLUE_ACCENT, pad=15)
    
    fig.suptitle('Refactoring Presence in Patches', fontsize=16, fontweight='bold', color=BLUE_ACCENT, y=1.05)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'refactoring_presence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(project_root, 'data', 'unified_data.csv')
    output_dir = os.path.join(project_root, 'output', 'RQ1')
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading dataset from: {csv_path}")
    data = load_dataset(csv_path)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"\nUnique Issues (after deduplication): {data['unique_issues']}")
    
    print("\nIssue Type Distribution (deduplicated by instance_id):")
    for issue_type, count in sorted(data['issue_type'].items(), key=lambda x: -x[1]):
        total = sum(data['issue_type'].values())
        pct = count / total * 100
        print(f"  {issue_type}: {count:,} ({pct:.1f}%)")
    
    print("\nGolden Patch Refactoring (deduplicated by instance_id):")
    golden_true = data['golden_has_refactoring']['True']
    golden_false = data['golden_has_refactoring']['False']
    golden_total = golden_true + golden_false
    print(f"  Has Refactoring: {golden_true:,} ({golden_true/golden_total*100:.1f}%)")
    print(f"  No Refactoring: {golden_false:,} ({golden_false/golden_total*100:.1f}%)")
    
    print("\nAgent Patch Refactoring:")
    agent_true = data['agent_has_refactoring']['True']
    agent_false = data['agent_has_refactoring']['False']
    agent_total = agent_true + agent_false
    print(f"  Has Refactoring: {agent_true:,} ({agent_true/agent_total*100:.1f}%)")
    print(f"  No Refactoring: {agent_false:,} ({agent_false/agent_total*100:.1f}%)")
    
    # Generate plots
    print("\n=== Generating Figures ===")
    plot_issue_type_distribution(data, output_dir)
    plot_refactoring_presence(data, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

