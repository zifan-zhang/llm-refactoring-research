"""
RQ1 Analysis: Compare refactoring behavior across different Agent frameworks
Generates visualizations for:
1. Refactoring rate by framework
2. Refactoring types distribution by framework
3. Framework comparison summary
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
    """Load dataset and extract framework-specific refactoring statistics."""
    framework_stats = defaultdict(lambda: {
        'total_patches': 0,
        'patches_with_refactoring': 0,
        'total_refactoring_count': 0,
        'refactoring_types': defaultdict(int),
        'solved_count': 0,
        'solved_with_refactoring': 0
    })
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            framework = row.get('agent_framework', '')
            if not framework:
                continue
            
            stats = framework_stats[framework]
            stats['total_patches'] += 1
            
            # Check if solved (handle both numeric and string formats)
            is_solved_raw = row.get('is_issue_solved', '')
            is_solved = is_solved_raw in ('1', 'resolved')
            if is_solved:
                stats['solved_count'] += 1
            
            # Parse agent refactoring (handle both '1'/'0' and 'True'/'False')
            has_refactoring_raw = row.get('agent_has_refactoring', '')
            has_refactoring = has_refactoring_raw in ('1', 'True')
            if has_refactoring:
                stats['patches_with_refactoring'] += 1
                if is_solved:
                    stats['solved_with_refactoring'] += 1
            
            agent_ref = parse_refactoring_dict(row.get('agent_refactoring_type_count', '{}'))
            for ref_type, count in agent_ref.items():
                stats['refactoring_types'][ref_type] += count
                stats['total_refactoring_count'] += count
    
    return dict(framework_stats)


def plot_framework_overview(data, output_dir):
    """Create comprehensive overview of framework refactoring behavior."""
    frameworks = list(data.keys())
    
    # Calculate metrics
    metrics = []
    for fw in frameworks:
        stats = data[fw]
        rate = stats['patches_with_refactoring'] / stats['total_patches'] * 100 if stats['total_patches'] > 0 else 0
        avg_per_patch = stats['total_refactoring_count'] / stats['total_patches'] if stats['total_patches'] > 0 else 0
        unique_types = len(stats['refactoring_types'])
        metrics.append({
            'framework': fw,
            'rate': rate,
            'avg_per_patch': avg_per_patch,
            'unique_types': unique_types,
            'total': stats['total_refactoring_count'],
            'patches_with_ref': stats['patches_with_refactoring'],
            'total_patches': stats['total_patches']
        })
    
    # Sort by rate descending
    metrics.sort(key=lambda x: -x['rate'])
    frameworks_sorted = [m['framework'] for m in metrics]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = [BLUE_COLORS[0], BLUE_COLORS[1], BLUE_COLORS[2]]
    
    # 1. Refactoring Rate (Top Left)
    ax1 = axes[0, 0]
    rates = [m['rate'] for m in metrics]
    bars1 = ax1.bar(frameworks_sorted, rates, color=colors, edgecolor='white', linewidth=2)
    
    for bar, m in zip(bars1, metrics):
        ax1.annotate(f'{m["rate"]:.1f}%\n({m["patches_with_ref"]}/{m["total_patches"]})',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, color=BLUE_ACCENT)
    
    ax1.set_ylabel('Refactoring Rate (%)', fontsize=11, color=BLUE_ACCENT)
    ax1.set_title('Refactoring Rate by Framework', fontsize=13, fontweight='bold', color=BLUE_ACCENT)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(rates) * 1.35)
    
    # 2. Total Refactoring Operations (Top Right)
    ax2 = axes[0, 1]
    totals = [m['total'] for m in metrics]
    bars2 = ax2.bar(frameworks_sorted, totals, color=colors, edgecolor='white', linewidth=2)
    
    for bar, total in zip(bars2, totals):
        ax2.annotate(f'{total:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=BLUE_ACCENT)
    
    ax2.set_ylabel('Total Refactoring Operations', fontsize=11, color=BLUE_ACCENT)
    ax2.set_title('Total Refactoring Operations by Framework', fontsize=13, fontweight='bold', color=BLUE_ACCENT)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Average Refactoring per Patch (Bottom Left)
    ax3 = axes[1, 0]
    avgs = [m['avg_per_patch'] for m in metrics]
    bars3 = ax3.bar(frameworks_sorted, avgs, color=colors, edgecolor='white', linewidth=2)
    
    for bar, avg in zip(bars3, avgs):
        ax3.annotate(f'{avg:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=BLUE_ACCENT)
    
    ax3.set_ylabel('Avg Refactoring per Patch', fontsize=11, color=BLUE_ACCENT)
    ax3.set_title('Average Refactoring per Patch by Framework', fontsize=13, fontweight='bold', color=BLUE_ACCENT)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Unique Refactoring Types (Bottom Right)
    ax4 = axes[1, 1]
    unique_types = [m['unique_types'] for m in metrics]
    bars4 = ax4.bar(frameworks_sorted, unique_types, color=colors, edgecolor='white', linewidth=2)
    
    for bar, ut in zip(bars4, unique_types):
        ax4.annotate(f'{ut}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=BLUE_ACCENT)
    
    ax4.set_ylabel('Number of Unique Types', fontsize=11, color=BLUE_ACCENT)
    ax4.set_title('Unique Refactoring Types by Framework', fontsize=13, fontweight='bold', color=BLUE_ACCENT)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    fig.suptitle('Agent Framework Refactoring Comparison', fontsize=16, fontweight='bold', color=BLUE_ACCENT, y=1.02)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'framework_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_framework_refactoring_types(data, output_dir, top_n=12):
    """Create grouped bar chart showing top refactoring types by framework."""
    frameworks = list(data.keys())
    
    # Get all refactoring types and their total counts
    all_types = defaultdict(int)
    for fw, stats in data.items():
        for ref_type, count in stats['refactoring_types'].items():
            all_types[ref_type] += count
    
    # Get top N types
    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    top_type_names = [t[0] for t in top_types]
    
    # Build data for each framework
    x = np.arange(len(top_type_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    for i, (fw, color) in enumerate(zip(frameworks, BLUE_COLORS[:3])):
        counts = [data[fw]['refactoring_types'].get(t, 0) for t in top_type_names]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, counts, width, label=fw, color=color, edgecolor='white')
        
        # Add value labels for non-zero values
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.annotate(f'{count}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, color=color)
    
    ax.set_xlabel('Refactoring Type', fontsize=12, color=BLUE_ACCENT)
    ax.set_ylabel('Count', fontsize=12, color=BLUE_ACCENT)
    ax.set_title(f'Top {top_n} Refactoring Types by Agent Framework',
                fontsize=14, fontweight='bold', color=BLUE_ACCENT, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(top_type_names, rotation=45, ha='right', fontsize=10)
    ax.legend(title='Framework', fontsize=11, title_fontsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'framework_refactoring_types.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_framework_pie_charts(data, output_dir):
    """Create pie charts showing refactoring distribution within each framework."""
    frameworks = sorted(data.keys())
    n_frameworks = len(frameworks)
    
    fig, axes = plt.subplots(1, n_frameworks, figsize=(6 * n_frameworks, 6))
    
    if n_frameworks == 1:
        axes = [axes]
    
    for ax, fw in zip(axes, frameworks):
        stats = data[fw]
        with_ref = stats['patches_with_refactoring']
        without_ref = stats['total_patches'] - with_ref
        total = stats['total_patches']
        
        sizes = [with_ref, without_ref]
        labels_display = ['Has Refactoring', 'No Refactoring']
        colors = [BLUE_COLORS[1], BLUE_COLORS[3]]
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct=lambda pct: f'{pct:.1f}%',
            colors=colors,
            startangle=90,
            explode=[0.03, 0],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        for i, autotext in enumerate(autotexts):
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
            autotext.set_color('white' if i == 0 else BLUE_ACCENT)
        
        legend_labels = [
            f'{labels_display[0]} (n={with_ref:,})',
            f'{labels_display[1]} (n={without_ref:,})'
        ]
        ax.legend(wedges, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=10)
        
        ax.set_title(f'{fw}\n(Total: {total:,})', fontsize=13, fontweight='bold', color=BLUE_ACCENT, pad=10)
    
    fig.suptitle('Refactoring Presence by Agent Framework', fontsize=15, fontweight='bold', color=BLUE_ACCENT, y=1.05)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'framework_refactoring_pies.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def generate_framework_summary_table(data, output_dir):
    """Generate CSV summary table for framework statistics."""
    rows = []
    for fw, stats in data.items():
        rate = stats['patches_with_refactoring'] / stats['total_patches'] * 100 if stats['total_patches'] > 0 else 0
        avg_per_patch = stats['total_refactoring_count'] / stats['total_patches'] if stats['total_patches'] > 0 else 0
        unique_types = len(stats['refactoring_types'])
        
        # Get top 3 refactoring types for this framework
        sorted_types = sorted(stats['refactoring_types'].items(), key=lambda x: -x[1])[:3]
        top_types = '; '.join([f"{t[0]}({t[1]})" for t in sorted_types])
        
        rows.append({
            'framework': fw,
            'total_patches': stats['total_patches'],
            'patches_with_refactoring': stats['patches_with_refactoring'],
            'refactoring_rate': f'{rate:.2f}%',
            'total_refactoring_ops': stats['total_refactoring_count'],
            'avg_refactoring_per_patch': f'{avg_per_patch:.4f}',
            'unique_refactoring_types': unique_types,
            'top_3_refactoring_types': top_types,
            'solved_count': stats['solved_count'],
            'solved_with_refactoring': stats['solved_with_refactoring']
        })
    
    # Sort by refactoring rate descending
    rows.sort(key=lambda x: -float(x['refactoring_rate'].rstrip('%')))
    
    output_path = os.path.join(output_dir, 'framework_refactoring_summary.csv')
    fieldnames = ['framework', 'total_patches', 'patches_with_refactoring', 'refactoring_rate',
                  'total_refactoring_ops', 'avg_refactoring_per_patch', 'unique_refactoring_types',
                  'top_3_refactoring_types', 'solved_count', 'solved_with_refactoring']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    print("\n=== Framework Refactoring Analysis Summary ===")
    print(f"Total frameworks: {len(data)}")
    
    for fw, stats in sorted(data.items(), key=lambda x: -x[1]['patches_with_refactoring'] / x[1]['total_patches'] if x[1]['total_patches'] > 0 else 0):
        rate = stats['patches_with_refactoring'] / stats['total_patches'] * 100 if stats['total_patches'] > 0 else 0
        avg = stats['total_refactoring_count'] / stats['total_patches'] if stats['total_patches'] > 0 else 0
        print(f"\n{fw}:")
        print(f"  Total patches: {stats['total_patches']}")
        print(f"  Patches with refactoring: {stats['patches_with_refactoring']} ({rate:.1f}%)")
        print(f"  Total refactoring operations: {stats['total_refactoring_count']}")
        print(f"  Avg refactoring per patch: {avg:.4f}")
        print(f"  Unique refactoring types: {len(stats['refactoring_types'])}")
    
    # Generate visualizations
    print("\n=== Generating Figures ===")
    plot_framework_overview(data, output_dir)
    plot_framework_refactoring_types(data, output_dir)
    plot_framework_pie_charts(data, output_dir)
    generate_framework_summary_table(data, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()




