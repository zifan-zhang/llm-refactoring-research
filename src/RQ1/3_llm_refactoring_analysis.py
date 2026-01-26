"""
RQ1 Analysis: Compare refactoring behavior across different LLM models
Generates visualizations for:
1. Refactoring rate by LLM model
2. Total refactoring counts by LLM model
3. Average refactoring types per patch by LLM model
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
    '#2B5278',  # Navy blue
    '#3B6FA0',  # Medium blue
    '#5088BC',  # Medium-light blue
    '#6B9FD4',  # Light blue
    '#85B4E0',  # Lighter blue
    '#A8CCE8',  # Pale blue
    '#C5DCF0',  # Very pale blue
]
BLUE_ACCENT = '#1E3A5F'
BLUE_GRADIENT = plt.cm.Blues


def parse_refactoring_dict(refactoring_str):
    """Parse refactoring type count from JSON string."""
    if not refactoring_str or refactoring_str == '{}':
        return {}
    try:
        return json.loads(refactoring_str)
    except json.JSONDecodeError:
        return {}


def load_dataset(csv_path):
    """Load dataset and extract LLM-specific refactoring statistics."""
    llm_stats = defaultdict(lambda: {
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
            llm_model = row.get('llm_model', '')
            if not llm_model:
                continue
            
            stats = llm_stats[llm_model]
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
    
    return dict(llm_stats)


def plot_refactoring_rate_by_llm(data, output_dir):
    """Create bar chart showing refactoring rate by LLM model."""
    # Calculate refactoring rate for each LLM
    llm_rates = []
    for llm, stats in data.items():
        rate = stats['patches_with_refactoring'] / stats['total_patches'] * 100 if stats['total_patches'] > 0 else 0
        llm_rates.append((llm, rate, stats['patches_with_refactoring'], stats['total_patches']))
    
    # Sort by rate descending
    llm_rates.sort(key=lambda x: -x[1])
    
    llms = [x[0] for x in llm_rates]
    rates = [x[1] for x in llm_rates]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create gradient colors
    colors = [plt.cm.Blues(0.3 + 0.6 * (i / len(llms))) for i in range(len(llms))]
    colors = colors[::-1]  # Reverse so highest rate gets darkest color
    
    bars = ax.bar(range(len(llms)), rates, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for i, (bar, (llm, rate, with_ref, total)) in enumerate(zip(bars, llm_rates)):
        ax.annotate(f'{rate:.1f}%\n({with_ref}/{total})',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, color=BLUE_ACCENT)
    
    ax.set_xlabel('LLM Model', fontsize=12, color=BLUE_ACCENT)
    ax.set_ylabel('Refactoring Rate (%)', fontsize=12, color=BLUE_ACCENT)
    ax.set_title('Refactoring Rate by LLM Model\n(Percentage of patches containing refactoring)',
                fontsize=14, fontweight='bold', color=BLUE_ACCENT, pad=15)
    ax.set_xticks(range(len(llms)))
    ax.set_xticklabels(llms, rotation=45, ha='right', fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(rates) * 1.25)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'llm_refactoring_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_refactoring_count_by_llm(data, output_dir):
    """Create bar chart showing total refactoring count by LLM model."""
    llm_counts = []
    for llm, stats in data.items():
        avg_per_patch = stats['total_refactoring_count'] / stats['total_patches'] if stats['total_patches'] > 0 else 0
        llm_counts.append((llm, stats['total_refactoring_count'], avg_per_patch, stats['total_patches']))
    
    # Sort by total count descending
    llm_counts.sort(key=lambda x: -x[1])
    
    llms = [x[0] for x in llm_counts]
    totals = [x[1] for x in llm_counts]
    avgs = [x[2] for x in llm_counts]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Total refactoring count
    ax1 = axes[0]
    colors = [plt.cm.Blues(0.3 + 0.6 * (i / len(llms))) for i in range(len(llms))][::-1]
    bars1 = ax1.bar(range(len(llms)), totals, color=colors, edgecolor='white', linewidth=1.5)
    
    for bar, total in zip(bars1, totals):
        ax1.annotate(f'{total}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color=BLUE_ACCENT)
    
    ax1.set_xlabel('LLM Model', fontsize=11, color=BLUE_ACCENT)
    ax1.set_ylabel('Total Refactoring Operations', fontsize=11, color=BLUE_ACCENT)
    ax1.set_title('Total Refactoring Operations by LLM', fontsize=13, fontweight='bold', color=BLUE_ACCENT)
    ax1.set_xticks(range(len(llms)))
    ax1.set_xticklabels(llms, rotation=45, ha='right', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: Average refactoring per patch
    ax2 = axes[1]
    # Re-sort by average
    llm_counts_by_avg = sorted(llm_counts, key=lambda x: -x[2])
    llms_avg = [x[0] for x in llm_counts_by_avg]
    avgs_sorted = [x[2] for x in llm_counts_by_avg]
    
    colors2 = [plt.cm.Blues(0.3 + 0.6 * (i / len(llms))) for i in range(len(llms))][::-1]
    bars2 = ax2.bar(range(len(llms)), avgs_sorted, color=colors2, edgecolor='white', linewidth=1.5)
    
    for bar, avg in zip(bars2, avgs_sorted):
        ax2.annotate(f'{avg:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color=BLUE_ACCENT)
    
    ax2.set_xlabel('LLM Model', fontsize=11, color=BLUE_ACCENT)
    ax2.set_ylabel('Avg Refactoring per Patch', fontsize=11, color=BLUE_ACCENT)
    ax2.set_title('Average Refactoring Operations per Patch', fontsize=13, fontweight='bold', color=BLUE_ACCENT)
    ax2.set_xticks(range(len(llms)))
    ax2.set_xticklabels(llms_avg, rotation=45, ha='right', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'llm_refactoring_counts.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_llm_refactoring_types_heatmap(data, output_dir, top_n=10):
    """Create heatmap showing top refactoring types by LLM model."""
    # Get all refactoring types and their total counts
    all_types = defaultdict(int)
    for llm, stats in data.items():
        for ref_type, count in stats['refactoring_types'].items():
            all_types[ref_type] += count
    
    # Get top N types
    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    top_type_names = [t[0] for t in top_types]
    
    # Sort LLMs by total refactoring
    llm_totals = [(llm, stats['total_refactoring_count']) for llm, stats in data.items()]
    llm_totals.sort(key=lambda x: -x[1])
    llms = [x[0] for x in llm_totals]
    
    # Build matrix
    matrix = np.zeros((len(llms), len(top_type_names)))
    for i, llm in enumerate(llms):
        for j, ref_type in enumerate(top_type_names):
            matrix[i, j] = data[llm]['refactoring_types'].get(ref_type, 0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=11, color=BLUE_ACCENT)
    
    # Set ticks
    ax.set_xticks(np.arange(len(top_type_names)))
    ax.set_yticks(np.arange(len(llms)))
    ax.set_xticklabels(top_type_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(llms, fontsize=10)
    
    # Add text annotations
    for i in range(len(llms)):
        for j in range(len(top_type_names)):
            value = int(matrix[i, j])
            if value > 0:
                text_color = 'white' if value > matrix.max() * 0.5 else BLUE_ACCENT
                ax.text(j, i, str(value), ha='center', va='center', color=text_color, fontsize=9)
    
    ax.set_title(f'Top {top_n} Refactoring Types by LLM Model',
                fontsize=14, fontweight='bold', color=BLUE_ACCENT, pad=15)
    ax.set_xlabel('Refactoring Type', fontsize=12, color=BLUE_ACCENT)
    ax.set_ylabel('LLM Model', fontsize=12, color=BLUE_ACCENT)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'llm_refactoring_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def generate_llm_summary_table(data, output_dir):
    """Generate CSV summary table for LLM statistics."""
    rows = []
    for llm, stats in data.items():
        rate = stats['patches_with_refactoring'] / stats['total_patches'] * 100 if stats['total_patches'] > 0 else 0
        avg_per_patch = stats['total_refactoring_count'] / stats['total_patches'] if stats['total_patches'] > 0 else 0
        unique_types = len(stats['refactoring_types'])
        
        rows.append({
            'llm_model': llm,
            'total_patches': stats['total_patches'],
            'patches_with_refactoring': stats['patches_with_refactoring'],
            'refactoring_rate': f'{rate:.2f}%',
            'total_refactoring_ops': stats['total_refactoring_count'],
            'avg_refactoring_per_patch': f'{avg_per_patch:.3f}',
            'unique_refactoring_types': unique_types,
            'solved_count': stats['solved_count'],
            'solved_with_refactoring': stats['solved_with_refactoring']
        })
    
    # Sort by refactoring rate descending
    rows.sort(key=lambda x: -float(x['refactoring_rate'].rstrip('%')))
    
    output_path = os.path.join(output_dir, 'llm_refactoring_summary.csv')
    fieldnames = ['llm_model', 'total_patches', 'patches_with_refactoring', 'refactoring_rate',
                  'total_refactoring_ops', 'avg_refactoring_per_patch', 'unique_refactoring_types',
                  'solved_count', 'solved_with_refactoring']
    
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
    print("\n=== LLM Refactoring Analysis Summary ===")
    print(f"Total LLM models: {len(data)}")
    
    for llm, stats in sorted(data.items(), key=lambda x: -x[1]['patches_with_refactoring'] / x[1]['total_patches'] if x[1]['total_patches'] > 0 else 0):
        rate = stats['patches_with_refactoring'] / stats['total_patches'] * 100 if stats['total_patches'] > 0 else 0
        print(f"\n{llm}:")
        print(f"  Total patches: {stats['total_patches']}")
        print(f"  Patches with refactoring: {stats['patches_with_refactoring']} ({rate:.1f}%)")
        print(f"  Total refactoring operations: {stats['total_refactoring_count']}")
    
    # Generate visualizations
    print("\n=== Generating Figures ===")
    plot_refactoring_rate_by_llm(data, output_dir)
    plot_refactoring_count_by_llm(data, output_dir)
    plot_llm_refactoring_types_heatmap(data, output_dir)
    generate_llm_summary_table(data, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()




