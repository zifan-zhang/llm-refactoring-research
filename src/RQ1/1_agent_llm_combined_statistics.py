"""
RQ1: Agent Framework and LLM Combined Statistics
Generate comprehensive statistics table combining agent frameworks and LLMs
Including: absolute values, ratios, per patch, and unique types
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from src.constant import ROOT_DIR, DATA_DIR


def load_unified_data():
    """Load unified data from CSV"""
    csv_path = DATA_DIR / "unified_data.csv"
    df = pd.read_csv(csv_path)
    return df


def analyze_refactoring_stats(df_subset, group_name):
    """
    Analyze refactoring statistics for a given data subset
    
    Args:
        df_subset: DataFrame subset
        group_name: Group name (for identification)
    
    Returns:
        dict: Statistics results
    """
    total_patches = len(df_subset)
    patches_with_refactoring = df_subset[df_subset['agent_has_refactoring'] == 1].shape[0]
    
    # Count total refactorings and types
    total_refactorings = 0
    all_types = []
    
    for idx, row in df_subset.iterrows():
        refactoring_types_str = row['agent_refactoring_type_count']
        
        try:
            if pd.notna(refactoring_types_str) and refactoring_types_str.strip():
                refactoring_types = json.loads(refactoring_types_str)
                
                for ref_type, count in refactoring_types.items():
                    total_refactorings += count
                    all_types.extend([ref_type] * count)
        
        except (json.JSONDecodeError, ValueError):
            continue
    
    unique_types = len(set(all_types))
    type_distribution = dict(Counter(all_types))
    
    # Calculate per patch metrics
    per_patch = total_refactorings / total_patches if total_patches > 0 else 0
    
    return {
        'group_name': group_name,
        'total_patches': total_patches,
        'patches_with_refactoring': patches_with_refactoring,
        'refactoring_ratio': patches_with_refactoring / total_patches if total_patches > 0 else 0,
        'total_refactorings': total_refactorings,
        'unique_types': unique_types,
        'per_patch': per_patch,
        'type_distribution': type_distribution
    }


def analyze_golden_refactoring_stats(df):
    """
    Analyze golden refactoring statistics with deduplication by instance_id
    
    Args:
        df: DataFrame containing golden refactoring data
    
    Returns:
        dict: Golden refactoring statistics
    """
    # Deduplicate by instance_id - keep the first occurrence
    df_dedup = df.drop_duplicates(subset=['instance_id'], keep='first')
    
    total_instances = len(df_dedup)
    instances_with_refactoring = df_dedup[df_dedup['golden_has_refactoring'] == 1].shape[0]
    
    # Count total refactorings and types
    total_refactorings = 0
    all_types = []
    
    for idx, row in df_dedup.iterrows():
        refactoring_types_str = row['golden_refactoring_type_count']
        
        try:
            if pd.notna(refactoring_types_str) and refactoring_types_str.strip():
                refactoring_types = json.loads(refactoring_types_str)
                
                for ref_type, count in refactoring_types.items():
                    total_refactorings += count
                    all_types.extend([ref_type] * count)
        
        except (json.JSONDecodeError, ValueError):
            continue
    
    unique_types = len(set(all_types))
    type_distribution = dict(Counter(all_types))
    
    # Calculate per instance metrics
    per_instance = total_refactorings / total_instances if total_instances > 0 else 0
    
    return {
        'total_instances': total_instances,
        'instances_with_refactoring': instances_with_refactoring,
        'refactoring_ratio': instances_with_refactoring / total_instances if total_instances > 0 else 0,
        'total_refactorings': total_refactorings,
        'unique_types': unique_types,
        'per_instance': per_instance,
        'type_distribution': type_distribution
    }


def generate_combined_table(df):
    """
    Generate comprehensive statistics table for agent frameworks and LLMs
    
    Returns:
        pd.DataFrame: Statistics table
    """
    results = []
    
    # Get all unique agent frameworks and LLMs
    frameworks = sorted(df['agent_framework'].dropna().unique())
    llms = sorted(df['llm_model'].dropna().unique())
    
    # 1. Generate statistics for each framework + LLM combination
    for framework in frameworks:
        for llm in llms:
            subset = df[(df['agent_framework'] == framework) & (df['llm_model'] == llm)]
            if len(subset) > 0:
                stats = analyze_refactoring_stats(subset, f"{framework}+{llm}")
                stats['agent_framework'] = framework
                stats['llm_model'] = llm
                stats['aggregation_type'] = 'Combination'
                results.append(stats)
    
    # 2. Aggregated by Agent Framework
    for framework in frameworks:
        subset = df[df['agent_framework'] == framework]
        if len(subset) > 0:
            stats = analyze_refactoring_stats(subset, f"{framework} (All LLMs)")
            stats['agent_framework'] = framework
            stats['llm_model'] = 'ALL'
            stats['aggregation_type'] = 'By Framework'
            results.append(stats)
    
    # 3. Aggregated by Base LLM
    for llm in llms:
        subset = df[df['llm_model'] == llm]
        if len(subset) > 0:
            stats = analyze_refactoring_stats(subset, f"All Frameworks+{llm}")
            stats['agent_framework'] = 'ALL'
            stats['llm_model'] = llm
            stats['aggregation_type'] = 'By LLM'
            results.append(stats)
    
    # 4. Generate overall statistics for agents
    stats = analyze_refactoring_stats(df, "Overall (Agents)")
    stats['agent_framework'] = 'ALL'
    stats['llm_model'] = 'ALL'
    stats['aggregation_type'] = 'Overall'
    results.append(stats)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Adjust column order
    columns_order = [
        'aggregation_type',
        'agent_framework',
        'llm_model',
        'total_patches',
        'patches_with_refactoring',
        'refactoring_ratio',
        'total_refactorings',
        'unique_types',
        'per_patch'
    ]
    
    result_df = result_df[columns_order]
    
    return result_df, results


def format_table_for_display(result_df):
    """
    Format table for better display
    """
    display_df = result_df.copy()
    
    # Format percentages
    display_df['refactoring_ratio'] = display_df['refactoring_ratio'].apply(lambda x: f"{x:.2%}")
    
    # Format per_patch
    display_df['per_patch'] = display_df['per_patch'].apply(lambda x: f"{x:.2f}")
    
    # Rename columns to English
    display_df.columns = [
        'Aggregation Type',
        'Agent Framework',
        'LLM Model',
        'Total Patches',
        'Patches w/ Refactoring',
        'Refactoring Ratio',
        'Total Refactorings',
        'Unique Types',
        'Per Patch'
    ]
    
    return display_df


def generate_type_distribution_summary(results_with_distributions):
    """
    Generate summary of refactoring type distribution
    """
    summary = []
    
    for item in results_with_distributions:
        if item['type_distribution']:
            sorted_types = sorted(
                item['type_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            summary.append({
                'group_name': item['group_name'],
                'aggregation_type': item.get('aggregation_type', 'N/A'),
                'agent_framework': item['agent_framework'],
                'llm_model': item['llm_model'],
                'top_5_types': sorted_types[:5]
            })
    
    return summary


def main():
    """Main function"""
    print("=" * 80)
    print("Refactoring Statistics: Aggregated by Framework, LLM, and Human Baseline")
    print("=" * 80)
    
    # Load data
    df = load_unified_data()
    print(f"\nLoaded {len(df)} records in total")
    
    # Generate comprehensive statistics table
    result_df, results_with_distributions = generate_combined_table(df)
    
    # Analyze golden refactoring statistics (with deduplication)
    print("\n" + "=" * 80)
    print("Analyzing Human Baseline (Golden Patch) Statistics")
    print("=" * 80)
    golden_stats = analyze_golden_refactoring_stats(df)
    
    # Add golden stats to results for complete JSON output
    golden_stats_entry = {
        'group_name': 'Human Baseline (Golden Patch)',
        'aggregation_type': 'Human Baseline',
        'agent_framework': 'N/A',
        'llm_model': 'N/A',
        'total_patches': golden_stats['total_instances'],
        'patches_with_refactoring': golden_stats['instances_with_refactoring'],
        'refactoring_ratio': golden_stats['refactoring_ratio'],
        'total_refactorings': golden_stats['total_refactorings'],
        'unique_types': golden_stats['unique_types'],
        'per_patch': golden_stats['per_instance'],
        'type_distribution': golden_stats['type_distribution']
    }
    results_with_distributions.append(golden_stats_entry)
    
    # Create output directory
    output_dir = ROOT_DIR / "output" / "RQ1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw data (with detailed type_distribution)
    output_json = output_dir / "agent_llm_combined_statistics.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results_with_distributions, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed statistics saved to: {output_json}")
    
    # Save golden refactoring statistics
    output_golden_json = output_dir / "golden_refactoring_statistics.json"
    with open(output_golden_json, 'w', encoding='utf-8') as f:
        json.dump(golden_stats, f, indent=2, ensure_ascii=False)
    print(f"Golden refactoring statistics saved to: {output_golden_json}")
    
    # Save CSV
    output_csv = output_dir / "agent_llm_combined_statistics.csv"
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"CSV table saved to: {output_csv}")
    
    # Generate formatted display table
    display_df = format_table_for_display(result_df)
    output_display_csv = output_dir / "agent_llm_combined_statistics_display.csv"
    display_df.to_csv(output_display_csv, index=False, encoding='utf-8')
    print(f"Display table saved to: {output_display_csv}")
    
    # Generate refactoring type distribution summary
    type_summary = generate_type_distribution_summary(results_with_distributions)
    output_types = output_dir / "refactoring_type_distribution.json"
    with open(output_types, 'w', encoding='utf-8') as f:
        json.dump(type_summary, f, indent=2, ensure_ascii=False)
    print(f"Refactoring type distribution saved to: {output_types}")
    
    # Print preview - show aggregated results
    print("\n" + "=" * 80)
    print("Aggregated Statistics Preview:")
    print("=" * 80)
    
    # Show by Framework
    print("\n[Aggregated by Agent Framework]")
    framework_df = display_df[display_df['Aggregation Type'] == 'By Framework']
    print(framework_df.to_string(index=False))
    
    # Show by LLM
    print("\n[Aggregated by Base LLM]")
    llm_df = display_df[display_df['Aggregation Type'] == 'By LLM']
    print(llm_df.to_string(index=False))
    
    # Show overall
    print("\n[Overall Agent Statistics]")
    overall_df = display_df[display_df['Aggregation Type'] == 'Overall']
    print(overall_df.to_string(index=False))
    
    # Print key statistics
    print("\n" + "=" * 80)
    print("Key Statistics Summary:")
    print("=" * 80)
    
    total_row = result_df[result_df['aggregation_type'] == 'Overall'].iloc[0]
    print(f"\nAll Agents (Overall):")
    print(f"  Total patches: {total_row['total_patches']}")
    print(f"  Patches with refactoring: {total_row['patches_with_refactoring']} ({total_row['refactoring_ratio']:.2%})")
    print(f"  Total refactorings: {total_row['total_refactorings']}")
    print(f"  Unique types: {total_row['unique_types']}")
    print(f"  Average refactorings per patch: {total_row['per_patch']:.2f}")
    
    print(f"\nHuman Baseline (Golden Patch):")
    print(f"  Total instances: {golden_stats['total_instances']}")
    print(f"  Instances with refactoring: {golden_stats['instances_with_refactoring']} ({golden_stats['refactoring_ratio']:.2%})")
    print(f"  Total refactorings: {golden_stats['total_refactorings']}")
    print(f"  Unique types: {golden_stats['unique_types']}")
    print(f"  Average refactorings per instance: {golden_stats['per_instance']:.2f}")
    
    # Print top refactoring types in golden data
    if golden_stats['type_distribution']:
        print(f"\n  Top 10 refactoring types in Golden Patch:")
        sorted_types = sorted(
            golden_stats['type_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (ref_type, count) in enumerate(sorted_types[:10], 1):
            print(f"    {i}. {ref_type}: {count}")
    
    print("\n" + "=" * 80)
    print("Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
