"""
RQ1 Data Preparation
Analyze patch apply status and refactoring statistics
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from src.constant import ROOT_DIR, DATA_DIR, PATCH_APPLY_OUTPUT_DIR


def load_unified_data():
    """Load unified data from CSV"""
    csv_path = DATA_DIR / "unified_data.csv"
    df = pd.read_csv(csv_path)
    return df


def get_all_agents():
    """Get all agent names from patch apply results directory"""
    patch_dir = Path(PATCH_APPLY_OUTPUT_DIR)
    agents = [d.name for d in patch_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    return sorted(agents)


def get_all_instances():
    """Get all instance IDs from index.json"""
    index_path = DATA_DIR / "index.json"
    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['all_ids']


def analyze_patch_apply_status():
    """
    Analyze patch apply status from agent_commit_mapping.json files
    
    Returns:
        dict: Statistics for different patch statuses
    """
    agents = get_all_agents()
    instances = get_all_instances()
    
    total_expected = len(agents) * len(instances)
    
    status_count = {
        'success': 0,
        'skipped': 0,  # Empty patches
        'failed': 0,   # Failed to apply
        'error': 0,
        'other': 0
    }
    
    for agent in agents:
        mapping_path = Path(PATCH_APPLY_OUTPUT_DIR) / agent / "agent_commit_mapping.json"
        
        if not mapping_path.exists():
            status_count['error'] += len(instances)
            continue
        
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            
            for mapping in mappings:
                status = mapping.get('status', 'other')
                if status in status_count:
                    status_count[status] += 1
                else:
                    status_count['other'] += 1
        
        except Exception:
            status_count['error'] += len(instances)
    
    return {
        "total_patches": int(total_expected),
        "total_agents": len(agents),
        "total_instances": len(instances),
        "success_patches": int(status_count['success']),
        "empty_patches": int(status_count['skipped']),
        "fail_patches": int(status_count['failed']),
        "error_patches": int(status_count['error']),
        "other_patches": int(status_count['other']),
        "success_ratio": float(status_count['success'] / total_expected) if total_expected > 0 else 0,
        "empty_ratio": float(status_count['skipped'] / total_expected) if total_expected > 0 else 0,
        "fail_ratio": float(status_count['failed'] / total_expected) if total_expected > 0 else 0,
        "error_ratio": float(status_count['error'] / total_expected) if total_expected > 0 else 0
    }


def analyze_agent_refactoring(df):
    """
    Analyze agent refactoring statistics from successfully applied patches
    
    Returns:
        dict: Statistics including total count and unique types
    """
    total_count = 0
    all_types = []
    
    for idx, row in df.iterrows():
        refactoring_types_str = row['agent_refactoring_type_count']
        
        try:
            if pd.notna(refactoring_types_str) and refactoring_types_str.strip():
                refactoring_types = json.loads(refactoring_types_str)
                
                for ref_type, count in refactoring_types.items():
                    total_count += count
                    all_types.append(ref_type)
            
        except (json.JSONDecodeError, ValueError):
            continue
    
    unique_types = list(set(all_types))
    type_distribution = dict(Counter(all_types))
    
    instances_with_refactoring = df[df['agent_has_refactoring'] == 1].shape[0]
    total_instances = df.shape[0]
    
    return {
        "total_refactorings": int(total_count),
        "unique_types_count": len(unique_types),
        "unique_types": sorted(unique_types),
        "type_distribution": type_distribution,
        "instances_with_refactoring": int(instances_with_refactoring),
        "total_instances": int(total_instances),
        "refactoring_ratio": float(instances_with_refactoring / total_instances) if total_instances > 0 else 0
    }


def analyze_golden_refactoring(df):
    """
    Analyze golden refactoring statistics (deduplicated by instance_id)
    
    Returns:
        dict: Statistics including total count and unique types
    """
    df_unique = df.drop_duplicates(subset=['instance_id'])
    
    total_count = 0
    all_types = []
    
    for idx, row in df_unique.iterrows():
        refactoring_types_str = row['golden_refactoring_type_count']
        
        try:
            if pd.notna(refactoring_types_str) and refactoring_types_str.strip():
                refactoring_types = json.loads(refactoring_types_str)
                
                for ref_type, count in refactoring_types.items():
                    total_count += count
                    all_types.append(ref_type)
            
        except (json.JSONDecodeError, ValueError):
            continue
    
    unique_types = list(set(all_types))
    type_distribution = dict(Counter(all_types))
    
    instances_with_refactoring = df_unique[df_unique['golden_has_refactoring'] == 1].shape[0]
    total_instances = df_unique.shape[0]
    
    return {
        "total_refactorings": int(total_count),
        "unique_types_count": len(unique_types),
        "unique_types": sorted(unique_types),
        "type_distribution": type_distribution,
        "instances_with_refactoring": int(instances_with_refactoring),
        "total_unique_instances": int(total_instances),
        "refactoring_ratio": float(instances_with_refactoring / total_instances) if total_instances > 0 else 0
    }


def main():
    """Main function to prepare RQ1 data"""
    print("RQ1 Data Preparation")
    print("=" * 60)
    
    patch_apply_stats = analyze_patch_apply_status()
    print(f"\nPatch Apply Status:")
    print(f"  Total: {patch_apply_stats['total_patches']} "
          f"({patch_apply_stats['total_agents']} agents × {patch_apply_stats['total_instances']} instances)")
    print(f"  Success: {patch_apply_stats['success_patches']} ({patch_apply_stats['success_ratio']:.1%})")
    print(f"  Empty: {patch_apply_stats['empty_patches']} ({patch_apply_stats['empty_ratio']:.1%})")
    print(f"  Failed: {patch_apply_stats['fail_patches']} ({patch_apply_stats['fail_ratio']:.1%})")
    
    df = load_unified_data()
    agent_refactoring_stats = analyze_agent_refactoring(df)
    print(f"\nAgent Refactoring:")
    print(f"  Total: {agent_refactoring_stats['total_refactorings']}")
    print(f"  Unique types: {agent_refactoring_stats['unique_types_count']}")
    print(f"  With refactoring: {agent_refactoring_stats['instances_with_refactoring']}/{agent_refactoring_stats['total_instances']} "
          f"({agent_refactoring_stats['refactoring_ratio']:.1%})")
    
    golden_refactoring_stats = analyze_golden_refactoring(df)
    print(f"\nGolden Refactoring (deduplicated):")
    print(f"  Total: {golden_refactoring_stats['total_refactorings']}")
    print(f"  Unique types: {golden_refactoring_stats['unique_types_count']}")
    print(f"  With refactoring: {golden_refactoring_stats['instances_with_refactoring']}/{golden_refactoring_stats['total_unique_instances']} "
          f"({golden_refactoring_stats['refactoring_ratio']:.1%})")
    
    results = {
        "patch_apply_status": patch_apply_stats,
        "agent_refactoring": agent_refactoring_stats,
        "golden_refactoring": golden_refactoring_stats
    }
    
    output_dir = ROOT_DIR / "output" / "RQ1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "0_data_preparation.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
