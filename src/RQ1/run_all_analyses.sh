#!/bin/bash
# Run all RQ1 analysis scripts in sequence
# Usage: bash src/RQ1/run_all_analyses.sh

echo "========================================"
echo "RQ1 数据分析和可视化 - 批量运行"
echo "========================================"
echo ""

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

# Check if data file exists
if [ ! -f "data/unified_data.csv" ]; then
    echo "错误: 找不到数据文件 data/unified_data.csv"
    echo "请先运行数据生成脚本: python -m src.core.3_unified_data_builder"
    exit 1
fi

# Create output directory
mkdir -p output/RQ1

echo "数据源: data/unified_data.csv"
echo "输出目录: output/RQ1/"
echo ""

# Array of scripts
scripts=(
    "1_issue_refactoring_overview.py"
    "2_refactoring_types_comparison.py"
    "3_llm_refactoring_analysis.py"
    "4_framework_refactoring_analysis.py"
    "5_comprehensive_rq1_analysis.py"
    "6_generate_comprehensive_figures.py"
)

# Run each script
total=${#scripts[@]}
count=0

for script in "${scripts[@]}"; do
    count=$((count + 1))
    echo "========================================"
    echo "[$count/$total] 运行: $script"
    echo "========================================"
    
    python "src/RQ1/$script"
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "错误: $script 运行失败 (退出码: $exit_code)"
        echo "请检查错误信息并修复问题"
        exit $exit_code
    fi
    
    echo ""
done

echo "========================================"
echo "✓ 所有分析脚本运行完成!"
echo "========================================"
echo ""
echo "生成的文件位于: output/RQ1/"
echo ""
echo "文件统计:"
ls -1 output/RQ1/ | wc -l | xargs echo "  总文件数:"
ls -1 output/RQ1/*.png 2>/dev/null | wc -l | xargs echo "  图形文件 (PNG):"
ls -1 output/RQ1/*.csv 2>/dev/null | wc -l | xargs echo "  数据文件 (CSV):"
ls -1 output/RQ1/*.tex 2>/dev/null | wc -l | xargs echo "  LaTeX表格 (TEX):"
ls -1 output/RQ1/*.txt 2>/dev/null | wc -l | xargs echo "  文本报告 (TXT):"
echo ""
echo "查看完整列表: ls -lh output/RQ1/"
echo ""
