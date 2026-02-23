# An Empirical Study of Code Agent Refactoring in Tangled Commits on Multi-SWE-bench Java Projects

Reproduction guide for the experiment

## Setup

```bash
git clone <this-repo>
cd llm-refactoring-research
pip install -r requirements.txt
```

## Data

Download datasets, RefactoringMiner, and clone Java repos:

```bash
./scripts/download_research_data.sh
env.sh
```

This creates `data/`, `repos/`, and `tools/` (RefactoringMiner, optional git-lfs) and add RefactoringMiner and git-lfs to the PATH

## Pipeline (run from repo root)

Run in order:

1. **Apply patches**  
   `python -m src.core.0_patch_applier`

2. **Golden refactorings detection**  
   `python -m src.core.1_golden_refactoring_detector`

3. **Agent refactorings detection**  
   `python -m src.core.2_agent_refactoring_detector`

4. **Create unified dataset**  
   `python -m src.core.3_unified_data_builder`  
   → writes `data/unified_data.csv`  
   **Then manually remove the two outlier rows** (version-rollback patches): rows where `(instance_id, agent_name)` is `(fasterxml__jackson-databind-4320, MopenHands_Gemini-2.5-Pro)` and `(fasterxml__jackson-databind-4087, MopenHands_Gemini-2.5-Pro)`. See `doc/outlier_investigation_report.md`.

   **Dependency note:** Step 4 cannot run on pipeline outputs alone. It also needs **external data** from `./scripts/download_research_data.sh`: `data/index.json` (task difficulty), `data/issue_types.xlsx` (issue type labels), `data/Multi-SWE-bench/java/` (golden patches & issue text), and `data/evaluation_logs_new/` (compilation logs and final reports). The script also provides `data/java_experiment/` (agent predictions), which step 1 uses to produce `data/patch_apply_results/` that step 4 reads. So run the download script first, then steps 1–4 in order.

5. **RQ1 – statistics & heatmaps**  
   `python -m src.RQ1.0_data_preparation`  
   `python -m src.RQ1.1_agent_llm_combined_statistics`  
   `python -m src.RQ1.2_refactoring_heatmap_generator`

6. **RQ2 – impact analysis**  
   `python -m src.RQ2.1_multicollinearity_diagnosis`  
   `python -m src.RQ2.2_simple_refactoring_presence_analysis`  
   `python -m src.RQ2.3_refactoring_logistic_regression_analysis`

7. **RQ3 – LLM experiments** (optional; needs API keys)  
   Judgement / patch selection / assessment / refinement:  
   `python -m src.RQ3.1_run_patch_judgement_experiment`  
   `python -m src.RQ3.2_run_patch_selection_experiment`  
   `python -m src.RQ3.3_run_refactoring_assessment_experiment`  
   `python -m src.RQ3.4_run_patch_refinement_experiment`

## Outputs

- **Patch apply**: `data/patch_apply_results/<agent>/`
- **Refactoring detection**: `data/refactoring_detection_results/agent/`, `golden/`
- **Unified data**: `data/unified_data.csv`, `data/refactoring_summary.json`
- **RQ1/RQ2**: figures and tables under `data/` or script-defined paths

## Data sources

- Multi-SWE-bench (Java): Hugging Face `ByteDance-Seed/Multi-SWE-bench`
- Experiment artifacts: Hugging Face `Azusa434/LLM-Refactoring-Research`
- RefactoringMiner: [tsantalis/RefactoringMiner](https://github.com/tsantalis/RefactoringMiner) v3.0.13

## Our fixed version of Multi-SWE-bench's evaluation framework: https://github.com/zifan-zhang/mswebench-reproduce

- Used to evaluate the original agent patches, golden patches and refined agent patches for java instances.
