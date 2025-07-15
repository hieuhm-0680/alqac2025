#!/usr/bin/env python3
"""
Experiment runner for ALQAC2025 retrieval evaluation.

This script:
1. Generates configuration files using gen_config.py
2. Runs each configuration using main.py
3. Evaluates results using evaluate.py
4. Organizes outputs in separate folders for each experiment
"""

import os
import json
import yaml
import shutil
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys
from tqdm import tqdm


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], cwd: str = None, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"🚀 Running: {' '.join(cmd)}")
    if cwd:
        print(f"   Working directory: {cwd}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture_output,
        text=True,
        check=False
    )

    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        if result.stderr:
            print(f"   Error: {result.stderr}")
        if result.stdout:
            print(f"   Output: {result.stdout}")
    else:
        print("✅ Command completed successfully")

    return result


def load_config_summary(config_dir: str) -> Dict:
    """Load the configuration summary file."""
    summary_file = os.path.join(config_dir, "config_summary.yaml")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Config summary not found: {summary_file}")

    with open(summary_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_single_experiment(config_info: Dict, config_dir: str, results_base_dir: str,
                          ground_truth_file: str, workspace_dir: str) -> bool:
    """Run a single experiment with the given configuration."""
    config_name = config_info['filename'].replace('.yaml', '')
    experiment_dir = os.path.join(results_base_dir, config_name)

    print(f"\n{'='*60}")
    print(f"Running experiment: {config_name}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"{'='*60}")

    ensure_directory(experiment_dir)

    source_config = os.path.join(config_dir, config_info['filename'])
    dest_config = os.path.join(experiment_dir, config_info['filename'])

    with open(source_config, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    output_file = os.path.join(experiment_dir, "retrieval_results.json")
    config_data['pipeline']['save_results_path'] = output_file
    config_data['pipeline']['save_local_path'] = os.path.join(
        experiment_dir, "local_retriever_results.json")
    config_data['pipeline']['save_global_path'] = os.path.join(
        experiment_dir, "global_retriever_results.json")
    config_data['pipeline']['save_fused_path'] = os.path.join(
        experiment_dir, "fused_results.json")
    config_data['pipeline']['save_reranked_path'] = os.path.join(
        experiment_dir, "reranked_results.json")

    with open(dest_config, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False,
                  sort_keys=False, indent=2)

    print(f"📋 Config modified and saved to: {dest_config}")

    main_cmd = [
        sys.executable, "main.py",
        "--config", dest_config
    ]

    main_result = run_command(main_cmd, cwd=workspace_dir)
    if main_result.returncode != 0:
        print(f"❌ Failed to run main.py for {config_name}")
        print("🔄 Attempting to build index and retry experiment...")
        build_index_cmd = [sys.executable, "main.py", "--build_indexes", "1", "--config", dest_config]
        build_result = run_command(build_index_cmd, cwd=workspace_dir)
        if build_result.returncode == 0:
            print("✅ Index built successfully. Skipping retry of main.py since evaluation is included.")
            main_result = build_result  # Use build_result as main_result for output checks
        else:
            print(f"❌ Failed to build index for {config_name}")
            with open(os.path.join(experiment_dir, "error_log.txt"), 'w') as f:
                f.write(f"Command: {' '.join(build_index_cmd)}\n")
                f.write(f"Return code: {build_result.returncode}\n")
                f.write(f"STDOUT:\n{build_result.stdout}\n")
                f.write(f"STDERR:\n{build_result.stderr}\n")
            return False

    # Check if output file was created
    if not os.path.exists(output_file):
        print(f"❌ Output file not created: {output_file}")
        return False

    # Run evaluation
    evaluation_file = os.path.join(experiment_dir, "evaluation_results.json")
    ground_truth_path = os.path.join(workspace_dir, ground_truth_file)
    eval_cmd = [
        sys.executable, "evaluate.py",
        "retrieval", output_file, ground_truth_path,
        "--save-to", evaluation_file
    ]

    eval_result = run_command(eval_cmd, cwd=workspace_dir)
    if eval_result.returncode != 0:
        print(f"❌ Failed to run evaluation for {config_name}")
        # Save error log
        with open(os.path.join(experiment_dir, "eval_error_log.txt"), 'w') as f:
            f.write(f"Command: {' '.join(eval_cmd)}\n")
            f.write(f"Return code: {eval_result.returncode}\n")
            f.write(f"STDOUT:\n{eval_result.stdout}\n")
            f.write(f"STDERR:\n{eval_result.stderr}\n")
        return False

    summary = {
        "experiment_name": config_name,
        "timestamp": datetime.now().isoformat(),
        "config_file": config_info['filename'],
        "config_details": {
            "enable_local": config_info['enable_local'],
            "enable_global": config_info['enable_global'],
            "enable_reranker": config_info['enable_reranker'],
            "local_lexical": config_info['local_lexical'],
            "local_semantic": config_info['local_semantic'],
            "global_lexical": config_info['global_lexical'],
            "global_semantic": config_info['global_semantic'],
            "lexical_config": config_info.get('lexical_config', {})
        },
        "files": {
            "config": config_info['filename'],
            "retrieval_results": "retrieval_results.json",
            "evaluation_results": "evaluation_results.json"
        },
        "status": "completed"
    }

    with open(os.path.join(experiment_dir, "experiment_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Experiment {config_name} completed successfully")
    print(f"📊 Results saved in: {experiment_dir}")

    return True


def run_all_experiments(config_dir: str = "src/config/generated_configs",
                        results_base_dir: str = "experiments",
                        ground_truth_file: str = "dataset/alqac25_train.json",
                        workspace_dir: str = None,
                        config_filter: Optional[str] = None) -> None:
    """Run experiments for all generated configurations."""

    if workspace_dir is None:
        workspace_dir = os.getcwd()

    print(f"🔬 Starting ALQAC2025 Experiments")
    print(f"📁 Config directory: {config_dir}")
    print(f"📁 Results directory: {results_base_dir}")
    print(f"📄 Ground truth file: {ground_truth_file}")
    print(f"🏠 Workspace directory: {workspace_dir}")

    # Ensure results directory exists
    ensure_directory(results_base_dir)

    # Load configuration summary
    try:
        summary = load_config_summary(config_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Run 'python gen_config.py' first to generate configurations.")
        return

    configs = summary['configs']

    if config_filter:
        configs = [c for c in configs if config_filter.lower()
                   in c['filename'].lower()]
        print(
            f"🔍 Filtered to {len(configs)} configs matching '{config_filter}'")

    print(f"📊 Total configurations to run: {len(configs)}")

    if not os.path.exists(os.path.join(workspace_dir, ground_truth_file)):
        print(f"❌ Ground truth file not found: {ground_truth_file}")
        return

    # Run experiments
    successful_experiments = 0
    failed_experiments = 0

    for i, config_info in enumerate(tqdm(configs, desc="Running experiments", unit="exp"), 1):
        print(f"\n🔄 Progress: {i}/{len(configs)}")

        success = run_single_experiment(
            config_info, config_dir, results_base_dir,
            ground_truth_file, workspace_dir
        )

        if success:
            successful_experiments += 1
        else:
            failed_experiments += 1

    # Create overall summary
    overall_summary = {
        "experiment_run": {
            "timestamp": datetime.now().isoformat(),
            "total_configs": len(configs),
            "successful_experiments": successful_experiments,
            "failed_experiments": failed_experiments,
            "success_rate": successful_experiments / len(configs) if configs else 0,
        },
        "config_directory": config_dir,
        "results_directory": results_base_dir,
        "ground_truth_file": ground_truth_file
    }

    with open(os.path.join(results_base_dir, "experiment_run_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"🎯 EXPERIMENT RUN SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful experiments: {successful_experiments}")
    print(f"❌ Failed experiments: {failed_experiments}")
    print(f"📊 Success rate: {successful_experiments/len(configs)*100:.1f}%")
    print(f"📁 Results saved in: {results_base_dir}")
    print(
        f"📄 Summary: {os.path.join(results_base_dir, 'experiment_run_summary.json')}")


def generate_configs_if_needed(config_dir: str) -> None:
    """Generate configurations if they don't exist."""
    summary_file = os.path.join(config_dir, "config_summary.yaml")

    if not os.path.exists(summary_file):
        print("📝 Generating configurations...")
        result = run_command(
            [sys.executable, "gen_config.py", "--output-dir", config_dir])
        if result.returncode != 0:
            raise RuntimeError("Failed to generate configurations")
        print("✅ Configurations generated successfully")
    else:
        print("✅ Configurations already exist")


def list_experiment_results(results_dir: str = "experiments") -> None:
    """List all experiment results with summary."""
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return

    summary_file = os.path.join(results_dir, "experiment_run_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        print(f"🎯 EXPERIMENT RUN SUMMARY")
        print(f"{'='*60}")
        run_info = summary['experiment_run']
        print(f"📅 Run timestamp: {run_info['timestamp']}")
        print(f"📊 Total configs: {run_info['total_configs']}")
        print(f"✅ Successful: {run_info['successful_experiments']}")
        print(f"❌ Failed: {run_info['failed_experiments']}")
        print(f"📈 Success rate: {run_info['success_rate']*100:.1f}%")
        print()

    # List individual experiment directories
    experiment_dirs = [d for d in os.listdir(results_dir)
                       if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('config_')]

    print(f"📁 Individual experiment results ({len(experiment_dirs)} total):")
    print("-" * 60)

    for exp_dir in sorted(experiment_dirs):
        exp_path = os.path.join(results_dir, exp_dir)
        summary_file = os.path.join(exp_path, "experiment_summary.json")

        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                exp_summary = json.load(f)

            status = exp_summary.get('status', 'unknown')
            status_emoji = "✅" if status == "completed" else "❌"

            print(f"{status_emoji} {exp_dir}")

            # Check if evaluation results exist
            eval_file = os.path.join(exp_path, "evaluation_results.json")
            if os.path.exists(eval_file):
                with open(eval_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)

                metrics = eval_data.get('metrics', {})
                if metrics:
                    # Show key metrics
                    recall_5 = metrics.get('recall@5', 0)
                    precision_5 = metrics.get('precision@5', 0)
                    print(
                        f"    📊 Recall@5: {recall_5:.4f}, Precision@5: {precision_5:.4f}")
        else:
            print(f"❓ {exp_dir} (no summary)")


def main():
    parser = argparse.ArgumentParser(
        description="Run ALQAC2025 retrieval experiments")
    parser.add_argument("--config-dir", default="src/config/generated_configs",
                        help="Directory containing generated config files")
    parser.add_argument("--results-dir", default="experiments",
                        help="Directory to save experiment results")
    parser.add_argument("--ground-truth", default="dataset/alqac25_train.json",
                        help="Path to ground truth file")
    parser.add_argument("--filter", type=str,
                        help="Filter configs by name (substring match)")
    parser.add_argument("--generate-configs", action="store_true",
                        help="Generate configurations if they don't exist")
    parser.add_argument("--list-results", action="store_true",
                        help="List existing experiment results")
    parser.add_argument("--workspace-dir", default=None,
                        help="Workspace directory (default: current directory)")

    args = parser.parse_args()

    if args.list_results:
        list_experiment_results(args.results_dir)
        return

    if args.generate_configs:
        generate_configs_if_needed(args.config_dir)
        return

    try:
        # Auto-generate configs if needed
        generate_configs_if_needed(args.config_dir)

        # Run experiments
        run_all_experiments(
            config_dir=args.config_dir,
            results_base_dir=args.results_dir,
            ground_truth_file=args.ground_truth,
            workspace_dir=args.workspace_dir,
            config_filter=args.filter
        )

    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
