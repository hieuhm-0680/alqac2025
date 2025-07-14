import json
import os
from src.evaluator import Evaluator

def eval_retrieval(predictions_file, ground_truth_file, output_file=None):
    """Evaluate retrieval predictions against ground truth."""
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    if not predictions or not ground_truth:
        print("❌ Failed to load data files.")
        return

    print(
        f"📊 Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth items")

    print("⚡ Running evaluation...")
    results = Evaluator.evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        task_type="retrieval",
        save_to_file=output_file,
        k_values=[1, 3, 5, 10],
        predictions_path=predictions_file,
        ground_truth_path=ground_truth_file
    )
    print("\n" + "="*50)
    print("📄 RETRIEVAL EVALUATION RESULTS")
    print("="*50)
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")
    print(f"\n💾 Results saved to: {output_file}")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    print(f"\n📄 Saved file structure:")
    print(f"  - timestamp: {saved_data['timestamp']}")
    print(f"  - task_type: {saved_data['task_type']}")
    print(f"  - predictions_file: {saved_data['predictions_file']}")
    print(f"  - ground_truth_file: {saved_data['ground_truth_file']}")
    print(f"  - total_metrics: {saved_data['summary']['total_metrics']}")
    print(f"  - best_metric: {saved_data['summary']['best_metric']}")
    print(f"  - worst_metric: {saved_data['summary']['worst_metric']}")

    return results
    

def main():
    import sys

    if len(sys.argv) < 3:
        print("📋 ALQAC2025 Prediction File Evaluator")
        print(
            "Usage: python evaluate.py <task_type> <prediction_file> [ground_truth_file] [--save-to output_file]")
        print("  task_type: 'retrieval' or 'qa'")
        print("  prediction_file: path to JSON prediction file")
        print(
            "  ground_truth_file: optional path to ground truth file for coverage analysis")
        print("  --save-to: optional path to save validation results as JSON")
        print("\nExample:")
        print("  python evaluate.py retrieval output/final_results.json")
        print("  python evaluate.py retrieval output/final_results.json dataset/alqac25_train.json --save-to validation_report.json")
        return

    task_type = sys.argv[1]
    prediction_file = sys.argv[2]

    if task_type not in ['retrieval', 'qa']:
        print(f"❌ Invalid task type: {task_type}. Must be 'retrieval' or 'qa'")
        return

    ground_truth_file = None
    save_to_file = None

    for i, arg in enumerate(sys.argv[3:], 3):
        if arg == '--save-to' and i + 1 < len(sys.argv):
            save_to_file = sys.argv[i + 1]
        elif not arg.startswith('--') and ground_truth_file is None:
            ground_truth_file = arg
    if ground_truth_file and not os.path.exists(ground_truth_file):
        print(f"❌ Ground truth file not found: {ground_truth_file}")
        return
    if not os.path.exists(prediction_file):
        print(f"❌ Prediction file not found: {prediction_file}")
        return

    if task_type == 'retrieval':
        eval_retrieval(prediction_file, ground_truth_file, save_to_file)
    elif task_type == 'qa':
        pass
   
if __name__ == "__main__":
    main()