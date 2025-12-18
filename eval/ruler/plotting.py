import os
import re
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

PREDICTIONS_DIR = os.path.join(os.env.get("NATSL_ROOT", ""), "predictions")


def extract_score(summary_csv_path):
    """
    Extract the numeric Score value from a summary-*.csv file.
    Expected format:
        Tasks,niah_single_1
        Score,6.6
        Nulls,0/500
    """
    with open(summary_csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0].strip().lower() == "score":
                return float(row[1])
    raise ValueError(f"No Score found in {summary_csv_path}")

def collect_results(predictions_dir):
    """
    Returns:
        results[model][context_length] = score
    """
    results = defaultdict(dict)

    for model_name in sorted(os.listdir(predictions_dir)):
        model_dir = os.path.join(predictions_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        for run_dir in os.listdir(model_dir):
            match = re.match(r"ruler-(\d+)", run_dir)
            if not match:
                continue

            context_length = int(match.group(1))
            run_path = os.path.join(model_dir, run_dir)

            # find summary-*.csv
            summary_files = [
                f for f in os.listdir(run_path)
                if f.startswith("summary-") and f.endswith(".csv")
            ]
            if not summary_files:
                continue

            # assume one summary file per run
            summary_path = os.path.join(run_path, summary_files[0])
            score = extract_score(summary_path)

            results[model_name][context_length] = score

    return results

def plot_results(results):
    plt.figure(figsize=(7, 4))

    all_contexts = set()

    for model_name, scores in results.items():
        contexts = sorted(scores.keys())
        values = [scores[c] for c in contexts]
        all_contexts.update(contexts)

        plt.plot(
            contexts,
            values,
            marker="o",
            linewidth=2,
            label=model_name
        )

    all_contexts = sorted(all_contexts)

    plt.xscale("log", base=2)
    plt.xticks(all_contexts, [str(c) for c in all_contexts])

    plt.xlabel("Context length (tokens)")
    plt.ylabel("RULER score")
    plt.title("RULER performance over context length")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("plot.png")

if __name__ == "__main__":
    results = collect_results(PREDICTIONS_DIR)
    plot_results(results)


