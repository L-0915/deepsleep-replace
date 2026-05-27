"""Merge OpenBookQA results into existing benchmark result files."""
import glob
import json
import os

RESULTS_DIR = "data/eval/benchmark_results"

MERGE_MAP = {
    "ds_b0.1": "ds_b0.1_obqa",
    "ds_b0.5": "ds_b0.5_obqa",
    "qw_b0.1": "qw_b0.1_obqa",
    "qw_b0.5": "qw_b0.5_obqa",
}

for main_prefix, obqa_prefix in MERGE_MAP.items():
    # Find latest main result file (has 4 benchmarks)
    main_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{main_prefix}*.json")), reverse=True)
    # Filter out obqa files
    main_files = [f for f in main_files if "_obqa_" not in f]

    # Find latest obqa result file
    obqa_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{obqa_prefix}*.json")), reverse=True)

    if not main_files:
        print(f"Skip {main_prefix}: no main result file found")
        continue
    if not obqa_files:
        print(f"Skip {main_prefix}: no OpenBookQA result file found")
        continue

    with open(main_files[0]) as f:
        main_data = json.load(f)
    with open(obqa_files[0]) as f:
        obqa_data = json.load(f)

    # Merge obqa results into main
    for task_name, task_results in obqa_data.get("results", {}).items():
        if "openbookqa" in task_name:
            main_data.setdefault("results", {})[task_name] = task_results
            print(f"  Added {task_name} to {os.path.basename(main_files[0])}")

    # Write merged data back
    with open(main_files[0], "w") as f:
        json.dump(main_data, f, indent=2, ensure_ascii=False)
    print(f"  Merged -> {main_files[0]}")

print("Done.")
