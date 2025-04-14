import os
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_result_paths", nargs="+", help="List of geneval_results.jsonl paths")
    parser.add_argument("--output_csv", type=str, default="compbench_summary.csv", help="Path to save the CSV summary")
    return parser.parse_args()

def main():
    args = parse_args()
    rows = []

    for result_path in args.ckpt_result_paths:
        if not os.path.exists(result_path):
            print(f"[!] Missing: {result_path}")
            continue

        ckpt_dir = os.path.dirname(result_path)
        df = pd.read_json(result_path, lines=True)
        acc_per_task = df.groupby("tag")["correct"].mean()

        row = {"checkpoint": ckpt_dir}
        row.update(acc_per_task.to_dict())  # {'color': 0.85, 'shape': 0.92, ...}
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output_csv, index=False)
    print(f"[✓] Saved to {args.output_csv}")

if __name__ == "__main__":
    main()
