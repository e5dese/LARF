import os
import json
import pandas as pd
import argparse

def parse_results(models, base_dir="safe_test", output_csv="log_parsed_summary.csv"):
    records = []

    for model in models:
        eval_dir = os.path.join(base_dir, model, "eval_results")
        if not os.path.isdir(eval_dir):
            print(f"Warning: {eval_dir} does not exist. Skipping...")
            continue

        for root, dirs, files in os.walk(eval_dir):
            for f in files:
                if not f.endswith(".json"):
                    continue
                filepath = os.path.join(root, f)
                temp_dir = os.path.basename(root)  # temp_0, temp_1, temp_1_run2, etc.

                with open(filepath, 'r') as fh:
                    data = json.load(fh)

                # 统计 unsafe 数量
                total = len(data)
                unsafe_count = sum(1 for item in data if item.get("score", "").strip().lower().startswith("unsafe"))

                # 提取数据集名称
                dataset_name = f.split('-')[-1].replace('.json', '')

                # 温度大类
                temp_group = '0.0' if 'temp_0' in temp_dir else '1.0'

                records.append({
                    'Model': model,
                    'Temp': temp_group,
                    'Dataset': dataset_name,
                    'Run': temp_dir,
                    'Unsafe_Count': unsafe_count,
                    'Total': total
                })

    if not records:
        print("No results found.")
        return

    print(f"Successfully parsed {len(records)} evaluation records.\n")

    df = pd.DataFrame(records)

    # 按 Model, Temp, Dataset 分组，计算均值
    summary = df.groupby(['Model', 'Temp', 'Dataset']).agg(
        Unsafe_Avg=('Unsafe_Count', 'mean'),
        Total=('Total', 'mean'),
        Run_Count=('Run', 'count')
    ).reset_index()

    summary['Unsafe_Rate(%)'] = (summary['Unsafe_Avg'] / summary['Total'] * 100).round(2)

    summary['Result_Str'] = summary.apply(
        lambda row: f"{row['Unsafe_Avg']:.1f}/{int(row['Total'])} ({row['Unsafe_Rate(%)']}%)"
        if row['Run_Count'] > 1
        else f"{int(row['Unsafe_Avg'])}/{int(row['Total'])} ({row['Unsafe_Rate(%)']}%)",
        axis=1
    )

    pivot_df = summary.pivot_table(
        index=['Model', 'Temp'],
        columns='Dataset',
        values='Result_Str',
        aggfunc='first'
    ).fillna('-')

    print("=" * 90)
    print(f"Safety Evaluation Summary")
    print("Format: unsafe_count/total (unsafe_rate%) | Temp=1.0 is average of multiple runs")
    print("=" * 90)
    print(pivot_df.to_markdown())
    print("=" * 90)

    pivot_df.to_csv(output_csv)
    print(f"\nSaved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='Model folder names to parse')
    parser.add_argument('--base_dir', default='safe_test', help='Base directory')
    parser.add_argument('--output_csv', default='log_parsed_summary_0416.csv', help='Output CSV file')
    args = parser.parse_args()

    parse_results(args.models, args.base_dir, args.output_csv)
