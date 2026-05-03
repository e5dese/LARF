import re
import pandas as pd

def parse_log_and_generate_report(log_file="llama_guard_log_4.11.txt"):
    records = []
    
    try:
        # 读取整个日志文件
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 {log_file}")
        return

    # 核心：使用正则表达式匹配评测结果行
    # 匹配格式示例: "42/400 个回答判定为 unsafe (safe_test/Qwen.../xxx-direct.json)"
    # 考虑到日志拷贝时可能产生的换行符，使用 \s* 匹配空白和换行
    pattern = re.compile(r'(\d+)/(\d+)\s*个回答判定为\s*unsafe\s*\((.*?\.json)\)')
    matches = pattern.findall(content)

    if not matches:
        print("未能从日志中提取到有效数据，请确认日志格式或内容是否正确。")
        return

    for match in matches:
        unsafe_count = int(match[0])
        total_count = int(match[1])
        file_path = match[2]
        
        # 解析文件路径提取元数据
        # 例如: safe_test/Qwen2.5-7B-Instruct/eval_results/temp_1/Qwen2.5-7B-Instruct-temp1.0-direct.json
        path_parts = file_path.split('/')
        if len(path_parts) < 5:
            continue
            
        model_name = path_parts[1]
        temp_dir = path_parts[3]  # 获取 temp_0, temp_1, temp_1_run2 等
        filename = path_parts[4]
        
        # 提取数据集名称 (直接取减号分割的最后一部分并去掉 .json)
        dataset_name = filename.split('-')[-1].replace('.json', '')
        
        # 区分温度大类：包含 temp_0 的归为 0.0，其他的归为 1.0
        temp_group = '0.0' if 'temp_0' in temp_dir else '1.0'

        records.append({
            'Model': model_name,
            'Temp': temp_group,
            'Dataset': dataset_name,
            'Run': temp_dir,
            'Unsafe_Count': unsafe_count,
            'Total': total_count
        })

    print(f"✅ 成功从日志中解析出 {len(records)} 条评测记录！\n正在计算均值并生成表格...")

    # 转换为 DataFrame
    df = pd.DataFrame(records)
    
    # 按照 Model, Temp, Dataset 分组，计算 Unsafe 的平均值
    summary = df.groupby(['Model', 'Temp', 'Dataset']).agg(
        Unsafe_Avg=('Unsafe_Count', 'mean'),
        Total=('Total', 'mean'),
        Run_Count=('Run', 'count') # 记录包含了几次 run，用于区分显示格式
    ).reset_index()

    # 计算不安全率 (%)
    summary['Unsafe_Rate(%)'] = (summary['Unsafe_Avg'] / summary['Total'] * 100).round(2)

    # 格式化输出字符串：多次运行的显示 1 位小数，单次运行显示整数
    summary['Result_Str'] = summary.apply(
        lambda row: f"{row['Unsafe_Avg']:.1f}/{int(row['Total'])} ({row['Unsafe_Rate(%)']}%)"
        if row['Run_Count'] > 1
        else f"{int(row['Unsafe_Avg'])}/{int(row['Total'])} ({row['Unsafe_Rate(%)']}%)",
        axis=1
    )

    # 生成透视表 (Pivot Table) 排列为二维表格格式
    pivot_df = summary.pivot_table(
        index=['Model', 'Temp'],
        columns='Dataset',
        values='Result_Str',
        aggfunc='first'
    ).fillna('-')

    # 控制台打印 Markdown 格式表格
    print("\n" + "="*90)
    print(f"安全评测结果汇总表 - 数据源: {log_file}")
    print("格式: 不安全数/总数 (不安全率%) | Temp=1为多次运行均值")
    print("="*90)
    print(pivot_df.to_markdown())
    print("="*90)
    
    # 导出为 CSV 文件供 Excel 使用
    csv_filename = "log_parsed_summary_llama_guard_4.13.csv"
    pivot_df.to_csv(csv_filename)
    print(f"\n📊 整理好的表格已保存至: {csv_filename}")

if __name__ == "__main__":
    parse_log_and_generate_report("llama_guard_log_4.13.txt")