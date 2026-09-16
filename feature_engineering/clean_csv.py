import pandas as pd
import numpy as np
import sys

def drop_na_or_zero_variance(df, na_thresh=0.5, var_tol=0.0):
    """
    删除 DataFrame 中以下列：
    - 缺失率超过 na_thresh 的列（默认 50%）；
    - 方差小于等于 var_tol 的列（默认 0，即常数列）。

    参数：
    - df: 输入的 Pandas DataFrame；
    - na_thresh: 缺失率阈值，范围为 0 到 1；
    - var_tol: 方差阈值。

    返回：
    - df_cleaned: 清理后的 DataFrame；
    - dropped_cols: 被删除的列名列表。
    """
    # 计算每列的缺失率
    missing_rate = df.isna().mean()
    cols_to_drop_na = missing_rate[missing_rate > na_thresh].index.tolist()

    # 计算每列的方差
    variance = df.var(skipna=True)
    cols_to_drop_var = variance[variance <= var_tol].index.tolist()

    # 合并要删除的列
    cols_to_drop = set(cols_to_drop_na + cols_to_drop_var)

    # 删除这些列
    df_cleaned = df.drop(columns=cols_to_drop)

    return df_cleaned, list(cols_to_drop)

def main(input_file, output_file):
    try:
        # 读取 CSV 文件
        df = pd.read_csv(input_file)
        print(f"成功读取文件：{input_file}")
    except Exception as e:
        print(f"读取文件时出错：{e}")
        sys.exit(1)

    # 清理数据
    df_cleaned, dropped_cols = drop_na_or_zero_variance(df, na_thresh=0.5, var_tol=0.0)

    # 输出被删除的列
    print(f"\n被删除的列：{dropped_cols}")

    # 保存清理后的数据到新的 CSV 文件
    try:
        df_cleaned.to_csv(output_file, index=False)
        print(f"\n清理后的数据已保存到：{output_file}")
    except Exception as e:
        print(f"保存文件时出错：{e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python clean_csv.py <输入文件路径> <输出文件路径>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    main(input_file, output_file)
