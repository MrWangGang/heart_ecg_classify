import pandas as pd
import ast
import os

# 设置 PTB-XL 数据集的路径
path = 'datasets'

# 检查路径是否存在
if not os.path.exists(path):
    print(f"错误: 路径 '{path}' 不存在。请修改 'path' 变量为你的数据集根目录。")
    exit()

# 加载 ptbxl_database.csv，其中包含文件名和诊断信息
Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# 加载 scp_statements.csv，用于将详细诊断代码映射到超类
agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    """
    将详细诊断代码聚合为诊断超类。
    """
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return sorted(list(set(tmp)))

# 应用聚合函数，创建 'diagnostic_superclass' 列
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# 创建一个新的 DataFrame，只包含文件名和标签
result_df = pd.DataFrame({
    'file_name': Y['filename_hr'],
    'label': Y['diagnostic_superclass'].apply(lambda x: ';'.join(x))
})

# --- 新增的修改：处理文件名 ---
# 使用 str.split('/') 按斜杠分割字符串，然后取最后一个元素
result_df['file_name'] = result_df['file_name'].str.split('/').str[-1]
# -----------------------------

# 将空标签替换为 NaN，以便后续删除
result_df['label'] = result_df['label'].replace('', pd.NA)
# 删除有空值的行
result_df.dropna(how='any', inplace=True)

# 保存为 CSV 文件
output_filename = './datasets/label.csv'
result_df.to_csv(output_filename, index=False)

print(f"成功生成文件: {output_filename}")
print(f"原始数据总行数: {len(Y)}")
print(f"删除空值后，最终保存的行数: {len(result_df)}")
print("\n文件内容示例:")
print(result_df.head())