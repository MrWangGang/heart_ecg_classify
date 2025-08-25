import matplotlib
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
matplotlib.use("tkagg")
def visualize_label_counts(csv_path, figsize_pixels=(1024, 780), dpi=100):
    if not os.path.exists(csv_path):
        print(f"错误: 文件 '{csv_path}' 不存在。请检查文件路径。")
        return

    try:
        # 1. 读取 CSV 文件
        df = pd.read_csv(csv_path)

        # 2. 删除标签为空的行
        df.dropna(subset=['label'], inplace=True)

        # 3. 处理多标签：将标签字符串按分号分割成列表，然后展开
        df['label'] = df['label'].str.split(';')
        exploded_df = df.explode('label')

        # 4. 统计每个标签的出现次数
        label_counts = exploded_df['label'].value_counts()

        # 5. 设置图像尺寸
        figsize_inches = (figsize_pixels[0] / dpi, figsize_pixels[1] / dpi)
        plt.figure(figsize=figsize_inches)

        # 6. 使用 seaborn 绘制条形图
        sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')

        # 7. 添加标题和标签，使图表更具可读性
        plt.title('Distribution of Diagnostic Superclass Labels', fontsize=16)
        plt.xlabel('Label Superclass', fontsize=12)
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xticks(rotation=45, ha='right')  # 旋转 x 轴标签，防止重叠

        # 8. 在每个条形上添加数值标签
        for index, value in enumerate(label_counts.values):
            plt.text(index, value + 50, str(value), ha='center', va='bottom', fontsize=10)

        plt.tight_layout() # 自动调整布局，防止标签被切断
        plt.show()

    except Exception as e:
        print(f"处理文件或绘图时发生错误: {e}")

# --- 主程序入口 ---
if __name__ == '__main__':
    csv_file_path = './datasets/label.csv'
    visualize_label_counts(csv_file_path, figsize_pixels=(1024, 780))