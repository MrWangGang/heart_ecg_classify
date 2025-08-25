import matplotlib
import numpy as np
import wfdb
import os
from matplotlib import pyplot as plt
import seaborn as sns
matplotlib.use("tkagg")
def visualize_ecg_single_file(file_path, figsize=(10.24, 8.0), title='ECG Signal Visualization'):
    if not os.path.exists(file_path + '.dat'):
        print(f"错误: 文件 '{file_path}.dat' 或 '{file_path}.hea' 不存在。")
        return

    try:
        signal, meta = wfdb.rdsamp(file_path)

        num_channels = signal.shape[1]

        # 创建子图，设置 figsize 为 10.24 x 8.0 英寸
        fig, axes = plt.subplots(num_channels, 1, figsize=figsize, sharex=True)

        for i in range(num_channels):
            sns.lineplot(x=np.arange(signal.shape[0]), y=signal[:, i], ax=axes[i], color='b')
            axes[i].set_ylabel(f'Lead {i+1}', rotation=0, ha='right')
            axes[i].grid(True, linestyle='--', alpha=0.6)
            axes[i].tick_params(axis='y', labelsize=8)

        fig.suptitle(title, fontsize=16)
        axes[-1].set_xlabel('Time Steps', fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"处理文件时发生错误: {e}")


# --- 示例使用 ---
if __name__ == "__main__":
    file_to_visualize = './datasets/records500/00001_hr'
    visualize_ecg_single_file(file_to_visualize, figsize=(10.24, 8.0))