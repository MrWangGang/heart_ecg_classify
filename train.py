import pandas as pd # 导入pandas库，用于数据处理和分析，特别是读取CSV文件
import numpy as np # 导入numpy库，用于数值计算，特别是处理数组和矩阵
import wfdb # 导入wfdb库，用于读取和处理PhysioNet数据库中的ECG数据文件（例如.dat和.hea文件）
import os # 导入os库，用于与操作系统交互，例如文件路径操作和目录创建
import torch # 导入torch库，PyTorch深度学习框架的核心库
import torch.nn as nn # 导入torch.nn模块，包含构建神经网络所需的各种层（如卷积层、全连接层）
from torch.utils.data import Dataset, DataLoader # 导入Dataset和DataLoader，用于创建自定义数据集和高效地批量加载数据
from sklearn.model_selection import train_test_split # 导入train_test_split，用于将数据集划分为训练集和验证集
from sklearn.preprocessing import MultiLabelBinarizer # 导入MultiLabelBinarizer，用于将多标签类别转换为二元（one-hot）表示
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score # 导入评估指标，用于评估模型性能
import random # 导入random库，用于生成随机数，设置随机种子
from tqdm import tqdm # 导入tqdm库，用于在循环中显示进度条，提供更好的用户体验
import matplotlib.pyplot as plt # 导入matplotlib.pyplot，用于数据可视化，特别是绘制损失曲线和指标曲线

# --- 1. 设置全局随机种子函数 ---
def set_seed(seed):
    """
    设置全局随机种子，确保实验的可复现性。
    Args:
        seed (int): 随机种子值。
    """
    random.seed(seed) # 设置Python内置random模块的随机种子
    np.random.seed(seed) # 设置Numpy的随机种子
    torch.manual_seed(seed) # 设置PyTorch CPU的随机种子
    if torch.cuda.is_available(): # 检查CUDA（GPU）是否可用
        torch.cuda.manual_seed_all(seed) # 设置所有PyTorch GPU的随机种子
        # torch.backends.cudnn.deterministic = True 确保每次运行卷积操作结果一致，可能略微降低性能
        # torch.backends.cudnn.benchmark = False 关闭cuDNN的自动优化，确保确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"全局随机种子已设置为: {seed}") # 打印设置成功的消息

# --- 2. 定义自定义 Dataset 类 ---
class ECGDataset(Dataset):
    """
    自定义ECG数据集类，继承自torch.utils.data.Dataset。
    用于加载ECG信号数据和对应的多标签。
    """
    def __init__(self, metadata_df, ecg_data_path, input_length=5000,
                 mean=None, std=None, is_train=True):
        """
        初始化ECGDataset。
        Args:
            metadata_df (pd.DataFrame): 包含文件路径和标签的元数据DataFrame。
            ecg_data_path (str): ECG数据文件所在的根目录路径。
            input_length (int): 每个ECG信号的目标长度，不足则填充，超出则截断。
            mean (np.ndarray, optional): 信号的均值，用于标准化。训练时计算，测试时传入。
            std (np.ndarray, optional): 信号的标准差，用于标准化。训练时计算，测试时传入。
            is_train (bool): 是否为训练集，用于决定是否计算均值和标准差。
        """
        self.metadata = metadata_df # 存储包含ECG文件信息和标签的DataFrame
        self.ecg_data_path = ecg_data_path # 存储ECG数据文件的根路径
        self.input_length = input_length # 存储每个ECG信号的目标长度
        self.mlb = MultiLabelBinarizer() # 初始化MultiLabelBinarizer，用于处理多标签分类问题

        # 将DataFrame中的'label'列（字符串，如"MI;STTC"）拆分为列表，然后fit_transform为二元标签
        labels_list = self.metadata['label'].str.split(';').tolist()
        self.labels = self.mlb.fit_transform(labels_list) # 对所有标签进行fit和transform，得到二元编码的标签数组
        print(f"标签类别: {self.mlb.classes_}") # 打印MultiLabelBinarizer学习到的所有标签类别

        self.mean = mean # 存储用于标准化的均值
        self.std = std # 存储用于标准化的标准差
        self.is_train = is_train # 标记当前数据集是否为训练集

        # 如果是训练集，则计算均值和标准差；验证集和测试集则使用训练集计算好的参数
        if self.is_train:
            self._calculate_normalization_params()

    def _calculate_normalization_params(self):
        """
        计算整个训练集的均值和标准差。
        这些参数将用于后续的信号标准化，确保数据分布一致。
        """
        all_signals = [] # 用于存储所有ECG信号的列表
        print("正在计算数据集的均值和标准差...")
        # 遍历数据集中的每个样本，读取ECG信号并进行预处理（填充/截断）
        for idx in tqdm(range(len(self)), desc="计算标准化参数"):
            file_name = self.metadata.iloc[idx]['file_name'] # 获取当前样本的文件名
            full_path = os.path.join(self.ecg_data_path, file_name) # 构建完整的文件路径
            try:
                # 使用wfdb.rdsamp读取ECG信号。signal是信号数据，_是元数据（这里不需要）
                signal, _ = wfdb.rdsamp(full_path)
                # 如果信号长度小于目标长度，则在末尾进行零填充
                if signal.shape[0] < self.input_length:
                    # np.pad((前填充, 后填充), (左填充, 右填充), mode='constant')
                    signal = np.pad(signal, ((0, self.input_length - signal.shape[0]), (0, 0)), mode='constant')
                else:
                    # 如果信号长度大于目标长度，则截断到目标长度
                    signal = signal[:self.input_length]
                all_signals.append(signal) # 将处理后的信号添加到列表中
            except Exception as e:
                # 捕获读取或处理文件时可能发生的错误
                print(f"警告: 计算标准化参数时处理文件 {full_path} 出错: {e}")
                continue # 跳过当前文件，继续处理下一个

        if all_signals: # 确保all_signals列表不为空
            # 将所有信号垂直堆叠成一个大的Numpy数组
            all_signals = np.concatenate(all_signals, axis=0)
            # 计算所有信号的均值，axis=0表示按列（即每个导联）计算均值，keepdims=True保持维度
            self.mean = np.mean(all_signals, axis=0, keepdims=True)
            # 计算所有信号的标准差
            self.std = np.std(all_signals, axis=0, keepdims=True)
            # 防止标准差为零导致除以零的错误，将零标准差替换为一个非常小的正数
            self.std[self.std == 0] = 1e-8
            print("均值和标准差计算完成。")
        else:
            # 如果没有有效信号，则均值和标准差设为None
            self.mean = None
            self.std = None
            print("警告: 未能计算标准化参数，数据集中没有有效信号。")

    def get_normalization_params(self):
        """
        返回计算出的均值和标准差。
        这个方法主要用于在训练集计算完参数后，将其传递给验证集或测试集。
        """
        return self.mean, self.std

    def __len__(self):
        """
        返回数据集中样本的总数。
        PyTorch DataLoader会调用此方法来确定迭代次数。
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        根据索引idx获取一个样本（ECG信号和对应的标签）。
        PyTorch DataLoader会调用此方法来获取每个批次的数据。
        """
        file_name = self.metadata.iloc[idx]['file_name'] # 根据索引获取文件名
        label = self.labels[idx] # 根据索引获取对应的二元编码标签

        full_path = os.path.join(self.ecg_data_path, file_name) # 构建完整的文件路径
        try:
            signal, _ = wfdb.rdsamp(full_path) # 读取ECG信号

            # 信号长度处理：填充或截断到input_length
            if signal.shape[0] < self.input_length:
                signal = np.pad(signal, ((0, self.input_length - signal.shape[0]), (0, 0)), mode='constant')
            else:
                signal = signal[:self.input_length]

            # 使用预先计算的均值和标准差进行标准化
            if self.mean is not None and self.std is not None:
                signal = (signal - self.mean) / self.std
            else:
                # 如果没有提供全局均值和标准差（例如，在测试阶段未传入），则在当前样本上计算并标准化
                mean = np.mean(signal, axis=0, keepdims=True)
                std = np.std(signal, axis=0, keepdims=True)
                std[std == 0] = 1e-8 # 防止除以零
                signal = (signal - mean) / std

            # 将Numpy数组转换为PyTorch张量，并调整维度顺序
            # 原始signal形状可能是 (length, channels)，PyTorch Conv1D需要 (batch_size, channels, length)
            signal_tensor = torch.tensor(signal, dtype=torch.float32).permute(1, 0) # permute(1, 0) 将 (length, channels) 变为 (channels, length)
            label_tensor = torch.tensor(label, dtype=torch.float32) # 将标签转换为浮点型张量

            return signal_tensor, label_tensor # 返回处理后的信号张量和标签张量

        except Exception as e:
            # 捕获处理文件时可能发生的错误
            print(f"处理文件 {full_path} 时出错: {e}")
            return None # 返回None，以便在collate_fn中过滤掉无效样本

# --- 3. 定义 Conv1D 分类模型 ---
class ECGClassifier(nn.Module):
    """
    基于一维卷积神经网络 (Conv1D) 的ECG分类模型。
    该模型设计用于处理ECG时间序列数据，通过多层卷积和池化提取特征，
    最后通过全连接层进行分类。
    """
    def __init__(self, num_channels, num_classes):
        """
        初始化ECGClassifier模型。
        Args:
            num_channels (int): 输入ECG信号的导联数量（通道数）。
            num_classes (int): 分类任务的类别数量。
        """
        super(ECGClassifier, self).__init__() # 调用父类nn.Module的构造函数
        self.conv_layers = nn.Sequential( # 定义卷积层序列
            # 第一层卷积：输入通道num_channels，输出64通道，卷积核大小7，填充3（保持输出长度不变）
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(), # ReLU激活函数
            nn.MaxPool1d(kernel_size=3, stride=2), # 最大池化层，池化核大小3，步长2（下采样）

            # 第二层卷积：输入64通道，输出128通道，卷积核大小5，填充2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            # 第三层卷积：输入128通道，输出256通道，卷积核大小3，填充1
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        # 自适应平均池化层：将每个通道的特征图池化为长度为1的向量
        # 无论输入特征图的长度是多少，输出都将是 (batch_size, 256, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_layers = nn.Sequential( # 定义全连接层序列
            # 第一个全连接层：输入256（来自avgpool），输出128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout层，以0.5的概率随机置零神经元，防止过拟合
            # 第二个全连接层（输出层）：输入128，输出num_classes（对应多标签分类的类别数）
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        定义模型的前向传播过程。
        Args:
            x (torch.Tensor): 输入张量，形状通常为 (batch_size, num_channels, input_length)。
        Returns:
            torch.Tensor: 模型的输出，形状为 (batch_size, num_classes)。
        """
        x = self.conv_layers(x) # 数据通过卷积层序列
        x = self.avgpool(x) # 数据通过自适应平均池化层，将特征图压缩为向量
        x = torch.flatten(x, 1) # 将多维张量展平为二维张量，从第二个维度开始展平（保留batch_size）
        x = self.fc_layers(x) # 展平后的数据通过全连接层序列
        return x # 返回模型的最终输出

# --- 4. 绘图函数 ---
def plot_metrics(history, save_dir):
    """
    绘制训练和验证过程中的损失曲线以及各项分类指标曲线。
    Args:
        history (dict): 包含训练历史记录的字典，包括损失、类别名称和每epoch的指标。
        save_dir (str): 保存图表的目录路径。
    """
    epochs = range(1, len(history['train_loss']) + 1) # 获取epoch的范围，用于X轴

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6)) # 创建一个新图表，设置大小
    plt.plot(epochs, history['train_loss'], label='Train Loss') # 绘制训练损失
    plt.plot(epochs, history['val_loss'], label='Validation Loss') # 绘制验证损失
    plt.title('Training and Validation Loss Over Epochs') # 设置图表标题
    plt.xlabel('Epochs') # 设置X轴标签
    plt.ylabel('Loss') # 设置Y轴标签
    plt.legend() # 显示图例
    plt.grid(True) # 显示网格
    plt.savefig(os.path.join(save_dir, 'loss_curve.png')) # 保存图表为PNG文件
    plt.show() # 显示图表

    # 绘制每类别的Precision, Recall, F1-score曲线
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 创建一个包含1行3列子图的图表
    fig.suptitle('Per-Class Metrics Over Epochs', fontsize=16) # 设置整个图表的总标题
    metrics = ['precision', 'recall', 'f1-score'] # 定义要绘制的指标名称
    titles = ['Precision', 'Recall', 'F1-score'] # 定义子图的标题
    class_names = history['class_names'] # 获取类别名称
    colors = plt.colormaps.get_cmap('tab10') # 获取颜色映射，用于区分不同类别的曲线

    for i, metric in enumerate(metrics): # 遍历每个指标
        ax = axes[i] # 获取当前指标对应的子图
        for j, cls_name in enumerate(class_names): # 遍历每个类别
            # 从历史记录中提取当前类别和指标的所有epoch值
            values = [h['class_metrics'][cls_name][metric] for h in history['epoch_history']]
            ax.plot(epochs, values, label=cls_name, color=colors(j)) # 绘制曲线
        ax.set_title(titles[i]) # 设置子图标题
        ax.set_xlabel('Epochs') # 设置X轴标签
        ax.set_ylabel('Score') # 设置Y轴标签
        ax.grid(True) # 显示网格
        if i == 0: # 只在第一个子图（Precision）上显示图例，避免重复
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 将图例放置在图表外部

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整子图布局，避免重叠，rect参数调整总标题和子图的间距
    plt.savefig(os.path.join(save_dir, 'per_class_metrics_curves.png')) # 保存图表
    plt.show() # 显示图表

    # 绘制总体（Micro Avg）Precision, Recall, F1-score曲线
    fig, ax = plt.subplots(figsize=(10, 6)) # 创建一个新图表
    ax.set_title('Overall Metrics (Micro Avg) Over Epochs') # 设置标题
    ax.set_xlabel('Epochs') # 设置X轴标签
    ax.set_ylabel('Score') # 设置Y轴标签
    overall_metrics = ['precision', 'recall', 'f1'] # 定义要绘制的总体指标
    titles = ['Overall Precision (Micro Avg)', 'Overall Recall (Micro Avg)', 'Overall F1-score (Micro Avg)'] # 定义图例标题

    for i, metric in enumerate(overall_metrics): # 遍历每个总体指标
        # 从历史记录中提取当前总体指标的所有epoch值
        values = [h['overall_metrics'][metric] for h in history['epoch_history']]
        ax.plot(epochs, values, label=titles[i]) # 绘制曲线

    ax.legend() # 显示图例
    ax.grid(True) # 显示网格
    plt.tight_layout() # 调整布局
    plt.savefig(os.path.join(save_dir, 'overall_micro_metrics_curves.png')) # 保存图表
    plt.show() # 显示图表

# --- 5. 主程序入口 ---
if __name__ == "__main__":
    # 当脚本直接运行时，执行以下代码块
    SEED = 42 # 定义随机种子，用于确保实验的可复现性
    set_seed(SEED) # 调用设置随机种子的函数

    csv_file_path = './datasets/label.csv' # 标签CSV文件的路径
    ecg_data_path = './datasets/records500' # ECG数据文件（.dat和.hea）的根目录路径
    BATCH_SIZE = 32 # 训练和验证时每个批次处理的样本数量
    EPOCHS = 50 # 训练的总轮数
    LEARNING_RATE = 0.001 # 优化器的学习率

    REPORT_DIR = './report' # 报告和保存模型文件的目录
    os.makedirs(REPORT_DIR, exist_ok=True) # 创建报告目录，如果已存在则不报错
    print(f"已确保报告目录存在: {REPORT_DIR}")

    MODEL_SAVE_PATH = os.path.join(REPORT_DIR, 'best_ecg_model_with_params.pth') # 最佳模型保存路径

    # 检查CUDA（GPU）是否可用，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 '{csv_file_path}' 不存在。请确保文件路径正确。")
        exit() # 如果文件不存在，则退出程序
    df_metadata = pd.read_csv(csv_file_path) # 读取CSV元数据文件到DataFrame

    # 将数据集划分为训练集和验证集，80%训练，20%验证
    train_df, val_df = train_test_split(df_metadata, test_size=0.2, random_state=SEED)

    # 训练集初始化：is_train=True，会在内部计算均值和标准差
    train_dataset = ECGDataset(train_df, ecg_data_path, is_train=True)
    # 获取训练集计算出的均值和标准差，用于后续验证集和测试集的标准化
    train_mean, train_std = train_dataset.get_normalization_params()

    # 验证集初始化：is_train=False，并传入训练集的均值和标准差进行标准化
    val_dataset = ECGDataset(val_df, ecg_data_path, mean=train_mean, std=train_std, is_train=False)

    def collate_fn(batch):
        """
        自定义collate_fn函数，用于DataLoader处理批次数据。
        主要目的是过滤掉ECGDataset.__getitem__中返回None的无效样本。
        """
        batch = [item for item in batch if item is not None] # 过滤掉批次中为None的样本
        if not batch: # 如果过滤后批次为空，则返回None
            return None, None
        signals, labels = zip(*batch) # 将信号和标签分别解包
        signals = torch.stack(signals) # 将信号张量堆叠成一个批次张量
        labels = torch.stack(labels) # 将标签张量堆叠成一个批次张量
        return signals, labels # 返回批次信号和标签

    # 创建训练数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    # 创建验证数据加载器
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    NUM_CHANNELS = 12 # ECG信号的导联数量（通道数），根据数据实际情况确定
    NUM_CLASSES = len(train_dataset.mlb.classes_) # 分类任务的类别数量，由MultiLabelBinarizer确定

    model = ECGClassifier(num_channels=NUM_CHANNELS, num_classes=NUM_CLASSES) # 实例化ECG分类模型
    criterion = nn.BCEWithLogitsLoss() # 定义损失函数：BCEWithLogitsLoss适用于多标签分类，它结合了Sigmoid和二元交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # 定义优化器：Adam优化器
    model.to(device) # 将模型移动到指定的设备（CPU或GPU）

    # 初始化历史记录字典，用于存储训练过程中的各项指标
    history = {
        'train_loss': [], # 存储每epoch的训练损失
        'val_loss': [], # 存储每epoch的验证损失
        'class_names': train_dataset.mlb.classes_, # 存储所有类别名称
        'epoch_history': [] # 存储每epoch的详细分类指标（包括每类和总体指标）
    }
    best_val_loss = float('inf') # 初始化最佳验证损失为无穷大，用于保存最佳模型

    print("\n开始训练...")
    for epoch in range(EPOCHS): # 遍历每个训练epoch
        # --- 训练阶段 ---
        model.train() # 将模型设置为训练模式（启用Dropout等）
        train_loss = 0.0 # 初始化当前epoch的训练损失
        tqdm_train = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]') # 创建训练进度条
        for signals, labels in tqdm_train: # 遍历训练数据加载器
            if signals is None: # 如果collate_fn返回None（表示批次中没有有效样本），则跳过
                continue
            signals, labels = signals.to(device), labels.to(device) # 将数据移动到指定设备
            optimizer.zero_grad() # 清除之前计算的梯度
            outputs = model(signals) # 前向传播，获取模型输出（logits）
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 更新模型参数
            train_loss += loss.item() # 累加当前批次的损失
            tqdm_train.set_postfix({'loss': f'{loss.item():.4f}'}) # 更新进度条的后缀信息

        # --- 验证阶段 ---
        model.eval() # 将模型设置为评估模式（禁用Dropout等）
        val_loss = 0.0 # 初始化当前epoch的验证损失
        all_val_preds = [] # 存储所有验证样本的预测结果
        all_val_labels = [] # 存储所有验证样本的真实标签
        tqdm_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Validation]') # 创建验证进度条
        with torch.no_grad(): # 在验证阶段禁用梯度计算，节省内存和计算
            for signals, labels in tqdm_val: # 遍历验证数据加载器
                if signals is None: # 如果collate_fn返回None，则跳过
                    continue
                signals, labels = signals.to(device), labels.to(device) # 将数据移动到指定设备
                outputs = model(signals) # 前向传播
                loss = criterion(outputs, labels) # 计算损失
                val_loss += loss.item() # 累加当前批次的损失
                preds = torch.sigmoid(outputs) > 0.5 # 将模型的logits通过Sigmoid函数转换为概率，然后阈值0.5转换为二元预测
                all_val_preds.append(preds.cpu().numpy()) # 将预测结果从GPU移到CPU并转换为Numpy数组
                all_val_labels.append(labels.cpu().numpy()) # 将真实标签从GPU移到CPU并转换为Numpy数组
                tqdm_val.set_postfix({'loss': f'{loss.item():.4f}'}) # 更新进度条的后缀信息

        avg_train_loss = train_loss / len(train_loader) # 计算平均训练损失
        avg_val_loss = val_loss / len(val_loader) # 计算平均验证损失

        history['train_loss'].append(avg_train_loss) # 记录平均训练损失
        history['val_loss'].append(avg_val_loss) # 记录平均验证损失

        print(f"\n--- Epoch [{epoch+1}/{EPOCHS}] ---")
        print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

        # 保存最佳模型 (基于验证损失)
        if avg_val_loss < best_val_loss: # 如果当前验证损失优于历史最佳
            best_val_loss = avg_val_loss # 更新最佳验证损失
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True) # 确保模型保存目录存在

            # 创建一个字典来保存模型状态字典、训练集的均值、标准差和MultiLabelBinarizer的类别信息
            checkpoint = {
                'model_state_dict': model.state_dict(), # 模型的参数（权重和偏置）
                'mean': train_mean, # 训练集均值
                'std': train_std, # 训练集标准差
                'mlb_classes': train_dataset.mlb.classes_ # MultiLabelBinarizer学习到的类别名称
            }
            torch.save(checkpoint, MODEL_SAVE_PATH) # 保存checkpoint到文件
            print(f"验证损失下降 ({best_val_loss:.4f})，已保存最佳模型和标准化参数到 '{MODEL_SAVE_PATH}'")

        # 将所有批次的预测结果和真实标签拼接起来，用于计算整体指标
        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_labels = np.concatenate(all_val_labels, axis=0)

        # 定义类别名称的中文映射，使报告更易读
        class_mapping = {
            'CD':   '传导障碍',
            'HYP':  '心室肥大',
            'MI':   '心肌梗死',
            'NORM': '正常心电',
            'STTC': '复极异常'
        }

        # 根据mlb.classes_获取对应的中文类别名称列表
        target_names_chinese = [class_mapping.get(cls, cls) for cls in history['class_names']]
        # 生成分类报告，output_dict=True使其返回一个字典，方便后续处理
        report = classification_report(all_val_labels, all_val_preds, target_names=target_names_chinese, zero_division=0, output_dict=True)
        print("\n验证集 Classification Report:")
        # 打印可读的分类报告
        print(classification_report(all_val_labels, all_val_preds, target_names=target_names_chinese, zero_division=0))

        # 提取并存储每个类别的Precision, Recall, F1-score
        class_metrics = {cls: {
            'precision': report[class_mapping.get(cls, cls)]['precision'],
            'recall': report[class_mapping.get(cls, cls)]['recall'],
            'f1-score': report[class_mapping.get(cls, cls)]['f1-score']
        } for cls in history['class_names']}

        # 计算并存储总体（Micro Average）Precision, Recall, F1-score
        # Micro Average适用于多标签分类，它计算的是所有样本和所有类别的总体的TP, FP, FN
        overall_metrics = {
            'precision': precision_score(all_val_labels, all_val_preds, average='micro', zero_division=0),
            'recall': recall_score(all_val_labels, all_val_preds, average='micro', zero_division=0),
            'f1': f1_score(all_val_labels, all_val_preds, average='micro', zero_division=0)
        }

        # 将当前epoch的详细指标添加到历史记录中
        history['epoch_history'].append({
            'class_metrics': class_metrics,
            'overall_metrics': overall_metrics
        })

    print("\n训练完成!")
    # 训练结束后，调用绘图函数可视化训练过程中的各项指标
    plot_metrics(history, REPORT_DIR)

