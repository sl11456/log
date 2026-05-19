import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_results(true_values, predicted_values, save_path,depth):
    """
    绘制真实值和预测值的散点图和曲线图，并保存到指定路径。

    Parameters:
    true_values (array-like): 真实值
    predicted_values (array-like): 预测值
    save_path (str): 保存结果的文件路径
    """

    # 配置字体
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']

    # 确保输入为 numpy 数组
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # 创建图形和子图
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

    # 散点图
    axs[0].scatter(true_values, predicted_values, alpha=0.7, edgecolors='k', s=50, c='blue', marker='o')
    axs[0].plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--', lw=2)
    axs[0].set_xlabel('True Values', fontsize=14)
    axs[0].set_ylabel('Predicted Values', fontsize=14)
    axs[0].set_title('Scatter Plot', fontsize=16)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].set_aspect('equal', adjustable='box')  # 保持x轴和y轴的比例

    # 曲线图
    axs[1].plot(depth, true_values, label='True Values', color='blue', linewidth=1)
    axs[1].plot(depth, predicted_values, label='Predicted Values', color='red', linewidth=0.5)
    axs[1].set_xlabel('Points number', fontsize=14)
    axs[1].set_ylabel('Values', fontsize=14)
    axs[1].set_title('Line Plot', fontsize=16)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # print(f"图像已保存到 {save_path}")


def print_log(content, arg, log_file_path="output/"):
    # 确保日志文件夹存在
    log_file_path2 = log_file_path + arg.model_name + "_log.txt"
    os.makedirs(os.path.dirname(log_file_path2), exist_ok=True)
    # 打开文件并写入内容
    with open(log_file_path2, 'a') as log_file:
        log_file.write(content + "\n")
