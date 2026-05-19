import argparse
import os
import pickle
import joblib
import pandas as pd
import numpy as np
import pywt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as Data
from tool_for_test import print_log
import numpy as np




def load_excel_files1(testwell, directory, target_column, sequence_length):
    """
    读取指定目录下的所有xlsx文件并合并成一个DataFrame
    """
    print("开始读取Excel文件...")
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    total_files = len(files)
    data_frames = []
    X_train_val, y_train_val = [],[]
    for i, file in enumerate(files):
        progress = (i + 1) / total_files * 100
        if file != testwell + '.xlsx' :
            print(f"正在读取文件: {file} (全文件读取进度：{progress:.2f}%)")
            df = pd.read_excel(os.path.join(directory, file))
            X, y = create_time_series(df, target_column, sequence_length)
            y1 = np.log10(y)
            X_train_val.append(X[:,:,:])
            y_train_val.append(y1)
            
    print("Excel文件读取完成。")
    return np.concatenate(X_train_val, axis=0),np.concatenate(y_train_val, axis=0)

def load_excel_files2(testwell, directory, target_column, sequence_length):
    """
    读取指定目录下的所有xlsx文件并合并成一个DataFrame，这次是测试数据
    """
    print("开始读取Excel文件...")
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    total_files = len(files)
    data_frames = []
    X_train_val, y_train_val = [],[]
    for i, file in enumerate(files):
        print(file)
        if file == testwell + '.xlsx':
            df = pd.read_excel(os.path.join(directory, file))
            X, y = create_time_series(df, target_column, sequence_length)
            y1 = np.log10(y)
            X_train_val.append(X[:,:,:])
            y_train_val.append(y1)
    print("Excel文件读取完成。")
    return np.concatenate(X_train_val, axis=0),np.concatenate(y_train_val, axis=0)
    # 归一化特征数据（X）
#     X_data = data.drop(columns=[target_column])
#     X_scaled = pd.DataFrame(scaler_X.fit_transform(X_data), columns=X_data.columns)
#
#     # 归一化目标数据（y）
#     y_data = data[[target_column]]
#     y_scaled = pd.DataFrame(scaler_y.fit_transform(y_data), columns=[target_column])
#

def create_time_series(data, target_column, sequence_length):
    """
    将DataFrame转换为时序数据
    """
    print("开始转换为时序数据...")
    X, y = [], []
    total_sequences = len(data) - sequence_length
    for i in range(total_sequences):
        X.append(data.iloc[i:i + sequence_length].drop([target_column], axis=1).values)
        # X.append(data.iloc[i:i + sequence_length].drop([target_column, 'CNL'], axis=1).values)
        y.append(data.iloc[i + sequence_length - 1][target_column])
    print("时序数据转换完成。")
    return np.array(X), np.array(y)

def create_time_series_mid(data, target_column, sequence_length):
    """
    将DataFrame转换为时序数据
    """
    print("开始转换为时序数据...")
    X, y = [], []
    total_sequences = len(data) - sequence_length
    for i in range(total_sequences):
        X.append(data.iloc[i:i + sequence_length].drop(target_column, axis=1).values)
        y.append(data.iloc[i + int(sequence_length/2) - 1][target_column])
    print("时序数据转换完成。")
    return np.array(X), np.array(y)

def split_data(X, y):
    """
    将数据分为训练集、验证集和测试集
    """
    print("开始划分数据集...")
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42)
    # print("数据集划分完成。")
    # return X_train, X_val, X_test, y_train, y_val, y_test

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    print("数据集划分完成。")
    return X_train, X_val, y_train, y_val


def create_data_loaders(X_train, X_val, y_train, y_val, batch_size=32):
    """
    创建数据加载器
    """
    print("开始创建数据加载器...")
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("数据加载器创建完成。")

    return train_loader, val_loader

class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, data):
        # 计算最小值和最大值
        if torch.is_tensor(data):
            data = data.numpy()
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)

    def nor(self, data):
        if torch.is_tensor(data):
            data = data.numpy()
            flag = 1
        else:
            flag = 0
        # 进行Min-Max归一化
        normalized_data = (data - self.min_val) / (self.max_val - self.min_val + 0.00000001)
        # normalized_data = data
        if flag == 1:
            return torch.from_numpy(normalized_data)
        else:
            return normalized_data

    def inverse_nor(self, data):
        if torch.is_tensor(data):
            data = data.numpy()
            flag = 1
        else:
            flag = 0
        inver_normalized_data = data * (self.max_val - self.min_val + 0.00000001) + self.min_val
        # inver_normalized_data=data
        if flag == 1:
            return torch.from_numpy(inver_normalized_data)
        else:
            return inver_normalized_data

def main(testwell, directory, target_column, sequence_length, batch_size=32):
    #############普通非小波分解################
    print("开始数据处理流程...")

    # 读取所有xlsx文件并合并# 将数据转换为时序数据
    X_train_val, y_train_val = load_excel_files1(testwell, os.path.join(directory, "全部数据"),target_column, sequence_length)
    # 划分训练集、验证集和测试集
    mm_x = Normalizer()
    mm_y = Normalizer()
    mm_x.fit(X_train_val)
    mm_y.fit(y_train_val)
    X_train_val_nor = mm_x.nor(X_train_val)
    y_train_val_nor = mm_y.nor(y_train_val)
    X_train, X_val, y_train, y_val = split_data(X_train_val_nor, y_train_val_nor)
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(X_train, X_val, y_train, y_val, batch_size)
###############################################################################
    x_test, y_test = load_excel_files2(testwell, os.path.join(directory, "全部数据"),target_column, sequence_length)
    x_test_nor = mm_x.nor(x_test)
    y_test_nor = mm_y.nor(y_test)
    test_nor_dataset = TensorDataset(torch.tensor(x_test_nor, dtype=torch.float32), torch.tensor(y_test_nor, dtype=torch.float32))
    test_nor_loader = DataLoader(test_nor_dataset, batch_size=batch_size, shuffle=False)
    
    # 打印数据集的形状
    print(f'注意：以下为输入情况：')
    print(f'训练集: X={X_train.shape}, y={y_train.shape}')
    print(f'验证集: X={X_val.shape}, y={y_val.shape}')
    print("数据预处理流程完成")
    outfile = os.path.join(directory, testwell)

    with open(os.path.join(outfile, 'mm_x.pickle'), "wb") as file:
        pickle.dump(mm_x, file)
    with open(os.path.join(outfile, 'mm_y.pickle'), "wb") as file:
        pickle.dump(mm_y, file)
    with open(os.path.join(outfile, 'train_loader.pkl'), 'wb') as f:
        pickle.dump(train_loader, f)
    with open(os.path.join(outfile, 'val_loader.pkl'), 'wb') as f:
        pickle.dump(val_loader, f)
    with open(os.path.join(outfile, 'test_loader.pkl'), 'wb') as f:
        pickle.dump(test_nor_loader, f)

    return train_loader, val_loader

def wavelet_and_decoder_for_y(sig00):
    sig0=sig00.squeeze()
    f_L = np.zeros(sig0.shape)
    f_H = np.zeros(sig0.shape)
    for i in range(sig0.shape[0]):
        sig = sig0[i, :]
        wavelet = 'db1'
        # 使用pywavelets库进行小波分解
        coeffs = pywt.wavedec(sig, wavelet, level=5)
        # 定义阈值，保留低频部分
        threshold = 6.04  # 该数值的计算公式为 threshold = 0.5 * np.max(np.abs(coeffs_x1[1]))
        # 阈值处理并进行小波重构
        coeffs_x1_thresholded = [c if np.abs(c).max() > threshold else np.zeros_like(c) for c in coeffs]
        recon2 = pywt.waverec(coeffs_x1_thresholded, wavelet)
        # 阈值处理并进行小波重构
        coeffs_x1_thresholded = [c if np.abs(c).max() < threshold else np.zeros_like(c) for c in coeffs]
        recon3 = pywt.waverec(coeffs_x1_thresholded, wavelet)
        f_L[i, :] = recon2
        f_H[i, :] = recon3
    f_L_new = torch.from_numpy(f_L).to(sig00.device).to(torch.float).unsqueeze(-1)
    f_H_new = torch.from_numpy(f_H).to(sig00.device).to(torch.float).unsqueeze(-1)
    return f_L_new, f_H_new
def pre_for_multi_scale(train_loader, batch_size, drop_last_yes_or_not):
    # f_L_new, f_H_new = wavelet_and_decoder(train_loader.dataset.tensors[0])
    # result1 = torch.stack((f_L_new, f_H_new), dim=1)
    f_L_new, f_H_new = wavelet_and_decoder_for_y(train_loader.dataset.tensors[1])
    result2 = torch.stack((f_L_new, f_H_new), dim=1)
    train_data2 = Data.TensorDataset(train_loader.dataset.tensors[0], result2)
    train_loader2 = torch.utils.data.DataLoader(dataset=train_data2, batch_size=batch_size, shuffle=False,
                                                drop_last=drop_last_yes_or_not)

    return train_loader2

def main2(testwell, directory, target_column, sequence_length, batch_size):
    ##############多尺度的小波分解数据集分析################
    print("开始数据处理流程...")

    # 读取所有xlsx文件并合并# 将数据转换为时序数据
    X_train_val, y_train_val = load_excel_files1(testwell, os.path.join(directory, "全部数据"),target_column, sequence_length)
    # 划分训练集、验证集和测试集
    y1 = np.expand_dims(np.expand_dims(y_train_val, axis=1), axis=2)
    y_train_val = np.repeat(y1, sequence_length, axis=1)

    mm_x = Normalizer()
    mm_y = Normalizer()
    mm_x.fit(X_train_val)
    mm_y.fit(y_train_val)
    X_train_val_nor = mm_x.nor(X_train_val)
    y_train_val_nor = mm_y.nor(y_train_val)
    X_train, X_val, y_train, y_val = split_data(X_train_val_nor, y_train_val_nor)
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(X_train, X_val, y_train, y_val, batch_size)
###############################################################################
    x_test, y_test = load_excel_files2(testwell, os.path.join(directory, "全部数据"),target_column, sequence_length)
    y2 = np.expand_dims(np.expand_dims(y_test, axis=1), axis=2)
    y_test = np.repeat(y2, sequence_length, axis=1)
    
    x_test_nor = mm_x.nor(x_test)
    y_test_nor = mm_y.nor(y_test)
    test_nor_dataset = TensorDataset(torch.tensor(x_test_nor, dtype=torch.float32), torch.tensor(y_test_nor, dtype=torch.float32))
    test_nor_loader = DataLoader(test_nor_dataset, batch_size=batch_size, shuffle=False)
    
    # 打印数据集的形状
    print(f'注意：以下为输入情况：')
    print(f'训练集: X={X_train.shape}, y={y_train.shape}')
    print(f'验证集: X={X_val.shape}, y={y_val.shape}')
    print("数据预处理流程完成")
    outfile = os.path.join(directory, testwell)
    
    train_loader2 = pre_for_multi_scale(train_loader,batch_size,True)
    val_loader2  = pre_for_multi_scale(val_loader, batch_size,False)
    test_loader2  = pre_for_multi_scale(test_nor_loader, batch_size,False)

    with open(os.path.join(outfile, 'mm_x2.pickle'), "wb") as file:
        pickle.dump(mm_x, file)
    with open(os.path.join(outfile, 'mm_y2.pickle'), "wb") as file:
        pickle.dump(mm_y, file)
    with open(os.path.join(outfile, 'train_loader2.pkl'), 'wb') as f:
        pickle.dump(train_loader2, f)
    with open(os.path.join(outfile, 'val_loader2.pkl'), 'wb') as f:
        pickle.dump(val_loader2, f)
    with open(os.path.join(outfile, 'test_loader2.pkl'), 'wb') as f:
        pickle.dump(test_loader2, f)

    return train_loader, val_loader


def load_data_loaders(args):
    """
    从存储的路径中加载数据加载器
    """
    # 读取存储的目录路径
    save_directory = 'data_save/本次数据读取的缓存'
    with open(os.path.join(save_directory, 'train_loader.pkl'), 'rb') as f:
        train_loader = pickle.load(f)
    with open(os.path.join(save_directory, 'val_loader.pkl'), 'rb') as f:
        val_loader = pickle.load(f)
    with open(os.path.join(save_directory, 'test_loader.pkl'), 'rb') as f:
        test_loader = pickle.load(f)

    print("数据加载器已从目录加载: ", save_directory)
    print_log(f"数据加载器已从目录加载: {save_directory}", args)
    return train_loader, val_loader, test_loader


def parse_int_list(arg):
    return [int(x) for x in arg.split(',')]


def get_parameters(name):
    parser = argparse.ArgumentParser(description='训练模型的脚本')
    ## model
    parser.add_argument('--num_layers', type=int, default=4, help='层的数量')
    parser.add_argument('--dropout', type=float, default=0.2, help='丢失概率')

    ## kan
    parser.add_argument('--grid_size', type=int, default=200, help='grid')

    ## TCN
    parser.add_argument('--num_channels', type=parse_int_list, default=[25, 50, 25])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--args.dropout')

    ## transformer
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_space', type=int, default=32)

    # training
    parser.add_argument('--num_epochs', type=int, default=100, help='训练的轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')

    # data
    # DEN输入维度为14, CNL输入维度为12
    if name == 'DEN':
        parser.add_argument('--model_name', type=str, default="Transformer_KAN", help='选择一个：LSTM,TCN,Transformer,Transformer_KAN,BiLSTM,GRU')
        parser.add_argument('--hidden_size', type=int, default=32, help='隐藏层的神经元数量')
        parser.add_argument('--input_directory', type=str, default=r'data_save\data\data_den', help='输入地址')
        parser.add_argument('--predict_target', type=str, default='DEN', help='预测目标')
        parser.add_argument('--input_size', type=int, default=7, help='输入特征的维度')
        parser.add_argument('--sequence_length', type=int, default=40, help='时序数据的长度')
        parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
        parser.add_argument('--output_size', type=int, default=1, help='输出特征的维度')
    
    if name == 'CNL':
        parser.add_argument('--model_name', type=str, default="Transformer_KAN", help='选择一个：LSTM,TCN,Transformer,Transformer_KAN,BiLSTM,GRU')
        parser.add_argument('--hidden_size', type=int, default=32, help='隐藏层的神经元数量')
        parser.add_argument('--input_directory', type=str, default=r'data_save\data\data_cnl', help='输入地址')
        parser.add_argument('--predict_target', type=str, default='CNL', help='预测目标')
        parser.add_argument('--input_size', type=int, default=11, help='输入特征的维度')
        parser.add_argument('--sequence_length', type=int, default=40, help='时序数据的长度')
        parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
        parser.add_argument('--output_size', type=int, default=1, help='输出特征的维度')
    
    if name == 'RT':
        parser.add_argument('--model_name', type=str, default="MultiScale_Attention_Res_shrink_LSTM", help='选择一个：LSTM,TCN,Transformer,Transformer_KAN,BiLSTM,GRU')
        parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层的神经元数量')
        parser.add_argument('--input_directory', type=str, default=r'data_save\data\data_rt', help='输入地址')
        parser.add_argument('--predict_target', type=str, default='RT', help='预测目标')
        parser.add_argument('--input_size', type=int, default=103, help='输入特征的维度')
        parser.add_argument('--sequence_length', type=int, default=80, help='时序数据的长度')
        parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
        parser.add_argument('--output_size', type=int, default=1, help='输出特征的维度')
    args = parser.parse_args()
    return args

