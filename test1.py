# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:00:02 2024

@author: Administrator
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import torch
from model_BiLSTM import BiLSTM
from model_GRU import GRU
from model_LSTM import LSTM
from model_TCN import TemporalConvNet
from model_Trans_KAN import TimeSeriesTransformer_ekan
from model_Transformer import TransformerModel
from tool_for_pre import get_parameters, create_time_series
from tool_for_test import plot_results
from matplotlib import pyplot as plt
from tool_for_pre import*

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
args = get_parameters()
y_label = '密度'
if args.predict_target=='CNL':
    y_label ='中子'
else:
    y_label ='密度'

def read_excel_files(directory):
    """
    读取指定目录下的所有Excel文件，并返回文件名列表
    """
    excel_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            excel_files.append(os.path.join(directory, filename))
    return excel_files

def define_your_model(args):
    """
    根据参数定义模型
    """
    # 根据args.model_name选择不同的模型
    if args.model_name == 'GRU':
        model = GRU(input_dim=args.input_size, hidden_dim=args.hidden_size, num_layers=args.num_layers, output_dim=args.output_size)
    elif args.model_name == 'LSTM_ekan':
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size)
    elif args.model_name == 'BiLSTM':
        model = BiLSTM(input_dim=args.input_size, hidden_dim=args.hidden_size, num_layers=args.num_layers, output_dim=args.output_size)
    elif args.model_name == 'TCN':
        model = TemporalConvNet(num_inputs=args.input_size, num_outputs=args.hidden_size, num_channels=args.num_channels, kernel_size=args.kernel_size, dropout=args.dropout)
    elif args.model_name == 'Transformer':
        model = TransformerModel(args.input_size, args.hidden_size, args.num_layers, args.output_size)
    elif args.model_name == 'Transformer_KAN':
        model = TimeSeriesTransformer_ekan(input_dim=args.input_size, num_heads=args.num_heads, num_layers=args.num_layers, num_outputs=args.output_size, hidden_space=args.hidden_space, dropout_rate=args.dropout)
    else:
        raise ValueError('Please choose a correct model name')
    return model

model_file_path = "models_save/val_model.pth"
with open("data_save/mm_x.pickle", "rb") as file:
    mm_x = pickle.load(file)
with open("data_save/mm_y.pickle", "rb") as file:
    mm_y = pickle.load(file)
args = get_parameters("Transformer_KAN")
input_directory = args.input_directory
input_directory = os.path.join(input_directory, "测试集")
excel_files = read_excel_files(input_directory)
# 获取当前时间
current_time = datetime.now()
# 格式化时间字符串
formatted_time = current_time.strftime("--%H--%M--")
# 逐个处理每个Excel文件
for file_path in excel_files:
    print(f"处理文件: {file_path}")
    last_directory = os.path.basename(file_path)
    # 读取Excel文件为DataFrame
    data = pd.read_excel(file_path)
    # 创建时序数据
    sequence_length = args.sequence_length
    target_column = args.predict_target
    X_test, y_test = create_time_series(data, target_column, sequence_length)
    X_test_nor = mm_x.nor(X_test)
    y_test_nor = mm_y.nor(y_test)
    
    print(f'测试井{last_directory}的shape: X={X_test.shape}, y={y_test.shape}')
    # 创建单独的数据加载器
    batch_size = args.batch_size   
    test_nor_dataset = TensorDataset(torch.tensor(X_test_nor, dtype=torch.float32), torch.tensor(y_test_nor, dtype=torch.float32))
    test_nor_loader = DataLoader(test_nor_dataset, batch_size=batch_size, shuffle=False)
    
    # 调用测试函数，这里假设您的测试函数是test(args)
    model = define_your_model(args)
    # 使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(model_file_path))
    # 迭代测试集，对每个batch进行预测
    model.eval()
    print("模型已加载")
    # 预测与实际值
    all_predictions = []
    all_targets = y_test
    depth = X_test[:,0,0]
    print("开始测试模型")
    with torch.no_grad():
        for inputs, targets in test_nor_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            
    # 反归一化并转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_predictions = mm_y.inverse_nor(all_predictions)
    
    # 计算各项指标
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    abs_err = np.mean(all_targets-all_predictions)
    rela_err = np.mean((all_targets-all_predictions)/all_targets)
    # 保存结果到Excel
    results_df = pd.DataFrame({
        'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'Root Mean Squared Error', 'R^2 Score', 'abs Error','rela Error'],
        'Value': [mse, mae, rmse, r2, abs_err, rela_err]
    })
    
    print(f"R2 for test_data is: {r2:.5f},MAE for test_data is: {abs_err:.5f},MRE for test_data is: {rela_err:.5f}")
    # 提取模型文件的目录
    model_dir = os.path.dirname(model_file_path)
    model_dir = os.path.join(model_dir, f"全部井测试-{formatted_time}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # 开始存储结果
    excel_save_path = os.path.join(model_dir, f"{last_directory}--{round(r2, 2)}.xlsx")
    results_df.to_excel(excel_save_path, index=False)

    # 定义保存文件的路径
    plot_save_path = os.path.join(model_dir, f"{last_directory}--{round(r2, 2)}.png")
    # 保存路径
    plot_results(all_targets, all_predictions, plot_save_path,depth)
    
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'SimHei'
    plt.figure()
    plt.plot(depth, all_predictions, label='预测值', color='blue', linestyle='-',linewidth=1)
    plt.plot(depth, all_targets, label='真实值', color='red', linestyle='-',linewidth=0.5)
    # half_length = int(np.floor(len(depth) / 2))
    # plt.plot(depth[half_length:], all_predictions[half_length:], label='真实值', color='blue', linestyle='-')
    # plt.plot(depth[half_length:], all_targets[half_length:], label='预测值', color='red', linestyle='-')
    plt.xlabel('深度')
    plt.ylabel(y_label)
    plt.title('密度的预测对比图')
    plt.legend()
    loss_plot_file_path = os.path.join(model_dir, f"测试井{last_directory}的预测图.png")
    plt.savefig(loss_plot_file_path)  # 保存图像到指定目录
    print(f"损失图已保存至 {loss_plot_file_path}")
    
