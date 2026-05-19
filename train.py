import torch
import torch.nn as nn
import torch.optim as optim
from data_pre import data_pre_process
from model_BiLSTM import BiLSTM
from model_GRU import GRU
from model_LSTM import LSTM
from model_TCN import TemporalConvNet
from model_Trans_KAN import TimeSeriesTransformer_ekan
from model_Trans_KAN_large import TimeSeriesTransformer_ekan_large
from model_Transformer import TransformerModel
from tool_for_test import print_log
from tool_for_pre import get_parameters, load_data_loaders, pre_for_multi_scale
from tool_for_train import train_model
from tool_for_model import MultiScale_Attention_Res_shrink_LSTM
import os
from datetime import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tool_for_pre import*
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tool_for_test import plot_results
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman']

targetname='RT'
testwell = '郑36-8'

print(targetname+testwell)

args = get_parameters(targetname)
outfile = os.path.join(args.input_directory, testwell)
if not os.path.exists(outfile):
    os.makedirs(outfile)
    
# 数据预处理（初次运行即可，运行后结果保存到data_save文件夹内）
data_pre_process(testwell,targetname)

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
    elif args.model_name == 'TimeSeriesTransformer_ekan_large':
        model = TimeSeriesTransformer_ekan_large(input_dim=args.input_size, num_heads=args.num_heads, num_layers=args.num_layers, num_outputs=args.output_size, hidden_space=args.hidden_space, dropout_rate=args.dropout)
    elif args.model_name == 'MultiScale_Attention_Res_shrink_LSTM':
        model = MultiScale_Attention_Res_shrink_LSTM(input_size=args.input_size, hidden_size=args.hidden_space, num_layers=args.num_layers, output_size=args.output_size, batch_size= args.batch_size, seq_length = args.sequence_length, num_heads=args.num_heads)
    else:
        print('please choose correct model name')
    return model

# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_log(f"The device being used is: {device}",args)

# 读取数据
if targetname=='RT':
    with open(os.path.join(outfile, 'train_loader2.pkl'), "rb") as file:
        train_loader = pickle.load(file)
    with open(os.path.join(outfile, 'val_loader2.pkl'), "rb") as file:
        val_loader = pickle.load(file)
    with open(os.path.join(outfile, 'test_loader2.pkl'), "rb") as file:
        test_nor_loader = pickle.load(file)
else:
    with open(os.path.join(outfile, 'train_loader.pkl'), "rb") as file:
        train_loader = pickle.load(file)
    with open(os.path.join(outfile, 'val_loader.pkl'), "rb") as file:
        val_loader = pickle.load(file)
    with open(os.path.join(outfile, 'test_loader.pkl'), "rb") as file:
        test_nor_loader = pickle.load(file)
with open(os.path.join(outfile, 'mm_x.pickle'), "rb") as file:
    mm_x = pickle.load(file)
with open(os.path.join(outfile, 'mm_y.pickle'), "rb") as file:
    mm_y = pickle.load(file)


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2
def r2_loss1(output, target):
    output_tensor = torch.from_numpy(output)
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output_tensor) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2
# 训练模型
# test(args, model_file_path)
best_score = -99999
best_epoch=-1
num_epochs = args.num_epochs
# 使用 ReduceLROnPlateau 调度器

save_path = os.path.join(outfile, "models_save" + datetime.now().strftime("--%d--%H--%M--%S"))
final_model_file_path = os.path.join(save_path, 'lstm_model_final.pth')
os.makedirs(save_path, exist_ok=True)

circle=1
best_r2=-999
for i in range(circle):
    print(f"第{i+1}次循环++++++++++++++++++++++++++++++++++++++++++++")
    model = define_your_model(args).to(device)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=10, verbose=True)
    train_losses = []
    val_losses = []
    val_mres = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs1, targets1 in train_loader:
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            outputs1 = model(inputs1[:,:,1:])
            loss = criterion(outputs1, targets1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        test_loss = 0
        val_mre = 0
        with torch.no_grad():
            for inputs2, targets2 in val_loader:
                inputs2, targets2 = inputs2.to(device), targets2.to(device)
                outputs2 = model(inputs2[:,:,1:])
                loss = criterion(outputs2, targets2)
                mre = torch.mean((outputs2-targets2)/(targets2+0.01) )
                val_loss += loss.item()
                val_mre += mre.item()
            for inputs3, targets3 in test_nor_loader:
                inputs3, targets3 = inputs3.to(device), targets3.to(device)
                outputs3 = model(inputs3[:,:,1:])
                loss_t = r2_loss(outputs3, targets3) 
                test_loss += loss_t.item()
        test_loss /= len(test_nor_loader)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_mre /= len(val_loader)
        val_mres.append(val_mre)
        
        if val_loss > best_score:
            best_score = val_loss
            best_epoch = epoch
            best_model = model.state_dict()
    
        # 调度器根据验证集的损失调整学习率
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, test_loader R2: {test_loss:.4f}, lr:{lr:.4f}')
        print_log(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}',args)
        # 每5个epoch保存一次模型
        if (epoch + 1) % 2 == 0:
            model_file_path = os.path.join(save_path, f'{args.model_name}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_file_path)
            # test_main(args, model_file_path)
            # print(f"模型参数已保存至 {model_file_path}")

    if best_score > best_r2:
        best_r2 = best_score
        print(f"最好的测试loss为{best_r2:.4f}")
        # 保存最佳模型
        torch.save(best_model,  os.path.join(outfile, 'val_model.pth'))
        print(f"最好模型已保存至 models_save/val_model.pth, R2 for val_data is: {best_r2:.4f} 在第{i+1}个循环，第{best_epoch+1}个epoch")
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', linestyle='-', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_file_path = os.path.join(save_path, 'loss_plot.png')
        plt.savefig(loss_plot_file_path)  # 保存图像到指定目录
        print(f"损失图已保存至 {loss_plot_file_path}")
        
        
        
print("训练完成")
#############################################test##################################################################
#############################################test##################################################################

y_label = '密度'
if args.predict_target=='CNL':
    y_label ='中子'
elif args.predict_target=='DEN':
    y_label ='密度'
elif args.predict_target=='RT':
    y_label ='Log RT'

def read_excel_files(directory):
    """
    读取指定目录下的所有Excel文件，并返回文件名列表
    """
    excel_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            excel_files.append(os.path.join(directory, filename))
    return excel_files

# 获取当前时间
current_time = datetime.now()
# 格式化时间字符串
formatted_time = current_time.strftime("--%H--%M--")
# 调用测试函数，这里假设您的测试函数是test(args)
model_test = define_your_model(args).to(device)
model_file_path = os.path.join(outfile, 'val_model.pth')
model_test.load_state_dict(torch.load(model_file_path))
# 迭代测试集，对每个batch进行预测
model_test.eval()
print("模型已加载")
# 预测与实际值
    
predictions = []
depth = mm_x.inverse_nor(test_nor_loader.dataset.tensors[0])[:,0,0]
print("开始测试模型")
with torch.no_grad():
    for inputs_test, targets_test in test_nor_loader:
        inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
        outputs_test = model_test(inputs_test[:,:,1:])
        predictions.extend(outputs_test.cpu().numpy())
        
predictions = np.array(predictions)
# 反归一化并转换为numpy数组
all_targets = mm_y.inverse_nor(test_nor_loader.dataset.tensors[1])
all_predictions = mm_y.inverse_nor(predictions)


if args.predict_target=='RT':
    targets_multi=test_nor_loader.dataset.tensors[1][:,0,:,:]+test_nor_loader.dataset.tensors[1][:,1,:,:]
    predictions_multi=predictions[:,0,:,:]+predictions[:,1,:,:]
    all_targets = mm_y.inverse_nor(targets_multi)[:,0,0]
    all_predictions = mm_y.inverse_nor(predictions_multi)[:,0,0]

# 计算各项指标
mse = mean_squared_error(all_targets, all_predictions)
mae = mean_absolute_error(all_targets, all_predictions)
rmse = np.sqrt(mse)
r2 = r2_loss1(all_predictions,all_targets)
abs_err = np.mean(all_targets.numpy()-all_predictions)
rela_err = np.mean((all_targets.numpy()-all_predictions)/all_targets.numpy())
# 保存结果到Excel
metrics_df = pd.DataFrame({
'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'Root Mean Squared Error', 'R^2 Score', 'Absolute Error', 'Relative Error'],
'Value': [mse, mae, rmse, r2, abs_err, rela_err]})

# 创建包含预测值和真实值的 DataFrame
predictions_df = pd.DataFrame({
    'True Value': all_targets,
    'Predicted Value': all_predictions})
# 将两个 DataFrame 合并
print(f"R2 for test_data is: {r2:.5f},MAE for test_data is: {abs_err:.5f},MRE for test_data is: {rela_err:.5f}")
# 提取模型文件的目录
model_dir = os.path.dirname(model_file_path)
model_dir = os.path.join(model_dir, f"全部井测试-{formatted_time}")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# 开始存储结果
excel_save_path1 = os.path.join(model_dir, f"{testwell}--指标26.xlsx")
metrics_df.to_excel(excel_save_path1, index=False)
excel_save_path2 = os.path.join(model_dir, f"{testwell}--数据26.xlsx")
predictions_df.to_excel(excel_save_path2, index=False)

# 定义保存文件的路径
plot_save_path = os.path.join(model_dir, f"{testwell}--{round(r2.item(), 2)}.png")
# 保存路径
depth1=np.arange(len(all_predictions))
plot_results(all_targets, all_predictions, plot_save_path,depth1)


depth1=np.arange(len(all_predictions))
mpl.rcParams['font.family'] = 'SimHei'
plt.figure()
plt.plot(depth[:], all_predictions[:], label='预测值', color='blue', linestyle='-',linewidth=1)
plt.plot(depth[:], all_targets[:], label='真实值', color='red', linestyle='-',linewidth=0.5)
# half_length = int(np.floor(len(depth) / 2))
# plt.plot(depth[half_length:], all_predictions[half_length:], label='真实值', color='blue', linestyle='-')
# plt.plot(depth[half_length:], all_targets[half_length:], label='预测值', color='red', linestyle='-')
plt.xlabel('深度')
plt.ylabel(y_label)
plt.title(y_label+'的预测对比图')
plt.legend()
loss_plot_file_path = os.path.join(model_dir, f"测试井{testwell}的预测图.png")
plt.savefig(loss_plot_file_path)  # 保存图像到指定目录
plt.show()
print(f"损失图已保存至 {loss_plot_file_path}")

print(testwell)
