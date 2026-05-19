import os
from datetime import datetime
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from torch import optim
import numpy as np

# from test import test_main
from tool_for_test import print_log
from tool_for_pre import pre_for_multi_scale


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, args):
    train_losses = []
    val_losses = []
    val_mres = []
    
    best_score = 99999
    best_epoch=-1
    
    # 使用 ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    save_path = os.path.join("models_save", args.model_name + datetime.now().strftime("--%d--%H--%M--%S"))
    final_model_file_path = os.path.join(save_path, 'lstm_model_final.pth')
    os.makedirs(save_path, exist_ok=True)
    
    # train_loader2=pre_for_multi_scale(train_loader,32,True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        val_mre = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mre = torch.mean((outputs-targets)/(targets+0.01) )
                val_loss += loss.item()
                val_mre += mre.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_mre /= len(val_loader)
        val_mres.append(val_mre)
        
        if val_loss < best_score:
            best_score = val_loss
            best_epoch = epoch
            best_model = model.state_dict()

        # 调度器根据验证集的损失调整学习率
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, lr:{lr:.4f}')
        print_log(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}',args)
        # 每5个epoch保存一次模型
        if (epoch + 1) % 2 == 0:
            model_file_path = os.path.join(save_path, f'{args.model_name}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_file_path)
            # test_main(args, model_file_path)
            # print(f"模型参数已保存至 {model_file_path}")

    print("训练完成")

    # 最终保存模型
    torch.save(model.state_dict(), final_model_file_path)
    print(f"最终训练完epoch模型已保存至 {final_model_file_path}")
    # 保存最佳模型
    torch.save(best_model, "models_save/val_model.pth")
    print(f"最好模型已保存至 models_save/val_model.pth, R2 for val_data is: {best_score:.2f} 在第{best_epoch+1}个epoch")

    # 绘制损失图并保存
    # 配置字体
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', linestyle='-', marker='o')
    plt.plot(range(1, num_epochs + 1), val_mres, label='Validation MRE', color='black', linestyle='-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_file_path = os.path.join(save_path, 'loss_plot.png')
    plt.savefig(loss_plot_file_path)  # 保存图像到指定目录
    print(f"损失图已保存至 {loss_plot_file_path}")
    return final_model_file_path
