from tool_for_pre import main, main2, get_parameters
import torch
import numpy as np
import os
import pandas as pd

def data_pre_process(testwell,targetname):
    args = get_parameters(targetname)
    directory = args.input_directory # 替换为你的目录路径
    target_column = args.predict_target  # 替换为你的目标列名称
    sequence_length = args.sequence_length  # 替换为你的时序数据长度
    batch_size = args.batch_size  # 替换为你的批次大小

    train_loader, val_loader = main(testwell, directory, target_column, sequence_length, batch_size)
    train_loader2, val_loader2 = main2(testwell, directory, target_column, sequence_length, batch_size)
    
    
