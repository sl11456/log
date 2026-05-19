import numpy as np
import pywt
import torch
from torch import nn
from tool_for_pre import  pre_for_multi_scale

def get_model(model_name,input_size, hidden_size, num_layers, output_size, batch_size, seq_length):
    if model_name=='LSTM':
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
    elif model_name=='GUR':
        model = GRU(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
    elif model_name=='ResidualLSTM':
        model = ResidualLSTM(input_size, hidden_size, num_layers)
    elif model_name == 'SelfAttention_LSTM':
        model = SelfAttention_LSTM(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
    elif model_name =='SelfAttention_Res_LSTM':
        model = SelfAttention_Res_LSTM(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    elif model_name =='MultiScale_Attention_LSTM':
        model = MultiScale_Attention_LSTM(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    elif model_name =='MultiScale_Attention_Res_LSTM':
        model = MultiScale_Attention_Res_LSTM(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    elif model_name =='MMD-LSTM':
        model = MultiScale_Attention_Res_shrink_LSTM(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    elif model_name =='RNN':
        model = RNN(input_size,hidden_size,num_layers)
    elif model_name =='HRNN':
        model = HierarchicalRNN(input_size,hidden_size,num_layers)
    elif model_name =='BiGRU':
        model = BiGRU(input_size,hidden_size,num_layers)
    elif model_name =='BiLSTM':
        model = BiLSTM(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    elif model_name =='Attention_Rnn':
        model = Attention_Rnn(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    elif model_name =='Attention_GRU':
        model = Attention_GRU(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    elif model_name == 'Attention_LSTM':
        model = Attention_LSTM(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
    elif model_name == 'Attention_BiLSTM':
        model = Attention_BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
    elif model_name == 'Res_Attention_LSTM':
        model = Res_Attention_LSTM(input_size, hidden_size, num_layers)
    elif model_name == 'Res_Attention_BiLSTM':
        model = Res_Attention_BiLSTM(input_size, hidden_size, num_layers)
    elif model_name =='DMS-LSTM':
        model = MultiScale_LSTM(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    elif model_name =='MS-LSTM':
        model = Res_Shrink_LSTM(input_size, hidden_size, num_layers, output_size)
    elif model_name =='Multi_Res_Shrink_LSTM':
        model = Multiple_Res_Shrink_LSTM(input_size, hidden_size, num_layers, output_size)
    elif model_name =='Res_LSTM':
        model = Res_LSTM(input_size, hidden_size, num_layers)
    elif model_name =='Res_RNN':
        model = Res_RNN(input_size, hidden_size, num_layers)

    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)
    return model


class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length) -> None:
        super(LSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=1 # 单向LSTM
        self.relu = nn.ReLU()

        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=0.5) # LSTM层
        self.fc=nn.Linear(hidden_size,output_size) # 全连接层

    def forward(self,x):
        #设置初始隐藏状态，也可以不设置
        #batch_size, seq_len = x.size[0], x.size[1]    # x.shape=(604,3,3)
        # h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        #堆叠模型层
        output, _ = self.lstm(x) # x(批数量, 序列长度, 特征数)到output(批数量, 序列长度, 神经元数)
        output = self.relu(output)
        pred = self.fc(output)  # output(批数量, 序列长度, 神经元数)到pred(批数量, 序列长度, 1)
        pred = pred  # pred(批数量, 序列长度, 1)，之所以这么选是因为这样每个时间步输出到同一个全链接
        return pred


class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ResidualLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.linear_projection = nn.Linear(input_size, hidden_size)
    def forward(self, x):
        # LSTM层
        out, _ = self.lstm(x)
        # 线性投影
        x_proj = self.linear_projection(x)
        # 残差连接
        out = self.relu(out + x_proj)
        # 全连接层
        out = self.fc(out)

        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,batch_size,seq_length):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.fc(out)  # 取最后一个时间步的输出

        return out

class SelfAttention_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length,num_heads=4) -> None:
        super(SelfAttention_LSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=1 # 单向LSTM
        self.relu = nn.ReLU()

        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True) # LSTM层
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)#注意力层
        self.fc=nn.Linear(hidden_size*2,1) # 全连接层

    def forward(self, input_data):

        output, (hidden, cell) = self.lstm(input_data)
        output = self.relu(output)
        # 计算注意力权重、计算注意力向量
        linear_output = output.transpose(0, 1)  # 将时间步和批次维度交换
        attention_output, attn_weights = self.attention(linear_output, linear_output, linear_output)
        attention_output = attention_output.transpose(0, 1)  # 将时间步和批次维度交换
        # 将自注意力输出结果与LSTM输出结果进行拼接
        combined_output = torch.cat([output, attention_output], dim=2)
        output = self.fc(combined_output)
        output = output  # pred(批数量, 序列长度, 1)只取最后一个序列的预测结果作为我们的预测结果
        return output

class SelfAttention_Res_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length,num_heads=4) -> None:
        super(SelfAttention_Res_LSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=1 # 单向LSTM
        self.relu = nn.ReLU()

        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True) # LSTM层
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)#注意力层
        self.fc=nn.Linear(hidden_size*2,1) # 全连接层
        self.linear_projection = nn.Linear(input_size, hidden_size)
    def forward(self, input_data):

        output, (hidden, cell) = self.lstm(input_data)
        # 线性投影
        x_proj = self.linear_projection(input_data)
        # 残差连接
        output = self.relu(output + x_proj)
        # 计算注意力权重、计算注意力向量
        linear_output = output.transpose(0, 1)  # 将时间步和批次维度交换
        attention_output, attn_weights = self.attention(linear_output, linear_output, linear_output)
        attention_output = attention_output.transpose(0, 1)  # 将时间步和批次维度交换
        # 将自注意力输出结果与LSTM输出结果进行拼接
        combined_output = torch.cat([output, attention_output], dim=2)
        output = self.fc(combined_output)
        output = output  # pred(批数量, 序列长度, 1)只取最后一个序列的预测结果作为我们的预测结果
        return output


def wavelet_and_decoder(sig00):
    # sig0=sig00.cpu().numpy()
    sig0=sig00
    f_L = np.zeros(sig0.shape)
    f_H = np.zeros(sig0.shape)
    for i in range(sig0.shape[0]):
        for j in range(sig0.shape[2]):
            sig = sig0[i, :, j]
            wavelet = 'db1'
            # 使用pywavelets库进行小波分解
            coeffs = pywt.wavedec(sig, wavelet, level=5)
            cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
            # 去掉
            coeffs2 = [cA5, cD5, cD4, np.zeros(cD3.shape), np.zeros(cD2.shape), np.zeros(cD1.shape)]
            recon2 = pywt.waverec(coeffs2, wavelet)
            # 去掉
            coeffs3 = [np.zeros(cA5.shape), np.zeros(cD5.shape), np.zeros(cD4.shape), cD3, cD2, cD1]
            recon3 = pywt.waverec(coeffs3, wavelet)
            f_L[i, :, j] = recon2
            f_H[i, :, j] = recon3
    f_L_new=torch.from_numpy(f_L).to(sig00.device).to(torch.float)
    f_H_new = torch.from_numpy(f_H).to(sig00.device).to(torch.float)
    return f_L_new, f_H_new

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

def wavelet_and_decoder_for_x(sig00):
    sig0=sig00.squeeze()
    f_L = np.zeros(sig0.shape)
    f_H = np.zeros(sig0.shape)
    wavelet = 'db1'
    for i in range(sig0.shape[0]):
        sig = sig0[i, :]
        # 使用pywavelets库进行小波分解
        coeffs = pywt.wavedec(sig, wavelet, level=5)
        cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
        # 去掉
        coeffs2 = [cA5, cD5, cD4, np.zeros(cD3.shape), np.zeros(cD2.shape), np.zeros(cD1.shape)]
        recon2 = pywt.waverec(coeffs2, wavelet)
        # 去掉
        coeffs3 = [np.zeros(cA5.shape), np.zeros(cD5.shape), np.zeros(cD4.shape), cD3, cD2, cD1]
        recon3 = pywt.waverec(coeffs3, wavelet)
        f_L[i, :] = recon2[:,:-1]
        f_H[i, :] = recon3[:,:-1]
    differ=(f_L+f_H-sig0.numpy()).sum()
    if differ > 1:
        print("小波分解出问题了！")
    f_L_new = torch.from_numpy(f_L).to(sig00.device).to(torch.float).unsqueeze(-1)
    f_H_new = torch.from_numpy(f_H).to(sig00.device).to(torch.float).unsqueeze(-1)
    return f_L_new, f_H_new

class MultiScale_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length,num_heads=4) -> None:
        super(MultiScale_LSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=1 # 单向LSTM
        self.relu = nn.ReLU()

        self.lstm1=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True) # LSTM层
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)  # LSTM层
        self.fc1=nn.Linear(hidden_size,1) # 全连接层
        self.fc2 = nn.Linear(hidden_size, 1)  # 全连接层


    def forward(self, input_data):
        #多尺度层

        #合流层
        f_all=input_data
        #高频LSTM层
        output_a, (hidden1, cell1)  = self.lstm1(f_all)
        output_a = self.relu(output_a)
        output1 = self.fc1(output_a)
        #低频LSTM层
        output_d, (hidden2, cell2)  = self.lstm2(f_all)
        output_d = self.relu(output_d)
        output2 = self.fc2(output_d)
        #输出
        pred = torch.stack([output1, output2], dim=1)
        return pred

class MultiScale_Attention_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length,num_heads=4) -> None:
        super(MultiScale_Attention_LSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=1 # 单向LSTM
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.lstm1=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True) # LSTM层
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)  # LSTM层
        self.fc1 = nn.Linear(hidden_size,1) # 全连接层
        self.fc2 = nn.Linear(hidden_size, 1)  # 全连接层
        self.attention1 = nn.MultiheadAttention(hidden_size, num_heads)  # 注意力层
        self.attention2 = nn.MultiheadAttention(hidden_size, num_heads)  # 注意力层

    def forward(self, input_data):
        # 多尺度层
        # 合流层
        f_all = input_data
        # 高频LSTM层
        output_a, (_, _) = self.lstm1(f_all)
        # 高频LSTM层-注意力层
        linear_output1 = output_a.transpose(0, 1)  # 将时间步和批次维度交换
        attention_output1, attn_weights1 = self.attention1(linear_output1, linear_output1, linear_output1)
        attention_output1 = attention_output1.transpose(0, 1)  # 将时间步和批次维度交换
        output11 = output_a + attention_output1
        # 高频流-全连接激活层
        output12 = self.relu1(output11)
        output13 = self.fc1(output12)

        # 低频流-LSTM层
        output_d, (_, _) = self.lstm2(f_all)
        # 低频流-注意力层
        linear_output2 = output_d.transpose(0, 1)  # 将时间步和批次维度交换
        attention_output2, attn_weights2 = self.attention2(linear_output2, linear_output2, linear_output2)
        attention_output2 = attention_output2.transpose(0, 1)  # 将时间步和批次维度交换
        output21 = output_d + attention_output2
        # 低频流-全连接激活层
        output22 = self.relu2(output21)
        output23 = self.fc2(output22)
        # 输出
        pred = torch.stack([output13, output23], dim=1)


        return pred

class MultiScale_Attention_Res_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length,num_heads=4) -> None:
        super(MultiScale_Attention_Res_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_directions = 1  # 单向LSTM
        self.relu11 = nn.ReLU()
        self.relu12 = nn.ReLU()
        self.relu21 = nn.ReLU()
        self.relu22 = nn.ReLU()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True)  # LSTM层
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True)  # LSTM层
        self.fc1 = nn.Linear(hidden_size, 1)  # 全连接层
        self.fc2 = nn.Linear(hidden_size, 1)  # 全连接层
        self.attention1 = nn.MultiheadAttention(hidden_size, num_heads)  # 注意力层
        self.attention2 = nn.MultiheadAttention(hidden_size, num_heads)  # 注意力层
        self.linear_projection1 = nn.Linear(input_size, hidden_size)
        self.linear_projection2 = nn.Linear(input_size, hidden_size)

    def forward(self, input_data):
        # 多尺度层
        # 合流层
        f_all = input_data
        # 高频LSTM层
        output_a, (_, _) = self.lstm1(f_all)
        # 高频LSTM层-注意力层
        linear_output1 = output_a.transpose(0, 1)  # 将时间步和批次维度交换
        attention_output1, attn_weights1 = self.attention1(linear_output1, linear_output1, linear_output1)
        attention_output1 = attention_output1.transpose(0, 1)  # 将时间步和批次维度交换
        output11 = output_a + attention_output1
        # 高频流-残差
        x_proj1 = self.linear_projection1(f_all)
        output111 = self.relu11(output11 + x_proj1)
        # 高频流-全连接激活层
        output12 = self.relu12(output111)
        output13 = self.fc1(output12)

        # 低频流-LSTM层
        output_d, (_, _) = self.lstm2(f_all)
        # 低频流-注意力层
        linear_output2 = output_d.transpose(0, 1)  # 将时间步和批次维度交换
        attention_output2, attn_weights2 = self.attention2(linear_output2, linear_output2, linear_output2)
        attention_output2 = attention_output2.transpose(0, 1)  # 将时间步和批次维度交换
        output21 = output_d + attention_output2
        # 低频流-残差
        x_proj2 = self.linear_projection2(f_all)
        output211 = self.relu21(output21 + x_proj2)
        # 低频流-全连接激活层
        output22 = self.relu22(output211)
        output23 = self.fc2(output22)
        # 输出
        pred = torch.stack([output13, output23], dim=1)

        return pred

#Res_Shrink_LSTM的子模块
class Shrinkage(nn.Module):
      def __init__(self, input_size,hidden_size):
        super(Shrinkage, self).__init__()
        self.fc1 = nn.Sequential(
          nn.Linear(hidden_size, hidden_size),
          nn.BatchNorm1d(hidden_size),
          nn.ReLU(inplace=True),
          nn.Linear(hidden_size, 1),
          nn.Sigmoid()
        )
        # 对x在时间维度进行全局平均池化
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(hidden_size, 1)


      def forward(self, input):
        x_abs = torch.abs(input)
        GAP = self.GAP(x_abs.transpose(1, 2)).squeeze(2) #在时间维度上进行全局平均池化
        GAP_FC = self.fc2(GAP)
        alpha  = self.fc1(GAP)
        threshold = torch.mul(GAP_FC, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_abs), n_sub)
        result = x
        return result


#Res_Shrink_LSTM的主模块
class Res_Shrink_LSTM(nn.Module):#MM-LSTM
      def __init__(self, input_size, hidden_size, num_layers,output_size):
        super(Res_Shrink_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.shrinkage = Shrinkage(input_size,hidden_size)
        self.linear_projection = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

      def forward(self, x):
        residual = self.linear_projection(x)
        out, _ = self.lstm(x)
        out_shrinkage = self.shrinkage(out)#实现了shirink
        result = residual + out_shrinkage
        # result = residual
        result = self.fc(result)
        return result

#Multiple_Res_Shrink_LSTM的子模块
class Multiple_Shrinkage(nn.Module):
      def __init__(self, input_size,hidden_size,num_heads):
        super(Multiple_Shrinkage, self).__init__()
        self.fc1 = nn.Sequential(
          nn.Linear(hidden_size, hidden_size),
          nn.BatchNorm1d(hidden_size),
          nn.ReLU(inplace=True),
          nn.Linear(hidden_size, 1),
          nn.Sigmoid()
        )
        # 对x在时间维度进行全局平均池化
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.attention1 = nn.MultiheadAttention(hidden_size,num_heads)  # 注意力层


      def forward(self, input):
        x_abs = torch.abs(input)
        GAP = self.GAP(x_abs.transpose(1, 2)) #在时间维度上进行全局平均池化
        GAP = GAP.transpose(1, 2)
        #第一个FC
        GAP_FC = self.fc2(GAP)
        #第二个FC
            # 注意力层
        linear_output1=GAP.transpose(0, 1)  # 将时间步和批次维度交换
        attention_output1, attn_weights1 = self.attention1(linear_output1, linear_output1, linear_output1)
        GAP2 = attention_output1.transpose(0, 1)  # 将时间步和批次维度交换
        GAP2=GAP2.squeeze(1)
        alpha  = self.fc1(GAP2)
        alpha = alpha.unsqueeze(1)
        #两个FC相乘
        threshold = torch.mul(GAP_FC, alpha)
        #软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_abs), n_sub)
        result = x
        return result
#Multiple_Res_Shrink_LSTM的主模块
class Multiple_Res_Shrink_LSTM(nn.Module):#MMD-LSTM
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Multiple_Res_Shrink_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.shrinkage = Multiple_Shrinkage(input_size, hidden_size,num_heads=2)
        self.linear_projection = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        residual = self.linear_projection(x)
        out, _ = self.lstm(x)
        out_shrinkage = self.shrinkage(out)  # 实现了shirink
        result = residual + out_shrinkage
        result = self.fc(result)
        return result


class MultiScale_Attention_Res_shrink_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length,num_heads=4) -> None:
        super(MultiScale_Attention_Res_shrink_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_directions = 1  # 单向LSTM
        self.relu11 = nn.ReLU()
        self.relu12 = nn.ReLU()
        self.relu21 = nn.ReLU()
        self.relu22 = nn.ReLU()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=True)  # LSTM层
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=True)  # LSTM层
        self.fc1 = nn.Linear(hidden_size*2, 1)  # 全连接层
        self.fc2 = nn.Linear(hidden_size*2, 1)  # 全连接层
        self.attention1 = nn.MultiheadAttention(hidden_size*2, num_heads)  # 注意力层
        self.attention2 = nn.MultiheadAttention(hidden_size*2, num_heads)  # 注意力层
        self.linear_projection1 = Multiple_Res_Shrink_LSTM(input_size, hidden_size*2, num_layers, output_size)
        self.linear_projection2 = Multiple_Res_Shrink_LSTM(input_size, hidden_size*2, num_layers, output_size)

    def forward(self, input_data):
        # 多尺度层
        # 合流层
        f_all = input_data
        # 高频LSTM层
        output_a, (_, _) = self.lstm1(f_all)
        # 高频LSTM层-注意力层
        linear_output1 = output_a.transpose(0, 1)  # 将时间步和批次维度交换
        attention_output1, attn_weights1 = self.attention1(linear_output1, linear_output1, linear_output1)
        attention_output1 = attention_output1.transpose(0, 1)  # 将时间步和批次维度交换
        output11 = output_a + attention_output1
        # 高频流-残差
        x_proj1 = self.linear_projection1(f_all)
        output111 = self.relu11(output11 + x_proj1)
        # output111 = self.relu11(output111)
        # 高频流-全连接激活层
        output12 = self.relu12(output111)
        output13 = self.fc1(output12)

        # 低频流-LSTM层
        output_d, (_, _) = self.lstm2(f_all)
        # 低频流-注意力层
        linear_output2 = output_d.transpose(0, 1)  # 将时间步和批次维度交换
        attention_output2, attn_weights2 = self.attention2(linear_output2, linear_output2, linear_output2)
        attention_output2 = attention_output2.transpose(0, 1)  # 将时间步和批次维度交换
        output21 = output_d + attention_output2
        # 低频流-残差
        x_proj2 = self.linear_projection2(f_all)
        output211 = self.relu21(output21 + x_proj2)
        # output211 = self.relu21(output21)
        # 低频流-全连接激活层
        output22 = self.relu22(output211)
        output23 = self.fc2(output22)
        # 输出
        pred = torch.stack([output13, output23], dim=1)

        return pred

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 全连接层

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)  # output(批数量, 序列长度, 神经元数)到pred(批数量, 序列长度, 1)
        out=out
        return out



class HierarchicalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_hierarchy=3):
        super(HierarchicalRNN, self).__init__()
        self.num_hierarchy = num_hierarchy
        self.rnns = nn.ModuleList([nn.RNN(input_size, hidden_size, num_layers, batch_first=True) for _ in range(num_hierarchy)])
        self.fc = nn.Linear(192, 1)

    def forward(self, x):
        outputs = []
        for i in range(self.num_hierarchy):
            output, _ = self.rnns[i](x)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=2)  # 沿着最后一个维度拼接
        output = self.fc(outputs)
        return output

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 双向GRU的输出维度需要乘以2

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 初始化双向GRU的隐藏状态
        out, _ = self.gru(x, h0)  # 前向计算
        out = self.fc(out)  # 取最后一个时间步的输出并进行线性变换
        return out

class BiLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length) -> None:
        super(BiLSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=2 # 双向LSTM

        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True) # LSTM层
        self.fc=nn.Linear(hidden_size,1) # 全连接层

    def forward(self,x):
        #堆叠模型层
        output, _ = self.lstm(x) # x(批数量, 序列长度, 特征数)到output(批数量, 序列长度, 神经元数)
        pred = self.fc(output)  # output(批数量, 序列长度, 神经元数)到pred(批数量, 序列长度, 1)
        pred = pred  # (批数量, 序列长度, 1)只取最后一个序列的预测结果作为我们的预测结果
        return pred

class Attention_Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length):
        super(Attention_Rnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 全连接层
        self.attention = nn.MultiheadAttention(hidden_size, 4)  # 注意力层

    def forward(self, x):
        out, _ = self.rnn(x)
        # 计算注意力权重、计算注意力向量
        attention_output, attn_weights = self.attention(out, out, out)
        # 将自注意力输出结果与LSTM输出结果进行拼接
        # combined_output = torch.cat([out, attention_output], dim=2)
        output = out + attention_output  # 自注意力机制的输出与原模型的输出相加
        output = self.fc(output)
        return output

class Attention_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length):
        super(Attention_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 全连接层
        self.attention = nn.MultiheadAttention(hidden_size, 4)  # 注意力层

    def forward(self, x):
        out, _ = self.gru(x)
        # 计算注意力权重、计算注意力向量
        attention_output, attn_weights = self.attention(out, out, out)
        # 将自注意力输出结果与LSTM输出结果进行拼接
        # combined_output = torch.cat([out, attention_output], dim=2)
        output = out + attention_output  # 自注意力机制的输出与原模型的输出相加
        output = self.fc(output)
        return output

class Attention_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length):
        super(Attention_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 全连接层
        self.attention = nn.MultiheadAttention(hidden_size, 4)  # 注意力层

    def forward(self, x):
        out, _ = self.lstm(x)
        # 计算注意力权重、计算注意力向量
        attention_output, attn_weights = self.attention(out, out, out)
        # 将自注意力输出结果与LSTM输出结果进行拼接
        # combined_output = torch.cat([out, attention_output], dim=2)
        output = out + attention_output  # 自注意力机制的输出与原模型的输出相加
        output = self.fc(output)
        return output

class Res_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Res_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.linear_projection = nn.Linear(input_size, hidden_size)
    def forward(self, x):
        # LSTM层
        out, _ = self.lstm(x)
        # 线性投影
        x_proj = self.linear_projection(x)
        # 残差连接
        out = self.relu(out + x_proj)#输出与原模型的输出相加
        # 全连接层
        out = self.fc(out)

        return out
class Res_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Res_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.RNN = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.linear_projection = nn.Linear(input_size, hidden_size)
    def forward(self, x):
        # LSTM层
        out, _ = self.RNN(x)
        # 线性投影
        x_proj = self.linear_projection(x)
        # 残差连接
        out = self.relu(out + x_proj)#输出与原模型的输出相加
        # 全连接层
        out = self.fc(out)

        return out

class Res_Attention_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Res_Attention_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.linear_projection = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 4)  # 注意力层
    def forward(self, x):
        # LSTM层
        out, _ = self.lstm(x)
        # 注意力层
        attention_output, attn_weights = self.attention(out, out, out)
        # 线性投影
        x_proj = self.linear_projection(x)
        # 残差连接
        output = out + attention_output  # 自注意力机制的输出与原模型的输出相加
        out = self.relu(output + x_proj)
        # 全连接层
        out = self.fc(out)

        return out

class Res_Attention_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Res_Attention_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # 双向LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.linear_projection = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 4)  # 注意力层
    def forward(self, x):
        # LSTM层
        out, _ = self.lstm(x)
        # 注意力层
        attention_output, attn_weights = self.attention(out, out, out)
        # 线性投影
        x_proj = self.linear_projection(x)
        # 残差连接
        output = out + attention_output  # 自注意力机制的输出与原模型的输出相加
        out = self.relu(output + x_proj)
        # 全连接层
        out = self.fc(out)

        return out

class Attention_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length):
        super(Attention_BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_directions = 2  # 双向LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 全连接层
        self.attention = nn.MultiheadAttention(hidden_size, 4)  # 注意力层

    def forward(self, x):
        out, _ = self.lstm(x)
        # 计算注意力权重、计算注意力向量
        attention_output, attn_weights = self.attention(out, out, out)
        # 将自注意力输出结果与LSTM输出结果进行拼接
        # combined_output = torch.cat([out, attention_output], dim=2)
        output = out + attention_output  # 自注意力机制的输出与原模型的输出相加
        output = self.fc(output)
        return output