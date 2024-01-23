import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
from scipy.signal import convolve2d

# 定义模型
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        """  x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.sigmoid(x) """
        x = self.fc3(x)
        return x

# 定义超参数
input_size =  7# 你的输入特征维度
hidden_size = 60
output_size = 1  # 回归问题通常是单一输出
learning_rate = 0.001
epochs = 60

# 创建数据集
total_bits = 90000
target = np.asmatrix([1,2,2,2,1])
resp = np.asmatrix([0.00346311341209041,0.00266347902207479,	0.00309343823946191,	0.0591629837184036,	0.180446192639864,	0.249664568147311,	0.180446192639864,	0.0591629837184036,	0.00309343823946191,	0.00266347902207479,	0.00346311341209041])
random_array = np.random.choice([-1, 1], size=(1, total_bits))
readback_signal = convolve2d(random_array, resp, mode="same")
ideal_output = np.asmatrix(convolve2d(random_array, target, mode="same")) / np.sqrt(65)
np.random.seed(123)
#add noise to readback signal
for si in range(10, 24, 2):
    readback_signal_copy = readback_signal
    snr = 10 ** (si / 10.0)
    xpower = np.sum(readback_signal_copy ** 2) / len(readback_signal_copy.T)
    npower = xpower / snr
    noise = np.random.randn(len(readback_signal_copy.T)) * np.sqrt(npower)
    readback_signal_copy = readback_signal_copy + noise.T

    pad_unit = np.asmatrix([-1, -1, -1])
    pad_array = np.concatenate((pad_unit, readback_signal_copy,pad_unit), axis=1)
    window_size = 7
    stride = 1
    x = np.empty((total_bits, window_size))

    for i in range(0, pad_array.shape[1] - window_size + 1, stride):
        window_data = pad_array[:, i:i+window_size]
        # 将window数组变换为列向量，并添加到结果矩阵中
        tmp = window_data
        
        x[i, :] = tmp

    # 将结果矩阵转换为numpy数组
    #x = np.asmatrix(x)
    # 为x添加噪声
    """ snr_db = 
    signal_power = np.max(x)
    noise_power_db = signal_power - snr_db
    noise_power_linear = 10 ** (noise_power_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power_linear), x.shape) """
    y = ideal_output


    # 转换为 PyTorch 张量
    X_tensor = torch.FloatTensor(x)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    # 将数据移动到 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    # 定义数据集大小
    total_size = len(X_tensor)
    train_size = int(0.7 * total_size)
    val_size = int(0.3 * total_size)
    test_size = total_size - train_size - val_size

    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, val_size, test_size])

    # 创建数据加载器并将数据移动到 GPU 上
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 初始化模型，并将模型参数移动到 GPU 上
    model = CustomModel(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model.train
    # 训练模型
    for epoch in range(epochs):
        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # 在验证集上计算损失
        val_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_dataloader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                val_outputs = model(batch_X_val)
                val_loss += criterion(val_outputs, batch_y_val).item()

        # 计算平均验证集损失
        avg_val_loss = val_loss / len(val_dataloader)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}')


    torch.save(model, '/home/likefei/PR-NN_Detector_PRchannel/individual/nneq_snr' + str(si) + '.pt')
    print(str(si) + ' train finish\n')
# 在这里你可以使用训练好的模型进行测试
# 例如：with torch.no_grad(): predictions = model(new_data.to(device))
