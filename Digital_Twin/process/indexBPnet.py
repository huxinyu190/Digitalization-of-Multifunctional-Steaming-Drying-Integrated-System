
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import randint
import joblib
import os
from process import steamingProcess as sp

printvalue = 0

# 定义模型
class BPNet(nn.Module):
    def __init__(self, input_size=5, hidden_size=np.array([[15], [10]]), output_size=1):
         super().__init__()
         self.fc1 = nn.Linear(input_size, hidden_size[0][0])
         self.fc2 = nn.Linear(hidden_size[0][0], hidden_size[1][0])
         self.fc3 = nn.Linear(hidden_size[1][0], output_size)
         self.dropout = nn.Dropout(0.1)   # 训练时以概率 0.2 随机置零输入张量的元素，防止过拟合

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## 蒸汽流量预测BP神经网络模型的搭建
def mvBP(MC_raw, data_path, device = torch.device("cpu") ):
    # 导入数据
    data = pd.read_csv(data_path)
    data = np.array(data.iloc[:, :7].values)

    j = 0
    n = len(data)
    t = np.zeros(n)

    X0 = np.zeros(n)
    for i in range(0, n):
        if data[i][3] == 0:
            t[j] = data[i][0]
            X0[j] = MC_raw
            j += 1
        else:
            if data[i][0] - data[i - 1][0] > 0:
                t[j] = data[i][0] - data[i - 1][0]
                X0[j] = data[i - 1][5]
                j += 1

    j = 0
    tau = np.zeros(n)
    T = np.zeros(n)
    energy = np.zeros(n)
    for i in range(0, n - 1):
        if data[i + 1][0] - data[i][0] == 0:
            tau[j] = data[i][2]
            T[j] = data[i][4]
            energy[j] = data[i][6]
            j += 1

    data_t = t[:j]
    data_X0 = X0[:j]
    data_tau = tau[:j]
    data_T = T[:j]
    data_energy = energy[:j]

    data_Xeq = sp.steamingMC(data_t, data_tau, data_T, data_X0, MC_raw, 'X_eq')

    data_power = data_Xeq - data_X0
    data_quantity = data_power / data_t
    x = np.array([data_power, data_quantity]).T
    data_mv = data_energy * 0.9 / 0.748175531 / data_t / 3600 * 10000
    y = data_mv.reshape(-1, 1)
    random_state = randint(1, 100)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=random_state)
    # print("数据随机划分种子: {}".format(random_state)) # 输出随机种子  22   89   75

    # 数据标准化
    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train = scaler_x.transform(x_train)
    X_val = scaler_x.transform(x_val)
    Y_train = scaler_y.transform(y_train)
    Y_val = scaler_y.transform(y_val)

    # 转张量
    x_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    model = BPNet(2, np.array([[5], [3]]), 1)

    # 训练配置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500)   # 当评价指标不再改善时，减少学习率

    # 训练循环

    model.to(device)
    for epoch in range(3000):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = criterion(val_pred, y_val_t)
            scheduler.step(val_loss)
    return model, scaler_x, scaler_y, val_loss.item()

def Mv_pred(x, MC_raw,
            accuracy = 0.3,
            data_path = './process/The_Dihuang_index_variation_during_the_process.csv',
            scaler_x_path = './model_BP/Mv_scaler_x.joblib',
            scaler_y_path ='./model_BP/Mv_scaler_y.joblib',
            model_path = './model_BP/Mv_BP_model_weights.pth',
            device = "cpu"):   # 物料含水率预测
    '''
    :param x: 蒸制总时间，干制总时间，干制平均温度, 蒸制前含水率
    :param MC_raw: 原料的含水率
    :param accuracy: 预测精度
    :param data_path: BP训练数据路径
    :param scaler_x: BP输入x特征标准化器路径
    :param scaler_y: BP输入y特征标准化器路径
    :param model: BP模型路径
    :param device: cpu or gpu
    :return: 预测值
    '''

    x = np.array(x)
    t = x[:, 0]
    tau = x[:, 1]
    T = x[:, 2]
    X0 = x[:, 3]

    Xeq = sp.steamingMC(t, tau, T, X0, MC_raw, 'X_eq')
    po = Xeq - X0
    qua = po / t
    x = np.array((po, qua)).T

    device = torch.device(device)

    if os.path.isfile('./model_BP/Mv_BP_model_weights.pth'):
        pass
        # print('蒸汽流量预测模型已经存在。')
    else:
        print('蒸汽流量预测模型不存在，训练中...')
        real_original = np.array([4.577802747])
        pred_original = np.array([0])
        loss = 10
        while (loss > 0.05 or
               abs(real_original[0] - pred_original[0]) > accuracy):
            Mv_model, Mv_scaler_x, Mv_scaler_y, loss = mvBP(MC_raw, data_path, device)
            print(loss)
            test_data = Mv_scaler_x.transform(np.array([[16.22385017, 2.027981271]]))
            test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
            pred = Mv_model(test_tensor).to('cpu').detach().numpy()
            pred_original = Mv_scaler_y.inverse_transform(pred)


        # 保存标准化参数
        joblib.dump(Mv_scaler_x, scaler_x_path)  # 输入特征标准化器
        joblib.dump(Mv_scaler_y, scaler_y_path)  # 输出目标标准化器

        # 保存模型
        torch.save(Mv_model.state_dict(), model_path)  # 保存参数
        print('蒸汽流量预测模型训练完成。')

    # 加载模型
    Mv_scaler_x = joblib.load(scaler_x_path)
    Mv_scaler_y = joblib.load(scaler_y_path)
    Mv_model = (BPNet(2, np.array([[5], [3]]), 1))
    Mv_model.load_state_dict(torch.load(model_path))  # 加载模型参数
    Mv_model.eval()  # 切换为评估模式（关闭Dropout等）

    x = Mv_scaler_x.transform(x)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    pred = Mv_model(x).to('cpu').detach().numpy()
    return Mv_scaler_y.inverse_transform(pred).reshape(-1) / 10000

def indexvariationBP(raw, data_path, index_name, device = torch.device("cpu")):
    # 导入数据
    raw = np.array(raw)
    data = pd.read_csv(data_path)
    x = data.iloc[:, :5].values
    Index_id = {"MC": 0, "EC": 1, "WEP": 2, "AEP": 3, "L": 4, "b": 5}
    y = data.iloc[:, Index_id[index_name] + 5].values - raw[Index_id[index_name]]
    y = y.reshape(-1, 1)
    random_state = randint(1, 100)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    # 数据标准化
    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train = scaler_x.transform(x_train)
    X_test = scaler_x.transform(x_test)
    Y_train = scaler_y.transform(y_train)
    Y_test = scaler_y.transform(y_test)

    # 转张量
    x_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    x_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(Y_test, dtype=torch.float32).to(device)

    model = BPNet(5, np.array([[15], [10]]), 1)

    # 训练配置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500)   # 当评价指标不再改善时，减少学习率

    # 训练循环

    model.to(device)
    for epoch in range(3000):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(x_test_t)
            test_loss = criterion(test_pred, y_test_t)
            scheduler.step(test_loss)
    return model, scaler_x, scaler_y, test_loss.item()

def index_pred(x, raw, index_name,
               accuracy = 0.3,
               data_path = './process/The_Dihuang_index_variation_during_the_process.csv',
               device = "cpu"):
    '''
    :param x: 需要预测阶段的各个参数，蒸制总时间，蒸制次数，干制总时间，干制次数，干制平均温度
    :param raw: 原料的指标值
    :param index_name: 待预测指标名称缩写 {"MC":"含水率", "EC":"能耗", "WEP":"水提多糖含量", "AEP":"碱提多糖含量", "L":"明度", "b":"黄蓝度"}
    :param accuracy: 预测精度，训练好的BP神经网络模型loss的不超过accuracy
    :param data_path: BP训练数据路径
    :param device: cpu or gpu
    :return: 预测值
    '''

    Index_name =  {"MC":"含水率", "EC":"能耗", "WEP":"水提多糖含量", "AEP":"碱提多糖含量", "L":"明度", "b":"黄蓝度"}
    Index_id = {"MC":0, "EC":1, "WEP":2, "AEP":3, "L":4, "b":5}

    device = torch.device(device)
    global printvalue
    if printvalue == 0:
        printvalue = 1
        if os.path.isfile('./model_BP/{}_variation_BP_model_weights.pth'.format(index_name)):
            pass
            # print('物料{}预测模型已经存在。'.format(Index_name[index_name]))
        else:
            print('物料{}预测模型不存在，训练中...'.format(Index_name[index_name]))
            loss = 10
            while loss > accuracy:
                for i in range(50):
                    model_opt, scaler_x_opt, scaler_y_opt, loss_opt = indexvariationBP(raw, data_path, index_name, device)
                    if loss_opt < loss:
                        loss = loss_opt
                        model = model_opt
                        scaler_x = scaler_x_opt
                        scaler_y = scaler_y_opt
            # 保存标准化参数
            joblib.dump(scaler_x, './model_BP/{}_scaler_x.joblib'.format(index_name))  # 输入特征标准化器
            joblib.dump(scaler_y, './model_BP/{}_scaler_y.joblib'.format(index_name))  # 输出目标标准化器
            # 保存模型
            torch.save(model.state_dict(), './model_BP/{}_variation_BP_model_weights.pth'.format(index_name))  # 保存参数
            print('物料{}预测模型训练完成，损失值Loss：{:.4f}。'.format(Index_name[index_name], loss))

    # 加载模型
    scaler_x = joblib.load('./model_BP/{}_scaler_x.joblib'.format(index_name))
    scaler_y = joblib.load('./model_BP/{}_scaler_y.joblib'.format(index_name))
    model = (BPNet(5, np.array([[15], [10]]), 1))
    model.load_state_dict(torch.load('./model_BP/{}_variation_BP_model_weights.pth'.format(index_name)))  # 加载模型参数
    model.eval()  # 切换为评估模式（关闭Dropout等)

    x = scaler_x.transform(x)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    pred = model(x).to('cpu').detach().numpy()
    return scaler_y.inverse_transform(pred) + raw[Index_id[index_name]]



