
from process import steamingProcess as sP
from process import dryingProcess as dP
from process import indexBPnet as indexBP
import torch
import globals
import numpy as np
import time



t_steaming = 8                  # 蒸制的时间(h)
MC_raw = 21.098                 # 原料的含水率(%)
EC_raw = 0                      # 原料的能耗(kWh/kg)
WEP_raw = 49.87472              # 原料水提多糖含量(%)
AEP_raw = 2.67019               # 原料碱提多糖含量(%)
L_raw = 27.558                  # 原料明度L*
b_raw = 8.781                   # 原料黄蓝度b*
m_raw = 50.0                    # 原料质量(kg)
T_environment = 20              # 环境温度(℃)
globals.raw = [MC_raw, EC_raw, WEP_raw, AEP_raw, L_raw, b_raw, m_raw]
device = torch.device("cpu")    # 神经网络训练设备选择
accuracy = 0.10                 # 物料指标预测精度
data_path = 'process/The_Dihuang_index_variation_during_the_process.csv'  # 物料含水率数据路径
MCmodel_chose = 'peleg'         # 含水率预测模型选择  'BPNet' or 'peleg'
Mvmodel_chose = 'peleg'         # 蒸汽流量预测模型选择  'BPNet' or 'peleg'
globals.remove = []

def all_process(u):
    if u == 0:
        # XGY 1 蒸制 8h；干制 8h 75°C；3次循环
        cycle = 3
        steaming_time = 8
        drying_time = [8]
        T_dry = [75]
        MC_result_1 = [23.522, 14.844, 9.83]

    if u == 1:
        # XGY 2 蒸制 4h；干制 [2, 3, 3]h [75, 65, 55]°C；6次循环
        cycle = 6
        steaming_time = 4
        drying_time = [2, 3, 3]
        T_dry = [75, 65, 55]
        MC_result_1 = [28.698, 20.614, 17.342, 14.238, 11.684, 12.46]

    if u == 2:
        # XGY 3 蒸制 4h；干制 [2, 3, 3]h [65, 55, 45]°C；6次循环
        cycle = 4
        steaming_time = 4
        drying_time = [2, 3, 3]
        T_dry = [65, 55, 45]
        MC_result_1 = [30.12811, 28.594, 22.85833, 19.425]

    t_dehumidify = 60
    T_dry_avg = np.sum(T_dry) / len(T_dry)
    t_steaming_time = 0
    tau = 0
    X0 = MC_raw
    x = np.array((steaming_time, tau, T_dry_avg, X0)).T
    C = np.array([])
    for n in range(cycle):
        MC_result = MC_result_1[n]
        if Mvmodel_chose == 'BPNet':
            m_v = indexBP.Mv_pred(x, MC_raw)
        elif Mvmodel_chose == 'peleg':
            m_v = sP.mvpred(steaming_time, tau, T_dry_avg, X0, MC_raw, globals.remove)
            X_t = sP.steamingMC(steaming_time, tau, T_dry_avg, X0, MC_raw, 'X_t', globals.remove)
            X_eq = sP.steamingMC(steaming_time, tau, T_dry_avg, X0, MC_raw, 'X_eq', globals.remove)
        t_steaming_time = t_steaming_time + steaming_time
        m = sP.material_m(21.098, 50, X_t)
        # coff = dP.coff_binary_search([X_t], [MC_result], [m], drying_time, T_dry, [sP.material_m(globals.raw[0], globals.raw[6], X_eq), T_dry_avg], 20, 60, 0.1)
        # print(coff)
        # C = np.append(C, coff[0])
        E_xFan, E_Fan, E_xPTC, E_PTC, E_xPot, k, mark = dP.drying_process(X_t, m, drying_time,
                                                                          T_dry, [T_dry_avg, X_eq], T_environment,
                                                                          t_dehumidify)    # , True, coff[0]
        X0 = mark[-1][2]
        tau = tau + np.sum(drying_time)
        E_d = dP.Ed_pred(T_dry_avg, X_t, mark[-1][2], globals.remove)
        print(
            "第{}次循环系统蒸制能耗为：{:.2f} kW·h，物料的含水率为：{:.2f} %；干制能耗为：{:.2f} kW·h；物料的含水率为：{:.2f} %。".
            format(n + 1, sP.steaming_process(steaming_time, m_v), X_t, E_d, mark[-1][2]))
        # print(sP.steaming_process(steaming_time, m_v))
#     print(C)

if __name__ == '__main__':
    start = time.time()
    for i in range(3):
        all_process(i)
    end = time.time()
    print((end - start)/3600)

    # print(dP.Ed_fit())

#
# print(dP.coff_fit(21.098, globals.remove, 2, 20, 'process/The_Dihuang_index_variation_during_the_process.csv'))

# # 物料含水率的预测
# x = np.array([[8, 1, 0, 0, 75],[8, 2, 8, 1, 65],[8, 2, 8, 1, 55],[8, 2, 8, 1, 75]])
# if MCmodel_chose == 'BPNet':
#     MC_pred = indexBP.MC_pred(x, MC_raw).reshape(-1, 1)
#     print(MC_pred)
# elif MCmodel_chose == 'peleg':
#     MC_pred = sP.steamingMC(8, 0, 75, 21.098, 21.098, 'X_eq') # , [4, 6, 8]
#     print(MC_pred)

# x = np.array([[35.50039955, 4.437549944],[35.50039955, 8.875099889],[13.12287902, 3.280719754],[35.50039955, 8.875099889]])
# mvpred = indexBP.Mv_pred(x, MC_raw)
# print(mvpred)

# train_time = time.time()
# x = np.array([[8, 1, 0, 0, 75],[8, 2, 8, 1, 65],[8, 2, 8, 1, 55],[8, 2, 8, 1, 75]])
# b_pred = indexBP.index_pred(x, raw, 'b', accuracy).reshape(-1, 1)
# print(b_pred)
# print(time.time() - train_time)

# # 蒸制过程
# t = [8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
# tau = [0, 8, 16, 0, 8, 16, 24, 32, 40, 0, 8, 16, 24]
# T = [75, 75, 75, 65, 65, 65, 65, 65, 65, 55, 55, 55, 55]
# X0 = [21.098, 23.522, 14.844, 21.098, 24.698, 20.614, 17.342, 14.238, 11.684, 21.098, 30.12811, 28.594, 22.85833] # 30.12811   37.44167
# # X0 = [21.098, 24.66, 10.79, 21.098, 28.78, 22.99, 18.4, 15.67, 14.31, 21.098, 32.48, 27.93, 22.33]
# # X0 = [21.098, 25.78, 13.74, 21.098, 27.2, 22.05, 19.69, 16.84, 15.67, 21.098, 33.39, 28.42, 22.04]
#
#
# t = np.array(t)
# tau = np.array(tau)
# T = np.array(T)
# X0 = np.array(X0)
#
# x = np.array((t, tau, T, X0)).T



# if Mvmodel_chose == 'BPNet':
#     m_v = indexBP.Mv_pred(x, MC_raw)
#     print(sP.steaming_process(t, m_v))
# elif Mvmodel_chose == 'peleg':
#     m_v = sP.mvpred(t, tau, T, X0, MC_raw)
#     X_t = sP.steamingMC(t, tau, T, X0, MC_raw, 'X_t')
#     print(sP.steaming_process(t, m_v))
#     print(X_t)

# # 蒸汽物料含水率预测
# steaming_time = 4
# tau = 8
# T_dry_avg = 55
# X0 = 34.44167
# MC_raw = 21.098
#
# m_v = sP.mvpred(steaming_time, tau, T_dry_avg, X0, MC_raw)
# X_t = sP.steamingMC(steaming_time, tau, T_dry_avg, X0, MC_raw, 'X_t')
#
# print(sP.steaming_process(steaming_time, m_v), X_t)



# mark = np.array([[0, 0, 0, 0]])
# mark = np.append(mark, [[0, 1, 2, 3]], 0)
# mark = np.append(mark, [[0, 1, 2, 3]], 0)
# mark = np.delete(mark, 0, 0)
# print(mark)


# # 干制过程
# t_steaming_time = 0
# cycle = 0
#
# # # XGY 1
# # MC = [48.63313, 35.576, 27.05933]
# # # MC_result = [23.522, 14.844, 9.83]
# # t = [8]
# # T_dry = [75]
# # steaming_time = 8
# # t_dehumidify = 60
#
# # # XGY 2
# # MC = [38.446, 26.382, 27.612, 18.956, 20.255, 21.166]
# # # MC_result = [24.698, 20.614, 17.342, 14.238, 11.684, 12.46]
# # t = [2, 3, 3]
# # T_dry = [75, 65, 55]
# # steaming_time = 4
# # t_dehumidify = 60
#
# # XGY 3
# MC = [42.43143, 37.9, 32.575, 30.32743]
# # MC_result = [37.44167, 28.594, 22.85833, 19.425]
# t = [2, 3, 3]
# T_dry = [65, 55, 45]
# steaming_time = 4
# t_dehumidify = 60
#
#
# for i in range(len(MC)):
#     cycle = cycle + 1
#     t_steaming_time = t_steaming_time + steaming_time
#     E_xFan, E_Fan, E_xPTC, E_PTC, E_xPot, k, mark = dP.drying_process(MC[i], sP.material_m(MC_raw, m_raw, MC[i]), t, T_dry, t_steaming_time, T_environment, t_dehumidify)
#     print("第{}次循环系统总能耗为：{:.2f} kW·h；物料的含水率为：{:.2f} %。".format(cycle ,(E_Fan+E_PTC) * 0.0002778 / 1000, mark[-1][2]))


# print((sP.material_m(MC_raw, m_raw, 30.32743) - sP.material_m(MC_raw, m_raw, 19.425)) * 2100 * 0.0002778)













