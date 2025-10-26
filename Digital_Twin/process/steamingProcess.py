
import numpy as np
np.product = np.prod
import pandas as pd
from symfit import parameters, variables, exp, log, Fit, Model
from process import energyAnalysis as eA
import pickle
import os
from CoolProp import CoolProp


def steaming_process(t, m_v):
    """蒸制过程的参数"""
    t = t * 3600                # 系统运行的时间
    T_0 = 293.15                # 环境温度
    eta_steamGenerator = 0.90   # 蒸汽发生器效率
    state_point = np.array([[1, 2, 2, 1], [101325, 1002700, 198540, 101325], [293.15, 453.15, 453.15, 293.15]])  # 温度 293.15, 453.15, 393.15, 293.15

    exergy_loss_steamGenerator = eA.exergy_water(1, state_point, t, T_0, m_v) - eA.exergy_water(2, state_point, t, T_0, m_v)
    exergy_loss_valve = eA.exergy_water(2, state_point, t, T_0, m_v) - eA.exergy_water(3, state_point, t, T_0, m_v)
    exergy_loss_tank = eA.exergy_water(3, state_point, t, T_0, m_v) - eA.exergy_water(4, state_point, t, T_0, m_v)

    # print("蒸汽发生器的㶲损为：{:.2f} kJ".format(exergy_loss_steamGenerator / 1000))
    # print("蒸汽通过控制阀㶲损为：{:.2f} kJ".format(exergy_loss_valve / 1000))
    # print("蒸汽通过物料罐的㶲损为：{:.2f} kJ".format(exergy_loss_tank / 1000))
    # print("蒸汽发生器的能耗为：{:.2f} kW·h".format(eA.energy_of_steam_generator(t, m_v, eta_steamGenerator) * 0.0002778 / 1000))
    return eA.energy_of_steam_generator(t, m_v, eta_steamGenerator) * 0.0002778 / 1000

def steamingMCfit(MC_raw, remove, fitData = './process/The_Dihuang_index_variation_during_the_process.csv'):
    """蒸制过程物料含水率的参数拟合"""
    data = pd.read_csv(fitData)
    data = np.array(data.iloc[:, :6].values)

    j = 0
    n = len(data)
    t = np.zeros(n)

    X0 = np.zeros(n)
    X = np.zeros(n)
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
    for i in range(0, n - 1):
        if data[i + 1][0] - data[i][0] == 0:
            tau[j] = data[i][2]
            T[j] = data[i][4]
            X[j] = data[i][5]
            j += 1

    data_t = t[:j]
    data_X0 = X0[:j]
    data_tau = tau[:j]
    data_T = T[:j]
    data_X = X[:j]

    # 去除异常数据
    if remove.size > 0:
        data_t = np.delete(data_t, remove)
        data_tau = np.delete(data_tau, remove)
        data_T = np.delete(data_T, remove)
        data_X0 = np.delete(data_X0, remove)
        data_X = np.delete(data_X, remove)
    # print(remove)
    # print(data_t, data_tau, data_T, data_X0, data_X)

    t, tau, T, X0, X = variables('t, tau, T, X0, X')

    k, a1, b1, b2 = parameters('k, a1, b1, b2')
    expr = (1 - exp(-k * t)) * ((a1 * T + b1) * log(tau + 1) + b2) + X0 * exp(-k * t)

    # k1, k2, a1, b1, b2, a = parameters('k1, k2, a1, b1, b2, a')
    # expr = ((a1*T+b1)*log(tau+1)+b2-X0)*(a*exp(-k1*t)+(1-a)*exp(-k2*t))+X0


    model = Model({X: expr})

    # 设置参数初始值及边界
    # a1=-0.2335, b1=2.4007, a2=0.3830, b2=45.7341, C1=4.3970, k=0.2576

    k.value, k.min, k.max = 0.2576, 0.01, 5.0
    a1.value = -0.2335
    b1.value = 2.4007
    b2.value = 45.7341

    # k1.value = -0.165432
    # k2.value = 0.70963620
    # a1.value = 0.06085863
    # b1.value = -24.25652968
    # b2.value = 100.1976468
    # a.value = 0.09414


    fit = Fit(model, t=data_t, tau=data_tau, T=data_T, X0=data_X0, X=data_X)

    fit_result = fit.execute()
    return fit_result

def steamingMC(t, tau, T, X0, MC_raw, X_predict = 'X_t', remove = [], fitData = './process/The_Dihuang_index_variation_during_the_process.csv'):
    """
    蒸制过程物料含水率的预测
    :param t: 蒸制时间
    :param tau: 干制总时间
    :param T: 干制的平均温度
    :param X0: 蒸制前的物料含水率
    :param MC_raw: 原料含水率
    :param remove: 要删除的数据的索引
    :param fitData: 蒸制过程物料含水率的数据
    :param X_predict: 预测值类型，'X_t'表示蒸制后的物料含水率，'X_eq'表示最大物料含水率
    :return: 蒸制后的物料含水率预测值
    """
    t = np.array(t)
    tau = np.array(tau)
    T = np.array(T)
    X0 = np.array(X0)
    remove = np.array(remove)
    # print(remove)

    if os.path.isfile('./process/fit_result.pkl'):
        pass
        # print('物料含水率预测模型已经存在。')
    else:
        print('物料含水率预测模型拟合中...')
        fit_result = steamingMCfit(MC_raw, remove, fitData)
        print(fit_result)
        # print(f"R² = {fit_result.r_squared:.4f}")
        # 保存整个fit_result对象
        with open("process/fit_result.pkl", "wb") as f:
            pickle.dump(fit_result, f)
        # 加载恢复
    with open("process/fit_result.pkl", "rb") as f:
        fit_result = pickle.load(f)

    k = fit_result.params['k']
    a1 = fit_result.params['a1']
    b1 = fit_result.params['b1']
    b2 = fit_result.params['b2']
    # k1 = fit_result.params['k1']
    # k2 = fit_result.params['k2']
    # a1 = fit_result.params['a1']
    # b1 = fit_result.params['b1']
    # b2 = fit_result.params['b2']
    # a = fit_result.params['a']

    t = np.array(t)
    tau = np.array(tau)
    T = np.array(T)
    X0 = np.array(X0)

    if X_predict == 'X_t':
        return (1 - np.exp(-k * t)) * ((a1 * T + b1) * np.log(tau + 1) + b2) + X0 * np.exp(-k * t)
        # return ((a1 * T + b1) * np.log(tau + 1) + b2 - X0) * (a * np.exp(-k1 * t) + (1 - a) * np.exp(-k2 * t)) + X0
    elif X_predict == 'X_eq':
        return (a1 * T + b1) * np.log(tau + 1) + b2


def mvfit(MC_raw, remove = [], fitData = './process/The_Dihuang_index_variation_during_the_process.csv'):
    """蒸汽质量流量预测"""
    data = pd.read_csv(fitData)
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
    Xs = np.zeros(n)
    energy = np.zeros(n)
    for i in range(0, n - 1):
        if data[i + 1][0] - data[i][0] == 0:
            tau[j] = data[i][2]
            T[j] = data[i][4]
            energy[j] = data[i][6]
            Xs[j] = data[i][5]
            j += 1

    data_t = t[:j]
    data_X0 = X0[:j]
    data_tau = tau[:j]
    data_T = T[:j]
    data_energy = energy[:j]
    data_Xs = Xs[:j]
    # print(remove)


    remove = np.array(remove)

    if remove.size > 0:
        data_t = np.delete(data_t, remove)
        data_tau = np.delete(data_tau, remove)
        data_T = np.delete(data_T, remove)
        data_X0 = np.delete(data_X0, remove)
        data_energy = np.delete(data_energy, remove)
        data_Xs = np.delete(data_Xs, remove)
    # print(data_t, data_tau, data_T, data_X0, data_energy)

    data_Xeq = steamingMC(data_t, data_tau, data_T, data_X0, MC_raw, 'X_eq', remove)
    data_Xt = steamingMC(data_t, data_tau, data_T, data_X0, MC_raw, 'X_t', remove)

    # data_power = data_Xeq - data_X0
    # data_quantity = data_power / data_t
    data_mv = data_energy * 0.9 / ((CoolProp.PropsSI('H', 'T', 273.153+120, 'Q', 1,
                                      'Water') - CoolProp.PropsSI('H', 'T', 293.15, 'Q',
                                                                  0, 'Water')) * 0.0002778 / 1000) / data_t / 3600 * 10000# 453.15

    # print(data_t, data_tau, data_T, data_X0, data_energy, data_Xeq, data_mv)
    # print(data_power, data_quantity, data_mv)

    MCeq, Td, mv, MC_t= variables('MCeq, Td, mv, MC_t')
    a1, a2, a3, b = parameters('a1, a2, a3, b')

    # expr = (a1 * po * po + a2 * po + a3) * log(b * qua + exp(1))
    expr = a1*MCeq+a2*MC_t+a3*Td+b
    model = Model({mv: expr})

    # 设置参数初始值及边界
    # a1=0.0060, a2=-0.2880, a3=4.3711, b=19.4463

    b.value, b.min = 1, 1
    a1.value = 0.1
    a2.value = 0.1
    a3.value = 0.1

    fit = Fit(model, MCeq=data_Xeq, Td=data_T, mv=data_mv, MC_t=data_Xt)

    fit_result_mv = fit.execute()
    return fit_result_mv

def mvpred(t, tau, T, X0, MC_raw, remove = [], fitData = './process/The_Dihuang_index_variation_during_the_process.csv'):
    """
    蒸制过程物料含水率的预测
    :param t: 蒸制时间
    :param tau: 干制总时间
    :param T: 干制的平均温度
    :param X0: 蒸制前的物料含水率
    :param MC_raw: 原料含水率
    :param remove: 要删除的数据的索引
    :param fitData: 蒸制过程物料含水率的数据
    :param X_predict: 预测值类型，'X_t'表示蒸制后的物料含水率，'X_eq'表示最大物料含水率
    :return: 蒸制后的物料含水率预测值
    """
    t = np.array(t)
    tau = np.array(tau)
    T = np.array(T)
    X0 = np.array(X0)
    # remove = np.array(remove)

    Xeq = steamingMC(t, tau, T, X0, MC_raw, 'X_eq', remove)
    Xt = steamingMC(t, tau, T, X0, MC_raw, 'X_t', remove)
    # po = Xeq - X0
    # qua = po / t
    # print(remove)

    if os.path.isfile('./process/fit_result_mv.pkl'):
        pass
        # print('蒸汽流量预测模型已经存在。')
    else:
        print('蒸汽流量预测模型拟合中...')
        fit_result_mv = mvfit(MC_raw, remove, fitData)
        print(fit_result_mv)
        # print(f"R² = {fit_result.r_squared:.4f}")
        # 保存整个fit_result对象
        with open("process/fit_result_mv.pkl", "wb") as f:
            pickle.dump(fit_result_mv, f)
        # 加载恢复
    with open("process/fit_result_mv.pkl", "rb") as f:
        fit_result_mv = pickle.load(f)

    b = fit_result_mv.params['b']
    a1 = fit_result_mv.params['a1']
    a2 = fit_result_mv.params['a2']
    a3 = fit_result_mv.params['a3']

    # po = np.array(po)
    # qua = np.array(qua)

    # return (a1*po*po + a2*po + a3) * np.log(b*qua + np.exp(1)) / 10000
    return (a1*Xeq+a2*Xt+a3*T+b) / 10000
def material_m(MC_raw, m_raw, MC):
    """
    物料质量计算
    :param MC_raw: 原料含水率
    :param m_raw: 原料质量
    :return: 物料质量
    """
    MC = np.array(MC)
    MC = MC / 100
    MC_raw = MC_raw / 100
    m = m_raw * (1 - MC_raw) / (1 - MC)
    return m


if __name__ == '__main__':
    # 蒸汽物料含水率预测
    steaming_time = 4
    tau = 8
    T_dry_avg = 55
    X0 = 37.44167
    MC_raw = 21.098

    m_v = mvpred(steaming_time, tau, T_dry_avg, X0, MC_raw)
    X_t = steamingMC(steaming_time, tau, T_dry_avg, X0, MC_raw, 'X_t')

    print(X_t)