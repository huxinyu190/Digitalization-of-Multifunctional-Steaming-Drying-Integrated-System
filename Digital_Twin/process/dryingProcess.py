
import numpy as np
import pandas as pd
from process import energyAnalysis as eA
from CoolProp import CoolProp
from process import steamingProcess as sP
from symfit import parameters, variables, exp, log, Fit, Model
import pickle
import globals
import os
import csv

# m_air = 1.9169              # 空气质量流量(kg/s) 1.9169
v_air = 1.4859639           # 空气体积流量(m^3/s) 1.4859639
M_water = 18.015                # 水的摩尔质量 (g/mol)
M_air = 28.965                    # 空气摩尔质量(g/mol)
R = 8.31446261815324        # 气体常数(J/(mol\cdot K))
R_air = R/M_air                 # 干空气气体常数(kJ/(kg\cdot K))
R_water = R/M_water             # 水气体常数(kJ/(kg\cdot K))
V = 1.9                    # 系统内空气容积(m^3)   2.2
eta_PTC = 0.95              # PTC效率
eta_Fan = 0.85              # 风机效率


def waterEvaporate(AH, MC, m, T_dry, coff_pred = [0,0], state = True, coff = 0.0000013):
    """
    物料水分扩散到空气的数学模型
    :param RH: 相对湿度 (%)
    :param MC: 物料含水率 (%)
    :param T: 空气加热前的温度 (K)
    :param m: 物料的质量 (kg)
    :param T_dry: 干制的温度 (K)
    """

    T_dry = T_dry + 273.153
    A = np.pi*0.8**2*10/4                    # 物料与空气接触的面积 (m^2)
    alpha = 0.02
    t = V/v_air
    MC = MC/100
    p_v = AH*(101.325+0.32554)/(0.622+AH)
    if state:
        pass
    else:
        if os.path.isfile('./process/fit_result_coff.pkl'):
            pass
            # print('蒸汽流量预测模型已经存在。')
        else:
            fit_result_coff = coff_fit(globals.raw[0], globals.remove, 0.1, True, 20)
            print(fit_result_coff)
            # print(f"R² = {fit_result.r_squared:.4f}")
            # 保存整个fit_result对象
            with open("process/fit_result_coff.pkl", "wb") as f:
                pickle.dump(fit_result_coff, f)
            # 加载恢复
        with open("process/fit_result_coff.pkl", "rb") as f:
            fit_result_coff = pickle.load(f)

        a3 = fit_result_coff.params['a3']
        b3 = fit_result_coff.params['b3']
        c3 = fit_result_coff.params['c3']
        d3 = fit_result_coff.params['d3']
        f = fit_result_coff.params['f']

        coff = (a3 * coff_pred[0] ** 2 + b3 * coff_pred[1]+ c3 * coff_pred[0] + d3 * coff_pred[0] * coff_pred[1] + f) * 1e-5
        # print(coff)
        # coff = (-97.1131+2.2709*coff_pred[0]+3.9631*coff_pred[1]-0.0298*coff_pred[0]**2-0.0342*coff_pred[1]**2-0.0262*coff_pred[0]*coff_pred[1]) * 1e-5

    m_dry_air = eA.moisture_air_rho(40, 20, 101.325+0.32554) * V / (AH + 1)
    P_sat = CoolProp.PropsSI("P", "T", T_dry, "Q", 1, "Water")  # 水的饱和压力 (Pa)
    cp_air = CoolProp.PropsSI("C", "T", T_dry, "P", 101650.54, "Air")  # 空气的比热容 (J/(kg\cdot K))
    cp_water = CoolProp.PropsSI("C", "T", T_dry, "Q", 1, "Water")  # 水的比热容 (J/(kg\cdot K))
    r_water = CoolProp.PropsSI("H", "T", T_dry, "Q", 1, "Water") - CoolProp.PropsSI("H", "T", T_dry, "Q", 0, "Water")
    cp_moistureAir = (cp_air + AH*cp_water)
    # J = np.double(2 * alpha / (2-alpha) * (P_sat - p_v) * np.sqrt(MC*m/(2*np.pi*T_dry)) * coff)
    J = 2 * alpha / (2-alpha) * (P_sat - p_v) * np.sqrt(M_water/(2*np.pi*T_dry*R*1000)) * coff
    m_waterofTakeout = J * A * t
    m_waterinair = AH * m_dry_air + m_waterofTakeout
    T_out = T_dry - m_waterofTakeout * r_water / cp_moistureAir / (m_dry_air + AH*m_dry_air) - (1 - ((T_dry-273.153-35) / (75-35)))**1.85
    AH = m_waterinair / m_dry_air
    MC = (MC * m - m_waterofTakeout) / (m - m_waterofTakeout)
    return AH, T_out - 273.153, MC * 100, m - m_waterofTakeout




def drying_process(MC, m, t, T_dry, coff_pred = [4, 0], T_environment = 20, t_dehumidify = 60, state = False, coff = 0.0000013):
    """
    干制过程模型
    :param MC：物料干制前的含水率 (%)
    :param steaming_time: 总蒸制时间 (h)
    :param m: 物料的质量 (kg)
    :param t: 干制的时间 (h)
    :param T_dry: 干制的温度 (℃)
    :param T_environment = 20: 环境温度 (℃)
    :param t_dehumidify = 60: 除湿时间 (s)
    """

    # 变量初始化
    t_use = 0                                  # 实际使用时间
    i = 0                                      # 除湿循环计数
    mark = np.array([[0, 0, 0, 0]])
    t = np.array(t) * 3600                     # 干制时间换算为秒
    t_total = np.sum(t)                        # 干制总时间换算为秒
    k = np.array([0])                          # 空气循环计数
    E_xFan = np.array([0])                     # 空气通过风机获得的㶲
    E_Fan = np.array([0])                      # 风机的能耗
    E_xPTC = np.array([0])                     # 空气通过PTC获得的㶲
    E_xPot = np.array([0])                     # 空气通过物料罐获得的㶲
    E_PTC = np.array([0])                      # PTC的能耗
    t_circle = V / v_air                       # 循环时间
    T_dry_average = np.sum(T_dry) / len(T_dry) # 平均干制温度


    for j in range(len(t)):
        t_gradient = t[j]
        t_gradient_ues = 0
        T_gradient = T_dry[j]
        while t_gradient_ues < t_gradient:
            air_moisture = 40
            AH_in = eA.RHtoAH(air_moisture, T_environment)
            AH_out = AH_in
            # M_air_moisture = M_air * (1 - air_moisture/100) + M_water * air_moisture/100
            # R_air_moisture = R / M_air_moisture
            T_in = T_environment
            T_out = T_environment
            E_xFan = np.append(E_xFan, eA.energy_of_fan(t_dehumidify, v_air, eta_Fan) * eta_Fan)
            E_Fan = np.append(E_Fan, eA.energy_of_fan(t_dehumidify, v_air, eta_Fan))
            t_gradient_ues = t_gradient_ues + t_dehumidify
            n = 0
            E_xptc = np.array([0])
            E_xpot = np.array([0])
            E_ptc = np.array([0])
            while AH_out < eA.RHtoAH(95, T_gradient):
                delta_h = eA.moisture_air_HS(AH_in, 'H', T_in, 101.325+0.32554) - eA.moisture_air_HS(AH_in, 'H', T_gradient, 101.325+0.32554)
                delta_s = eA.moisture_air_HS(AH_in, 'S', T_in, 101.325+0.32554) - eA.moisture_air_HS(AH_in, 'S', T_gradient, 101.325+0.32554)
                E_xptc = np.append(E_xptc, t_circle * v_air * eA.moisture_air_rho(air_moisture, T_in, 101.325+0.32554) * (delta_h - T_environment * delta_s))
                E_ptc = np.append(E_ptc, - t_circle * v_air * eA.moisture_air_rho(air_moisture, T_in, 101.325+0.32554) * delta_h / eta_PTC)
                AH_out, T_out, MC, m = waterEvaporate(AH_in, MC, m, T_gradient, coff_pred, state, coff)
                delta_h = eA.moisture_air_HS(AH_in, 'H', T_gradient, 101.325+0.32554) - eA.moisture_air_HS(AH_out, 'H', T_out, 101.325+0.32554)
                delta_s = eA.moisture_air_HS(AH_in, 'S', T_gradient, 101.325+0.32554) - eA.moisture_air_HS(AH_out, 'S', T_out, 101.325+0.32554)
                E_xpot = np.append(E_xpot, t_circle * v_air * eA.moisture_air_rho(air_moisture, T_out , 101.325+0.32554) * (delta_h - T_environment * delta_s))
                t_gradient_ues = t_gradient_ues + t_circle
                AH_in = AH_out
                T_in = T_out
                n = n + 1
                if t_gradient_ues >= t_gradient:
                    break
            E_xPTC = np.append(E_xPTC, np.sum(E_xptc))
            E_PTC = np.append(E_PTC, np.sum(E_ptc))
            E_xPot = np.append(E_xPot, np.sum(E_xpot))
            mark = np.append(mark, [[AH_out, T_out, MC, m]], 0)
            i = i + 1
            k = np.append(k, n)
            t_use = t_use + t_gradient_ues
            # print(k)
            # print(t_gradient_ues)
            # print(AH_out, T_out, MC, m)
    E_xFan = np.append(E_xFan, eA.energy_of_fan(t_dehumidify, v_air, eta_Fan) * eta_Fan)
    E_Fan = np.append(E_Fan, eA.energy_of_fan(t_dehumidify, v_air, eta_Fan))
    mark = np.delete(mark, 0, 0)
    k = np.delete(k.T, 0).T
    # print('-----------------------')
    return np.sum(E_xFan), np.sum(E_Fan), np.sum(E_xPTC), np.sum(E_PTC), np.sum(E_xPot), k, mark

def coff_binary_search(MC, MC_result, m, t, T_dry, coff_pred, T_environment = 20, t_dehumidify = 60, accuracy = 1):
    """
    二分法求解coff
    """
    var = 1
    coff = np.array([])
    for i in range(len(MC_result)):
        R = 8.31446261815324
        M = 18.015
        Delta_m = sP.material_m(globals.raw[0], globals.raw[6], MC[i]) - sP.material_m(globals.raw[0], globals.raw[6], MC_result[i])
        A_m = np.pi * 0.8 ** 2 * 10 / 4
        ps = CoolProp.PropsSI("P", "T", np.sum(T_dry)/len(T_dry)+273.15, "Q", 1, "Water")  # 水的饱和压力 (Pa)
        pv = 1.17 * 1000
        coff_1 = 2*Delta_m / (A_m*np.sum(t)*3600*(ps-pv))*np.sqrt((2*np.pi*np.sum(T_dry)/len(T_dry)*R*1000)/M)*(2-0.02)/0.04
        coff_0 = Delta_m / (A_m*np.sum(t)*3600*(ps-pv))*np.sqrt((2*np.pi*np.sum(T_dry)/len(T_dry)*R*1000)/M)*(2-0.02)/0.04/2
        # coff_1 = 1e-4
        _, _, _, _, _, _, mark_0 = drying_process(MC[i], m[i], t, T_dry, coff_pred, T_environment, t_dehumidify, True, coff_0)
        MC_pred_result_0 = mark_0[-1][2]
        if abs(MC_pred_result_0 - MC_result[i]) < accuracy:
            coff = np.append(coff, coff_0)
            continue
        _, _, _, _, _, _, mark_1 = drying_process(MC[i], m[i], t, T_dry, coff_pred, T_environment, t_dehumidify, True, coff_1)
        MC_pred_result_1 = mark_1[-1][2]
        if abs(MC_pred_result_1 - MC_result[i]) < accuracy:
            coff = np.append(coff, coff_1)
            continue
        while var == 1:
            if MC_result[i] - MC_pred_result_1 > 0 and MC_pred_result_0 - MC_result[i] > 0:
                print("在范围内")
                break
            elif MC_result[i] - MC_pred_result_1 < 0:
                coff_0 = coff_1
                coff_1 = coff_1 + 5e-5
                _, _, _, _, _, _, mark_1 = drying_process(MC[i], m[i], t, T_dry, coff_pred, T_environment, t_dehumidify, True, coff_1)
                MC_pred_result_1 = mark_1[-1][2]
                if abs(MC_pred_result_1 - MC_result[i]) < accuracy:
                    break
            else:
                coff_1 = coff_0
                coff_0 = coff_0 / 2
                _, _, _, _, _, _, mark_0 = drying_process(MC[i], m[i], t, T_dry, coff_pred, T_environment, t_dehumidify, True, coff_0)
                MC_pred_result_0 = mark_0[-1][2]
                if abs(MC_pred_result_0 - MC_result[i]) < accuracy:
                    break
        if abs(MC_pred_result_0 - MC_result[i]) < accuracy:
            coff = np.append(coff, coff_0)
            continue
        if abs(MC_pred_result_1 - MC_result[i]) < accuracy:
            coff = np.append(coff, coff_1)
            continue
        coff_2 = (coff_0 + coff_1) / 2
        _, _, _, _, _, _, mark_2 = drying_process(MC[i], m[i], t, T_dry, coff_pred, T_environment, t_dehumidify, True, coff_2)
        MC_pred_result_2 = mark_2[-1][2]
        if abs(MC_pred_result_2 - MC_result[i]) < accuracy:
            coff = np.append(coff, coff_2)
            continue
        while var == 1:
            if MC_result[i] - MC_pred_result_2 > 0 and MC_pred_result_0 - MC_result[i] > 0:
                coff_1 = coff_2
                coff_2 = (coff_0 + coff_1) / 2
                _, _, _, _, _, _, mark_2 = drying_process(MC[i], m[i], t, T_dry, coff_pred, T_environment, t_dehumidify, True, coff_2)
                MC_pred_result_2 = mark_2[-1][2]
                if abs(MC_pred_result_2 - MC_result[i]) < accuracy:
                    coff = np.append(coff, coff_2)
                    break
            else:
                coff_0 = coff_2
                MC_pred_result_0 = MC_pred_result_2
                coff_2 = (coff_0 + coff_1) / 2
                _, _, _, _, _, _, mark_2 = drying_process(MC[i], m[i], t, T_dry, coff_pred, T_environment, t_dehumidify, True, coff_2)
                MC_pred_result_2 = mark_2[-1][2]
                if abs(MC_pred_result_2 - MC_result[i]) < accuracy:
                    coff = np.append(coff, coff_2)
                    break
        print(i)
    # coff = np.delete(coff, 0, 0)
    return coff

def coff_fit(MC_raw, remove = [], accuracy = 0.1, coff_fit_state = False, T_environment = 20, fitData = './process/The_Dihuang_index_variation_during_the_process.csv'):
    """
    拟合coff
    :coff_fit_state: True为使用预测数据进行训练；False为使用原始数据进行训练。
    :return:
    """
    data = pd.read_csv(fitData)
    data = np.array(data.iloc[:, :6].values)

    j = 0
    n = len(data)
    t = np.zeros(n)
    GY = np.array([])

    X0 = np.zeros(n)
    X = np.zeros(n)
    for i in range(0, n):
        if data[i][3] == 0:
            t[j] = data[i][0]
            X0[j] = MC_raw
            GY = np.append(GY,j)
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

    tau_s = np.zeros(len(data_t))
    j = 0
    GY = np.append(GY,0)
    for i in range(0, len(data_t)):
        if int(GY[j]) == i:
            tau_s[i] = data_t[int(GY[j])]
            j += 1
        else:
            tau_s[i] = data_t[int(GY[j-1])] + data_t[int(GY[j-1])] * (i - int(GY[j-1]))

    # 去除异常数据
    remove = np.array(remove)
    if remove.size > 0:
        data_t = np.delete(data_t, remove)
        data_tau = np.delete(data_tau, remove)
        data_T = np.delete(data_T, remove)
        data_X0 = np.delete(data_X0, remove)
        tau_s = np.delete(tau_s, remove)

    MC_eq = sP.steamingMC(data_t, data_tau, data_T, data_X0, MC_raw, 'X_eq', remove)
    # data_m_eq = sP.material_m(globals.raw[0], globals.raw[6], MC_eq)
    if os.path.isfile('./process/coff_result.csv'):
        pass
        # print('物料含水率预测模型已经存在。')
    else:
        print('蒸发系数求解中...')
        C = np.array([])
        for u in range(3):
            if u == 0:
                # XGY 1 蒸制 8h；干制 8h 75°C；3次循环
                cycle = 3
                steaming_time = 8
                drying_time = [8]
                T_dry = [75]
                Xt_false = [48.63313, 35.576, 27.05933]
                MC_result_1 = [23.522, 14.844, 9.83]
            if u == 1:
                # XGY 2 蒸制 4h；干制 [2, 3, 3]h [75, 65, 55]°C；6次循环
                cycle = 6
                steaming_time = 4
                drying_time = [2, 3, 3]
                T_dry = [75, 65, 55]
                Xt_false = [38.446, 26.382, 27.612, 18.956, 20.255, 21.166]
                MC_result_1 = [24.698, 20.614, 17.342, 14.238, 11.684, 12.46]
            if u == 2:
                # XGY 3 蒸制 4h；干制 [2, 3, 3]h [65, 55, 45]°C；6次循环
                cycle = 4
                steaming_time = 4
                drying_time = [2, 3, 3]
                T_dry = [65, 55, 45]
                Xt_false = [42.43143, 37.9, 32.575, 30.32743]
                MC_result_1 = [30.12811, 28.594, 22.85833, 19.425]
            t_dehumidify = 60
            T_dry_avg = np.sum(T_dry) / len(T_dry)
            t_steaming_time = 0
            tau = 0
            if coff_fit_state:
                X0 = MC_raw
                # x = np.array((steaming_time, tau, T_dry_avg, X0)).T
                for n in range(cycle):
                    MC_result = MC_result_1[n]
                    # m_v = sP.mvpred(steaming_time, tau, T_dry_avg, X0, MC_raw, remove)
                    X_t = sP.steamingMC(steaming_time, tau, T_dry_avg, X0, MC_raw, 'X_t', remove)
                    X_eq = sP.steamingMC(steaming_time, tau, T_dry_avg, X0, MC_raw, 'X_eq', remove)
                    t_steaming_time = t_steaming_time + steaming_time
                    m = sP.material_m(21.098, 50, X_t)
                    coff = coff_binary_search([X_t], [MC_result], [m], drying_time, T_dry, [T_dry_avg, X_eq], 20, 60,
                                                 accuracy)
                    # print(coff)
                    C = np.append(C, coff[0])
                    E_xFan, E_Fan, E_xPTC, E_PTC, E_xPot, k, mark = drying_process(X_t, m, drying_time,
                                                                                      T_dry, [T_dry_avg, X_eq],
                                                                                      T_environment,
                                                                                      t_dehumidify, True,
                                                                                      coff[0])  # , True, coff[0]
                    X0 = mark[-1][2]
                    tau = tau + np.sum(drying_time)
                    # print(
                    #     "第{}次循环系统蒸制能耗为：{:.2f} kW·h，物料的含水率为：{:.2f} %；干制能耗为：{:.2f} kW·h；物料的含水率为：{:.2f} %。".
                    #     format(n + 1, sP.steaming_process(steaming_time, m_v), X_t, (E_Fan + E_PTC) * 0.0002778 / 1000,
                    #            mark[-1][2]))
                    # print(sP.steaming_process(steaming_time, m_v))
                # print(C)
            else:
                X_eq = sP.steamingMC(steaming_time, tau, T_dry_avg, X0, MC_raw, 'X_eq', remove)
                m = sP.material_m(21.098, 50, Xt_false)
                MC_result = MC_result_1
                coff = coff_binary_search(Xt_false, MC_result, m, drying_time, T_dry,
                                          [T_dry_avg, X_eq], 20, 60,
                                          accuracy)
                # print(coff)
                C = np.append(C, coff)
                # tau = tau + np.sum(drying_time)
        C = C.reshape(-1, 1).tolist()
        C.insert(0, ["coff_result"])
        with open('./process/coff_result.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(C)
        print('蒸发系数求解完成。')

    C = pd.read_csv('./process/coff_result.csv')
    # print(C)
    C = np.array(C.iloc[:, :].values).reshape(-1) * 1e+5
    if remove.size > 0:
        C = np.delete(C, remove)


    data_t = tau_s
    coff, X_eq, T_d = variables('coff, X_eq, T_d')
    a3, b3, c3, d3, f, e3 = parameters('a3, b3, c3, d3, f, e3')# , d3, f, e3

    # expr = (a3*tau_s**2 + b3*X_eq + c3*tau_s + d3*tau_s*X_eq + e3*X_eq**2 +f)
    expr = (a3 * T_d ** 2 + b3 * X_eq + c3 * T_d + d3 * T_d * X_eq + f)
    # expr = (a3 * T_d + b3 * m_eq + c3)
    model = Model({coff: expr})

    a3.value = -0.0298
    b3.value = 3.9631
    c3.value = 2.2709
    d3.value = -0.0262
    f.value = -97.1131

    # print(C, data_t, MC_eq)
    fit = Fit(model, coff=C, T_d=data_T, X_eq=MC_eq)# sP.steamingMC(data_t, data_tau, data_T, data_X0, MC_raw, 'X_t', remove)

    fit_result_coff = fit.execute()
    print(fit_result_coff)
    return fit_result_coff

def Ed_fit(remove = [], fitData = './process/The_Dihuang_index_variation_during_the_process.csv'):
    '''
    拟合干制能耗
    :return:
    '''
    data = pd.read_csv(fitData)
    data = np.array(data.iloc[:, :7].values)
    n = len(data)
    j = 0
    X0_s = np.zeros(n)
    t = np.zeros(n)
    for i in range(0, n):
        if data[i][3] == 0:
            t[j] = data[i][0]
            X0_s[j] = globals.raw[0]
            j += 1
        else:
            if data[i][0] - data[i - 1][0] > 0:
                t[j] = data[i][0] - data[i - 1][0]
                X0_s[j] = data[i - 1][5]
                j += 1


    X0 = np.zeros(n)
    Xd = np.zeros(n)
    tau = np.zeros(n)
    j = 0
    T = np.zeros(n)
    energy = np.zeros(n)
    for i in range(0, n - 1):
        if data[i + 1][0] - data[i][0] == 0:
            T[j] = data[i][4]
            tau[j] = data[i][2]
            energy[j] = data[i+1][6]
            X0[j] = data[i+1][5]
            Xd[j] = data[i][5]
            j += 1

    data_Xd = X0[:j]
    data_T = T[:j]
    data_energy = energy[:j]
    data_X0s = X0_s[:j]
    data_t = t[:j]
    data_tau = tau[:j]
    data_X0 = Xd[:j]
    # data_X0[3] = 40.438715
    # data_X0[9] = 40.438715

    # print(data_t, data_tau, data_T, data_X0s, data_energy, data_Xd, data_X0)
    remove = np.array(remove)

    if remove.size > 0:
        data_t = np.delete(data_t, remove)
        data_tau = np.delete(data_tau, remove)
        data_T = np.delete(data_T, remove)
        data_X0s = np.delete(data_X0s, remove)
        data_energy = np.delete(data_energy, remove)
        data_Xd = np.delete(data_Xd, remove)
        data_X0 = np.delete(data_X0, remove)
    # data_X0d = Xs[:j]
    # print(remove)

    data_X0d = sP.steamingMC(data_t, data_tau, data_T, data_X0s, globals.raw[0], 'X_t', remove)
    data_Deltam = sP.material_m(globals.raw[0], globals.raw[6], data_X0d) - sP.material_m(globals.raw[0], globals.raw[6], data_Xd)

    # print(data_T, data_X0d, data_X0, data_energy, data_Xd, data_Deltam)
    E_d, Delta_m, T_d = variables('E_d, Delta_m, T_d')
    # a1, a2, b = parameters('a1, a2, b')
    # expr = a1 * T_d + a2 * Delta_m + b
    a1, a2, a3, b = parameters('a1, a2, a3, b')
    expr = a1 * T_d ** 2 + a3 * T_d + a2 * Delta_m + b
    model = Model({E_d: expr})

    # 设置参数初始值及边界
    b.value, b.min = 13.7919, 1
    a1.value = -0.1543
    a2.value = 0.5517
    a3.value = 0.5517

    fit = Fit(model, E_d=data_energy, T_d=data_T, Delta_m=data_Deltam)

    fit_result_Ed= fit.execute()
    return fit_result_Ed

def Ed_pred(T_d, MC_0, MC_d, remove, fitData = './process/The_Dihuang_index_variation_during_the_process.csv'):
    '''
    预测干制能耗
    :return:
    '''
    if os.path.isfile('./process/fit_result_Ed.pkl'):
        pass
        # print('蒸汽流量预测模型已经存在。')
    else:
        print('蒸汽流量预测模型拟合中...')
        fit_result_Ed = Ed_fit(remove, fitData)
        print(fit_result_Ed)
        # print(f"R² = {fit_result.r_squared:.4f}")
        # 保存整个fit_result对象
        with open("process/fit_result_Ed.pkl", "wb") as f:
            pickle.dump(fit_result_Ed, f)
        # 加载恢复
    with open("process/fit_result_Ed.pkl", "rb") as f:
        fit_result_Ed = pickle.load(f)

    Delta_m = sP.material_m(globals.raw[0], globals.raw[6], MC_0) - sP.material_m(globals.raw[0], globals.raw[6], MC_d)
    b = fit_result_Ed.params['b']
    a1 = fit_result_Ed.params['a1']
    a2 = fit_result_Ed.params['a2']
    a3 = fit_result_Ed.params['a3']

    # return a1 * T_d + a2 * Delta_m + b
    return a1 * T_d**2+ a3 * T_d+ a2 * Delta_m + b

if __name__ == '__main__':

    Ed_fit([], 'The_Dihuang_index_variation_during_the_process.csv')


    # print(coff_fit(21.098, [], 20, 'The_Dihuang_index_variation_during_the_process.csv'))

    # MC = 26.382
    # steaming_time = 8
    # E_xFan, E_Fan, E_xPTC, E_PTC, E_xPot, k, mark = drying_process(MC,sP.material_m(21.098, 50, MC), [2, 3, 3],
    #                                                                [75, 65, 55], steaming_time, 20, 60)
    # print(E_xFan, E_Fan, E_xPTC, E_PTC, E_xPot, k, mark)
    # t = [[3, 2, 2],[1, 2, 5],[1, 1, 6]]
    # print(t[-1][2])
    # t = np.array(t) * 3600  # 干制时间换算为秒
    # t_total = np.sum(t)  # 干制总时间换算为秒
    # print(t[1])
    # XGY 1
    # for o in range(3):
    #     if o == 0:
    #         MC = [48.63313, 35.576, 27.05933]
    #         # print(len(MC))
    #         MC_result = [23.522, 14.844, 9.83]
    #         t = [8]
    #         T_dry = [75]
    #     # XGY 2
    #     elif o == 1:
    #         MC = [38.446, 26.382, 27.612, 18.956, 20.255, 21.166]
    #         MC_result = [24.698, 20.614, 17.342, 14.238, 11.684, 12.46]
    #         # MC = [39.93]
    #         # MC_result = [24.698]
    #         t = [2, 3, 3]
    #         T_dry = [75, 65, 55]
    #     # XGY 3
    #     elif o == 2:
    #         MC = [42.43143, 37.9, 32.575, 30.32743]
    #         MC_result = [37.44167, 28.594, 22.85833, 19.425]
    #         # MC = [39.93]
    #         # MC_result = [34.44167]
    #         t = [2, 3, 3]
    #         T_dry = [65, 55, 45]
    #
    #
    #     m = sP.material_m(21.098, 50, MC)
    #     coff = coff_binary_search(MC, MC_result, m, t, T_dry, [4,0],20, 60, 0.1)
    #     print(coff)




