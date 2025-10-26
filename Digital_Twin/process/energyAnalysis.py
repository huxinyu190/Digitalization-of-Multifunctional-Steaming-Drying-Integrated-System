import numpy as np
import pandas as pd
from CoolProp import CoolProp

def exergy_water(n, statePoint, t, T_0, m_v):
    """计算n到m过程水蒸汽（水）的㶲损失"""
    enthalpy = CoolProp.PropsSI('H', 'T', statePoint[2][n - 1], 'P', statePoint[1][n - 1],
                                      'Water')
    entropy = CoolProp.PropsSI('S', 'T', statePoint[2][n - 1], 'P', statePoint[1][n - 1],
                                     'Water')
    exergy = (enthalpy - T_0 * entropy) * t * m_v
    return exergy

def energy_of_steam_generator(t, m_v, eta_steam_generator):
    """计算水经过蒸汽发生器的㶲损失"""
    delta_enthalpy = CoolProp.PropsSI('H', 'T', 273.153+170, 'Q', 1,
                                      'Water') - CoolProp.PropsSI('H', 'T', 293.15, 'Q',
                                                                  0, 'Water')
    delta_entropy = CoolProp.PropsSI('S', 'T', 273.153+170, 'Q', 1,
                                     'Water') - CoolProp.PropsSI('S', 'T', 293.15, 'Q',
                                                                 0, 'Water')
    energy_loss = delta_enthalpy * t * m_v / eta_steam_generator
    return energy_loss

M_water = 18.015                # 水的摩尔质量 (g/mol)
M_air = 28.965                 # 空气的摩尔质量 (g/mol)
R = 8.31446261815324            # 理想气体常数 (J/mol/K)

def RHtoAH(RH, T_0 = 20, reverse = True):
    T_0 = T_0 + 273.153
    if reverse:
        P_sat = CoolProp.PropsSI("P", "T", T_0, "Q", 1, "Water") / 1000       # 水的饱和压力 (kPa)
        AH = 0.622 * ((RH / 100 * P_sat) / (101.325 - 2.339752945174977))                         # 计算空气的绝对湿度 (g/g)
        return AH
    else:
        P_sat = CoolProp.PropsSI("P", "T", T_0, "Q", 1, "Water") / 1000
        RH = RH * (101.325 - 2.339752945174977) / (P_sat * 0.622) * 100
        return RH

def moisture_air_HS(AH, state = 'H', T_0 = 20, P_0 = 101.325):
    T_0 = T_0 + 273.153
    P_0 = P_0 * 1000
    water_enthalpy = CoolProp.PropsSI('H', 'T', T_0, 'P', P_0, 'Water')
    air_enthalpy = CoolProp.PropsSI('H', 'T', T_0, 'P', P_0, 'Air')
    moisture_air_enthalpy = air_enthalpy + water_enthalpy * AH                                  # 湿空气的比焓 (kJ/kg)
    water_entropy = CoolProp.PropsSI('S', 'T', T_0, 'P', P_0, 'Water')
    air_entropy = CoolProp.PropsSI('S', 'T', T_0, 'P', P_0, 'Air')
    moisture_air_entropy = air_entropy + water_entropy * AH                                     # 湿空气的熵 (kJ/kg/K)
    if state == 'H':
        return moisture_air_enthalpy
    elif state == 'S':
        return moisture_air_entropy

def energy_of_fan(t, v_air, eta_fan_generator):
    """计算风机的能量损失"""
    delta_P = 325.54                     # 风机的全压 (Pa)
    delta_energy = v_air * delta_P * t / eta_fan_generator
    return delta_energy

def moisture_air_rho(RH, T, P):
    T = T + 273.153
    P = P * 1000
    P_sat = CoolProp.PropsSI("P", "T", T, "Q", 1, "Water")  # 水的饱和压力 (Pa)
    P_v = RH * P_sat / 100
    P_d = P - P_v
    rho = P_d / (R / M_air * 1000 * T) + P_v / (R / M_water * 1000 * T)
    return rho






if __name__ == '__main__':
    T = 293.15
    P_sat = CoolProp.PropsSI("P", "T", 273.153+20, "Q", 1, "Water") / 1000
    print(P_sat)




