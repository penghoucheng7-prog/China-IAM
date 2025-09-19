#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

def calculate_rmse_and_std_for_years(model_data, observed_data, years=[2005, 2010, 2015, 2020]):
    results = {}  # 存储每个模型情景索引下的RMSE和标准差结果

    # 遍历模型数据的索引，这些索引可能代表了不同的模型情景或变量
    for index in model_data.index.unique():
        rmse_list = []  # 存储每个变量在四年的RMSE
        std_list = []  # 存储每个变量在四年的标准差
        mval_list = [] # 储存每个变量在四年的模型值
        normalized_rmse_list = [] # 储存每个变量在四年的标准化RMSE
        msed_list =[] #储存每个变量在四年的RMSE_D
        bias_list = []  # 储存每个变量的偏差

        # 检查模型数据和观测数据中是否存在相应的变量以及对应的年份数据
        for year in years:
            if str(year) not in model_data.columns or str(year) not in observed_data.columns:
                continue

            # 获取模型数据和观测数据中的对应年份的值
            model_values = model_data.loc[index, str(year)]
            observed_values = observed_data.loc[index, str(year)]

            # 如果模型值或观察值存在缺失值，则跳过该年份的计算
            if pd.isna(model_values) or pd.isna(observed_values):
                continue
            
            # 如果数据完整，则计算均方误差并存储每年的观测值
            if not np.isnan(model_values) and not np.isnan(observed_values):
                mse = (model_values - observed_values) ** 2
                mse_d = model_values - observed_values
                rmse = np.sqrt(mse)  # 计算RMSE
                rmse_list.append(rmse)
                std_list.append(observed_values)
                mval_list.append(model_values)
                msed_list.append(mse_d)

        # 计算每个变量在四年的RMSE和标准差
        if rmse_list:
            rmse_avg = np.nanmean(rmse_list)
            std_values = np.array(std_list).flatten()  # 将四年的观测值扁平化为一维数组
            std_avg = np.nanstd(std_values)  # 计算四年观测值的标准差
            mval_values = np.array(mval_list).flatten()  # 将四年的模型值扁平化为一维数组
            msed_values = np.array(msed_list).flatten()  # 将四年的MSED扁平化为一维数组
            aligned_mval_values = mval_values[:len(std_values)]
            bias = np.nanmean(aligned_mval_values) - np.nanmean(std_values)
            
            # 计算 NORMAL_bias
            normal_bias = bias / std_avg if std_avg != 0 else np.nan

            # 遍历整个 msed_values，对不为0的值减去 bias，然后平方
            squared_values = [((value - bias) ** 2 if value != 0 else 0) for value in msed_values]
            # 计算平均值
            c_mse = np.nanmean(squared_values)
            c_rmse = np.sqrt(c_mse)
            normal_rmse = rmse_avg / std_avg
            normal_crmse = c_rmse / std_avg
            
            # 添加结果到字典
            results[index] = {
                'RMSE': rmse_avg, 
                'STD': std_avg, 
                'NORMAL_RMSE': normal_rmse, 
                'BIAS': bias, 
                'NORMAL_bias': normal_bias,  # 添加 NORMAL_bias
                'CERTERED_RMSE': c_rmse, 
                'C_NORMAL_RMSE': normal_crmse
            }

    return results

