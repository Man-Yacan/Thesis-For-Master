# -*- coding: utf-8 -*-

"""
@author: ManYacan
@Email: myxc@live.cn
@Website: www.manyacan.com
@time: 2022/07/25 19:44
@Description: 引入数据分析所需的所有包、读取文件并合并为一个大df。
"""

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import hues
import os
import math

# matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False

# 定义常量
CHAR = [chr(i) for i in range(97, 123)]  # 获取26个英文字母，用于给子图编号
DATA_PATH, EXPORT_PATH, DPI = './data/', './export/', 300
LEVEL_EXPORT_PATH = EXPORT_PATH + 'Level/'
ROW_NUM, COL_NUM = 5, 3  # 根据子图个数准备布图网格
DATA_NAME_LIST = {  # 所有需要用到的文件名
    '上覆水环境因子': 'overlying_env.csv',
    '底泥环境因子': 'sediment_env.csv',
    '底泥粒径': 'sediment_size.csv',
    '样品CT值': 'sample_ct.csv',
    '三维荧光': '../export/EEMs.csv',
    '拟合曲线': '../export/fitting_formula.csv',
}
IMG_TYPE = '.svg'
# 读取csv文件
overlying_water_env_df = pd.read_csv(
    DATA_PATH + DATA_NAME_LIST['上覆水环境因子'], index_col=0)  # 上覆水环境因子
sediment_size_df = pd.read_csv(
    DATA_PATH + DATA_NAME_LIST['底泥粒径'], index_col=0).T  # 底泥粒径
sediment_env_df = pd.read_csv(
    DATA_PATH + DATA_NAME_LIST['底泥环境因子'], index_col=0)  # 底泥环境因子
fitting_formula = pd.read_csv(
    EXPORT_PATH + DATA_NAME_LIST['拟合曲线'], index_col=0)  # 标注曲线拟合方程
eems_df = pd.read_csv(
    EXPORT_PATH + DATA_NAME_LIST['三维荧光'], index_col=0)  # 三维荧光数据
ct_origin = pd.read_csv(DATA_PATH + DATA_NAME_LIST['样品CT值'])  # 样品ct值

# （D90-D10）/D50 用来描述污泥粒度的分布宽度，数值越大表明粒度分布越宽，反之，则表明粒度分布越窄。
sediment_env_df['S-Size-d((0.9-0.1)/0.5)'] = (sediment_env_df['S-Size-d(0.9)'] - sediment_env_df['S-Size-d(0.1)']).div(
    sediment_env_df['S-Size-d(0.5)'])

# 将样品ct值带入标准曲线拟合公式转化为绝对丰度，CT值全部带入拟合后的标准曲线，获取所有样本所有基因的拷贝数。y=kx+b，x=(y-b)/k
ct_long = ct_origin.melt(id_vars='ID')
ct_long.columns = ['ID', 'Gene', 'CT']
ct_df = ct_long.pivot_table(
    index='ID', columns='Gene', values='CT', aggfunc='mean')
gene_abundance = ct_df.apply(lambda x: (x - fitting_formula[x.name][1]) / fitting_formula[x.name][
    0])
gene_abundance = gene_abundance.applymap(
    lambda x: np.power(10, x) / 2 * 100 / 0.3)  # 转化为拷贝数
# 列排序
gene_list = ct_origin.columns.to_list()[1:]
gene_abundance = gene_abundance[gene_list]

# 合并数据表的预处理，合并后的表格应为90×133
all_df_0 = pd.concat([gene_abundance, sediment_env_df, eems_df,
                     overlying_water_env_df, sediment_size_df], axis=1)
all_df = all_df_0.copy()

# 向df中新增季节、河流两列，用于数据区分
all_df['Period'] = all_df.apply(lambda x: 'Dry' if 'D' in x.name else ('Wet' if 'W' in x.name else 'Level'),
                                axis=1) + ' Season'
all_df['River'] = all_df.apply(lambda x: 'Nanfei' if 'N' in x.name else ('Pai' if 'P' in x.name else 'Hangbu'),
                               axis=1) + ' River'
all_df['SortNum'] = all_df.index.str[3:].astype(float)

# 对季节、河流两列进行排序
river_order = CategoricalDtype(
    ['Nanfei River', 'Pai River', 'Hangbu River'],
    ordered=True
)
period_order = CategoricalDtype(
    ['Dry Season', 'Level Season', 'Wet Season'],
    ordered=True
)
all_df['River'] = all_df['River'].astype(river_order)
all_df['Period'] = all_df['Period'].astype(period_order)
all_df = all_df.sort_values(by=['Period', 'River', 'SortNum'])
all_df.pop('SortNum')

# 定义一些经常用到的变量
period_list = all_df['Period'].unique().tolist()
river_list = all_df['River'].unique().tolist()
gene_list = ['Bacterial 16S rRNA', 'Archaeal 16S rRNA', 'AOA_amoA', 'AOB_amoA', 'nxrA',
             'narG', 'napA', 'nirK', 'nirS', 'nosZ', 'norB', 'hzsA', 'hzsB', 'hzo', 'nifH']
unit = {
    'W-TN': 'mg/L',
    'W-NO_{3}^{-}': 'mg/L',
    'W-NO_{2}^{-}': 'mg/L',
    'W-NH_{4}^{+}': 'mg/L',
    'W-TP': 'mg/L',
    'W-PO_{4}^{3-}': 'mg/L',
    'W-COD': 'mg/L',
    'W-DO': 'mg/L',
    'W-pH': '-',
    'W-T': '℃',
    'S-TN': 'mg/kg',
    'S-TP': 'mg/kg',
    'S-pH': '-'
}
IMG_TYPE = '.svg'


# 定义常用函数
def calc_pca(x, n=2):
    """
    PCA函数
    :param x: 输入矩阵
    :param n: 降维后的矩阵维数
    :return: 降维后的矩阵
    """
    x.dropna(how='any', inplace=True)
    x = (x - x.mean()) / x.std(ddof=0)
    SIGMA = (x.T @ x) / x.shape[0]
    U, S, V = np.linalg.svd(SIGMA)
    P = U[:, :n]
    Z = x @ P
    Z = Z.rename(columns={
        0: 'PCA 0',
        1: 'PCA 1',
    })
    Z['River'] = Z.apply(lambda x: 'Nanfei' if 'N' in x.name else (
        'Pai' if 'P' in x.name else 'Hangbu'), axis=1) + ' River'
    Z['Period'] = Z.apply(lambda x: 'Dry' if x.name.startswith('D') else ('Level' if x.name.startswith('L') else 'Wet'),
                          axis=1) + ' Season'
    return Z
