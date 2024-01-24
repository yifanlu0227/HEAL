# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["figure.figsize"] = (5.2,6.5)
matplotlib.rcParams['figure.subplot.left'] = 0.2
matplotlib.rcParams['figure.subplot.bottom'] = 0.2
matplotlib.rcParams['figure.subplot.right'] = 0.8
matplotlib.rcParams['figure.subplot.top'] = 0.8

# 使用ggplot风格
plt.style.use('ggplot')

# 示例数据
data = {
    'HEAL': {'AP': 0.813, 'Params': 7.15 },
    'Late Fusion': {'AP': 0.685, 'Params': 7.3 },
    'AttFusion': {'AP': 0.659, 'Params': 49.22 },
    'DiscoNet': {'AP': 0.695, 'Params': 49.29 },
    'F-Cooper': {'AP': 0.484, 'Params': 49.22 },
    'V2X-ViT': {'AP': 0.753, 'Params': 54.62 },
    'CoBEVT': {'AP': 0.742, 'Params': 51.58 },
    'HM-ViT': {'AP': 0.755, 'Params': 83.34 },
}

# 创建一个散点图
fig, ax = plt.subplots()

# 为每个样本绘制散点和文本
for sample_name, values in data.items():
    ax.scatter(values['Params'], values['AP'], label=sample_name, s=69)
    if sample_name == 'HM-ViT':
        ax.text(values['Params']-8, values['AP']+0.015, sample_name, fontsize=12)#, weight='bold')
    elif sample_name == 'CoBEVT':
        ax.text(values['Params']+1.3, values['AP']-0.025, sample_name, fontsize=12)#, weight='bold')
    elif sample_name == 'V2X-ViT':
        ax.text(values['Params']+2.5, values['AP']-0.005, sample_name, fontsize=12)#, weight='bold')
    elif sample_name == 'HEAL':
        ax.text(values['Params']-3, values['AP']+0.016, sample_name, fontsize=13, weight='bold')
    elif sample_name == 'Late Fusion':
        ax.text(values['Params']-2, values['AP']-0.025, sample_name, fontsize=12)
    else:
        ax.text(values['Params']+2, values['AP']-0.025, sample_name, fontsize=12)#, weight='bold')

# 设置图的标题和坐标轴标签
ax.set_title('Scatter plot of AP vs Params')
ax.set_xlabel('Training Params (M)')
ax.set_ylabel('AP@0.7')
plt.xlim([0, 95])
plt.ylim([0.42, 0.88])

file_path =  "vis_result/AP_FPS/Scatter_plot_of_AP_vs_Params.png"

# 显示图形
plt.savefig(file_path, dpi=400)
plt.close()
print(f"Saving to {file_path}")


