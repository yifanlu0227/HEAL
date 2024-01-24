# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["figure.figsize"] = (5.2,5.5)
matplotlib.rcParams['figure.subplot.left'] = 0.2
matplotlib.rcParams['figure.subplot.bottom'] = 0.2
matplotlib.rcParams['figure.subplot.right'] = 0.8
matplotlib.rcParams['figure.subplot.top'] = 0.8

# 使用ggplot风格
plt.style.use('ggplot')

# 示例数据
data = {
    'STTR': {'AP': 0.905, 'FPS': 24.036 },
    'AttFusion': {'AP': 0.751, 'FPS': 23.985 },
    'DiscoNet': {'AP': 0.737, 'FPS': 24.702 },
    'F-Cooper': {'AP': 0.632, 'FPS': 27.100 },
    'V2X-ViT': {'AP': 0.79, 'FPS': 10.244 },
    'CoBEVT': {'AP': 0.821, 'FPS': 16.119 },
    'HM-ViT': {'AP': 0.873, 'FPS': 3.797 },
}

# 创建一个散点图
fig, ax = plt.subplots()

# 为每个样本绘制散点和文本
for sample_name, values in data.items():
    ax.scatter(values['FPS'], values['AP'], label=sample_name)
    if sample_name == 'DiscoNet':
        ax.text(values['FPS']-3, values['AP']-0.02, sample_name, fontsize=12, weight='bold')
    elif sample_name == 'F-Cooper':
        ax.text(values['FPS']-5, values['AP']+0.01, sample_name, fontsize=12, weight='bold')
    elif sample_name == 'STTR':
        ax.text(values['FPS']-3, values['AP']+0.01, sample_name, fontsize=13, weight='bold')
    else:
        ax.text(values['FPS']-3, values['AP']+0.01, sample_name, fontsize=12, weight='bold')

# 设置图的标题和坐标轴标签
ax.set_title('Scatter plot of AP vs FPS')
ax.set_xlabel('Inference FPS')
ax.set_ylabel('AP@0.7')
plt.xlim([0, 30])
plt.ylim([0.55, 0.98])

file_path =  "vis_result/AP_FPS/Scatter_plot_of_AP_vs_FPS.png"

# 显示图形
plt.savefig(file_path, dpi=400)
plt.close()
print(f"Saving to {file_path}")


