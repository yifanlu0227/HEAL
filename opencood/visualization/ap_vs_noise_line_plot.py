# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
methods = ['HMViT', 'V2XViT', 'CoBEVT','HEAL']

xaxis_names = ['Pose Noise Std (m & deg)', 'Pose Noise Std (m & deg)', "Compression Ratio", "Compression Ratio"]
yaxis_names = ['Performace AP50', 'Performace AP70','Performace AP50','Performace AP70']

noise_level = [0, 0.2, 0.4, 0.6, 0.8, 1.0]#, 1.2]
pose_error_performance_50 = {
    "HMViT":  [0.876,	0.855,	0.79,	0.72,	0.682,	0.652],#,	0.64],
    "V2XViT": [0.882,	0.859,	0.8,	0.743,	0.688,	0.652],#,	0.621],
    "CoBEVT": [0.885,	0.862,	0.793,	0.715,	0.663,	0.634],#,	0.605],
    "HEAL":   [0.894,	0.869,	0.802,	0.747,	0.716,	0.692],#,	0.683],
}
pose_error_performance_70 = {
    "HMViT":  [0.755,	0.667,	0.515,	0.427,	0.399,	0.386],#,	0.385],
    "V2XViT": [0.753,	0.685,	0.582,	0.512,	0.481,	0.459],#,	0.447],
    "CoBEVT": [0.742,	0.646,	0.499,	0.421,	0.398,	0.388],#,	0.394],
    "HEAL":   [0.813,	0.712,	0.586,	0.532,	0.523,	0.524],#,	0.528],
}

compression_ratio = [1, 2, 3, 4, 5, 6]
compression_ratio_performance_50 = {
    "HMViT":  [0.876,	0.869,	0.869,	0.876,	0.869,  0.872],
    "V2XViT": [0.882,	0.868,	0.881,	0.881,	0.873,	0.874],
    "CoBEVT": [0.885,	0.869,	0.886,	0.879,	0.887,	0.88],
    "HEAL":   [0.894,	0.896,	0.897,	0.897,	0.888,	0.879],
}
compression_ratio_performance_70 = {
    "HMViT":  [0.755,	0.732,	0.732,	0.743,	0.734,  0.735],
    "V2XViT": [0.753,	0.737,	0.744,	0.741,	0.737,	0.721],
    "CoBEVT": [0.742,	0.731,	0.728,	0.736,	0.737,	0.724],
    "HEAL":   [0.813,	0.821,	0.824,	0.822,	0.818,	0.806],
}

draw_list = [
    {'xaxis': noise_level, 'yaxis': pose_error_performance_50, "ylim":[0.0, 0.9]},
    {'xaxis': noise_level, 'yaxis': pose_error_performance_70, "ylim":[0.0, 0.85]},
    {'xaxis': compression_ratio, 'yaxis': compression_ratio_performance_50, 'xlabel':[1,2,4,8,16,32], "ylim":[0.77, 0.92]},
    {'xaxis': compression_ratio, 'yaxis': compression_ratio_performance_70, 'xlabel':[1,2,4,8,16,32], "ylim":[0.55, 0.88]},
]

color = {
    'HEAL': 'r',
    'CoBEVT': 'skyblue',
    'V2XViT': 'slategrey',
    'HMViT': 'mediumpurple',
}

# 创建一个大图
fig, axs = plt.subplots(1, 4, figsize=(12, 2.85))

# 对于每一个子图，绘制所有方法的折线图
for idx, ax in enumerate(axs):
    data = draw_list[idx]
    for method in methods:
        ax.plot(data['xaxis'], data['yaxis'][method], '-s', label=method, markersize=3, color=color[method])  # '-s' 表示线型为'-'，marker为's' (正方形)
    ax.set_xticks(data['xaxis'])
    if 'Pose' in xaxis_names[idx]:
        ax.set_xticks(data['xaxis'][::2])

    ax.set_xlabel(xaxis_names[idx])
    ax.set_ylabel(yaxis_names[idx])
    ax.set_ylim(data['ylim'])
    ax.grid(True)
    ax.tick_params(axis='both', colors='black')


    if 'xlabel' in data:
        ax.set_xticklabels(data['xlabel'])


    

    # if idx==0:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend((handles), (labels), loc='lower left')

# # 添加图例
# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=len(methods))#, bbox_to_anchor=(0.5, -0.15))

plt.tight_layout()
# plt.subplots_adjust(bottom=0.23)  # 调整底部空间以适应图例

file_path =  "vis_result/AP_PoseError/line_plot_of_AP_vs_PoseError.png"

# 显示图形
plt.savefig(file_path, dpi=500)
plt.close()
print(f"Saving to {file_path}")


