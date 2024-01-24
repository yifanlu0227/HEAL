import torch
from opencood.tools.heal_tools import get_model_path_from_dir
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns

def kernel_cosine_similarity(weight1, weight2):
    """
    weight: [C_out, C_int, kernel_H, kernel_W]
    """
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    result = cos(weight1, weight2)
    return torch.mean(result)


if __name__ == "__main__":
    models = [
        "opencood/logs_HEAL/Pyramid_m1_base_2023_08_06_18_27_04",
        "opencood/logs_HEAL/Pyramid_m1m2_base_finetune_from_m1_base",
        "opencood/logs_HEAL/Pyramid_m2_base_2023_08_06_18_30_21",
    ]

    param_names = [
        "pyramid_backbone.resnet.layer0.0.conv1.weight",
        "pyramid_backbone.resnet.layer1.0.conv1.weight",
        "pyramid_backbone.resnet.layer2.0.conv1.weight",
        "shrink_conv.layers.0.double_conv.0.weight",
        "pyramid_backbone.single_head_0.weight",
        "pyramid_backbone.single_head_1.weight",
        "pyramid_backbone.single_head_2.weight",
        "cls_head.weight",
        "reg_head.weight",
        "dir_head.weight",
    ]

    state_dict_list = [torch.load(get_model_path_from_dir(model), map_location='cpu') for model in models]
    for param_name in param_names:
        label_dict = {
            i: model.split("/")[-1].split("_base")[0]
            for (i, model) in enumerate(models)
        }
        

        M = len(models)
        labels = [label_dict[idx] for idx in range(M)]
        distance_matrix = np.zeros((M, M))

        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                dist = 1 - kernel_cosine_similarity(
                    state_dict_list[i][param_name], state_dict_list[j][param_name]
                )
                print(dist)
                distance_matrix[i, j] = dist

        # 画热力图
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            distance_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.tight_layout()
        plt.savefig(
            f"/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/vislog/weight_similarity/{param_name}.png"
        )
