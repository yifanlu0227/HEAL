import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from opencood.visualization.tsne.cka import linear_CKA, kernel_CKA
from opencood.visualization.tsne.tsne_roi import load_data
from opencood.visualization.tsne.mmd import mmd_rbf

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m)))

def compute_distribution_distance(A, B, num_bins=50):
    hist_A = torch.histc(A, bins=num_bins, min=0, max=0).float() + 1e-5  # 加一个小值防止除以0
    hist_B = torch.histc(B, bins=num_bins, min=0, max=0).float() + 1e-5
    
    hist_A /= hist_A.sum()
    hist_B /= hist_B.sum()
    
    return jensen_shannon_divergence(hist_A, hist_B)

def torch_cov(m):
    m_exp = m - m.mean(dim=0)
    return m_exp.t() @ m_exp / (m.size(0) - 1)

# def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#     '''
#     将源域数据和目标域数据转化为核矩阵，即上文中的K
#     Params: 
# 	    source: 源域数据（n * len(x))
# 	    target: 目标域数据（m * len(y))
# 	    kernel_mul: 
# 	    kernel_num: 取不同高斯核的数量
# 	    fix_sigma: 不同高斯核的sigma值
# 	Return:
# 		sum(kernel_val): 多个核矩阵之和
#     '''
#     n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
#     total = torch.cat([source, target], dim=0)#将source,target按列方向合并
#     #将total复制（n+m）份
#     total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#     #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
#     total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#     #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
#     L2_distance = ((total0-total1)**2).sum(2) 
#     #调整高斯核函数的sigma值
#     if fix_sigma:
#         bandwidth = fix_sigma
#     else:
#         bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
#     #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
#     bandwidth /= kernel_mul ** (kernel_num // 2)
#     bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
#     #高斯核函数的数学表达式
#     kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
#     #得到最终的核矩阵
#     return sum(kernel_val)#/len(kernel_val)

# def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#     '''
#     计算源域数据和目标域数据的MMD距离
#     Params: 
# 	    source: 源域数据（n * len(x))
# 	    target: 目标域数据（m * len(y))
# 	    kernel_mul: 
# 	    kernel_num: 取不同高斯核的数量
# 	    fix_sigma: 不同高斯核的sigma值
# 	Return:
# 		loss: MMD loss
#     '''
#     batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
#     kernels = guassian_kernel(source, target,
#         kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
#     #根据式（3）将核矩阵分成4部分
#     XX = kernels[:batch_size, :batch_size]
#     YY = kernels[batch_size:, batch_size:]
#     XY = kernels[:batch_size, batch_size:]
#     YX = kernels[batch_size:, :batch_size]
#     loss = torch.mean(XX + YY - XY -YX)
#     return loss#因为一般都是n==m，所以L矩阵一般不加入计算


if __name__ == "__main__":
    pt_files = ["/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/camera_base.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/lidar_base.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/hybrid_base.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/lidar_single.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/lidar_single_alignto_camera_base.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/lidar_single_alignto_lidar_base.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/lidar_single_alignto_hybrid_base.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/camera_single.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/camera_single_alignto_camera_base.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/camera_single_alignto_lidar_base.pt",
                "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/camera_single_alignto_hybrid_base.pt"]
    # pt_files = ["/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/camera_single.pt",
    #             "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/camera_single_alignto_camera_base.pt",
    #             "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/camera_single_alignto_lidar_base.pt",
    #             "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/camera_single_alignto_hybrid_base.pt"]
    
    label_dict = {i: pt_file.split("/")[-1].rstrip(".pt") for (i, pt_file) in enumerate(pt_files)}
    
    data, datasets, label = load_data(pt_files, random=False)

    M = len(datasets)
    distance_matrix = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            # todo: find a better distance metrics!
            # dist = compute_distribution_distance(datasets[i], datasets[j])
            dist = 1 - kernel_CKA(datasets[i].numpy(), datasets[j].numpy())
            print(dist)
            distance_matrix[i, j] = dist

    labels = [label_dict[idx] for idx in range(M)]

    # 画热力图
    plt.figure(figsize=(20, 20))
    sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.tight_layout()
    plt.savefig("/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs_HEAL/All_RoI_Feature_center/visualization/pairwise_cka")