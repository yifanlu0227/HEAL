import cv2
import numpy as np
import glob
import os


projnames = ['/GPFS/rhome/yifanlu/OpenCOOD/opencood/logs/OPV2V_npj_v2xvit_w_2022_09_05_18_52_53/vis_0.4_0.4_0_0_video_vis',
             '/GPFS/rhome/yifanlu/OpenCOOD/opencood/logs/OPV2V_npj_v2xvit_w_2022_09_05_18_52_53/vis_0.6_0.6_0_0_video_vis']
# projnames = ['/GPFS/rhome/yifanlu/OpenCOOD/opencood/logs/OPV2V_npj_disconet_w_2022_09_02_16_19_51/vis_0.4_0.4_0_0_video_vis',
#              '/GPFS/rhome/yifanlu/OpenCOOD/opencood/logs/OPV2V_npj_disconet_w_2022_09_02_16_19_51/vis_0.6_0.6_0_0_video_vis']
# projnames = ['/GPFS/rhome/yifanlu/OpenCOOD/opencood/logs/OPV2V_npj_v2vnet_robust_new/vis_0.4_0.4_0_0_video_vis',
#              '/GPFS/rhome/yifanlu/OpenCOOD/opencood/logs/OPV2V_npj_v2vnet_robust_new/vis_0.6_0.6_0_0_video_vis']
# projnames = ['/GPFS/rhome/yifanlu/OpenCOOD/opencood/logs/OPV2V_npj_ms_ba/vis_0.4_0.4_0_0_video_vis',
#              '/GPFS/rhome/yifanlu/OpenCOOD/opencood/logs/OPV2V_npj_ms_ba/vis_0.6_0.6_0_0_video_vis']

print(projnames)

for projname in projnames:
    img_array = []
    for filename in sorted(glob.glob(f'{projname}/3d_*'))[30:75]:
        print(filename)
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    size = (2560, 1920)
    out = cv2.VideoWriter(f'./result_video_cut_bev/v2xvit_{projname.split("/")[-1]}'+".mp4",cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()