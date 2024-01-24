
from torch.utils.data import Subset
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import os

v2x = True
if v2x:
    from opencood.visualization.draw_fancy.draw_fancy_datasetv2x import SimpleDataset
else:
    from opencood.visualization.draw_fancy.draw_fancy_dataset import SimpleDataset

COLOR = ['red','springgreen','dodgerblue', 'darkviolet']
COLOR_RGB = [ tuple([int(cc * 255) for cc in matplotlib.colors.to_rgb(c)]) for c in COLOR]
COLOR_PC = [tuple([int(cc*0.2 + 255*0.8) for cc in c]) for c in COLOR_RGB]
classes = ['agent1', 'agent2', 'agent3', 'agent4']

def main():
    ## basic setting
    dataset = SimpleDataset()
    data_dict_demo = dataset[0]
    cav_ids = list(data_dict_demo.keys())
    color_map = dict()
    posx = dict()
    posy = dict()
    for (idx, cav_id) in enumerate(cav_ids):
        color_map[cav_id] = COLOR[idx]
        posx[cav_id] = []
        posy[cav_id] = []
    recs = []
    for i in range(0,len(cav_ids)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=COLOR[i]))

   

    
    ## matplotlib setting
    plt.figure()

    ## draw
    print("loop over dataset")
    dataset_len = len(dataset)
    for idx in range(dataset_len):
        print(idx)
        base_data_dict = dataset[idx]


        plt.style.use('dark_background')
        plt.xlim((-100,120))
        plt.ylim((-70,250))
        plt.xticks([])
        plt.yticks([])
        if not v2x:
            plt.gca().invert_xaxis()
        plt.legend(recs,classes,loc='lower left')

        for cav_id, cav_content in base_data_dict.items():
            pos = cav_content['params']['lidar_pose'] # list [6,]
            posx[cav_id].append(pos[0])
            posy[cav_id].append(pos[1])

            start_idx = max(0, idx-10)
            end_idx = idx
            for inner_idx in range(start_idx, end_idx + 1):
                plt.scatter(np.array(posx[cav_id][inner_idx:inner_idx+1]), 
                            np.array(posy[cav_id][inner_idx:inner_idx+1]),
                            s=(inner_idx-start_idx + 1)*4, 
                            alpha=(1 - (end_idx - inner_idx) * 0.09),
                            c=color_map[cav_id])

        if v2x:
            save_path = f"./result_v2x/location_in_bev"
        else:
            save_path = f"./result/location_in_bev"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        plt.savefig(f"{save_path}/trajectory_{idx:02d}.png", dpi=300)
        plt.clf()
    

if __name__ == "__main__":
    main()