## New Style Yaml and Old Style Yaml

We introduce identifiers such as `m1`, `m2`, ... to indicate the modalities and models that an agent will use. 

However, yaml files without identifiers like `m1` (from [CoAlign](https://github.com/yifanlu0227/CoAlign) repository) still work in this repository. For example, [PointPillar Early Fusion](https://github.com/yifanlu0227/CoAlign/blob/main/opencood/hypes_yaml/opv2v/lidar_only_with_noise/pointpillar_early.yaml). 

Note that there will be some differences in the weight key names of their two models' checkpoint. For example, training with the `m1` identifier will assign some parameters's name with prefix like `encoder_m1.`, `backbone_m1`, etc. But since the model structures are the same, you can convert them using the `rename_model_dict_keys` function in `opencood/utils/model_utils.py`.

