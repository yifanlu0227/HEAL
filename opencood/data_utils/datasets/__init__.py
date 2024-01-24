from opencood.data_utils.datasets.late_fusion_dataset import getLateFusionDataset
from opencood.data_utils.datasets.late_heter_fusion_dataset import getLateheterFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import getEarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import getIntermediateFusionDataset
from opencood.data_utils.datasets.intermediate_2stage_fusion_dataset import getIntermediate2stageFusionDataset
from opencood.data_utils.datasets.intermediate_heter_fusion_dataset import getIntermediateheterFusionDataset

from opencood.data_utils.datasets.heter_infer.intermediate_heter_infer_fusion_dataset import getIntermediateheterinferFusionDataset

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from opencood.data_utils.datasets.basedataset.v2xsim_basedataset import V2XSIMBaseDataset
from opencood.data_utils.datasets.basedataset.dairv2x_basedataset import DAIRV2XBaseDataset
from opencood.data_utils.datasets.basedataset.v2xset_basedataset import V2XSETBaseDataset

def build_dataset(dataset_cfg, visualize=False, train=True):
    fusion_name = dataset_cfg['fusion']['core_method']
    dataset_name = dataset_cfg['fusion']['dataset']

    assert fusion_name in ['late', 'lateheter', 'intermediate', 'intermediate2stage', 'intermediateheter', 'early', 'intermediateheterinfer']
    assert dataset_name in ['opv2v', 'v2xsim', 'dairv2x', 'v2xset']

    fusion_dataset_func = "get" + fusion_name.capitalize() + "FusionDataset"
    fusion_dataset_func = eval(fusion_dataset_func)
    base_dataset_cls = dataset_name.upper() + "BaseDataset"
    base_dataset_cls = eval(base_dataset_cls)

    dataset = fusion_dataset_func(base_dataset_cls)(
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
