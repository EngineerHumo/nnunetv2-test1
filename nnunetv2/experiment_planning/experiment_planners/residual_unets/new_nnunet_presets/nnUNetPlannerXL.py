from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.residual_unet import ResidualEncoderUNet

from nnunetv2.experiment_planning.experiment_planners.residual_unets.ResEncUNet_planner import ResEncUNetPlanner


class nnUNetPlannerXL(ResEncUNetPlanner):
    """
    Target is 40 GB VRAM max -> A100 40GB, RTX 6000 Ada Generation
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetXLPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet
        # this is supposed to give the same GPU memory requirement as the default nnU-Net
        self.UNet_reference_val_3d = 4500000000
        self.UNet_reference_val_2d = 250000000

