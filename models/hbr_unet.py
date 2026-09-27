"""Public HBR-UNet model entry point.

The implementation extends the EGE-UNet backbone with the HLLK and BRR
components described in the manuscript.  Keeping the backbone implementation
in ``egeunet.py`` preserves compatibility with checkpoints created before the
manuscript name was shortened to HBR-UNet.
"""

from .egeunet import EGEUNet


class HBRUNet(EGEUNet):
    """HBR-UNet with the manuscript's default architecture settings."""

    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        c_list=(8, 16, 24, 32, 48, 64),
        bridge=True,
        gt_ds=True,
        use_high_level_large_kernel=True,
        use_decoder_brr=True,
        hllk_kernel_size=7,
        use_brr_region_branch=True,
        use_brr_boundary_branch=True,
    ):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            c_list=list(c_list),
            bridge=bridge,
            gt_ds=gt_ds,
            use_high_level_large_kernel=use_high_level_large_kernel,
            use_decoder_brr=use_decoder_brr,
            hllk_kernel_size=hllk_kernel_size,
            use_brr_region_branch=use_brr_region_branch,
            use_brr_boundary_branch=use_brr_boundary_branch,
        )
