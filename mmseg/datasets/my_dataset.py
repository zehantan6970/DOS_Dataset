# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyDataset(CustomDataset):
    

    CLASSES = ('_background_', 'faeces', 'socks', 'rope', 'plastic_bag' )

    PALETTE = [[0, 0, 0], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64]]

    def __init__(self, **kwargs):
        super(MyDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
