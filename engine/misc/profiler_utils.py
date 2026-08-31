"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from calflops import calculate_flops
from typing import Tuple

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:
    # Profiling must use the same HxW that was used to precompute positional
    # embeddings and decoder anchors. The old square collate base size caused
    # 640x640 features to be combined with 672x1184 positional embeddings.
    if cfg.eval_spatial_size is not None:
        height, width = cfg.eval_spatial_size
        input_shape = (1, 3, int(height), int(width))
    else:
        base_size = cfg.train_dataloader.collate_fn.base_size
        if isinstance(base_size, (list, tuple)):
            input_shape = (1, 3, int(base_size[0]), int(base_size[1]))
        else:
            input_shape = (1, 3, int(base_size), int(base_size))

    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(model=model_for_info,
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=False)
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
