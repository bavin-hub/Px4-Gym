from typing import Optional

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import numpy as np
import torch
import carb


class X152b(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "X152b",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.array] = None
    ) -> None:
        """[summary]
        """
        
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:

            local_assets_path = "/home/hao/Documents/isaac_sim_sense/Flyscene"
            self._usd_path = local_assets_path + "/x-152bV2.1.usd"
            
        add_reference_to_stage(self._usd_path, prim_path)
        scale = torch.tensor([1, 1, 1])

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale
        )
