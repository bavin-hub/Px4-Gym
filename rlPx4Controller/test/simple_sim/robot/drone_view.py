from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class X152bView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "X152bView"
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name        )

        self.physics_rotors = [RigidPrimView(prim_paths_expr=f"/World/X152b/m{i}_prop",
                                             name=f"m{i}_prop_view") for i in range(1, 5)]
