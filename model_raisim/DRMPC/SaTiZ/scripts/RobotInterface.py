import numpy as np

class RobotInterface():
    def __init__(self, cfg):
        self.cfg = cfg
        pass

    @staticmethod
    def LoadUrdfParam(cfg):
        # region Robot geometry params
        L_body = cfg["Robot"]["Geometry"]["L_body"]
        L_thigh = cfg["Robot"]["Geometry"]["L_thigh"]
        L_shank = cfg["Robot"]["Geometry"]["L_shank"]

        params = {"L_body": L_body, "L_thigh": L_thigh, "L_shank": L_shank}
        # endregion

        # region inertia & mass params
        inertia = cfg["Robot"]["Mass"]["inertia"]
        mass = cfg["Robot"]["Mass"]["mass"]
        massCenter = cfg["Robot"]["Mass"]["massCenter"]
        # endregion

        params.update({"body_inertia": inertia[0], "thigh_inertia": inertia[1], "shank_inertia": inertia[2]})
        params.update({"body_mass": mass[0], "thigh_mass": mass[1], "shank_mass": mass[2]})
        params.update({"body_mc": massCenter[0], "thigh_mc": massCenter[1], "shank_mc": massCenter[2]})

        return params