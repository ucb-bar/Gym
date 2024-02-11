import sys
import os
import math

import numpy as np
from isaacgym import gymapi, gymutil
import torch

from env import SimulationEnv, GenericAsset



asset = GenericAsset(None)

env = SimulationEnv(None)

env.loadAsset(asset)


# keys: dict_keys(['root_pos', 'root_rot', 'dof_pos', 'root_vel', 'root_ang_vel', 'dof_vel'])

frame_idx = 0

#env.run()



dof_positions = np.zeros(env.num_dofs)


while True:
    # step the physics
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    
    # env.pose.p = gymapi.Vec3(ref_motion["root_pos"][frame_idx, 0], ref_motion["root_pos"][frame_idx, 1], ref_motion["root_pos"][frame_idx, 2])
    # env.pose.r = gymapi.Quat(
    #     ref_motion["root_rot"][frame_idx, 1], 
    #     ref_motion["root_rot"][frame_idx, 2], 
    #     ref_motion["root_rot"][frame_idx, 3],
    #     ref_motion["root_rot"][frame_idx, 0]
    # )

    # flip direction in Isaac Gym
    dof_positions *= -1

    env.step(dof_positions)

print("Done")
self.stop()

