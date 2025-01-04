import pybullet_envs
import gym
import numpy as np
import random
import joblib
from copy import deepcopy


env1 = gym.make('AntBulletEnv-v0')

env1.seed(0)
env1.action_space.seed(0)
env1.observation_space.seed(0)

env1.reset()
p = env1.env._p

base_po = []  # position and orientation of base for each body
base_v = []  # velocity of base for each body
joint_states = []  # joint states for each body
for i in range(p.getNumBodies()):
    base_po.append(p.getBasePositionAndOrientation(i))
    base_v.append(p.getBaseVelocity(i))
    joint_states.append([p.getJointState(i, j) for j in range(p.getNumJoints(i))])

joblib.dump(
    {
        "base_po": base_po,
        "base_v": base_v,
        "joint_states": joint_states,
    },
    "checkpoint.pkl",
)

print(env1.reset())
