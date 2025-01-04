import pybullet_envs
import pybullet as p
import joblib
import gym

chkpt_dict = joblib.load("checkpoint.pkl")
# random.setstate(chkpt_dict["random_rng_state"])
# np.random.set_state(chkpt_dict["np_rng_state"])

env1 = gym.make('AntBulletEnv-v0')

env1.seed(0)
env1.action_space.seed(0)
env1.observation_space.seed(0)

env1.reset()
p1 = env1.env._p

base_po = chkpt_dict["base_po"]
base_v = chkpt_dict["base_v"]
joint_states = chkpt_dict["joint_states"]
# reset env2 to the current state of env1
for i in range(p1.getNumBodies()):
    p1.resetBasePositionAndOrientation(i, *base_po[i])
    p1.resetBaseVelocity(i, *base_v[i])
    for j in range(p1.getNumJoints(i)):
        p1.resetJointState(i, j, *joint_states[i][j][:2])

# p1.restoreState(chkpt_dict["pb_state"])
# env = chkpt_dict["env"]
# print(p.getNumBodies())
# breakpoint()

# env.seed(0)
# env.action_space.seed(0)
# env.observation_space.seed(0)

# env.reset()


print(env1.reset())
