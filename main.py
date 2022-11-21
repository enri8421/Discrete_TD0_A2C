import gym
import numpy as np
import torch
from absl import flags
import sys

from utils.agent import PolicyAgent
from utils.trajectory_generator import GenerateTransitions
from utils.configuration import get_config

FLAGS = flags.FLAGS
flags.DEFINE_string('env', None, 'environment name')
flags.DEFINE_string('env_name', None, 'environment name for gym')
flags.DEFINE_boolean("load_weight", False, 'start training from current stored weight')
flags.DEFINE_boolean("save_weight", False, 'save weight in apposite folder')
flags.DEFINE_integer("n_envs", None, 'number of parallel env')
flags.DEFINE_float("lr_value", None, "lr value net")
flags.DEFINE_float("lr_policy", None, "lr policy net")
flags.DEFINE_float("gamma", None, "discount factor")
flags.DEFINE_float("beta", None, "entropy regularization weight")
flags.mark_flag_as_required('env')
flags.mark_flag_as_required('env_name')

FLAGS(sys.argv)
config = get_config(FLAGS)

make_env = lambda: gym.make(FLAGS.env_name)
envs = [make_env() for _ in range(config.n_envs)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

agent = PolicyAgent(config.n_envs, 
                    envs[0].observation_space.shape, 
                    envs[0].action_space.n,
                    config.lr_value,
                    config.lr_policy,
                    config.gamma,
                    config.beta,
                    device, 
                    None,
                    config.load_path)

exp_source = exp_source = GenerateTransitions(envs, agent, True)

ongoing_rewards = np.zeros(config.n_envs)
tot_rewards = 0
game_finished = 0
for idx, transitions in enumerate(exp_source):

    ongoing_rewards += transitions.rewards

    game_finished += np.sum(transitions.dones)
    tot_rewards += np.sum(ongoing_rewards*transitions.dones)
    ongoing_rewards = ongoing_rewards*(1 - transitions.dones)

    if idx % 100 == 0:
        if game_finished > 0:
            print(f"Trainign performance after itr {idx}: {tot_rewards / game_finished}")
    if (idx % 10000 == 0) and config.save:
        agent.save_model(config.save_path)

    agent.update(transitions)
