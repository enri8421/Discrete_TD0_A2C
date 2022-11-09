import gym
import ptan
import numpy as np
import torch
from agent import PolicyAgent
from trajectory_generator import GenerateTransitions
# Log in to your W&B account
import wandb
wandb.login(key = "42822622ab75e399b67576b1ecd07f7ec017e542")
wandb.init(project="Lunar-Lander-v2")

n_envs = 64



#make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4"))
make_env = lambda: gym.make("LunarLander-v2")
envs = [make_env() for _ in range(n_envs)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = lambda x: x/255
agent = PolicyAgent(n_envs, envs[0].observation_space.shape, envs[0].action_space.n, device=device)
exp_source = GenerateTransitions(envs, agent, True)

ongoing_rewards = np.zeros(n_envs)
tot_rewards = 0
game_finished = 0
for idx, transitions in enumerate(exp_source):

    ongoing_rewards += transitions.rewards

    game_finished += np.sum(transitions.dones)
    tot_rewards += np.sum(ongoing_rewards * transitions.dones)
    ongoing_rewards = ongoing_rewards * (1 - transitions.dones)

    if idx % 100 == 0:
        if game_finished > 0:
            wandb.log({'score': tot_rewards / game_finished})
    if idx % 10000 == 0:
        agent.save_model()

    agent.update(transitions)
