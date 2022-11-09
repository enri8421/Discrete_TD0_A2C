import gym
import ptan
import numpy as np
import torch
from livelossplot import PlotLosses
from agent import PolicyAgent
from trajectory_generator import GenerateTransitions

n_envs = 64

#make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4"))
make_env = lambda: gym.make("CartPole-v1")
envs = [make_env() for _ in range(n_envs)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = lambda x: x/255
agent = PolicyAgent(n_envs, envs[0].observation_space.shape, envs[0].action_space.n, device=device)
exp_source = GenerateTransitions(envs, agent, True)

ongoing_rewards = np.zeros(n_envs)
tot_rewards = 0
game_finished = 0
plotlosses = PlotLosses(groups={'Score': ['score']})

for idx, transitions in enumerate(exp_source):

    ongoing_rewards += transitions.rewards

    game_finished += np.sum(transitions.dones)
    tot_rewards += np.sum(ongoing_rewards * transitions.dones)
    ongoing_rewards = ongoing_rewards * (1 - transitions.dones)

    if idx % 100 == 0:
        if game_finished > 0:
            plotlosses.update({'score': tot_rewards / game_finished})
            plotlosses.send()

    agent.update(transitions)
