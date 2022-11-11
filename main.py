import gym
import numpy as np
import torch
from agent import PolicyAgent
from trajectory_generator import GenerateNstepTransition
from wrappers import  wrap_dqn
# Log in to your W&B account
import wandb
wandb.login(key = "42822622ab75e399b67576b1ecd07f7ec017e542")
wandb.init(project="Breakout")

n_envs = 64



make_env = lambda: wrap_dqn(gym.make("BreakoutNoFrameskip-v4"))
#make_env = lambda: gym.make("LunarLander-v2")
envs = [make_env() for _ in range(n_envs)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = lambda x: x/255
agent = PolicyAgent(n_envs, envs[0].observation_space.shape, envs[0].action_space.n, device=device, preprocessor = transform, load_path=None)
exp_source = GenerateNstepTransition(5, envs, agent, 0.99)

ongoing_rewards = np.zeros(n_envs)
tot_rewards = 0
game_finished = 0
for idx, (transitions, score) in enumerate(exp_source):

    ongoing_rewards += score.rewards

    game_finished += np.sum(score.dones)
    tot_rewards += np.sum(ongoing_rewards * score.dones)
    ongoing_rewards = ongoing_rewards * (1 - score.dones)

    if idx % 100 == 0:
        if game_finished > 0:
            wandb.log({'score': tot_rewards / game_finished})
            pass
    if idx % 10000 == 0:
        agent.save_model()

    agent.update(transitions)