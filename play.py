import gym
import ptan
import numpy as np
import torch
from livelossplot import PlotLosses
from agent import PolicyAgent
from trajectory_generator import GenerateTransitions


def softmax(prob):
    prob_norm = np.exp(prob - np.max(prob))
    return prob_norm / prob_norm.sum()


env = gym.make("CartPole-v1")
env = ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4", render_mode='human'))
n_action = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

agent = PolicyAgent(1, env.observation_space.shape, env.action_space.n, device=device) # load_path="weights/cardpole/")
state = env.reset()

done = False
while not done:
    # env.render()
    prob = softmax(agent.act([state]))
    action = np.random.choice(n_action, p=prob[0])
    state, reward, done, _ = env.step(action)

