import gym
import numpy as np
import torch
from utils.agent import PolicyAgent
from absl import flags
import sys

env_string = {"CartPole" : "CartPole-v1", "LunarLander" : "LunarLander-v2", "Acrobot" : "Acrobot-v1"}

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'LunarLander', 'environment name')
FLAGS(sys.argv)


env = gym.make(env_string[FLAGS.env])

n_action = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

agent = PolicyAgent(1, env.observation_space.shape, env.action_space.n,
                    device=device, load_path=f"weights/{FLAGS.env}/")
state = env.reset()

done = False
while not done:
    env.render()
    logit = agent.act([state])
    prob_norm = np.exp(logit - np.max(logit))
    prob = prob_norm / prob_norm.sum()
    action = np.random.choice(n_action, p=prob[0])
    state, reward, done, _ = env.step(action)

