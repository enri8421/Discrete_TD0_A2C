import gym
from trajectory_generator import GenerateTransitions
import numpy as np
from collections import namedtuple

Transitions = namedtuple('Transition', ['states', 'actions', 'ref_vals'])

class ToyEnv(gym.Env):
    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=3)
        self.index_step = 0

    def reset(self):
        self.index_step = 0
        return self.index_step

    def step(self, action):
        self.index_step += 1
        return self.index_step % self.observation_space.n, float(action), self.observation_space == 10, {}


class BasicAgent:


    def act(self, states):
        return np.zeros((len(states), 3))

class GenerateNstepTransition:

    def __init__(self, N, envs, agent, gamma, apply_softmax=False):
        self.N = N
        self.n_envs = len(envs)
        self.envs = envs
        self.agent = agent
        self.gamma = gamma
        self.apply_softmax = apply_softmax

    @staticmethod
    def softmax(prob):
        prob_norm = np.exp(prob - np.max(prob))
        return prob_norm / prob_norm.sum()


    def get_initial_states(self):
        states = []
        for env in self.envs:
            states.append(env.reset())
        return states

    def make_env_step(self, env_id, probs):
        if self.apply_softmax:
            probs = self.softmax(probs)
        action = np.random.choice(len(probs), p=probs)
        next_state, reward, done, _ = self.envs[env_id].step(action)
        if done:
            next_state = self.envs[env_id].reset()
        return next_state, reward, done, action


    def initialize(self):

        current_states = self.get_initial_states()
        rewards = np.zeros((self.N, self.n_envs))
        dones = np.zeros((self.N, self.n_envs), dtype=np.bool_)
        actions = np.zeros((self.N, self.n_envs), dtype=np.int32)
        states = []

        for i in range(self.N):
            logits = self.agent.act(current_states)
            states.append(current_states)
            next_states = []
            for env_id in range(self.n_envs):
                next_state, reward, done, action = self.make_env_step(env_id, logits[env_id])
                next_states.append(next_state)
                actions[i, env_id] = action
                rewards[i, env_id] = reward
                dones[i, env_id] = done

            current_states = next_states

        return states, rewards, dones, actions, current_states

    def compute_ref_values(self, idx, rewards, dones, current_states):

        ref_values = np.zeros(self.n_envs)
        already_done = np.full(self.n_envs, False)
        discount = 1

        for i in range(self.N):
            curr_idx = (idx + i)%self.N
            ref_values += discount*np.where(already_done, np.zeros(self.n_envs), rewards[curr_idx])
            discount *= self.gamma
            already_done = np.logical_or(dones[curr_idx], already_done)

        next_vals = self.agent.get_vals(current_states)
        ref_values += discount*np.where(already_done, np.zeros(self.n_envs), next_vals)

        return ref_values


    def __iter__(self):
        states, rewards, dones, actions, current_states = self.initialize()
        curr_idx = 0

        while True:
            ref_vals = self.compute_ref_values(curr_idx, rewards, dones, current_states)
            transitions = Transitions(states[curr_idx], actions[curr_idx], ref_vals)
            yield transitions

            logits = self.agent.act(current_states)
            states[current_states] = current_states
            next_states = []
            for env_id in range(self.n_envs):
                next_state, reward, done, action = self.make_env_step(env_id, logits[env_id])
                next_states.append(next_state)
                actions[curr_idx, env_id] = action
                rewards[curr_idx, env_id] = reward
                dones[curr_idx, env_id] = done

            current_states = next_states


envs = [ToyEnv() for _ in range(3)]

agent = BasicAgent()

exp_source = GenerateNstepTransition(4, envs, agent, True)


limit = 15

for idx, tra in enumerate(exp_source):
    print(tra)
    if idx == limit:
        break



