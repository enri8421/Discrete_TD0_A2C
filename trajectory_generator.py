import numpy as np
from collections import namedtuple

Transitions = namedtuple('Transition', ['states', 'actions', 'ref_vals'])
Score = namedtuple('Score', ['rewards', 'dones'])


class GenerateNstepTransition:

    def __init__(self, N, envs, agent, gamma):
        self.N = N
        self.n_envs = len(envs)
        self.envs = envs
        self.agent = agent
        self.gamma = gamma
        self.obs_shape = (4, 84, 84)

        self.vec_reset = lambda: np.vectorize(self.reset, otypes=[np.uint8], signature=f"()->(4,84,84)")(
            np.arange(self.n_envs))
        self.vec_step = lambda actions: np.vectorize(self.step, otypes=[np.int8, np.float32, np.bool_],
                                                     signature=f"(),()->(4,84,84),(),()")(np.arange(self.n_envs),
                                                                                          actions)

    def reset(self, env_id):
        state = self.envs[env_id].reset()
        return state

    def step(self, env_id, action):
        state, reward, done, _ = self.envs[env_id].step(action)
        if done:
            state = self.envs[env_id].reset()
        return np.array(state), reward, done

    @staticmethod
    def get_actions(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return (probs.cumsum(-1) >= np.random.uniform(size=probs.shape[:-1])[..., None]).argmax(-1)

    def initialize(self):
        current_state = self.vec_reset()
        states = np.zeros((self.N, self.n_envs) + self.obs_shape, dtype=np.float32)
        rewards = np.zeros((self.N, self.n_envs), dtype=np.float32)
        dones = np.zeros((self.N, self.n_envs), dtype=np.bool_)
        actions = np.zeros((self.N, self.n_envs), dtype=np.int64)

        for i in range(self.N):
            states[i] = current_state

            logits = self.agent.act(current_state)
            action = self.get_actions(logits)

            next_state, reward, done = self.vec_step(action)

            actions[i] = action
            rewards[i] = reward
            dones[i] = done

            current_state = next_state

        return states, rewards, dones, actions, current_state

    def compute_ref_values(self, idx, rewards, dones, current_states):
        ref_values = np.zeros(self.n_envs)
        already_done = np.full(self.n_envs, False)
        discount = 1

        for i in range(self.N):
            curr_idx = (idx + i) % self.N
            ref_values += discount * np.where(already_done, np.zeros(self.n_envs), rewards[curr_idx])
            discount *= self.gamma
            already_done = np.logical_or(dones[curr_idx], already_done)

        next_vals = self.agent.get_vals(current_states)
        ref_values += discount * np.where(already_done, np.zeros(self.n_envs), next_vals.squeeze())

        return ref_values

    def __iter__(self):
        states, rewards, dones, actions, current_states = self.initialize()
        curr_idx = 0

        while True:
            ref_vals = self.compute_ref_values(curr_idx, rewards, dones, current_states)
            transitions = Transitions(states[curr_idx], actions[curr_idx], ref_vals)
            scores = Score(rewards[curr_idx], dones[curr_idx])
            yield transitions, scores

            logits = self.agent.act(current_states)
            action = self.get_actions(logits)

            next_state, reward, done = self.vec_step(action)

            actions[curr_idx] = action
            rewards[curr_idx] = reward
            dones[curr_idx] = done
            current_state = next_state
            curr_idx = (curr_idx + 1) % self.N
