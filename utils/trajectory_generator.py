import numpy as np
from collections import namedtuple


Transition = namedtuple('Transition', ['states', 'actions', 'next_states', 'rewards', 'dones'])


class GenerateTransitions:
    "Simple TD(0) trajectories generator"

    def __init__(self, envs, agent, apply_softmax=False):
        self.envs = envs
        self.agent = agent
        self.apply_softmax = apply_softmax

    def softmax(self, prob):
        prob_norm = np.exp(prob - np.max(prob))
        return prob_norm / prob_norm.sum()

    def __iter__(self):
        states = []

        for env in self.envs:
            state = env.reset()
            states.append(state)

        while True:
            actions, next_states, rewards, dones = [], [], [], []
            logits = self.agent.act(states)

            actions = []
            for prob, env in zip(logits, self.envs):
                if self.apply_softmax:
                    prob = self.softmax(prob)
                action = np.random.choice(len(prob), p=prob)

                next_state, reward, done, _ = env.step(action)

                if done:
                    next_state = env.reset()

                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

            yield Transition(states, np.array(actions, dtype=np.int32), next_states,
                             np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool))

            states = next_states
