import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ConvNN, SimpleNN


class PolicyAgent:
    def __init__(self, n, obs_shape, n_actions, lr_value = 0.001, lr_policy = 0.001, gamma = 0.99, beta = 0.01, device="cpu", preprocessor = None, load_path = None):
        self.n = n
        self.value_net = SimpleNN(obs_shape, 1).to(device)
        self.policy_net = SimpleNN(obs_shape, n_actions).to(device)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr_value, eps=1e-3)
        self.optimizer_poliy = optim.Adam(self.policy_net.parameters(), lr=lr_policy, eps=1e-3)
        self.gamma = gamma
        self.beta = beta
        self.device = device
        self.preprocessor = preprocessor

        if load_path is not None:
            self.value_net.load_state_dict(torch.load(load_path+"value_weigth.pt", map_location=torch.device(device) ))
            self.policy_net.load_state_dict(torch.load(load_path+"policy_weigth.pt", map_location=torch.device(device) ))

    def list_2_tensor(self, states):
        states_np = np.array([np.array(s, copy=False) for s in states], copy=False)
        states_t = torch.FloatTensor(states_np).float()

        if self.preprocessor is not None:
            states_t = self.preprocessor(states_t)
        states_t = states_t.to(self.device)
        return states_t

    @torch.no_grad()
    def act(self, states, T = 1):
        states = self.list_2_tensor(states)
        logits = T*self.policy_net(states)
        return logits.cpu().numpy()

    def unpack_transitions(self, transitions):
        states_t = self.list_2_tensor(transitions.states)
        next_states_t = self.list_2_tensor(transitions.next_states)
        rewards_t = torch.unsqueeze(torch.FloatTensor(transitions.rewards), dim=1).to(self.device)
        dones_t = torch.unsqueeze(torch.BoolTensor(transitions.dones), dim=1).to(self.device)
        actions_t = torch.LongTensor(transitions.actions).to(self.device)
        return states_t, next_states_t, rewards_t, dones_t, actions_t

    @torch.no_grad()
    def compute_ref_vals(self, next_states, rewards, dones):
        next_values = self.value_net(next_states)
        ref_value = rewards + self.gamma*torch.where(dones, torch.zeros((self.n, 1)).to(self.device), next_values)
        return ref_value

    def get_vals_and_logits(self, states):
        return self.value_net(states), self.policy_net(states)

    def compute_prob_and_entropy(self, logits, actions):
        log_props = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        log_prob_actions = log_props[range(self.n), actions]
        entropy = -(probs * log_props).sum(dim=1)
        return log_prob_actions, entropy

    def analyse_transitions(self, transitions):
        states, next_states, rewards, dones, actions = self.unpack_transitions(transitions)

        ref_vals = self.compute_ref_vals(next_states, rewards, dones)
        vals, logits = self.get_vals_and_logits(states)

        log_prob_actions, entropy = self.compute_prob_and_entropy(logits, actions)

        return vals, ref_vals, log_prob_actions, entropy

    def update(self, transitions):

        states, next_states, rewards, dones, actions = self.unpack_transitions(transitions)

        self.optimizer_value.zero_grad()
        self.optimizer_poliy.zero_grad()

        vals, ref_vals, log_prob_actions, entropy = self.analyse_transitions(transitions)

        # compute value loss and make update step
        vals_loss = F.mse_loss(vals, ref_vals)
        vals_loss.backward()
        self.optimizer_value.step()

        # compute policy loss and make update step
        advs = ref_vals - vals.detach()
        policy_loss = - (advs.squeeze() * log_prob_actions).mean() - self.beta * entropy.mean()
        policy_loss.backward()
        self.optimizer_poliy.step()

    def save_model(self, path = ""):
        torch.save(self.value_net.state_dict(), path + "value_weigth.pt")
        torch.save(self.policy_net.state_dict(), path + "policy_weigth.pt")


