import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.8


class ActorCritic(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.linear1 = nn.Linear(n_inputs, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)

        self.actor = nn.Linear(64, n_inputs)
        self.critic = nn.Linear(64, 1)

    # In a PyTorch model, you only have to define the forward pass. PyTorch computes the backwards pass for you!
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        return x

    # Only the Actor head
    def get_action_probs(self, x):
        x = self(x)
        x = self.actor(x)
        action_probs = torch.sigmoid(x)
        return action_probs

    # Only the Critic head
    def get_state_value(self, x):
        x = self(x)
        state_value = self.critic(x)
        return state_value

    # Both heads
    def evaluate_actions(self, x):
        x = self(x)
        action_probs = torch.sigmoid(self.actor(x))
        state_values = self.critic(x)
        return action_probs, state_values

    def calc_actual_state_values(self, rewards, dones, states):
        R = []
        rewards.reverse()

        # If we happen to end the set on a terminal state, set next return to zero
        if dones[-1] == True:
            next_return = 0

        # If not terminal state, bootstrap v(s) using our critic
        # TODO: don't need to estimate again, just take from last value of v(s) estimates
        else:
            s = torch.from_numpy(states[-1]).float().unsqueeze(0)
            next_return = self.get_state_value(s).data[0][0]

        # Backup from last state to calculate "true" returns for each state in the set
        R.append(next_return)
        dones.reverse()
        for r in range(1, len(rewards)):
            if not dones[r]:
                this_return = rewards[r] + next_return * GAMMA
            else:
                this_return = 0
            R.append(this_return)
            next_return = this_return

        R.reverse()
        state_values_true = torch.FloatTensor(R).unsqueeze(1)

        return state_values_true

    def reflect(self, states, actions, rewards, dones):

        # Calculating the ground truth "labels" as described above
        state_values_true = self.calc_actual_state_values(rewards, dones, states)

        s = torch.FloatTensor(states)
        action_probs, state_values_est = self.evaluate_actions(s)
        action_log_probs = action_probs.log()
        a = torch.LongTensor(actions).view(-1, self.n_actions)
        chosen_action_log_probs = action_log_probs.gather(1, a)

        # This is also the TD error
        advantages = state_values_true - state_values_est

        entropy = (action_probs * action_log_probs).sum(1).mean()
        action_gain = (chosen_action_log_probs * advantages).mean()
        value_loss = advantages.pow(2).mean()
        total_loss = value_loss - action_gain - 0.0001 * entropy

        return total_loss
