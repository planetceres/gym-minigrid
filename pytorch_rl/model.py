import operator
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import orthogonal

class FFPolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, masks, actions):
        value, x, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy, states

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Policy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super().__init__()

        self.action_space = action_space
        assert action_space.__class__.__name__ == "Discrete"
        num_outputs = action_space.n

        self.h = {
                    'XL': self.hidden_units(10),
                    'L': self.hidden_units(9),
                    'M': self.hidden_units(8),
                    'S': self.hidden_units(7),
                    'XS': self.hidden_units(6)
                    }

        self.fc1 = nn.Linear(num_inputs, self.h['XL'])
        self.fc2 = nn.Linear(self.h['XL'], self.h['L'])
        self.fc3 = nn.Linear(self.h['L'], self.h['M'])

        # Input size, hidden state size
        self.gru = nn.GRUCell(self.h['M'], self.h['M'])

        self.a_fc1 = nn.Linear(self.h['M'], self.h['S'])
        self.a_fc2 = nn.Linear(self.h['S'], self.h['S'])
        self.dist = Categorical(self.h['S'], num_outputs)

        self.v_fc1 = nn.Linear(self.h['M'], self.h['S'])
        self.v_fc2 = nn.Linear(self.h['S'], self.h['S'])
        self.v_fc3 = nn.Linear(self.h['S'], 1)

        self.train()
        self.reset_parameters()

    def hidden_units(self, scale):
        '''
        Hidden unit sizes scaled by 2^n
        Scale   Units       Size
            5       32          XXS
            6       64          XS
            7       128         S       *default
            8       256         M
            9       512         L
            10      1024        XL
        '''
        return int(2**scale)

    @property
    def state_size(self):
        """
        Size of the recurrent state of the model (propagated between steps)
        """
        return self.h['M']

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        orthogonal(self.gru.weight_ih.data)
        orthogonal(self.gru.weight_hh.data)
        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        batch_numel = reduce(operator.mul, inputs.size()[1:], 1)
        inputs = inputs.view(-1, batch_numel)

        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.tanh(x)

        assert inputs.size(0) == states.size(0)
        states = self.gru(x, states * masks)

        x = self.a_fc1(states)
        x = F.relu(x)
        x = self.a_fc2(x)
        actions = x

        x = self.v_fc1(states)
        x = F.relu(x)
        x = self.v_fc2(x)
        x = F.relu(x)
        x = self.v_fc3(x)
        value = x

        return value, actions, states

    def forward2(self, inputs, states, masks):
        batch_numel = reduce(operator.mul, inputs.size()[1:], 1)
        inputs = inputs.view(-1, batch_numel)

        x = self.fc1(inputs)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)

        assert inputs.size(0) == states.size(0)
        states = self.gru(x, states * masks)

        x = self.a_fc1(states)
        x = F.tanh(x)
        x = self.a_fc2(x)
        actions = x

        x = self.v_fc1(states)
        x = F.tanh(x)
        x = self.v_fc2(x)
        x = F.tanh(x)
        x = self.v_fc3(x)
        value = x

        return value, actions, states
