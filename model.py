import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=48, fc3_units=24):
        """
        Instantiates the network's architecture and its parameters.
        
        Params
        =======
        state_size (int): Dimension of each state
        action_size (int): Number of possible actions
        seed(int): Random seed
        fc1_unites (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Defines the size of the layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        
        
    def forward(self, state):
        """
        Builds a network that maps states to action_values.
        
        Params
        ========
        state (1-dimensional Tensor of size 37): 
        
        Returns
        ========
        action_values (1-dimensional Tensor of size 4): action_values associated to the state
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
#         x = self.fc3(x)
        
        return x
