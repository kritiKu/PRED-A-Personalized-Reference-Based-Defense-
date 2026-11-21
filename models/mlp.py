import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def get_model_params(model):
    """Extract model parameters as a flattened 1D tensor or list of parameter vectors."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_model_params(model, params_tensor):
    """Set model parameters from a flattened 1D tensor."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(params_tensor[offset:offset + numel].view_as(p.data))
        offset += numel
