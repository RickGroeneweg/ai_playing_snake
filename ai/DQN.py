import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """neural network used by the DQN agent"""

    def __init__(self, h, w, outputs):
        #TODO: incorporate h and w so that linear_input_size is not hard coded!!!!
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(10, 12, kernel_size=4, stride=1)
        
        linear_input_size = 2*2*12 +1 #+1 for hunger
        self.l1 = nn.Linear(linear_input_size, 50)
        self.l2 = nn.Linear(50, 40)
        
        self.head = nn.Linear(40, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, 
                x: torch.Tensor, 
                hunger: torch.Tensor #1x1
                ):
        
        
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_vector = x.view(x.size(0), -1)
 
        
        x = torch.cat((x_vector, hunger), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.head(x)