import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1) # Apply relu activation in forward()
        self.layer2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1) # Apply relu activation in forward()
        self.layer3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.layer4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1) # Apply relu activation in forward()
        self.layer5 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1) # Apply relu activation in forward()
        self.layer6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.layer7 = nn.Linear(401408, 4096) # Apply relu activation in forward()
        self.layer8 = nn.Linear(4096, 4096) # Apply relu activation in forward()
        self.layer9 = nn.Linear(4096, 4096) # Apply relu activation in forward()
        self.layer10 = nn.Linear(4096, 1000) # Apply log_softmax in forward()

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer6(x)
        x = torch.flatten(x, 1)
        x = self.layer7(x)
        x = F.relu(x)
        x = self.layer8(x)
        x = F.relu(x)
        x = self.layer9(x)
        x = F.relu(x)
        x = self.layer10(x)
        x = F.log_softmax(x, dim=1)
        return x

# Create model instance
model = NeuralNetwork()

# Print model summary
print(model)
