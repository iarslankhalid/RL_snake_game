import torch
import torch.nn as nn
import torch.optim as optim
import os

LOAD_PREVIOUS_MODEL = True

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="./model/model.pth"):
        os.makedirs('./model', exist_ok=True)
        model_params = {"state_dict": self.state_dict()}
        torch.save(model_params, file_name)

    def load(self, file_name="./model/model.pth"):
        model_params = torch.load(file_name, map_location=torch.device('cpu'))
        self.load_state_dict(model_params['state_dict'])

model = Linear_QNet(11, 256, 3)
# Load the weights (replace 'model.pth' with the actual path to your saved weights file)
model.load_state_dict(torch.load('model/model.pth', map_location=torch.device('cpu')))

# Print and compare the model weights
for name, param in model.named_parameters():
    file_weights = torch.load('model/model.pth', map_location=torch.device('cpu'))['state_dict'][name]
    
    if torch.all(torch.eq(param, file_weights)):
        print(f'{name} weights match.')
    else:
        print(f'{name} weights do not match.')
