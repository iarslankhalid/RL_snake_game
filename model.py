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

    def load_model(self, weights_path="./model/model.pth"):
        checkpoint = torch.load(weights_path)
        self.load_state_dict(checkpoint['state_dict'])
        

class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        
        if len(state.shape) == 1:
            # reshape --> (1, x)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)        
            done = (done, )
            
        # 1: predicted Q values with current state
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state))
                
            target[idx][torch.argmax(action).item()] = Q_new
            
        # 2: Q_new = Reward + gamma *  max(next predicted Q value)
        #pred.clone()
        #preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()
    