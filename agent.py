from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer, LOAD_PREVIOUS_MODEL
#
from collections import deque

import torch
import random
import numpy as np


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent():
    def __init__(self) -> None:
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        '''
 state[11] =   [danger straight, danger right, danger left,
               direction left, direction right, direction up, direction down,
               food left, food right, food up, food down]
        '''
        
        state = [
            # Danger straigt
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            
            # Danger Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            # Danger Left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            # Move directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y]  # food down 
        
        return np.array(state, dtype=int)
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    
    def get_action(self, state):
        # random moves ---> trade off between exploration and exploitation
        self.epsilon = 80 - self.n_game
        action = [0, 0, 0]
        
        if random.uniform(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  
            move = torch.argmax(prediction).item()
            action[move] = 1
            
        return action
    
    
    def train(self):
        plot_scores = []
        plot_mean_score = []
        total_score = 0
        record = 0
        game = SnakeGameAI()

        while True:
            # get old state
            old_state = self.get_state(game)

            # get action
            action = self.get_action(old_state)

            # perform move and get new move
            reward, done, score = game.play_step(action)
            new_state = self.get_state(game)

            # train short memory
            self.train_short_memory(old_state, action, reward, new_state, done)

            # remember
            self.remember(old_state, action, reward, new_state, done)

            if done:
                # train long memory
                game.reset()
                self.n_game += 1
                self.train_long_memory()

                if score > record:
                    record = score
                    self.model.save()

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / self.n_game
                plot_mean_score.append(mean_score)
                result = f"Game # {self.n_game}, Score: {score}, Average Score: {mean_score:.2f} --- Model's Best: {record}"

                filepath = './model/results.txt'
                with open(filepath, 'a') as file:
                    file.write(result)
                    file.write('\n')

                print(result)

                          
if __name__ == '__main__':        
    agent = Agent()
    
    if LOAD_PREVIOUS_MODEL:
        print("Loading Previously Trained Model...")
        agent.model.load_model()    # Default --> './model/model.pth'
        print("Model loaded sucessfully.")
    agent.train()
    