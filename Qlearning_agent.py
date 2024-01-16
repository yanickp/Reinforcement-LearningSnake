import torch
import random
import numpy as np
from collections import deque
from Direction import Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, tint_color
import threading
import agentParent as AgentParent
from pytorch_model_summary import summary


MAX_MEMORY = 10_000
BATCH_SIZE = 1_000
LR = 0.01 # todo optimize this parameter


class QLearningAgent(AgentParent.agent):

    def __init__(self, board_width, board_height, block_size, name="deepQ", layers=[256], targetNetwork=True,
                 inputSize=11, lr=LR):
        super().__init__(board_width, board_height, block_size, name)
        self.gamma = 0.99  # discount rate todo optimize this parameter
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.layers = layers
        self.inputSize = inputSize

        self.model = Linear_QNet(self.inputSize, layers, 3)
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma, targetNetwork=targetNetwork)
        # print('==== summary ====')
        # summary(self.model.to("cpu"), torch.tensor(self.get_state()))

    def loadBrain(self, path):
        self.loadedModel = True
        self.model.load_state_dict(torch.load(path))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0]
        # random moves: tradeoff exploration / exploitation

        if random.random() < max(0.0, 1 - self.n_games / self.epsilon) and not self.loadedModel:
            # Random exploration
            move = random.randint(0, 2)
            final_move[move] = 1
        # elif self.n_games % 2 == 0 and not self.loadedModel:
        #     # Random exploration
        #     print("Using model 30%")
        #     move = random.randint(0, 2)
        #     if move == random.randint(0, 2):
        #         state0 = torch.tensor(state, dtype=torch.float)
        #         prediction = self.model(state0)
        #         move = torch.argmax(prediction).item()
        #         final_move[move] = 1
        #     final_move[move] = 1
        else:
            # Exploitation using the model's prediction
            # print("Using model")
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def train(self, state, action, reward, next_state, done, score):
        # agent = Agent()
        # game = SnakeGameAI()
        if not self.isDead:
            # get old state
            # state_old = self.get_state(game)
            state_old = state
            # get move
            # final_move = self.get_action(state_old)
            final_move = action
            # perform move and get new state
            # reward, done, score = game.play_step(final_move)
            state_new = next_state

            if not self.loadedModel:  # if not loaded model, train
                # train short memory
                self.train_short_memory(state_old, final_move, reward, state_new, done)
                # remember
                self.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                # game.reset()
                self.n_games += 1
                self.isDead = True
                self.train_long_memory()
                # if self.n_games % 1 == 0:
                self.trainer.update_target()
                self.scores.append(score)
                self.mean_scores.append(np.mean(self.scores[-50:]))
                if score > self.record:
                    self.record = score
                    self.model.save(file_name=str(self.name) + 'H' + str(self.record) + '.pth')

                # plot_scores.append(score)
                self.total_score += score


if __name__ == '__main__':
    agent = QLearningAgent()
    agent.train()
