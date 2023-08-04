import torch
import random
import numpy as np
from collections import deque
from Direction import Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, tint_color
import threading

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self, board_width, board_height, block_size, name="deepQ", layers=[256]):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.layers = layers
        self.model = Linear_QNet(11, layers, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.name = name
        self.isDead = False
        self.TimeNotEaten = 0
        self.score = 0
        self.record = 0
        self.total_score = 0

        self.scores = []
        self.mean_scores = []

        self.loadedModel = False

        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.accent_color = tint_color(self.color, 50)  # tint by 50 for accent color

        self.board_width = board_width
        self.board_height = board_height
        self.BLOCK_SIZE = block_size

        # inittialize the snake
        self.direction = Direction.RIGHT

        self.head = Point(self.board_width / 2, self.board_height / 2)
        self.snake = [self.head,
                      Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y)]


    def loadBrain(self, path):
        self.model.load_state_dict(torch.load(path))
    def reset(self):
        self.direction = Direction.RIGHT
        self.isDead = False
        self.TimeNotEaten = 0
        self.head = Point(self.board_width / 2, self.board_height / 2)
        self.snake = [self.head,
                      Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y)]

        self.score = 0


    def get_state(self, food):
        head = self.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location 
            food.x < head.x,  # food left
            food.x > head.x,  # food right
            food.y < head.y,  # food up
            food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)

    def _move(self, action):
        # [straight, right, left]
        if not self.isDead:
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]  # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
            else:  # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

            self.direction = new_dir

            x = self.head.x
            y = self.head.y
            if self.direction == Direction.RIGHT:
                x += self.BLOCK_SIZE
            elif self.direction == Direction.LEFT:
                x -= self.BLOCK_SIZE
            elif self.direction == Direction.DOWN:
                y += self.BLOCK_SIZE
            elif self.direction == Direction.UP:
                y -= self.BLOCK_SIZE

            self.head = Point(x, y)
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.board_width - self.BLOCK_SIZE or pt.x < 0 or pt.y > self.board_height - self.BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

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
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 200 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
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

            if not self.loadedModel: # if not loaded model, train
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
                self.scores.append(score)
                self.mean_scores.append(np.mean(self.scores[-10:]))
                if score > self.record:
                    self.record = score
                    self.model.save(file_name='deepQ128128.pth')

                # plot_scores.append(score)
                self.total_score += score



if __name__ == '__main__':
    agent = Agent()
    agent.train()
