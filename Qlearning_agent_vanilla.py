import random
import numpy as np
# from collections import deque
# from game import SnakeGameAI, Direction, Point
# from model import Linear_QNet, QTrainer
# from helper import plot
import threading

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent_valilla:

    def __init__(self):
        self.n_games = 0
        self.q_table = {}
        self.epsilon = 0 # randomness
        self.record = 0
        self.total_score = 0

    def _get_q_value(self, state, action):
        # Get the Q-value for a specific state-action pair from the Q-table
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]  # Initialize Q-values for each action in the state
        action_index = np.argmax(action)
        return self.q_table[state][action_index]

    def _update_q_value(self, state, action, new_q_value):
        # Update the Q-value for a specific state-action pair in the Q-table
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]

        action_index = np.argmax(action)
        self.q_table[state][action_index] = new_q_value


    # def get_state(self, game):
    #     head = game.snake[0]
    #     point_l = Point(head.x - 20, head.y)
    #     point_r = Point(head.x + 20, head.y)
    #     point_u = Point(head.x, head.y - 20)
    #     point_d = Point(head.x, head.y + 20)
    #
    #     dir_l = game.direction == Direction.LEFT
    #     dir_r = game.direction == Direction.RIGHT
    #     dir_u = game.direction == Direction.UP
    #     dir_d = game.direction == Direction.DOWN
    #
    #     state = [
    #         # Danger straight
    #         (dir_r and game.is_collision(point_r)) or
    #         (dir_l and game.is_collision(point_l)) or
    #         (dir_u and game.is_collision(point_u)) or
    #         (dir_d and game.is_collision(point_d)),
    #
    #         # Danger right
    #         (dir_u and game.is_collision(point_r)) or
    #         (dir_d and game.is_collision(point_l)) or
    #         (dir_l and game.is_collision(point_u)) or
    #         (dir_r and game.is_collision(point_d)),
    #
    #         # Danger left
    #         (dir_d and game.is_collision(point_r)) or
    #         (dir_u and game.is_collision(point_l)) or
    #         (dir_r and game.is_collision(point_u)) or
    #         (dir_l and game.is_collision(point_d)),
    #
    #         # Move direction
    #         dir_l,
    #         dir_r,
    #         dir_u,
    #         dir_d,
    #
    #         # Food location
    #         game.food.x < game.head.x,  # food left
    #         game.food.x > game.head.x,  # food right
    #         game.food.y < game.head.y,  # food up
    #         game.food.y > game.head.y  # food down
    #         ]
    #
    #     return np.array(state, dtype=int)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state = tuple(state)
            move = np.argmax(self.q_table.get(state, [0, 0, 0]))  # Choose the action with the highest Q-value
        action_vector = [0, 0, 0]
        action_vector[move] = 1
        return action_vector


    def train(self, state, action, reward, next_state, done, score):
        plot_scores = []
        plot_mean_scores = []
        record = 0
        # agent = Agent()
        # game = SnakeGameAI()
        # while True:
            # Q-learning update rule
            # state = self.get_state(game)
        state = tuple(state)
        #
        # action = self.get_action(state)
        # reward, done, score = game.play_step(action)
        # next_state = self.get_state(game)

        next_state = tuple(next_state)

        current_q = self._get_q_value(state, action)
        max_next_q = max(self.q_table.get(next_state, [0, 0, 0])) # [0,0,0] is the default value if next_state is not in the Q-table
        new_q = current_q + 0.1 * (
                    reward + 0.99 * max_next_q - current_q)  # Learning rate = 0.1, discount factor = 0.99
        self._update_q_value(state, action, new_q)


        if done:
            if score > self.record:
                self.record = score

            # game.reset()
            self.n_games += 1
            if self.n_games > 10:
                print('')
            self.total_score += score
            mean_score = self.total_score / self.n_games
            print('Game', self.n_games, 'Score', score, 'Record:', self.record, 'Mean Score:', mean_score)


if __name__ == '__main__':
    agent = Agent_valilla()
    agent.train()
