import pickle
import random
import numpy as np
from Direction import Direction, Point
import threading
from helper import tint_color

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent_valilla:

    def __init__(self, board_width, board_height, block_size, name="vanilla"):
        self.n_games = 0  # amount of games played
        self.q_table = {}
        self.epsilon = 0  # randomness
        self.record = 0  # highest score
        self.total_score = 0  # overall score
        self.score = 0  # per game

        self.name = name
        self.isDead = False
        self.TimeNotEaten = 0

        self.loadedModel = False

        self.scores = []
        self.mean_scores = []

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
            food.x < self.head.x,  # food left
            food.x > self.head.x,  # food right
            food.y < self.head.y,  # food up
            food.y > self.head.y  # food down
        ]

        return np.array(state, dtype=int)

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

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_games
        change = random.randint(0, 100)
        if change < self.epsilon and not self.loadedModel: #or change < 5:
            move = random.randint(0, 2)
        else:
            state = tuple(state)
            move = np.argmax(self.q_table.get(state, [0, 0, 0]))  # Choose the action with the highest Q-value
        action_vector = [0, 0, 0]
        action_vector[move] = 1
        return action_vector

    def loadBrain(self, filename):
        with open(filename, 'rb') as brain:
            self.q_table = pickle.load(brain)
        self.loadedModel = True
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

    def train(self, state, action, reward, next_state, done, score):
        if not self.isDead:  # If the snake is dead, there is no need to train the Q-table
            state = tuple(state)
            next_state = tuple(next_state)

            current_q = self._get_q_value(state, action)
            max_next_q = max(
                self.q_table.get(next_state,
                                 [0, 0, 0]))  # [0,0,0] is the default value if next_state is not in the Q-table
            new_q = current_q + 0.1 * (
                    reward + 0.99 * max_next_q - current_q)  # Learning rate = 0.1, discount factor = 0.99
            self._update_q_value(state, action, new_q)

        if done:
            if score > self.record:
                self.record = score
                with open(f'model/{self.name}.pkl', 'wb') as fp:
                    pickle.dump(self.q_table, fp)

            # game.reset()
            self.isDead = True
            self.n_games += 1
            self.total_score += score
            mean_score = self.total_score / self.n_games
            self.scores.append(score)
            self.mean_scores.append(np.mean(self.scores[-50:]))
            # print(str(self.name) + 's Game', self.n_games, 'Score', score, 'Record:', self.record, 'Mean Score:', mean_score)


if __name__ == '__main__':
    agent = Agent_valilla()
    agent.train()
