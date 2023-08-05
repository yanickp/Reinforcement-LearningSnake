import numpy as np

from Direction import Direction, Point
import threading
from helper import tint_color
import random

BLOCK_SIZE = 20


class agent:
    def __init__(self, board_width, board_height, block_size, name):
        self.n_games = 0  # amount of games played

        self.epsilon = 100  # randomness
        self.record = 0  # highest score
        self.total_score = 0  # overall score
        self.score = 0  # per game

        self.food = None

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

    def reset(self):
        self.direction = Direction.RIGHT
        self.isDead = False
        self.TimeNotEaten = 0
        self.head = Point(self.board_width / 2, self.board_height / 2)
        self.snake = [self.head,
                      Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y)]

        self.score = 0

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

    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        if self.food is not None:
            food_left = self.food.x < head.x  # food left
            food_right = self.food.x > head.x  # food right
            food_up = self.food.y < head.y  # food up
            food_down = self.food.y > head.y  # food down
        else:
            food_left = False
            food_right = False
            food_up = False
            food_down = False

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
            food_left,
            food_right,
            food_up,
            food_down,
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

