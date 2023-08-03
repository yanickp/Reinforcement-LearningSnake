import pygame
import random
from collections import namedtuple
import numpy as np
from Qlearning_agent_vanilla import Agent_valilla
from Direction import Direction, Point


pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)





# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=1200, h=680):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.timesReset = 0

        self.agents = []

        self.reset()


    def reset(self):
        # init game state
        # self.direction = Direction.RIGHT
        #
        # self.head = Point(self.w/2, self.h/2)
        # self.snake = [self.head,
        #               Point(self.head.x-BLOCK_SIZE, self.head.y),
        #               Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        # self.score = 0
        for agent in self.agents:
            agent.reset()

        self.food = None
        self.timesReset += 1
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        centerX = self.w // 2
        centerY = self.h // 2

        if self.timesReset < 150:
            # Calculate the offset based on the number of times reset
            offset_blocks = self.timesReset // 15

            # Limit the offset to 3 blocks initially and increase it by 1 block every 20 times reset
            offset_blocks = max(offset_blocks, 3)

            # Calculate the range for spawning the food
            min_x = centerX - (3 + offset_blocks) * BLOCK_SIZE
            max_x = centerX + (3 + offset_blocks) * BLOCK_SIZE
            min_y = centerY - (3 + offset_blocks) * BLOCK_SIZE
            max_y = centerY + (3 + offset_blocks) * BLOCK_SIZE

            # Have food spawn in a random location within the calculated range
            x = random.randint(min_x // BLOCK_SIZE, max_x // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(min_y // BLOCK_SIZE, max_y // BLOCK_SIZE) * BLOCK_SIZE
        else:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        # print("food at: ", x, y)

        self.food = Point(x, y)

        # if food is in any of the snakes
        for agent in self.agents:
            if self.food in agent.snake:
                self._place_food()
        # if self.food in self.snake:
        #     self._place_food()


    def play_step(self, action, agent):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        agent._move(action) # update the head
        agent.snake.insert(0, agent.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if agent.is_collision() or self.frame_iteration > 100*len(agent.snake):
            game_over = True
            reward = -10
            return reward, game_over, agent.score

        # 4. place new food or just move
        if agent.head == self.food:
            agent.score += 1
            reward = 10
            self._place_food()
        else:
            # reward = -(self._distance_to_food() / 10)
            agent.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, agent.score






    def _update_ui(self):
        self.display.fill(BLACK)

        for index, agent in enumerate(self.agents):
            for pt in agent.snake:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

            text = font.render(str(agent.name) + "'s score: " + str(agent.score), True, WHITE)
            self.display.blit(text, [0, 20*index])

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # pygame.draw.line(self.display, RED, (self.head.x, self.head.y), (self.food.x, self.food.y), 2)
        # draw rectangle for posible food locations

        pygame.display.flip()




    # def getState(self):
    #     head = self.snake[0]
    #     point_l = Point(head.x - 20, head.y)
    #     point_r = Point(head.x + 20, head.y)
    #     point_u = Point(head.x, head.y - 20)
    #     point_d = Point(head.x, head.y + 20)
    #
    #     dir_l = self.direction == Direction.LEFT
    #     dir_r = self.direction == Direction.RIGHT
    #     dir_u = self.direction == Direction.UP
    #     dir_d = self.direction == Direction.DOWN
    #
    #     state = [
    #         # Danger straight
    #         (dir_r and self.is_collision(point_r)) or
    #         (dir_l and self.is_collision(point_l)) or
    #         (dir_u and self.is_collision(point_u)) or
    #         (dir_d and self.is_collision(point_d)),
    #
    #         # Danger right
    #         (dir_u and self.is_collision(point_r)) or
    #         (dir_d and self.is_collision(point_l)) or
    #         (dir_l and self.is_collision(point_u)) or
    #         (dir_r and self.is_collision(point_d)),
    #
    #         # Danger left
    #         (dir_d and self.is_collision(point_r)) or
    #         (dir_u and self.is_collision(point_l)) or
    #         (dir_r and self.is_collision(point_u)) or
    #         (dir_l and self.is_collision(point_d)),
    #
    #         # Move direction
    #         dir_l,
    #         dir_r,
    #         dir_u,
    #         dir_d,
    #
    #         # Food location
    #         self.food.x < self.head.x,  # food left
    #         self.food.x > self.head.x,  # food right
    #         self.food.y < self.head.y,  # food up
    #         self.food.y > self.head.y  # food down
    #     ]
    #
    #     return np.array(state, dtype=int)

if __name__ == '__main__':
    game = SnakeGameAI()
    game.agents = [Agent_valilla(game.w, game.h, BLOCK_SIZE), Agent_valilla(game.w, game.h, BLOCK_SIZE)]
    while True:
        for agent in game.agents:
            state = agent.get_state(game.food)
            action = agent.get_action(state)
            reward, game_over, score = game.play_step(action, agent)
            nextState = agent.get_state(game.food)
            agent.train(state, action, reward, nextState, game_over, score)
            if game_over:
                game.reset()
            game._update_ui()

# todo 1: make 2 snakes play against each other
# todo 2: make 2 snakes play against each other with different brains
