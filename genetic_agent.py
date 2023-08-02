import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

class GeneticAgents:
    def __init__(self):
        self.populationCount = 100
        self.n_games = 0
        self.epsilon = 0 # randomness

        self.mutationRate = 0.1

        # mutatable parameters
        self.hidden_layers = [256] # 1 hidden layer with 256 neurons, mutatable



        # todo make class diagram


