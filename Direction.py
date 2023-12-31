from collections import namedtuple
from enum import Enum


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Point = namedtuple('Point', 'x, y')
class Point(object):
    __slots__ = ('x', 'y')
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class Slope(object):
    __slots__ = ('vertical', 'horizontal')
    def __init__(self, vertical: int, horizontal: int):
        self.vertical = vertical
        self.horizontal = horizontal
