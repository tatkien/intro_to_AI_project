"""
Global Settings and Constants
"""
import pygame

# Screen settings
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1000
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)

# Paths
IMG_URL = "images/"
DATA_URL = "data/"

# Game settings
GRID_SIZE = 10
CELL_SIZE = 60

# Level configurations
LEVEL_CONFIG = {
    1: {
        'grid_size': 10,
        'cell_size': 85,
        'num_locations': 4,
        'num_tasks': 1,  # 1 passenger to deliver
        'name': 'Level 1 - Beginner'
    },
    2: {
        'grid_size': 15,
        'cell_size': 56,
        'num_locations': 6,
        'num_tasks': 2,  # 2 passengers to deliver
        'name': 'Level 2 - Intermediate'
    },
    3: {
        'grid_size': 20,
        'cell_size': 43,
        'num_locations': 8,
        'num_tasks': 3,  # 3 passengers to deliver
        'name': 'Level 3 - Advanced'
    }
}
