import pygame
import os
import ctypes
import sys

from settings import *
from src.scene_manager import SceneManager

# --- PREVENT STRETCHING/BLURRING (Windows Fix) ---
# This tells the OS to disable auto-scaling so the game runs at true 1:1 resolution.
try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass # Ignore if on Mac/Linux

def main():
    """
    Main game loop for the Taxi Driver Project
    Initializes Pygame, sets up the display, and manages scene transitions
    """
    # Initialize Pygame
    pygame.init()
    
    # Set up display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Taxi Driver Project")
    
    # Set up clock for frame rate control
    clock = pygame.time.Clock()
    
    # Initialize scene manager
    scene_manager = SceneManager()
    
    # Main game loop
    running = True
    while running:
        # Handle events
        event_list = pygame.event.get()
        for event in event_list:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        # Update current scene
        scene_manager.run(event_list)
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()