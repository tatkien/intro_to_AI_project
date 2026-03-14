import pygame

class Scene:
    """
    Base class for all scenes in the game
    Scenes represent different screens/states of the application
    """
    
    def __init__(self, manager):
        """
        Initialize the scene
        
        Args:
            manager: The SceneManager that controls scene transitions
        """
        self.manager = manager
    
    def update(self, event_list):
        """
        Update scene logic and handle events
        
        Args:
            event_list: List of pygame events to process
        """
        pass
    
    def draw(self):
        """
        Render the scene to the screen
        """
        pass
