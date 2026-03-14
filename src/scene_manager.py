import pygame
from src.scenes.menu import MenuScene
from src.scenes.taxi import TaxiScene

SCENES = {
    "menu": MenuScene,
    "taxi": TaxiScene
}


class SceneManager:
    """Manages scene transitions and game flow"""
    
    def __init__(self):
        self.current_scene = MenuScene(self)

    def switch_scene(self, scene_name, *args, **kwargs):
        """Switch to a different scene"""
        if scene_name in SCENES:
            self.current_scene = SCENES[scene_name](self, *args, **kwargs)
            print(f"Switched to {scene_name} scene")
        else:
            print(f"Scene '{scene_name}' not found!")

    def run(self, event_list):
        """Run the current scene's update and draw methods"""
        if self.current_scene:
            self.current_scene.update(event_list)
            self.current_scene.draw()
