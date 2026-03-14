import pygame
import sys
import os

from src.scenes.scene import Scene
from src.scenes.element import ThemedButton, Image, RuleBox
from settings import *

class MenuScene(Scene):
    """
    Main menu scene for the AI Taxi Project
    """
    
    def __init__(self, manager):
        super().__init__(manager)
        self.screen = pygame.display.get_surface()
        self.screen_width, self.screen_height = self.screen.get_size()
        
        # Load logo
        self.logo = None
        try:
            logo_url = os.path.join(IMG_URL, "logo.jpg")
            if os.path.exists(logo_url):
                logo_img = pygame.image.load(logo_url).convert()
                logo_width, logo_height = logo_img.get_size()
                # Scale logo to fit nicely
                max_width = 500
                if logo_width > max_width:
                    scale = max_width / logo_width
                    logo_width = int(logo_width * scale)
                    logo_height = int(logo_height * scale)
                logo_x = (self.screen_width - logo_width) // 2
                self.logo = Image(logo_url, logo_x, 50, size=(logo_width, logo_height))
        except Exception as e:
            print(f"Could not load logo: {e}")
        
        # Title text
        self.title_font = pygame.font.Font(None, 72)
        self.title_text = "AI Taxi Project"
        
        # Create buttons
        button_width = 300
        button_height = 60
        button_x = (self.screen_width - button_width) // 2
        start_y = 300 if self.logo is None else 350
        
        # Level selection buttons
        level_button_width = 90
        level_spacing = 10
        total_level_width = 3 * level_button_width + 2 * level_spacing
        level_start_x = (self.screen_width - total_level_width) // 2
        
        self.btn_level1 = ThemedButton(
            "Level 1",
            level_start_x, start_y,
            level_button_width, 50,
            font_size=24,
            action=lambda: self.start_game(1)
        )
        
        self.btn_level2 = ThemedButton(
            "Level 2",
            level_start_x + level_button_width + level_spacing, start_y,
            level_button_width, 50,
            font_size=24,
            action=lambda: self.start_game(2)
        )
        
        self.btn_level3 = ThemedButton(
            "Level 3",
            level_start_x + 2 * (level_button_width + level_spacing), start_y,
            level_button_width, 50,
            font_size=24,
            action=lambda: self.start_game(3)
        )
        
        self.btn_exit = ThemedButton(
            "Exit",
            button_x, start_y + 100,
            button_width, button_height,
            font_size=36,
            action=self.exit_game
        )
        
        # Info box
        info_text = [
            "Welcome to Taxi Driver!",
            "",
            "Select a Level:",
            "Level 1: 10x10 grid, 1 task",
            "Level 2: 15x15 grid, 2 continuous tasks",
            "Level 3: 20x20 grid, 3 continuous tasks",
            "",
            "Obstacles:",
            "- Walls",
            "- Mud (cost x3)",
            "",
            "Features:",
            "- BFS, DFS, and A* algorithms",
            "- Interactive visualization",
            "",
            "Controls:",
            "1. Select algorithm",
            "2. Click 'Run'",
            "3. 'Pause' to stop",
            "",
            "Arrow Keys: Move",
            "Space: Pick/Drop"
        ]
        
        info_width = 900
        info_x = (self.screen_width - info_width) // 2
        info_y = start_y + 180
        
        self.info_box = RuleBox(info_x, info_y, info_width, 450, info_text)
        self.info_text = info_text
        self.info_x = info_x
        self.info_y = info_y
        self.info_width = info_width
        self.info_height = 450
        
        # Scrolling for info box
        self.scroll_offset = 0
        self.max_scroll = 0
        
        self.buttons = [self.btn_level1, self.btn_level2, self.btn_level3, self.btn_exit]
    
    def start_game(self, level=1):
        """Start the taxi game scene with selected level"""
        self.manager.switch_scene("taxi", level=level)
    
    def exit_game(self):
        """Exit the application"""
        pygame.quit()
        sys.exit()
    
    def update(self, event_list):
        """Handle menu events"""
        for event in event_list:
            for button in self.buttons:
                button.check_click(event)
            
            # Handle mouse wheel for scrolling info box
            if event.type == pygame.MOUSEWHEEL:
                self.scroll_offset -= event.y * 20
                self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))
    
    def draw(self):
        """Render the menu"""
        # Background
        self.screen.fill((20, 20, 30))
        
        # Logo or title
        if self.logo:
            self.logo.draw(self.screen)
        else:
            title_surf = self.title_font.render(self.title_text, True, (255, 255, 255))
            title_rect = title_surf.get_rect(center=(self.screen_width // 2, 100))
            self.screen.blit(title_surf, title_rect)
        
        # Buttons
        for button in self.buttons:
            button.draw(self.screen)
        
        # Info box with scrolling
        self.draw_info_box()
    
    def draw_info_box(self):
        """Draw info box with scrolling support"""
        # Create background
        box_rect = pygame.Rect(self.info_x, self.info_y, self.info_width, self.info_height)
        pygame.draw.rect(self.screen, (40, 40, 50), box_rect, border_radius=10)
        pygame.draw.rect(self.screen, (100, 150, 200), box_rect, width=2, border_radius=10)
        
        # Create surface for scrollable content
        content_surface = pygame.Surface((self.info_width - 40, 2000), pygame.SRCALPHA)
        content_surface.fill((0, 0, 0, 0))
        
        # Render text lines
        font = pygame.font.Font(None, 24)
        y_pos = 10
        line_height = 25
        
        for line in self.info_text:
            if line.strip() == "":
                y_pos += line_height // 2
            else:
                color = (255, 255, 255)
                if line.endswith(":"):
                    color = (100, 200, 255)
                text_surf = font.render(line, True, color)
                content_surface.blit(text_surf, (20, y_pos))
                y_pos += line_height
        
        # Calculate max scroll
        total_content_height = y_pos + 10
        self.max_scroll = max(0, total_content_height - self.info_height)
        
        # Blit scrollable content
        content_rect = pygame.Rect(0, self.scroll_offset, self.info_width - 40, self.info_height - 20)
        self.screen.blit(content_surface, (self.info_x + 20, self.info_y + 10), content_rect)
        
        # Draw scrollbar if needed
        if self.max_scroll > 0:
            scrollbar_x = self.info_x + self.info_width - 15
            scrollbar_height = self.info_height - 20
            scrollbar_y = self.info_y + 10
            
            # Background track
            pygame.draw.rect(self.screen, (60, 60, 70), (scrollbar_x, scrollbar_y, 10, scrollbar_height), border_radius=5)
            
            # Thumb
            thumb_height = max(20, int(scrollbar_height * (self.info_height / total_content_height)))
            thumb_y = scrollbar_y + int((self.scroll_offset / self.max_scroll) * (scrollbar_height - thumb_height))
            pygame.draw.rect(self.screen, (100, 150, 200), (scrollbar_x, thumb_y, 10, thumb_height), border_radius=5)