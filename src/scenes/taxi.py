import pygame
import sys
import os
import time
import random
import numpy as np

from src.scenes.scene import Scene
from src.scenes.element import ThemedButton, RuleBox
from settings import IMG_URL
from config import Tile, TILE_PROPS
from src.search.algorithms import BreadthFirstSearch, DepthFirstSearch, AStarSearch

# Constants
PANEL_WIDTH = 450


class TaxiEnv:
    """
    Self-contained Taxi environment (no gymnasium dependency).
    State format: [taxi_row, taxi_col, passenger_states..., current_passenger]
      passenger_state: 0=waiting, 1=in taxi, 2=delivered
      current_passenger: -1 if empty, else index of passenger in taxi
    Actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff
    """

    def __init__(self, level=1):
        self.level = level
        self._configure_level(level)

        self.window_size = self.grid_size * self.cell_size
        self.last_action = 0

        # Multi-passenger system
        self.num_passengers = self.num_tasks
        self.passenger_locations = []
        self.destination_locations = []
        self.passenger_states = []
        self.current_passenger = -1

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self._generate_terrain()
        self._generate_locations()

        self.state = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure_level(self, level):
        if level == 1:
            self.grid_size = 10
            self.cell_size = 60
            self.num_locations = 4
            self.num_tasks = 1
        elif level == 2:
            self.grid_size = 15
            self.cell_size = 45
            self.num_locations = 6
            self.num_tasks = 2
        elif level == 3:
            self.grid_size = 20
            self.cell_size = 35
            self.num_locations = 10
            self.num_tasks = 3
        else:
            self.grid_size = 10
            self.cell_size = 60
            self.num_locations = 4
            self.num_tasks = 1

    # ------------------------------------------------------------------
    # Terrain & location generation
    # ------------------------------------------------------------------

    def _generate_terrain(self):
        self.grid.fill(Tile.EMPTY)

        if self.level == 1:
            walls = [
                (0, 6), (1, 8),
                (2, 2), (2, 3), (2, 4),
                (3, 9), (4, 9), (5, 9),
                (5, 5), (6, 5), (7, 5),
                (8, 2), (8, 3),
            ]
            muds = [
                (7, 7), (7, 8),
                (4, 3), (4, 4),
                (6, 8), (6, 9),
            ]
        elif self.level == 2:
            walls = [
                (1, 5), (2, 5), (3, 5),
                (5, 2), (5, 3), (5, 4),
                (7, 7), (7, 8), (7, 9), (7, 10),
                (9, 12), (10, 12), (11, 12),
                (3, 11), (4, 11),
                (12, 3), (12, 4), (12, 5),
            ]
            muds = [
                (2, 7), (2, 8), 
                (3, 2), (3, 7), (3, 8), 
                (4, 1), (4, 7), (4, 8),
                (14, 8), (13, 8), (14, 9),
                (6, 5), (6, 6), (6, 7),
                (9, 5), (9, 6), (10, 5), (10, 6),
                (2, 13), (2, 14), (3, 13),
            ]
        else:
            walls = [
                (2, 5), (3, 5), (4, 5), (5, 5),
                (7, 2), (7, 3), (7, 4),
                (10, 10), (10, 11), (10, 12), (10, 13),
                (12, 7), (13, 7), (14, 7),
                (5, 15), (6, 15), (7, 15),
                (15, 5), (15, 6), (15, 7), (15, 8),
                (17, 12), (17, 13), (17, 14),
            ]
            muds = [
                (1, 3), (1, 4), (1, 5),
                (2, 3),
                (3, 1), (3, 2), (3, 3), (3, 8), (3, 9), (3, 10),
                (4, 8), (4, 9), (4, 10),
                (5, 8), (5, 9), (5, 10),
                (3, 18), (4, 18), (5, 18),
                (18, 3), (18, 4), (18, 5),
                (7, 17), (7, 19),
                (8, 8), (8, 9), (8, 19),
                (9, 8), (9, 9),
                (11, 15), (11, 16),
                (12, 0), (12, 1), (12, 3), (12, 4), (12, 15), (12, 16),
                (16, 10), (16, 11), (17, 10), (17, 11),
            ]

        self.walls = walls
        self.muds = muds
        for row, col in walls:
            self.grid[row, col] = Tile.WALL
        for row, col in muds:
            self.grid[row, col] = Tile.MUD

    def _generate_locations(self):
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
            (255, 64, 128),   # pink
            (64, 255, 128),   # mint
        ]

        # Level 3 uses explicit positions (10 locs across a 20x20 grid)
        if self.level == 3:
            positions = [
                (1, 1),
                (1, 10),
                (1, 18),
                (6, 1),
                (6, 18),
                (10, 1),
                (10, 18),
                (16, 1),
                (16, 18),
                (18, 10),
            ]
        else:
            spacing = self.grid_size // 3
            positions = []

            for i in range(self.num_locations):
                row = (i // 2) * spacing + 1
                col = (i % 2) * (self.grid_size - 2) + 1
                row = min(max(row, 1), self.grid_size - 2)
                col = min(max(col, 1), self.grid_size - 2)

                attempts = 0
                while self.grid[row, col] == Tile.WALL and attempts < 100:
                    offset = (attempts // 4) + 1
                    direction = attempts % 4
                    if direction == 0:
                        col = min(col + offset, self.grid_size - 2)
                    elif direction == 1:
                        row = min(row + offset, self.grid_size - 2)
                    elif direction == 2:
                        col = max(col - offset, 1)
                    else:
                        row = max(row - offset, 1)
                    attempts += 1

                positions.append((row, col))

        self.locs = {i: positions[i] for i in range(self.num_locations)}
        self.loc_colors = {i: colors[i] for i in range(self.num_locations)}

    # ------------------------------------------------------------------
    # Search interface
    # ------------------------------------------------------------------

    def is_goal_state(self, state):
        passenger_states = state[2:2 + self.num_passengers]
        return all(ps == 2 for ps in passenger_states)

    def get_successors(self, state):
        # Save env state
        orig_state = self.state
        orig_pass_locs = self.passenger_locations[:]
        orig_dest_locs = self.destination_locations[:]
        orig_pass_states = self.passenger_states[:]
        orig_current = self.current_passenger

        # Apply given state
        self.state = list(state)
        self.passenger_states = list(state[2:2 + self.num_passengers])
        self.current_passenger = state[-1]

        successors = []
        for action in range(6):
            next_state, reward, _, _, _ = self._simulate_step(action)
            cost = -reward
            successors.append((action, tuple(next_state), cost))

        # Restore env state
        self.state = orig_state
        self.passenger_locations = orig_pass_locs
        self.destination_locations = orig_dest_locs
        self.passenger_states = orig_pass_states
        self.current_passenger = orig_current

        return successors

    def _simulate_step(self, action):
        row, col = self.state[0], self.state[1]
        passenger_states = list(self.passenger_states)
        current_passenger = self.current_passenger
        reward = -1
        terminated = False

        if action < 4:
            new_row, new_col = row, col
            if action == 0:
                new_row = min(row + 1, self.grid_size - 1)
            elif action == 1:
                new_row = max(row - 1, 0)
            elif action == 2:
                new_col = min(col + 1, self.grid_size - 1)
            elif action == 3:
                new_col = max(col - 1, 0)

            tile_props = TILE_PROPS[self.grid[new_row, new_col]]
            if tile_props['walkable']:
                row, col = new_row, new_col
                reward = -tile_props['cost']
            else:
                reward = -1

        elif action == 4:  # Pickup
            if current_passenger == -1:
                picked_up = False
                for i in range(self.num_passengers):
                    if passenger_states[i] == 0:
                        pass_loc = self.locs[self.passenger_locations[i]]
                        if (row, col) == pass_loc:
                            passenger_states[i] = 1
                            current_passenger = i
                            reward = -1
                            picked_up = True
                            break
                if not picked_up:
                    reward = -10
            else:
                reward = -10

        elif action == 5:  # Dropoff
            if current_passenger != -1:
                dest_loc = self.locs[self.destination_locations[current_passenger]]
                if (row, col) == dest_loc:
                    passenger_states[current_passenger] = 2
                    current_passenger = -1
                    reward = -1
                    if all(ps == 2 for ps in passenger_states):
                        terminated = True
                else:
                    reward = -10
            else:
                reward = -10

        next_state = [row, col] + passenger_states + [current_passenger]
        return next_state, reward, terminated, False, {}

    def heuristic(self, state):
        taxi_row, taxi_col = state[0], state[1]
        passenger_states = state[2:2 + self.num_passengers]
        current_passenger = state[-1]

        total_dist = 0

        if current_passenger != -1:
            dest_loc = self.locs[self.destination_locations[current_passenger]]
            total_dist += abs(taxi_row - dest_loc[0]) + abs(taxi_col - dest_loc[1]) + 1

        # Waiting passengers: must travel pickup->destination + 2 actions (pickup + dropoff)
        waiting_pickups = []
        for i in range(self.num_passengers):
            if passenger_states[i] == 0:
                pass_loc = self.locs[self.passenger_locations[i]]
                dest_loc = self.locs[self.destination_locations[i]]
                total_dist += (
                    abs(pass_loc[0] - dest_loc[0]) + abs(pass_loc[1] - dest_loc[1]) + 2
                )
                waiting_pickups.append(pass_loc)

        # If taxi is empty and there are waiting passengers,
        # add the minimum distance to the nearest pickup location
        if current_passenger == -1 and waiting_pickups:
            min_to_pickup = min(
                abs(taxi_row - pr) + abs(taxi_col - pc)
                for pr, pc in waiting_pickups
            )
            total_dist += min_to_pickup

        return total_dist

    # ------------------------------------------------------------------
    # Environment control
    # ------------------------------------------------------------------

    def reset(self):
        self.passenger_locations = []
        self.destination_locations = []
        self.passenger_states = []

        used_pickups = set()
        for _ in range(self.num_passengers):
            pass_idx = random.randint(0, self.num_locations - 1)
            while pass_idx in used_pickups:
                pass_idx = random.randint(0, self.num_locations - 1)
            used_pickups.add(pass_idx)

            dest_idx = random.randint(0, self.num_locations - 1)
            while dest_idx == pass_idx:
                dest_idx = random.randint(0, self.num_locations - 1)

            self.passenger_locations.append(pass_idx)
            self.destination_locations.append(dest_idx)
            self.passenger_states.append(0)

        self.current_passenger = -1

        while True:
            taxi_row = random.randint(0, self.grid_size - 1)
            taxi_col = random.randint(0, self.grid_size - 1)
            if self.grid[taxi_row, taxi_col] != Tile.WALL:
                break

        self.state = [taxi_row, taxi_col] + self.passenger_states + [self.current_passenger]
        return self.state, {}

    def step(self, action):
        next_state, reward, terminated, truncated, info = self._simulate_step(action)
        self.state = next_state
        self.passenger_states = list(next_state[2:2 + self.num_passengers])
        self.current_passenger = next_state[-1]
        if action < 4:
            self.last_action = action

        if terminated:
            print(f"All {self.num_passengers} passengers delivered!")

        return self.state, reward, terminated, truncated, info

class TaxiScene(Scene):
    """
    Interactive Taxi Scene with AI Search Algorithms
    Allows users to run BFS, DFS, and A* on the taxi environment
    """
    
    def __init__(self, manager, level=1):
        super().__init__(manager)
        self.screen = pygame.display.get_surface()
        self.screen_width, self.screen_height = self.screen.get_size()
        self.level = level
        
        # Initialize environment with level
        self.env = TaxiEnv(level=level)
        # Scale cell size to fit available screen space so grid expands with window
        padding = 40
        avail_w = max(0, self.screen_width - PANEL_WIDTH - padding * 2)
        avail_h = max(0, self.screen_height - padding * 2)
        # compute max cell size that fits horizontally and vertically
        max_cell_w = max(20, avail_w // self.env.grid_size) if self.env.grid_size > 0 else 60
        max_cell_h = max(20, avail_h // self.env.grid_size) if self.env.grid_size > 0 else 60
        # choose the smaller to ensure it fits both dimensions
        self.env.cell_size = min(max_cell_w, max_cell_h)
        # update dependent window size
        self.env.window_size = self.env.grid_size * self.env.cell_size
        self.reset_environment()
        
        # State management
        self.is_running = False
        self.is_paused = False
        self.solution_path = []
        self.current_step = 0
        self.step_delay = 300  # milliseconds between steps
        self.last_step_time = 0
        self.selected_algorithm = None
        self.algorithm_name = "None"
        
        # Statistics
        self.nodes_expanded = 0
        self.path_length = 0
        self.path_cost = 0
        self.solution_time = 0
        self.max_frontier_size = 0
        
        # Scrolling for statistics panel
        self.scroll_offset = 0
        self.max_scroll = 0
        
        # Load textures
        self.load_textures()
        
        # Create UI elements
        self.create_ui_elements()
        
    def reset_environment(self):
        """Reset the taxi environment to a new state"""
        obs, info = self.env.reset()
        # State now includes passenger states and current_passenger
        self.start_state = tuple(self.env.state)
        self.current_state = self.start_state
        self.is_running = False
        self.solution_path = []
        self.current_step = 0
        
    def load_textures(self):
        """Load and prepare textures for rendering"""
        cell_size = self.env.cell_size
        
        try:
            # Load road textures
            raw_road = pygame.image.load(os.path.join(IMG_URL, "road.jpg"))
            self.road = pygame.transform.scale(raw_road, (cell_size, cell_size))
            
            # Load wall texture (horizontal only)
            raw_wall = pygame.image.load(os.path.join(IMG_URL, "wall.png")).convert_alpha()
            self.wall = pygame.transform.scale(raw_wall, (cell_size, cell_size))
            
            # Load mud texture
            raw_mud = pygame.image.load(os.path.join(IMG_URL, "mud.jpg"))
            self.mud = pygame.transform.scale(raw_mud, (cell_size, cell_size))
            
        except pygame.error as e:
            print(f"Error loading textures: {e}")
            self.road = None
            self.wall = None
            self.mud = None
        
        # Load game sprites
        try:
            raw_taxi = pygame.image.load(os.path.join(IMG_URL, "taxi.png"))
            taxi_base = pygame.transform.scale(raw_taxi, (cell_size - 20, cell_size - 20))
            self.taxi_images = {
                0: taxi_base,  # South
                1: pygame.transform.rotate(taxi_base, 180),  # North
                2: pygame.transform.rotate(taxi_base, 270),  # East
                3: pygame.transform.rotate(taxi_base, 90)   # West
            }
        except pygame.error:
            self.taxi_images = None
        
        try:
            raw_dest = pygame.image.load(os.path.join(IMG_URL, "dest.png")).convert_alpha()
            self.dest_image = pygame.transform.scale(raw_dest, (cell_size - 20, cell_size - 20))
        except pygame.error:
            self.dest_image = None
        
        try:
            raw_pass = pygame.image.load(os.path.join(IMG_URL, "passenger.png"))
            self.passenger_image = pygame.transform.scale(raw_pass, (cell_size - 20, cell_size - 20))
        except pygame.error:
            self.passenger_image = None
    
    def create_ui_elements(self):
        """Create UI buttons and info panels"""
        panel_x = self.screen_width - PANEL_WIDTH + 20
        button_width = PANEL_WIDTH - 40
        button_height = 45
        spacing = 12
        
        # Buttons at the BOTTOM
        y_offset = self.screen_height - 420
        
        # Algorithm selection buttons
        self.btn_bfs = ThemedButton(
            "BFS", panel_x, y_offset, button_width, button_height,
            action=lambda: self.select_algorithm("BFS")
        )
        y_offset += button_height + spacing
        
        self.btn_dfs = ThemedButton(
            "DFS", panel_x, y_offset, button_width, button_height,
            action=lambda: self.select_algorithm("DFS")
        )
        y_offset += button_height + spacing
        
        self.btn_astar = ThemedButton(
            "A*", panel_x, y_offset, button_width, button_height,
            action=lambda: self.select_algorithm("A*")
        )
        y_offset += button_height + spacing * 2
        
        # Control buttons
        self.btn_run = ThemedButton(
            "Run", panel_x, y_offset, button_width, button_height,
            action=self.run_search
        )
        y_offset += button_height + spacing
        
        self.btn_pause = ThemedButton(
            "Pause", panel_x, y_offset, button_width, button_height,
            action=self.toggle_pause
        )
        self.pause_button_y = y_offset
        y_offset += button_height + spacing
        
        self.btn_reset = ThemedButton(
            "Reset", panel_x, y_offset, button_width, button_height,
            action=self.reset_scene
        )
        y_offset += button_height + spacing
        
        self.btn_back = ThemedButton(
            "Back to Menu", panel_x, y_offset, button_width, button_height,
            action=self.back_to_menu
        )
        
        self.buttons = [
            self.btn_bfs, self.btn_dfs, self.btn_astar,
            self.btn_run, self.btn_pause, self.btn_reset, self.btn_back
        ]
    
    def select_algorithm(self, algorithm_name):
        """Select search algorithm"""
        self.algorithm_name = algorithm_name
        if algorithm_name == "BFS":
            self.selected_algorithm = BreadthFirstSearch(self.env)
        elif algorithm_name == "DFS":
            self.selected_algorithm = DepthFirstSearch(self.env)
        elif algorithm_name == "A*":
            self.selected_algorithm = AStarSearch(self.env)
        print(f"Selected algorithm: {algorithm_name}")
    
    def run_search(self):
        """Execute the selected search algorithm"""
        if self.selected_algorithm is None:
            print("Please select an algorithm first!")
            return
        
        print(f"\nRunning {self.algorithm_name}...")
        print(f"Initial state: {self.start_state}")
        
        # Reset pause state
        self.is_paused = False
        self.btn_pause.text = "Pause"
        
        # Run the search
        start_time = time.time()
        self.solution_path = self.selected_algorithm.search(self.start_state)
        end_time = time.time()
        
        # Store statistics
        self.solution_time = end_time - start_time
        self.nodes_expanded = self.selected_algorithm.nodes_expanded
        self.path_cost = self.selected_algorithm.path_cost
        self.max_frontier_size = self.selected_algorithm.max_frontier_size
        
        if self.solution_path:
            self.path_length = len(self.solution_path)
            print(f"Solution found! Path length: {self.path_length}, Path cost: {self.path_cost}")
            print(f"Time: {self.solution_time:.3f}s, Nodes expanded: {self.nodes_expanded}, Max frontier size: {self.max_frontier_size}")
            self.is_running = True
            self.current_step = 0
            # Set environment state with new format: [taxi_row, taxi_col, passenger_states..., current_passenger]
            self.env.state = list(self.start_state)
            self.env.passenger_states = list(self.start_state[2:2+self.env.num_passengers])
            self.env.current_passenger = self.start_state[-1]
            self.env.last_action = 0
        else:
            print(f"No solution found! Max frontier size: {self.max_frontier_size}")
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        # Update button text
        if self.is_paused:
            self.btn_pause.text = "Resume"
        else:
            self.btn_pause.text = "Pause"
    
    def reset_scene(self):
        """Reset the entire scene"""
        self.reset_environment()
        self.nodes_expanded = 0
        self.path_length = 0
        self.path_cost = 0
        self.solution_time = 0
        self.max_frontier_size = 0
        self.is_paused = False
        self.btn_pause.text = "Pause"
        self.scroll_offset = 0
        print("Scene reset!")
    
    def back_to_menu(self):
        """Return to main menu"""
        from src.scenes.menu import MenuScene
        self.manager.current_scene = MenuScene(self.manager)
    
    def update(self, event_list):
        """Update scene state and handle events"""
        # Handle UI button clicks and scroll
        for event in event_list:
            for button in self.buttons:
                button.check_click(event)
            
            # Handle mouse wheel for scrolling statistics panel
            if event.type == pygame.MOUSEWHEEL:
                # Scroll up (negative) or down (positive)
                self.scroll_offset -= event.y * 20
                self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))
            
            # Manual control with keyboard
            if event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_DOWN:
                    action = 0
                elif event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_SPACE:
                    current_passenger = self.env.state[-1]
                    action = 5 if current_passenger != -1 else 4
                
                if action is not None and not self.is_running:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    if terminated:
                        print("Goal reached manually!")
                        self.reset_environment()
        
        # Execute solution path
        if self.is_running and not self.is_paused:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_step_time >= self.step_delay:
                if self.current_step < len(self.solution_path):
                    action = self.solution_path[self.current_step]
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self.current_step += 1
                    self.last_step_time = current_time
                    
                    if terminated:
                        print("Goal reached!")
                        self.is_running = False
                else:
                    self.is_running = False
    
    def draw(self):
        """Render the scene"""
        self.screen.fill((40, 40, 40))
        
        # Calculate grid offset to center it
        grid_width = self.env.grid_size * self.env.cell_size
        grid_height = self.env.grid_size * self.env.cell_size
        grid_x = (self.screen_width - PANEL_WIDTH - grid_width) // 2
        grid_y = (self.screen_height - grid_height) // 2
        
        # Draw grid
        self.draw_grid(grid_x, grid_y)
        
        # Draw side panel
        self.draw_side_panel()
        
        # Draw statistics
        self.draw_statistics()
        
        # Draw buttons
        for button in self.buttons:
            button.draw(self.screen)
        
    
    def draw_grid(self, offset_x, offset_y):
        """Draw the taxi environment grid with textures"""
        cell_size = self.env.cell_size
        grid_width = self.env.grid_size * cell_size
        grid_height = self.env.grid_size * cell_size
        
        # Get current state - new format: [taxi_row, taxi_col, passenger_states..., current_passenger]
        taxi_row, taxi_col = self.env.state[0], self.env.state[1]
        passenger_states = self.env.state[2:2+self.env.num_passengers]
        current_passenger = self.env.state[-1]
        
        # Draw cells
        for row in range(self.env.grid_size):
            for col in range(self.env.grid_size):
                x = offset_x + col * cell_size
                y = offset_y + row * cell_size
                
                # Get tile type from grid
                tile_type = self.env.grid[row, col]
                
                if tile_type == Tile.WALL:
                    # Draw road underneath so transparent pixels in the wall PNG
                    # show road rather than the dark background
                    if self.road:
                        self.screen.blit(self.road, (x, y))
                    else:
                        pygame.draw.rect(self.screen, (80, 80, 80), (x, y, cell_size, cell_size))
                    if self.wall:
                        self.screen.blit(self.wall, (x, y))
                    else:
                        pygame.draw.rect(self.screen, (60, 60, 60), (x, y, cell_size, cell_size))
                elif tile_type == Tile.MUD:
                    # Draw mud texture
                    if self.mud:
                        self.screen.blit(self.mud, (x, y))
                    else:
                        pygame.draw.rect(self.screen, (139, 90, 43), (x, y, cell_size, cell_size))
                else:
                    # Draw road texture (Tile.EMPTY)
                    if self.road:
                        self.screen.blit(self.road, (x, y))
                    else:
                        pygame.draw.rect(self.screen, (80, 80, 80), (x, y, cell_size, cell_size))
        
        # Draw grid lines
        for i in range(self.env.grid_size + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen, (100, 100, 100),
                (offset_x + i * cell_size, offset_y),
                (offset_x + i * cell_size, offset_y + grid_height)
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen, (100, 100, 100),
                (offset_x, offset_y + i * cell_size),
                (offset_x + grid_width, offset_y + i * cell_size)
            )
        
        # Draw location markers
        for idx, (l_row, l_col) in self.env.locs.items():
            color = self.env.loc_colors[idx]
            x = offset_x + l_col * cell_size + 5
            y = offset_y + l_row * cell_size + 5
            rect = pygame.Rect(x, y, cell_size - 10, cell_size - 10)
            pygame.draw.rect(self.screen, color, rect, width=3)
        
        # Draw destinations for all undelivered passengers
        for i in range(self.env.num_passengers):
            if passenger_states[i] != 2:  # Not yet delivered
                dest_idx = self.env.destination_locations[i]
                dest_row, dest_col = self.env.locs[dest_idx]
                pass_color = self.env.loc_colors[self.env.passenger_locations[i]]
                if self.dest_image:
                    x = offset_x + dest_col * cell_size + 10
                    y = offset_y + dest_row * cell_size + 10
                    self.screen.blit(self.dest_image, (x, y))
                else:
                    font = pygame.font.Font(None, 28)
                    text = font.render(f"D{i}", True, pass_color)
                    x = offset_x + dest_col * cell_size + cell_size // 2
                    y = offset_y + dest_row * cell_size + cell_size // 2
                    text_rect = text.get_rect(center=(x, y))
                    self.screen.blit(text, text_rect)
        
        # Draw passengers (only those waiting at locations)
        for i in range(self.env.num_passengers):
            if passenger_states[i] == 0:  # Waiting at location
                pass_idx = self.env.passenger_locations[i]
                p_row, p_col = self.env.locs[pass_idx]
                pass_color = self.env.loc_colors[pass_idx]
                if self.passenger_image:
                    x = offset_x + p_col * cell_size + 10
                    y = offset_y + p_row * cell_size + 10
                    self.screen.blit(self.passenger_image, (x, y))
                else:
                    font = pygame.font.Font(None, 28)
                    text = font.render(f"P{i}", True, pass_color)
                    x = offset_x + p_col * cell_size + cell_size // 2
                    y = offset_y + p_row * cell_size + cell_size // 2
                    text_rect = text.get_rect(center=(x, y))
                    self.screen.blit(text, text_rect)
        
        # Draw taxi
        if self.taxi_images:
            taxi_img = self.taxi_images.get(self.env.last_action, self.taxi_images[0])
            x = offset_x + taxi_col * cell_size + 10
            y = offset_y + taxi_row * cell_size + 10
            
            # Tint taxi if passenger is inside
            if current_passenger != -1:
                tinted = taxi_img.copy()
                tinted.fill((0, 200, 0, 100), special_flags=pygame.BLEND_RGBA_MULT)
                self.screen.blit(tinted, (x, y))
            else:
                self.screen.blit(taxi_img, (x, y))
        else:
            # Draw simple rectangle
            color = (255, 200, 0) if current_passenger == -1 else (0, 200, 0)
            x = offset_x + taxi_col * cell_size + 10
            y = offset_y + taxi_row * cell_size + 10
            rect = pygame.Rect(x, y, cell_size - 20, cell_size - 20)
            pygame.draw.rect(self.screen, color, rect)
    
    def should_use_vertical_wall(self, row, col):
        """
        Determine if wall should use vertical texture based on neighbors
        This creates more natural-looking wall patterns
        """
        # Check if there are walls above/below (vertical alignment)
        has_vertical_neighbor = False
        if row > 0 and self.env.grid[row - 1, col] == Tile.WALL:
            has_vertical_neighbor = True
        if row < self.env.grid_size - 1 and self.env.grid[row + 1, col] == Tile.WALL:
            has_vertical_neighbor = True
        
        # Check if there are walls left/right (horizontal alignment)
        has_horizontal_neighbor = False
        if col > 0 and self.env.grid[row, col - 1] == Tile.WALL:
            has_horizontal_neighbor = True
        if col < self.env.grid_size - 1 and self.env.grid[row, col + 1] == Tile.WALL:
            has_horizontal_neighbor = True
        
        # Use vertical texture if walls align vertically
        if has_vertical_neighbor and not has_horizontal_neighbor:
            return True
        
        # Default to horizontal, or alternate for isolated walls
        return (row + col) % 2 == 1
    
    def draw_side_panel(self):
        """Draw the side panel background"""
        panel_rect = pygame.Rect(
            self.screen_width - PANEL_WIDTH, 0,
            PANEL_WIDTH, self.screen_height
        )
        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        pygame.draw.line(
            self.screen, (100, 100, 100),
            (self.screen_width - PANEL_WIDTH, 0),
            (self.screen_width - PANEL_WIDTH, self.screen_height),
            2
        )
    
    def draw_statistics(self):
        """Draw algorithm statistics and passenger/destination info"""
        panel_x = self.screen_width - PANEL_WIDTH + 20
        y_start = 20  # Start at top of panel
        
        # Create a surface for scrollable content
        stats_area_height = self.screen_height - 440  # Space between top and buttons
        stats_surface = pygame.Surface((PANEL_WIDTH - 40, 2000), pygame.SRCALPHA)  # Large surface for content
        stats_surface.fill((0, 0, 0, 0))
        
        font = pygame.font.Font(None, 30)
        font_small = pygame.font.Font(None, 26)
        font_tiny = pygame.font.Font(None, 22)
        
        # Draw on the stats_surface instead of screen
        surf_y = 0
        
        # Title
        title_text = font.render("Game Info", True, (100, 200, 255))
        stats_surface.blit(title_text, (0, surf_y))
        surf_y += 30
        
        # Level info
        level_text = font_small.render(f"Level {self.level} ({self.env.grid_size}x{self.env.grid_size})", True, (200, 200, 200))
        stats_surface.blit(level_text, (0, surf_y))
        surf_y += 22

        # Taxi start position (static)
        taxi_start = (self.start_state[0], self.start_state[1])
        taxi_text = font_small.render(f"Taxi start: {taxi_start}", True, (200, 200, 200))
        stats_surface.blit(taxi_text, (0, surf_y))
        surf_y += 25

        # Count delivered passengers
        delivered_count = sum(1 for ps in self.env.passenger_states if ps == 2)
        tasks_text = font_small.render(f"Delivered: {delivered_count}/{self.env.num_passengers}", True, (200, 200, 200))
        stats_surface.blit(tasks_text, (0, surf_y))
        surf_y += 30
        
        # Passenger & Destination Locations
        separator = font_small.render("--- Passengers ---", True, (150, 150, 255))
        stats_surface.blit(separator, (0, surf_y))
        surf_y += 25
        
        for i in range(self.env.num_passengers):
            status = self.env.passenger_states[i]
            pass_loc_idx = self.env.passenger_locations[i]
            dest_loc_idx = self.env.destination_locations[i]
            pass_pos = self.env.locs[pass_loc_idx]
            dest_pos = self.env.locs[dest_loc_idx]
            pass_color = self.env.loc_colors[pass_loc_idx]
            
            # Status indicator
            status_str = "Waiting" if status == 0 else ("In Taxi" if status == 1 else "Delivered")
            status_color = (255, 200, 0) if status == 0 else ((0, 255, 0) if status == 1 else (100, 100, 100))
            
            # Passenger info
            pass_text = font_tiny.render(f"P{i}: {status_str}", True, status_color)
            stats_surface.blit(pass_text, (0, surf_y))
            surf_y += 18
            
            loc_text = font_tiny.render(f"  From: ({pass_pos[0]}, {pass_pos[1]})", True, (180, 180, 180))
            stats_surface.blit(loc_text, (0, surf_y))
            surf_y += 16
            
            dest_text = font_tiny.render(f"  To: ({dest_pos[0]}, {dest_pos[1]})", True, (180, 180, 180))
            stats_surface.blit(dest_text, (0, surf_y))
            surf_y += 20
        
        surf_y += 10
        
        # Algorithm Stats
        separator2 = font_small.render("--- Algorithm ---", True, (150, 150, 255))
        stats_surface.blit(separator2, (0, surf_y))
        surf_y += 25
        
        stats_lines = [
            f"Algorithm: {self.algorithm_name}",
            f"Nodes Expanded: {self.nodes_expanded}",
            f"Max Frontier Size: {self.max_frontier_size}",
            f"Path Length: {self.path_length}",
            f"Path Cost: {self.path_cost}",
            f"Time: {self.solution_time:.3f}s",
            f"Step: {self.current_step}/{len(self.solution_path) if self.solution_path else 0}"
        ]
        
        for line in stats_lines:
            text = font_tiny.render(line, True, (220, 220, 220))
            stats_surface.blit(text, (0, surf_y))
            surf_y += 18
        
        # Calculate max scroll based on content height
        self.max_scroll = max(0, surf_y - stats_area_height)
        
        # Blit the scrollable surface to screen with offset
        stats_rect = pygame.Rect(0, self.scroll_offset, PANEL_WIDTH - 40, stats_area_height)
        self.screen.blit(stats_surface, (panel_x, y_start), stats_rect)
        
        # Draw scroll indicator if content is scrollable
        if self.max_scroll > 0:
            scroll_bar_x = self.screen_width - 15
            scroll_bar_height = stats_area_height
            scroll_bar_y = y_start
            
            # Background track
            pygame.draw.rect(self.screen, (60, 60, 60), (scroll_bar_x, scroll_bar_y, 10, scroll_bar_height))
            
            # Scrollbar thumb
            thumb_height = max(20, int(scroll_bar_height * (stats_area_height / surf_y)))
            thumb_y = scroll_bar_y + int((self.scroll_offset / self.max_scroll) * (scroll_bar_height - thumb_height))
            pygame.draw.rect(self.screen, (150, 150, 255), (scroll_bar_x, thumb_y, 10, thumb_height), border_radius=5)
