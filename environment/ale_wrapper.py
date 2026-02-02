import gymnasium as gym
import collections
import heapq
import numpy as np
import copy
from typing import List, Tuple

class TaxiStateExtractor:
    """Extract state information from Taxi-v3 environment"""
    def __init__(self, env):
        self.env = env
    
    def extract_state(self, state):
        """
        Convert Taxi state encoding to components
        Taxi-v3 state encoding: taxi_row, taxi_col, passenger_location, destination
        State is a single integer that encodes: 5*5*5*4 = 500 possible states
        
        Decoding:
        - Taxi row: state // 100
        - Taxi col: (state // 20) % 5
        - Passenger location: (state // 4) % 5 (0-3: locations, 4: in taxi)
        - Destination: state % 4
        """
        taxi_row, taxi_col, passenger_loc, destination = tuple(self.env.unwrapped.decode(state))
        
        return {
            'taxi_row': taxi_row,
            'taxi_col': taxi_col,
            'passenger_loc': passenger_loc,  # 0-3: at location, 4: in taxi
            'destination': destination,  # 0: R, 1: G, 2: Y, 3: B
        }


class TaxiWrapper(gym.Wrapper):
    """
    Wrapper for Taxi-v3 environment to support search algorithms.
    
    Taxi-v3 Goal: Pick up passenger and drop them at destination
    
    Actions:
        0: Move South
        1: Move North
        2: Move East
        3: Move West
        4: Pickup passenger
        5: Dropoff passenger
    
    Locations (coordinates):
        R (Red):    (0, 0)
        G (Green):  (0, 4)
        Y (Yellow): (4, 0)
        B (Blue):   (4, 3)
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.state_extractor = TaxiStateExtractor(env)
        
        # Action space for Taxi
        self.available_actions = [0, 1, 2, 3, 4, 5]
        
        # Coordinates mapping (for passenger locations and destinations)
        self.locations = {
            0: (0, 0),  # R (Red)
            1: (0, 4),  # G (Green)
            2: (4, 0),  # Y (Yellow)
            3: (4, 3),  # B (Blue)
        }
    
    def reset(self, **kwargs):
        """Reset environment and return state"""
        state, info = self.env.reset(**kwargs)
        return state, info
    
    def step(self, action):
        """Execute action and return new state"""
        state, reward, terminated, truncated, info = self.env.step(action)
        return state, reward, terminated, truncated, info
    
    def get_symbolic_state(self, state):
        """
        Extract state components for search algorithms.
        Returns state as tuple for hashing in visited sets.
        """
        state_dict = self.state_extractor.extract_state(state)
        return (state_dict['taxi_row'], state_dict['taxi_col'], 
                state_dict['passenger_loc'], state_dict['destination'])
    
    def is_goal_state(self, state):
        """
        Check if state is a goal state.
        Goal: Passenger delivered to destination
        
        Args:
            state: Can be either int (encoded state) or tuple (taxi_row, taxi_col, passenger_loc, destination)
        """
        if isinstance(state, int):
            state = self.get_symbolic_state(state)
        
        taxi_row, taxi_col, passenger_loc, destination = state
        
        # Goal achieved when:
        # 1. Passenger is not in taxi (passenger_loc != 4)
        # 2. Passenger location matches destination
        # After successful dropoff, passenger_loc becomes the destination index
        
        # In Taxi-v3, after successful dropoff, the episode terminates
        # So goal is when passenger was picked up (loc=4) and taxi is at destination
        dest_row, dest_col = self.locations[destination]
        
        return (passenger_loc == destination and taxi_row == dest_row and taxi_col == dest_col)
    def _calculate_cost(self, current_state, action, next_state):
        """
        Calculate cost of taking an action from current_state to next_state.
        
        Args:
            current_state: int (encoded state) or tuple
            action: action taken
            next_state: int (encoded state) or tuple
        Returns:
            cost: int
        """
        if action in [0, 1, 2, 3]:  # Movement actions
            return 1 #if self._non_legit_move(*self.get_symbolic_state(current_state)[:2], action) == False else 10
        elif action == 4:  # Pickup
            taxi_row, taxi_col, passenger_loc, destination = self.get_symbolic_state(current_state)
            if passenger_loc < 4:
                pass_row, pass_col = self.locations[passenger_loc]
                if taxi_row == pass_row and taxi_col == pass_col:
                    return 1  # Successful pickup
                else:
                    return 10  # Illegal pickup
            else:
                return 10  # Illegal pickup (passenger already in taxi)
        
        taxi_row, taxi_col, passenger_loc, destination = self.get_symbolic_state(current_state)
        dest_row, dest_col = self.locations[destination]
        if passenger_loc == 4:
            if taxi_row == dest_row and taxi_col == dest_col:
                return -20  # Successful dropoff (negative cost as reward)
            else:
                return 10  # Illegal dropoff
        else:
            return 10  # Illegal dropoff (no passenger in taxi)
        
    def get_successors(self, state) -> List[Tuple[int, Tuple, float]]:
        """
        Get possible successor states from current state.
        
        Args:
            state: int (encoded state) or tuple
        
        Returns:
            List of (action, next_state, cost) tuples
        """
        if isinstance(state, tuple):
            # Convert tuple back to encoded state for simulation
            taxi_row, taxi_col, passenger_loc, destination = state
            encoded_state = taxi_row * 100 + taxi_col * 20 + passenger_loc * 4 + destination
        else:
            encoded_state = state
        
        successors = []
        
        # Try each action
        for action in self.available_actions:
            next_state = self._simulate_action(encoded_state, action)
            cost = self._calculate_cost(encoded_state, action, next_state)
            successors.append((action, next_state, cost))
        
        return successors
    
    def _non_legit_move(self, taxi_row, taxi_col, action) -> bool:
        """
        Check if an action is illegal in the Taxi environment.
        
        Args:
            state: int (encoded state) or tuple
            action: action to check
        Returns:
            True if action is illegal, False otherwise
        """
        # Define walls in the Taxi environment
        non_legit_action = {
            2: [(0,1), (1,1), (3,0), (4,0), (3,2), (4,2)],  # East
            3: [(0,2), (1,2), (3,1), (4,1), (3,3), (4,3)],  # West
        }
        if action in non_legit_action:
            if (taxi_row, taxi_col) in non_legit_action[action]:
                return True
        return False
        
    def _simulate_action(self, state, action) -> Tuple:
        """
        Simulate an action in Taxi environment analytically.
        This is approximate - for exact simulation, would need to step through env.
        
        For now, returns the encoded state after basic movement.
        """
        taxi_row, taxi_col, passenger_loc, destination = self.get_symbolic_state(state)
        
        if self._non_legit_move(taxi_row, taxi_col, action):
            # Illegal move, stay in place
            return (taxi_row, taxi_col, passenger_loc, destination)
        
        if action == 0:  # South
            taxi_row = min(4, taxi_row + 1)
        elif action == 1:  # North
            taxi_row = max(0, taxi_row - 1)
        elif action == 2:  # East
            taxi_col = min(4, taxi_col + 1)
        elif action == 3:  # West
            taxi_col = max(0, taxi_col - 1)
        elif action == 4:  # Pickup
            # Check if taxi is at a pickup location and passenger is there
            for loc_idx, (loc_row, loc_col) in self.locations.items():
                if taxi_row == loc_row and taxi_col == loc_col and passenger_loc == loc_idx:
                    passenger_loc = 4  # In taxi
        elif action == 5:  # Dropoff
            # Check if taxi is at destination and passenger is in taxi
            dest_row, dest_col = self.locations[destination]
            if taxi_row == dest_row and taxi_col == dest_col and passenger_loc == 4:
                passenger_loc = destination  # Delivered
        
        return (taxi_row, taxi_col, passenger_loc, destination)
    
    def manhattan_distance(self, state):
        """Manhattan distance heuristic for Taxi"""
        if isinstance(state, int):
            state = self.get_symbolic_state(state)
        
        taxi_row, taxi_col, passenger_loc, destination = state
        dest_row, dest_col = self.locations[destination]
        
        # If passenger in taxi, distance to destination
        if passenger_loc == 4:
            return abs(taxi_row - dest_row) + abs(taxi_col - dest_col)
        
        # Otherwise, distance to passenger + distance from passenger to destination
        pass_row, pass_col = self.locations[passenger_loc]
        dist_to_passenger = abs(taxi_row - pass_row) + abs(taxi_col - pass_col)
        dist_passenger_to_dest = abs(pass_row - dest_row) + abs(pass_col - dest_col)
        
        return dist_to_passenger + dist_passenger_to_dest
    
    def heuristic(self, state):
        """Default heuristic (Manhattan distance)"""
        return self.manhattan_distance(state)
    
    
    